"""
inference.py — Inference wrappers for the XGBoost prototype.

PredictWrapper  : thin wrapper for local/notebook use (F8, U7, U30).
ModelServer     : production-grade wrapper with input validation,
                  standardised JSON response envelope, and friendly
                  error messages (V3).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# INFERENCE WRAPPER (FIX 8 + UPGRADE 7 + 30)
# ─────────────────────────────────────────────

class PredictWrapper:
    """Thin inference wrapper: LabelEncoder decode (F8), unseen-cat warnings (U7), log-untransform (U30).

    Usage:  artifact = joblib.load(path); w = PredictWrapper(artifact)
            w.predict(df) / w.predict_proba(df) / w.predict_with_threshold(df)
    """

    def __init__(self, artifact: dict) -> None:
        self.pipeline        = artifact["pipeline"]
        self.le              = artifact.get("label_encoder")
        self.threshold       = artifact.get("best_threshold", 0.5)
        self.metric          = artifact.get("metric")
        self.ohe_cat_cols    = artifact.get("ohe_cat_cols", artifact.get("cat_cols", []))
        self.te_cat_cols     = artifact.get("te_cat_cols", [])
        self.cat_cols        = self.ohe_cat_cols + self.te_cat_cols
        self._known_categories: dict[str, set] = artifact.get("known_categories", {})
        self._log_transformed: bool = artifact.get("log_transformed", False)
        self._drift_monitor = artifact.get("drift_monitor")

    def _warn_unseen(self, X: "pd.DataFrame") -> None:
        for col in self.cat_cols:
            if col not in X.columns:
                continue
            known    = self._known_categories.get(col, set())
            seen_now = set(X[col].dropna().unique())
            unseen   = seen_now - known
            if unseen:
                log.warning(
                    "Unseen categories in '%s' (handle_unknown='ignore' → OHE zeros): %s",
                    col, sorted(unseen),
                )

    def predict(self, X: "pd.DataFrame") -> np.ndarray:
        self._warn_unseen(X)
        raw = self.pipeline.predict(X)
        if self._log_transformed:
            raw = np.expm1(raw)
        return self.le.inverse_transform(raw.astype(int)) if self.le is not None else raw

    def predict_proba(self, X: "pd.DataFrame") -> np.ndarray:
        self._warn_unseen(X)
        return self.pipeline.predict_proba(X)

    def predict_with_threshold(
        self, X: "pd.DataFrame", threshold: float | None = None
    ) -> np.ndarray:
        """Apply a custom probability threshold (binary classification only)."""
        self._warn_unseen(X)
        t    = threshold if threshold is not None else self.threshold
        prob = self.pipeline.predict_proba(X)[:, 1]
        raw  = (prob >= t).astype(int)
        return self.le.inverse_transform(raw) if self.le is not None else raw

    def check_drift(
        self,
        X: "pd.DataFrame",
        y: np.ndarray | None = None,
        segment_cols: list[str] | None = None,
    ) -> dict:
        """Run the persisted drift monitor against an inference batch."""
        if self._drift_monitor is None:
            return {"checked": False, "reason": "artifact has no drift_monitor"}
        self._warn_unseen(X)
        predictions = self.pipeline.predict(X)
        proba = self.pipeline.predict_proba(X) if hasattr(self.pipeline, "predict_proba") else None
        return self._drift_monitor.check(
            X,
            y_new=y,
            predictions=predictions,
            prediction_proba=proba,
            segment_cols=segment_cols,
        ).to_dict()


# ─────────────────────────────────────────────
# V3 — ModelServer: production-grade inference wrapper
# ─────────────────────────────────────────────

class ModelServer:
    """
    V3: Production inference wrapper around PredictWrapper.

    Adds three things the raw PredictWrapper lacks for API deployment:

    1. Input validation
       ─────────────────
       • Checks that all expected feature columns are present in the request.
       • Reports *exactly* which columns are missing or extra so callers can
         fix their payloads without guesswork.
       • Validates that the dataframe contains at least one row.

    2. Standardised JSON response envelope
       ──────────────────────────────────────
       Every call to predict() / predict_proba() returns a dict (JSON-ready) of
       the form:

       {
         "model_version":    "<timestamp>_<run_id>",
         "task":             "classification" | "regression",
         "metric":           "auprc" | "roc_auc" | "macro_f1" | "r2" | ...,
         "threshold":        0.3142,               # classification only
         "n_rows":           128,
         "predictions":      [0, 1, 0, ...],       # decoded labels (classification)
                                                   # or float values (regression)
         "probabilities":    [[0.91, 0.09], ...],  # classification only; null otherwise
         "positive_proba":   [0.09, ...],          # binary classification only; null otherwise
         "warnings":         []                    # non-fatal issues (unseen cats, etc.)
       }

    3. Friendly error messages
       ──────────────────────────
       All exceptions are caught and re-raised as ValueError with a message that
       names the caller's mistake (missing column, wrong dtype, empty dataframe)
       rather than a raw sklearn/numpy traceback.

    Usage
    ─────
    .. code-block:: python

        import joblib
        from inference import ModelServer

        artifact = joblib.load("models/model_20240101_abc123.joblib")
        server   = ModelServer(artifact)

        # From a dict (e.g. a parsed JSON POST body)
        response = server.predict({"V1": [-1.36], "V2": [0.22], ...})

        # From a DataFrame
        response = server.predict(new_df)

        # Direct JSON string
        json_str = server.predict_json(new_df)
    """

    # Keys that are persisted inside the joblib artifact
    _ARTIFACT_KEYS = (
        "pipeline", "num_cols", "ohe_cat_cols", "te_cat_cols",
        "metric", "label_encoder", "best_threshold",
        "run_id", "timestamp", "task", "log_transformed",
        "known_categories", "eval_metrics",
    )

    def __init__(self, artifact: dict) -> None:
        """
        Parameters
        ──────────
        artifact : dict
            The dict loaded from a joblib artifact produced by this training script.
        """
        self._wrapper      = PredictWrapper(artifact)
        self._run_id       = artifact.get("run_id", "unknown")
        self._timestamp    = artifact.get("timestamp", "unknown")
        self._task         = artifact.get("task", "classification")
        self._metric       = artifact.get("metric")
        self._eval_metrics = artifact.get("eval_metrics", {})
        self._log_tf       = artifact.get("log_transformed", False)
        self._drift_monitor = artifact.get("drift_monitor")

        # Expected feature columns (original, pre-preprocessor)
        num_cols     = artifact.get("num_cols", [])
        ohe_cat_cols = artifact.get("ohe_cat_cols", artifact.get("cat_cols", []))
        te_cat_cols  = artifact.get("te_cat_cols", [])
        self._expected_cols: list[str] = num_cols + ohe_cat_cols + te_cat_cols

        self._model_version = f"{self._timestamp}_{self._run_id}"

        log.info(
            "[ModelServer] Initialised — version=%s, task=%s, features=%d",
            self._model_version, self._task, len(self._expected_cols),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        data: "pd.DataFrame | dict",
        threshold: float | None = None,
    ) -> dict:
        """
        Run inference and return a standardised response dict.

        Parameters
        ──────────
        data      : pd.DataFrame or dict (column → list of values)
        threshold : override the model's tuned threshold (binary classification)

        Returns
        ───────
        dict — see class docstring for the full schema.
        """
        df, warnings = self._prepare(data)
        threshold_used = threshold if threshold is not None else self._wrapper.threshold

        try:
            if self._task == "classification":
                raw_preds = self._wrapper.predict_with_threshold(df, threshold=threshold_used)
                probas_2d = self._wrapper.predict_proba(df)
                preds_list     = raw_preds.tolist()
                probas_list    = probas_2d.tolist()
                pos_proba_list = (
                    probas_2d[:, 1].tolist() if probas_2d.shape[1] == 2 else None
                )
            else:
                raw_preds      = self._wrapper.predict(df)
                preds_list     = raw_preds.tolist()
                probas_list    = None
                pos_proba_list = None
                threshold_used = None

        except Exception as exc:
            raise ValueError(
                f"[ModelServer] Inference failed: {exc}\n"
                "Check that input dtypes and column names match the training data."
            ) from exc

        return self._envelope(
            n_rows         = len(df),
            predictions    = preds_list,
            probabilities  = probas_list,
            positive_proba = pos_proba_list,
            threshold      = threshold_used,
            warnings       = warnings,
        )

    def predict_json(
        self,
        data: "pd.DataFrame | dict",
        threshold: float | None = None,
        indent: int | None = 2,
    ) -> str:
        """Convenience wrapper — returns the response as a JSON string."""
        return json.dumps(self.predict(data, threshold=threshold), indent=indent)

    def info(self) -> dict:
        """Return model metadata (version, task, metric, eval scores, features)."""
        metric_name = self._metric.name if self._metric is not None else "unknown"
        return {
            "model_version":   self._model_version,
            "task":            self._task,
            "metric":          metric_name,
            "threshold":       self._wrapper.threshold,
            "n_features":      len(self._expected_cols),
            "feature_names":   self._expected_cols,
            "eval_metrics":    self._eval_metrics,
            "log_transformed": self._log_tf,
        }

    def check_drift(
        self,
        data: "pd.DataFrame | dict",
        y: list | np.ndarray | None = None,
        segment_cols: list[str] | None = None,
    ) -> dict:
        """Validate input and return production drift/skew diagnostics."""
        if self._drift_monitor is None:
            return {"checked": False, "reason": "artifact has no drift_monitor"}
        df, warnings = self._prepare(data)
        result = self._wrapper.check_drift(df, y=np.asarray(y) if y is not None else None, segment_cols=segment_cols)
        result["warnings"] = warnings
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare(self, data: "pd.DataFrame | dict") -> tuple["pd.DataFrame", list[str]]:
        """Convert input → DataFrame, validate columns, collect warnings."""
        import pandas as _pd

        # ── 1. Coerce to DataFrame ─────────────────────────────────────────
        if isinstance(data, dict):
            try:
                df = _pd.DataFrame(data)
            except Exception as exc:
                raise ValueError(
                    f"[ModelServer] Could not convert dict to DataFrame: {exc}"
                ) from exc
        elif isinstance(data, _pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                f"[ModelServer] 'data' must be a pandas DataFrame or dict, "
                f"got {type(data).__name__}."
            )

        # ── 2. Non-empty check ─────────────────────────────────────────────
        if len(df) == 0:
            raise ValueError("[ModelServer] Input DataFrame is empty (0 rows).")

        warnings: list[str] = []

        # ── 3. Column validation ───────────────────────────────────────────
        if self._expected_cols:
            input_cols   = set(df.columns)
            expected_set = set(self._expected_cols)

            missing = expected_set - input_cols
            extra   = input_cols - expected_set

            if missing:
                raise ValueError(
                    f"[ModelServer] Missing required column(s): "
                    f"{sorted(missing)}.\n"
                    f"Expected features: {self._expected_cols}"
                )
            if extra:
                msg = (
                    f"[ModelServer] Extra column(s) in input will be ignored: "
                    f"{sorted(extra)}."
                )
                log.warning(msg)
                warnings.append(msg)
                df = df[self._expected_cols]

        # ── 4. Unseen-category warnings (delegate to PredictWrapper) ──────
        try:
            self._wrapper._warn_unseen(df)
        except Exception:
            pass  # non-fatal

        return df, warnings

    def _envelope(
        self,
        n_rows: int,
        predictions: list,
        probabilities: list | None,
        positive_proba: list | None,
        threshold: float | None,
        warnings: list[str],
    ) -> dict:
        """Assemble the standardised response dict."""
        metric_name = self._metric.name if self._metric is not None else "unknown"
        return {
            "model_version":  self._model_version,
            "task":           self._task,
            "metric":         metric_name,
            "threshold":      threshold,
            "n_rows":         n_rows,
            "predictions":    predictions,
            "probabilities":  probabilities,
            "positive_proba": positive_proba,
            "warnings":       warnings,
        }
