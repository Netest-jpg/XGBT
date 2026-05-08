"""Uncertainty estimation for trained XGBoost pipelines.

Provides split-conformal prediction for classification/regression and optional
XGBoost quantile regression intervals for regression tasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

log = logging.getLogger(__name__)


@dataclass
class UncertaintyReport:
    enabled: bool
    task: str
    alpha: float
    status: str
    summary: dict[str, Any]
    output_csv: str | None = None
    output_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("nan")
    q = np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)
    return float(np.quantile(scores, min(q, 1.0), method="higher"))


def _classification_sets(
    pipeline,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    calib_proba = pipeline.predict_proba(X_calib)
    test_proba = pipeline.predict_proba(X_test)
    y_calib_arr = np.asarray(y_calib).astype(int)
    nonconformity = 1.0 - calib_proba[np.arange(len(y_calib_arr)), y_calib_arr]
    qhat = _conformal_quantile(nonconformity, alpha)
    threshold = 1.0 - qhat

    rows = []
    set_sizes = []
    for i, row in enumerate(test_proba):
        labels = np.flatnonzero(row >= threshold).astype(int).tolist()
        if not labels:
            labels = [int(np.argmax(row))]
        set_sizes.append(len(labels))
        rows.append({
            "row_id": i,
            "prediction_set": labels,
            "set_size": len(labels),
            "confidence": float(np.max(row)),
            "entropy": float(-(row * np.log(np.clip(row, 1e-12, 1.0))).sum()),
        })
    summary = {
        "method": "split_conformal_classification",
        "qhat": qhat,
        "probability_threshold": float(threshold),
        "mean_prediction_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
    }
    return pd.DataFrame(rows), summary


def _regression_intervals(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
    alpha: float,
    quantile_low: float,
    quantile_high: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_calib = np.asarray(pipeline.predict(X_calib), dtype=float)
    pred_test = np.asarray(pipeline.predict(X_test), dtype=float)
    residual_scores = np.abs(np.asarray(y_calib, dtype=float) - pred_calib)
    qhat = _conformal_quantile(residual_scores, alpha)

    out = pd.DataFrame({
        "row_id": np.arange(len(X_test)),
        "prediction": pred_test,
        "conformal_lower": pred_test - qhat,
        "conformal_upper": pred_test + qhat,
        "conformal_radius": qhat,
    })
    summary: dict[str, Any] = {
        "method": "split_conformal_regression",
        "qhat": qhat,
    }

    if y_test is not None:
        y_test_arr = np.asarray(y_test, dtype=float)
        covered = (y_test_arr >= out["conformal_lower"].to_numpy()) & (y_test_arr <= out["conformal_upper"].to_numpy())
        summary["conformal_empirical_coverage"] = float(np.mean(covered))

    try:
        low_pipe = clone(pipeline)
        high_pipe = clone(pipeline)
        for p, q in ((low_pipe, quantile_low), (high_pipe, quantile_high)):
            model = p.named_steps["model"]
            model.set_params(objective="reg:quantileerror", quantile_alpha=float(q))
            p.fit(X_train, y_train)
        out["quantile_lower"] = np.asarray(low_pipe.predict(X_test), dtype=float)
        out["quantile_upper"] = np.asarray(high_pipe.predict(X_test), dtype=float)
        summary["quantile_regression"] = {
            "status": "completed",
            "lower_alpha": quantile_low,
            "upper_alpha": quantile_high,
        }
    except Exception as exc:
        summary["quantile_regression"] = {
            "status": "skipped",
            "reason": str(exc),
        }
        log.warning("[uncertainty] Quantile regression skipped: %s", exc)

    return out, summary


def estimate_uncertainty(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
    task: str,
    output_dir: Path,
    run_id: str,
    alpha: float = 0.10,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    enabled: bool = True,
) -> UncertaintyReport:
    """Estimate and persist uncertainty diagnostics for a fitted pipeline."""
    if not enabled:
        return UncertaintyReport(False, task, alpha, "disabled", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if task == "classification":
            df, summary = _classification_sets(pipeline, X_calib, y_calib, X_test, alpha)
        else:
            df, summary = _regression_intervals(
                pipeline, X_train, y_train, X_calib, y_calib, X_test, y_test,
                alpha, quantile_low, quantile_high,
            )
    except Exception as exc:
        log.warning("[uncertainty] Estimation failed: %s", exc)
        return UncertaintyReport(True, task, alpha, "failed", {"reason": str(exc)})

    csv_path = output_dir / f"uncertainty_{run_id}.csv"
    json_path = output_dir / f"uncertainty_{run_id}.json"
    df.to_csv(csv_path, index=False)
    report = UncertaintyReport(True, task, alpha, "completed", summary, str(csv_path), str(json_path))
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    log.info("[uncertainty] Report saved → %s", csv_path)
    return report
