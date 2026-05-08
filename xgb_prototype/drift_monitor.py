"""Continuous drift monitoring against a training reference dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class DriftCheckResult:
    check_id: int
    drifted_features: list[str]
    pvalues: dict[str, float]
    label_drift: dict[str, Any]
    prediction_drift: dict[str, Any]
    data_quality_drift: dict[str, Any]
    serving_training_skew: dict[str, Any]
    segment_drift: dict[str, Any]
    concept_drift: dict[str, Any]
    calibration_drift: dict[str, Any]
    novel_class_emergence: dict[str, Any]
    severity: str
    consecutive_drift_checks: int
    alert: bool
    retraining_recommended: bool
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContinuousDriftMonitor:
    """Monitor persistent feature drift in new batches of data.

    The monitor compares each new batch against training reference distributions.
    It raises an alert only when drift persists for `persistence` consecutive
    checks, reducing noise from a single unusual batch.
    """

    reference_numerical: dict[str, list[float]]
    reference_categorical: dict[str, dict[str, int]]
    reference_missing_rates: dict[str, float] = field(default_factory=dict)
    reference_dtypes: dict[str, str] = field(default_factory=dict)
    reference_labels: dict[str, int] = field(default_factory=dict)
    reference_predictions: list[float] = field(default_factory=list)
    reference_prediction_classes: dict[str, int] = field(default_factory=dict)
    reference_calibration: dict[str, float] = field(default_factory=dict)
    reference_feature_target_links: dict[str, float] = field(default_factory=dict)
    alpha: float = 0.05
    persistence: int = 3
    min_feature_drift_ratio: float = 0.10
    retrain_feature_ratio: float = 0.25
    retrain_severity: str = "high"
    checks_seen: int = 0
    consecutive_drift_checks: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_reference_data(
        cls,
        X_ref: pd.DataFrame,
        num_cols: list[str],
        cat_cols: list[str],
        alpha: float = 0.05,
        persistence: int = 3,
        min_feature_drift_ratio: float = 0.10,
        retrain_feature_ratio: float = 0.25,
        retrain_severity: str = "high",
        max_reference_values: int = 50_000,
        random_state: int = 42,
        y_ref: pd.Series | None = None,
        predictions_ref: np.ndarray | pd.Series | None = None,
        prediction_proba_ref: np.ndarray | pd.Series | None = None,
    ) -> "ContinuousDriftMonitor":
        rng = np.random.default_rng(random_state)
        reference_numerical: dict[str, list[float]] = {}
        for col in num_cols:
            if col not in X_ref.columns:
                continue
            values = X_ref[col].dropna().astype(float).to_numpy()
            if len(values) > max_reference_values:
                idx = rng.choice(len(values), size=max_reference_values, replace=False)
                values = values[idx]
            reference_numerical[col] = values.tolist()

        reference_categorical: dict[str, dict[str, int]] = {}
        for col in cat_cols:
            if col not in X_ref.columns:
                continue
            counts = X_ref[col].dropna().astype(str).value_counts().to_dict()
            reference_categorical[col] = {str(k): int(v) for k, v in counts.items()}

        reference_labels = (
            {str(k): int(v) for k, v in pd.Series(y_ref).dropna().astype(str).value_counts().to_dict().items()}
            if y_ref is not None else {}
        )
        reference_predictions: list[float] = []
        reference_prediction_classes: dict[str, int] = {}
        if prediction_proba_ref is not None:
            arr = np.asarray(prediction_proba_ref)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                arr = arr[:, 1]
            reference_predictions = pd.Series(arr).dropna().astype(float).tolist()
        elif predictions_ref is not None:
            pred_series = pd.Series(predictions_ref).dropna()
            if pd.api.types.is_numeric_dtype(pred_series):
                reference_predictions = pred_series.astype(float).tolist()
            reference_prediction_classes = {
                str(k): int(v) for k, v in pred_series.astype(str).value_counts().to_dict().items()
            }

        reference_calibration = {}
        if y_ref is not None and reference_predictions:
            reference_calibration = _binary_calibration(pd.Series(y_ref), np.asarray(reference_predictions))

        reference_feature_target_links: dict[str, float] = {}
        if y_ref is not None:
            y_num = pd.to_numeric(pd.Series(y_ref), errors="coerce")
            if y_num.notna().sum() > 10:
                for col, values in reference_numerical.items():
                    if col not in X_ref.columns:
                        continue
                    x_num = pd.to_numeric(X_ref[col], errors="coerce")
                    pair = pd.concat([x_num, y_num], axis=1).dropna()
                    if len(pair) > 10 and pair.iloc[:, 0].nunique() > 1:
                        reference_feature_target_links[col] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))

        return cls(
            reference_numerical=reference_numerical,
            reference_categorical=reference_categorical,
            reference_missing_rates={str(c): float(v) for c, v in X_ref.isna().mean().to_dict().items()},
            reference_dtypes={str(c): str(t) for c, t in X_ref.dtypes.to_dict().items()},
            reference_labels=reference_labels,
            reference_predictions=reference_predictions,
            reference_prediction_classes=reference_prediction_classes,
            reference_calibration=reference_calibration,
            reference_feature_target_links=reference_feature_target_links,
            alpha=alpha,
            persistence=max(1, int(persistence)),
            min_feature_drift_ratio=float(min_feature_drift_ratio),
            retrain_feature_ratio=float(retrain_feature_ratio),
            retrain_severity=str(retrain_severity).lower(),
        )

    @property
    def feature_count(self) -> int:
        return len(self.reference_numerical) + len(self.reference_categorical)

    def check(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series | np.ndarray | None = None,
        predictions: pd.Series | np.ndarray | None = None,
        prediction_proba: pd.Series | np.ndarray | None = None,
        segment_cols: list[str] | None = None,
    ) -> DriftCheckResult:
        """Compare one new data batch against the training reference."""
        self.checks_seen += 1
        drifted: list[str] = []
        pvalues: dict[str, float] = {}
        segment_cols = segment_cols or []

        for col, ref_values in self.reference_numerical.items():
            if col not in X_new.columns:
                continue
            new_values = X_new[col].dropna().astype(float).to_numpy()
            if len(ref_values) == 0 or len(new_values) == 0:
                continue
            _, pval = scipy_stats.ks_2samp(np.asarray(ref_values), new_values)
            pvalues[col] = float(pval)
            if pval < self.alpha:
                drifted.append(col)

        for col, ref_counts in self.reference_categorical.items():
            if col not in X_new.columns:
                continue
            new_counts_dict = X_new[col].dropna().astype(str).value_counts().to_dict()
            all_cats = sorted(set(ref_counts) | set(new_counts_dict))
            if len(all_cats) < 2:
                continue
            ref_counts_arr = np.array([ref_counts.get(cat, 0) for cat in all_cats], dtype=float)
            new_counts_arr = np.array([new_counts_dict.get(cat, 0) for cat in all_cats], dtype=float)
            if ref_counts_arr.sum() == 0 or new_counts_arr.sum() == 0:
                continue
            ref_smoothed = ref_counts_arr + 1e-6
            expected = ref_smoothed / ref_smoothed.sum() * new_counts_arr.sum()
            if len(expected) < 2:
                continue
            _, pval = scipy_stats.chisquare(new_counts_arr, f_exp=expected)
            pvalues[col] = float(pval)
            if pval < self.alpha:
                drifted.append(col)

        label_drift = self._label_drift(y_new)
        prediction_drift = self._prediction_drift(predictions, prediction_proba)
        data_quality_drift = self._data_quality_drift(X_new)
        serving_training_skew = self._serving_training_skew(X_new)
        segment_drift = self._segment_drift(X_new, segment_cols)
        concept_drift = self._concept_drift(X_new, y_new)
        calibration_drift = self._calibration_drift(y_new, prediction_proba)
        novel_class_emergence = self._novel_class_emergence(y_new)

        ratio = len(drifted) / max(self.feature_count, 1)
        extra_drift_channels = [
            label_drift.get("drift_detected"),
            prediction_drift.get("drift_detected"),
            data_quality_drift.get("drift_detected"),
            serving_training_skew.get("skew_detected"),
            bool(segment_drift.get("drifted_segments")),
            concept_drift.get("drift_detected"),
            calibration_drift.get("drift_detected"),
            bool(novel_class_emergence.get("novel_classes")),
        ]
        if ratio >= self.min_feature_drift_ratio:
            self.consecutive_drift_checks += 1
        elif any(extra_drift_channels):
            self.consecutive_drift_checks += 1
        else:
            self.consecutive_drift_checks = 0

        severity = self._overall_severity(ratio, extra_drift_channels)
        alert = self.consecutive_drift_checks >= self.persistence
        retraining_recommended = alert and (
            ratio >= self.retrain_feature_ratio or self._severity_rank(severity) >= self._severity_rank(self.retrain_severity)
        )
        recommendation = self._recommendation(ratio, severity, alert, retraining_recommended)

        result = DriftCheckResult(
            check_id=self.checks_seen,
            drifted_features=sorted(drifted),
            pvalues=pvalues,
            label_drift=label_drift,
            prediction_drift=prediction_drift,
            data_quality_drift=data_quality_drift,
            serving_training_skew=serving_training_skew,
            segment_drift=segment_drift,
            concept_drift=concept_drift,
            calibration_drift=calibration_drift,
            novel_class_emergence=novel_class_emergence,
            severity=severity,
            consecutive_drift_checks=self.consecutive_drift_checks,
            alert=alert,
            retraining_recommended=retraining_recommended,
            recommendation=recommendation,
        )
        self.history.append(result.to_dict())
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuousDriftMonitor":
        return cls(**data)

    def _severity(self, ratio: float) -> str:
        if ratio >= self.retrain_feature_ratio:
            return "high"
        if ratio >= self.min_feature_drift_ratio:
            return "medium"
        if ratio > 0:
            return "low"
        return "none"

    def _overall_severity(self, feature_ratio: float, channel_flags: list[bool]) -> str:
        base = self._severity(feature_ratio)
        if self._severity_rank(base) >= 2:
            return base
        active_channels = sum(bool(v) for v in channel_flags)
        if active_channels >= 3:
            return "high"
        if active_channels >= 1:
            return "medium"
        return base

    def _label_drift(self, y_new: pd.Series | np.ndarray | None) -> dict[str, Any]:
        if y_new is None or not self.reference_labels:
            return {"checked": False}
        return _chisquare_counts(self.reference_labels, pd.Series(y_new).dropna().astype(str).value_counts().to_dict(), self.alpha)

    def _prediction_drift(
        self,
        predictions: pd.Series | np.ndarray | None,
        prediction_proba: pd.Series | np.ndarray | None,
    ) -> dict[str, Any]:
        if prediction_proba is not None and self.reference_predictions:
            arr = np.asarray(prediction_proba)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                arr = arr[:, 1]
            arr = pd.Series(arr).dropna().astype(float).to_numpy()
            if len(arr) and self.reference_predictions:
                _, pval = scipy_stats.ks_2samp(np.asarray(self.reference_predictions), arr)
                return {"checked": True, "type": "probability", "pvalue": float(pval), "drift_detected": bool(pval < self.alpha)}
        if predictions is not None and self.reference_prediction_classes:
            new_counts = pd.Series(predictions).dropna().astype(str).value_counts().to_dict()
            out = _chisquare_counts(self.reference_prediction_classes, new_counts, self.alpha)
            out["type"] = "class"
            return out
        return {"checked": False}

    def _data_quality_drift(self, X_new: pd.DataFrame) -> dict[str, Any]:
        changed = {}
        new_missing = X_new.isna().mean().to_dict()
        for col, ref_rate in self.reference_missing_rates.items():
            if col not in X_new.columns:
                continue
            delta = float(new_missing.get(col, 0.0) - ref_rate)
            if abs(delta) >= max(0.05, self.alpha):
                changed[col] = {"reference": ref_rate, "new": float(new_missing.get(col, 0.0)), "delta": delta}
        all_null = [str(c) for c in X_new.columns if X_new[c].isna().all()]
        return {"checked": True, "drift_detected": bool(changed or all_null), "missing_rate_changes": changed, "all_null_columns": all_null}

    def _serving_training_skew(self, X_new: pd.DataFrame) -> dict[str, Any]:
        ref_cols = set(self.reference_dtypes)
        new_cols = set(map(str, X_new.columns))
        dtype_changes = {}
        for col in sorted(ref_cols & new_cols):
            new_dtype = str(X_new[col].dtype)
            if self.reference_dtypes[col] != new_dtype:
                dtype_changes[col] = {"reference": self.reference_dtypes[col], "new": new_dtype}
        return {
            "checked": True,
            "skew_detected": bool(ref_cols - new_cols or new_cols - ref_cols or dtype_changes),
            "missing_columns": sorted(ref_cols - new_cols),
            "extra_columns": sorted(new_cols - ref_cols),
            "dtype_changes": dtype_changes,
        }

    def _segment_drift(self, X_new: pd.DataFrame, segment_cols: list[str]) -> dict[str, Any]:
        drifted_segments = []
        for seg_col in segment_cols:
            if seg_col not in X_new.columns:
                continue
            for seg_value, batch in X_new.groupby(seg_col, dropna=True):
                if len(batch) < 20:
                    continue
                drifted_features = []
                for col, ref_values in self.reference_numerical.items():
                    if col not in batch.columns or not ref_values:
                        continue
                    new_values = batch[col].dropna().astype(float).to_numpy()
                    if len(new_values) < 10:
                        continue
                    _, pval = scipy_stats.ks_2samp(np.asarray(ref_values), new_values)
                    if pval < self.alpha:
                        drifted_features.append({"feature": col, "pvalue": float(pval)})
                if drifted_features:
                    drifted_segments.append({"segment_column": seg_col, "segment_value": str(seg_value), "drifted_features": drifted_features[:10]})
        return {"checked": bool(segment_cols), "drifted_segments": drifted_segments}

    def _concept_drift(self, X_new: pd.DataFrame, y_new: pd.Series | np.ndarray | None) -> dict[str, Any]:
        if y_new is None or not self.reference_feature_target_links:
            return {"checked": False, "reason": "requires labels and reference feature-target links"}
        y_num = pd.to_numeric(pd.Series(y_new), errors="coerce").reset_index(drop=True)
        changed = {}
        for col, ref_corr in self.reference_feature_target_links.items():
            if col not in X_new.columns:
                continue
            x_num = pd.to_numeric(X_new[col], errors="coerce").reset_index(drop=True)
            pair = pd.concat([x_num, y_num], axis=1).dropna()
            if len(pair) < 20 or pair.iloc[:, 0].nunique() <= 1:
                continue
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            delta = abs(corr - ref_corr)
            if delta >= 0.20:
                changed[col] = {"reference_corr": ref_corr, "new_corr": corr, "abs_delta": delta}
        return {"checked": True, "drift_detected": bool(changed), "changed_feature_target_links": changed}

    def _calibration_drift(
        self,
        y_new: pd.Series | np.ndarray | None,
        prediction_proba: pd.Series | np.ndarray | None,
    ) -> dict[str, Any]:
        if y_new is None or prediction_proba is None or not self.reference_calibration:
            return {"checked": False}
        arr = np.asarray(prediction_proba)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr[:, 1]
        cal = _binary_calibration(pd.Series(y_new), arr)
        ece_delta = cal.get("ece", 0.0) - self.reference_calibration.get("ece", 0.0)
        brier_delta = cal.get("brier_score", 0.0) - self.reference_calibration.get("brier_score", 0.0)
        return {
            "checked": True,
            "drift_detected": bool(ece_delta > 0.05 or brier_delta > 0.05),
            "reference": self.reference_calibration,
            "new": cal,
            "ece_delta": float(ece_delta),
            "brier_score_delta": float(brier_delta),
        }

    def _novel_class_emergence(self, y_new: pd.Series | np.ndarray | None) -> dict[str, Any]:
        if y_new is None or not self.reference_labels:
            return {"checked": False}
        new_labels = set(pd.Series(y_new).dropna().astype(str).unique())
        ref_labels = set(self.reference_labels)
        return {"checked": True, "novel_classes": sorted(new_labels - ref_labels)}

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(str(severity).lower(), 3)

    def _recommendation(
        self,
        ratio: float,
        severity: str,
        alert: bool,
        retraining_recommended: bool,
    ) -> str:
        pct = ratio * 100
        if retraining_recommended:
            return (
                f"Retraining recommended: {pct:.1f}% of monitored features are drifting "
                f"with persistent {severity} severity."
            )
        if alert:
            return (
                f"Persistent drift alert: {pct:.1f}% of monitored features drifted for "
                f"{self.consecutive_drift_checks} consecutive checks. Increase review frequency."
            )
        if severity != "none":
            return (
                f"Drift observed in {pct:.1f}% of monitored features. Waiting for "
                f"{self.persistence} consecutive checks before alerting."
            )
        return "No drift action recommended."


def _chisquare_counts(ref_counts: dict[str, int], new_counts_raw: dict[Any, int], alpha: float) -> dict[str, Any]:
    new_counts = {str(k): int(v) for k, v in new_counts_raw.items()}
    all_keys = sorted(set(ref_counts) | set(new_counts))
    if len(all_keys) < 2:
        return {"checked": True, "drift_detected": False, "pvalue": None, "reason": "single category"}
    ref = np.array([ref_counts.get(k, 0) for k in all_keys], dtype=float)
    new = np.array([new_counts.get(k, 0) for k in all_keys], dtype=float)
    if ref.sum() == 0 or new.sum() == 0:
        return {"checked": True, "drift_detected": False, "pvalue": None, "reason": "empty distribution"}
    expected = (ref + 1e-6) / (ref + 1e-6).sum() * new.sum()
    _, pval = scipy_stats.chisquare(new, f_exp=expected)
    return {
        "checked": True,
        "drift_detected": bool(pval < alpha),
        "pvalue": float(pval),
        "reference_distribution": {k: int(v) for k, v in zip(all_keys, ref)},
        "new_distribution": {k: int(v) for k, v in zip(all_keys, new)},
    }


def _binary_calibration(y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy()
    p = np.asarray(y_proba, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], 0.0, 1.0)
    if len(np.unique(y)) != 2 or len(y) == 0:
        return {}
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        bin_mask = (p >= lo) & (p < hi)
        if not bin_mask.any():
            continue
        ece += bin_mask.mean() * abs(float(y[bin_mask].mean()) - float(p[bin_mask].mean()))
    return {"ece": float(ece), "brier_score": float(np.mean((p - y) ** 2))}
