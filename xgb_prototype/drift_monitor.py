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

        return cls(
            reference_numerical=reference_numerical,
            reference_categorical=reference_categorical,
            alpha=alpha,
            persistence=max(1, int(persistence)),
            min_feature_drift_ratio=float(min_feature_drift_ratio),
            retrain_feature_ratio=float(retrain_feature_ratio),
            retrain_severity=str(retrain_severity).lower(),
        )

    @property
    def feature_count(self) -> int:
        return len(self.reference_numerical) + len(self.reference_categorical)

    def check(self, X_new: pd.DataFrame) -> DriftCheckResult:
        """Compare one new data batch against the training reference."""
        self.checks_seen += 1
        drifted: list[str] = []
        pvalues: dict[str, float] = {}

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

        ratio = len(drifted) / max(self.feature_count, 1)
        if ratio >= self.min_feature_drift_ratio:
            self.consecutive_drift_checks += 1
        else:
            self.consecutive_drift_checks = 0

        severity = self._severity(ratio)
        alert = self.consecutive_drift_checks >= self.persistence
        retraining_recommended = alert and (
            ratio >= self.retrain_feature_ratio or self._severity_rank(severity) >= self._severity_rank(self.retrain_severity)
        )
        recommendation = self._recommendation(ratio, severity, alert, retraining_recommended)

        result = DriftCheckResult(
            check_id=self.checks_seen,
            drifted_features=sorted(drifted),
            pvalues=pvalues,
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
