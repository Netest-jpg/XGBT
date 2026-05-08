"""metrics.py — MetricConfig dataclass and automatic metric selection.

Extended metric catalogue
─────────────────────────
Classification (binary & multi-class)
  Probability-based : roc_auc, auprc, log_loss, cross_entropy (alias),
                      brier_score, ece
  Threshold-based   : accuracy, balanced_accuracy,
                      precision_{micro,macro,weighted},
                      recall_{micro,macro,weighted},
                      f1_{micro,macro,weighted},
                      specificity, hamming_loss
  (legacy shorthands kept: macro_f1, weighted_f1)

Regression
  r2, mae, mse, rmse, median_absolute_error
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    hamming_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .settings import IMBALANCE_THRESHOLD, METRIC_NAME

log = logging.getLogger(__name__)


# ── ECE helper ────────────────────────────────────────────────────────────────

def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins, binary)."""
    y_true  = np.asarray(y_true).astype(int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 0.0, 1.0)
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n       = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if not mask.any():
            continue
        acc     = y_true[mask].mean()
        conf    = y_proba[mask].mean()
        ece_val += mask.sum() / n * abs(acc - conf)
    return float(ece_val)


# ── Specificity helper ────────────────────────────────────────────────────────

def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate (binary only; returns NaN for multi-class)."""
    labels = np.unique(y_true)
    if len(labels) != 2:
        return float("nan")
    neg_label = labels[0]
    tn    = int(((y_true == neg_label) & (y_pred == neg_label)).sum())
    fp    = int(((y_true == neg_label) & (y_pred != neg_label)).sum())
    denom = tn + fp
    return tn / denom if denom > 0 else float("nan")


# ── MetricConfig ──────────────────────────────────────────────────────────────

_PROBA_METRICS = frozenset({
    "roc_auc", "auprc", "log_loss", "cross_entropy",
    "brier_score", "ece",
})

_LOWER_IS_BETTER = frozenset({
    "log_loss", "cross_entropy", "brier_score", "ece", "hamming_loss",
    "mse", "rmse", "mae", "median_absolute_error",
})

_EVAL_METRIC_MAP: dict[str, str] = {
    "roc_auc":               "auc",
    "auprc":                 "aucpr",
    "log_loss":              "logloss",
    "cross_entropy":         "logloss",
    "brier_score":           "logloss",
    "ece":                   "logloss",
    "accuracy":              "error",
    "balanced_accuracy":     "error",
    "precision_micro":       "mlogloss",
    "precision_macro":       "mlogloss",
    "precision_weighted":    "mlogloss",
    "recall_micro":          "mlogloss",
    "recall_macro":          "mlogloss",
    "recall_weighted":       "mlogloss",
    "f1_micro":              "mlogloss",
    "f1_macro":              "mlogloss",
    "macro_f1":              "mlogloss",
    "f1_weighted":           "mlogloss",
    "weighted_f1":           "mlogloss",
    "specificity":           "error",
    "hamming_loss":          "mlogloss",
    "r2":                    "rmse",
    "mae":                   "mae",
    "mse":                   "rmse",
    "rmse":                  "rmse",
    "median_absolute_error": "mae",
}


@dataclass
class MetricConfig:
    name: str
    direction: str                           # "maximize" | "minimize"
    needs_proba: bool
    scale_pos_weight: float | None
    eval_metric: str
    score_kwargs: dict[str, Any] = field(default_factory=dict)

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> float:
        """Always higher-is-better so Optuna can uniformly maximise."""
        raw = self._raw_score(np.asarray(y_true), y_pred, y_proba)
        return -raw if self.direction == "minimize" else raw

    def score_display(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> float:
        """Human-readable unsigned value for logs and reports."""
        return self._raw_score(np.asarray(y_true), y_pred, y_proba)

    def _raw_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> float:
        n = self.name

        # Probability-based
        if n == "roc_auc":
            return float(roc_auc_score(y_true, y_proba))
        if n == "auprc":
            return float(average_precision_score(y_true, y_proba))
        if n in ("log_loss", "cross_entropy"):
            return float(log_loss(y_true, y_proba))
        if n == "brier_score":
            p = y_proba[:, 1] if (y_proba is not None and y_proba.ndim > 1) else y_proba
            return float(brier_score_loss(y_true, p))
        if n == "ece":
            p = y_proba[:, 1] if (y_proba is not None and y_proba.ndim > 1) else y_proba
            return _ece(y_true, p)

        # Threshold-based classification
        if n == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if n == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if n == "precision_micro":
            return float(precision_score(y_true, y_pred, average="micro",    zero_division=0))
        if n == "precision_macro":
            return float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
        if n == "precision_weighted":
            return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        if n == "recall_micro":
            return float(recall_score(y_true, y_pred, average="micro",    zero_division=0))
        if n == "recall_macro":
            return float(recall_score(y_true, y_pred, average="macro",    zero_division=0))
        if n == "recall_weighted":
            return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        if n == "f1_micro":
            return float(f1_score(y_true, y_pred, average="micro",    zero_division=0))
        if n in ("f1_macro", "macro_f1"):
            return float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
        if n in ("f1_weighted", "weighted_f1"):
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if n == "specificity":
            return _specificity(y_true, y_pred)
        if n == "hamming_loss":
            return float(hamming_loss(y_true, y_pred))

        # Regression
        if n == "r2":
            return float(r2_score(y_true, y_pred))
        if n == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        if n == "mse":
            return float(mean_squared_error(y_true, y_pred))
        if n == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if n == "median_absolute_error":
            return float(median_absolute_error(y_true, y_pred))

        raise ValueError(
            f"Unknown metric: '{n}'. "
            f"Supported: {sorted(_EVAL_METRIC_MAP.keys())}"
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def _make(
    name: str,
    *,
    scale_pos_weight: float | None = None,
    score_kwargs: dict[str, Any] | None = None,
) -> MetricConfig:
    return MetricConfig(
        name=name,
        direction="minimize" if name in _LOWER_IS_BETTER else "maximize",
        needs_proba=name in _PROBA_METRICS,
        scale_pos_weight=scale_pos_weight,
        eval_metric=_EVAL_METRIC_MAP.get(name, "logloss"),
        score_kwargs=score_kwargs or {},
    )


# ── Supported metric names (exposed to config.yaml) ──────────────────────────

SUPPORTED_CLASSIFICATION_METRICS = frozenset({
    "roc_auc", "auprc", "log_loss", "cross_entropy", "brier_score", "ece",
    "accuracy", "balanced_accuracy",
    "precision_micro",  "precision_macro",  "precision_weighted",
    "recall_micro",     "recall_macro",     "recall_weighted",
    "f1_micro",         "f1_macro",         "macro_f1",
    "f1_weighted",      "weighted_f1",
    "specificity",      "hamming_loss",
})

SUPPORTED_REGRESSION_METRICS = frozenset({
    "r2", "mae", "mse", "rmse", "median_absolute_error",
})


def select_metric(y: pd.Series, task: str) -> MetricConfig:
    """Auto-select best Optuna objective metric. Respects METRIC_NAME override."""

    # Regression
    if task == "regression":
        if METRIC_NAME not in ({"auto"} | SUPPORTED_REGRESSION_METRICS):
            log.warning(
                "Requested metric='%s' not supported for regression; using R².", METRIC_NAME
            )
            return _make("r2")
        chosen = "r2" if METRIC_NAME == "auto" else METRIC_NAME
        log.info("Metric selected: %s (regression)", chosen)
        return _make(chosen)

    # Classification — inspect class distribution
    classes      = np.unique(y)
    n_classes    = len(classes)
    class_counts = np.array([(y == c).sum() for c in classes])
    minority_ratio = class_counts.min() / class_counts.sum()
    spw            = class_counts.max() / max(class_counts.min(), 1)

    # Manual override
    if METRIC_NAME != "auto":
        if METRIC_NAME not in SUPPORTED_CLASSIFICATION_METRICS:
            log.warning(
                "Unsupported metric='%s'; falling back to auto-select.", METRIC_NAME
            )
        else:
            spw_val = spw if METRIC_NAME in ("auprc", "brier_score", "ece") else None
            mc = _make(METRIC_NAME, scale_pos_weight=spw_val)
            log.info("Metric selected from config: %s", METRIC_NAME)
            return mc

    # Auto binary
    if n_classes == 2:
        if minority_ratio >= IMBALANCE_THRESHOLD:
            log.info(
                "Metric: roc_auc (binary balanced, minority=%.2f%%)", minority_ratio * 100
            )
            return _make("roc_auc")
        log.info(
            "Metric: auprc (binary imbalanced, minority=%.2f%%, spw=%.1f)",
            minority_ratio * 100, spw,
        )
        return _make("auprc", scale_pos_weight=spw)

    # Auto multi-class
    uniform       = 1.0 / n_classes
    max_deviation = abs(class_counts / class_counts.sum() - uniform).max()
    if max_deviation < uniform * 0.5:
        log.info("Metric: f1_macro (multi-class balanced, %d classes)", n_classes)
        return _make("f1_macro")
    log.info("Metric: f1_weighted (multi-class skewed, %d classes)", n_classes)
    return _make("f1_weighted")


# ── Comprehensive metric suite for evaluation reports ────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    *,
    is_binary: bool = True,
) -> dict[str, float]:
    """Full extended metric suite called by evaluation.py.

    Returns unsigned display values — never Optuna-flipped.
    """
    out: dict[str, float] = {}

    out["accuracy"]          = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["hamming_loss"]      = float(hamming_loss(y_true, y_pred))

    for avg in ("micro", "macro", "weighted"):
        out[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        out[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        out[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )

    if is_binary:
        out["specificity"] = _specificity(y_true, y_pred)

    if y_proba is not None:
        proba_1d = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba_1d))
        except Exception:
            pass
        try:
            out["auprc"]       = float(average_precision_score(y_true, proba_1d))
            out["brier_score"] = float(brier_score_loss(y_true, proba_1d))
            out["ece"]         = _ece(y_true, proba_1d)
        except Exception:
            pass
        try:
            out["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass

    return out


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Full extended metric suite for regression."""
    return {
        "r2":                    float(r2_score(y_true, y_pred)),
        "mae":                   float(mean_absolute_error(y_true, y_pred)),
        "mse":                   float(mean_squared_error(y_true, y_pred)),
        "rmse":                  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }