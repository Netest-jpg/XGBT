"""metrics.py — MetricConfig dataclass and automatic metric selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    r2_score,
    roc_auc_score,
)

from .settings import IMBALANCE_THRESHOLD, METRIC_NAME

log = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    name: str
    direction: str
    needs_proba: bool
    scale_pos_weight: float | None
    eval_metric: str

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> float:
        if self.name == "r2":
            return r2_score(y_true, y_pred)
        if self.name == "roc_auc":
            return roc_auc_score(y_true, y_proba)
        if self.name == "auprc":
            return average_precision_score(y_true, y_proba)
        if self.name == "macro_f1":
            return f1_score(y_true, y_pred, average="macro", zero_division=0)
        if self.name == "weighted_f1":
            return f1_score(y_true, y_pred, average="weighted", zero_division=0)
        raise ValueError(f"Unknown metric: {self.name}")


def select_metric(y: pd.Series, task: str) -> MetricConfig:
    """Auto-select best Optuna objective metric. Respects METRIC_NAME override."""
    if task == "regression":
        if METRIC_NAME not in ("auto", "r2"):
            log.warning("Requested metric='%s' not supported for regression; using R².", METRIC_NAME)
        log.info("Metric selected: R² (regression)")
        return MetricConfig(name="r2", direction="maximize",
                            needs_proba=False, scale_pos_weight=None, eval_metric="rmse")

    classes      = np.unique(y)
    n_classes    = len(classes)
    class_counts = np.array([(y == c).sum() for c in classes])
    minority_ratio = class_counts.min() / class_counts.sum()

    # Manual override
    if METRIC_NAME != "auto":
        supported = {"roc_auc", "auprc", "macro_f1", "weighted_f1"}
        if METRIC_NAME not in supported:
            log.warning("Unsupported metric='%s'; falling back to auto-select.", METRIC_NAME)
        else:
            _override_map = {
                "roc_auc":    MetricConfig("roc_auc",    "maximize", True,  None, "auc"),
                "auprc":      MetricConfig("auprc",      "maximize", True,
                                           class_counts.max() / class_counts.min(), "aucpr"),
                "macro_f1":   MetricConfig("macro_f1",   "maximize", False, None, "mlogloss"),
                "weighted_f1":MetricConfig("weighted_f1","maximize", False, None, "mlogloss"),
            }
            mc = _override_map[METRIC_NAME]
            log.info("Metric selected from config: %s", METRIC_NAME)
            return mc

    # Binary auto
    if n_classes == 2:
        if minority_ratio >= IMBALANCE_THRESHOLD:
            log.info("Metric: ROC-AUC (binary balanced, minority=%.2f%%)", minority_ratio * 100)
            return MetricConfig("roc_auc", "maximize", True, None, "auc")
        spw = class_counts.max() / class_counts.min()
        log.info("Metric: AUPRC (binary imbalanced, minority=%.2f%%, spw=%.1f)",
                 minority_ratio * 100, spw)
        return MetricConfig("auprc", "maximize", True, spw, "aucpr")

    # Multi-class auto
    uniform       = 1.0 / n_classes
    max_deviation = abs(class_counts / class_counts.sum() - uniform).max()
    if max_deviation < uniform * 0.5:
        log.info("Metric: macro-F1 (multi-class balanced, %d classes)", n_classes)
        return MetricConfig("macro_f1", "maximize", False, None, "mlogloss")
    log.info("Metric: weighted-F1 (multi-class skewed, %d classes)", n_classes)
    return MetricConfig("weighted_f1", "maximize", False, None, "mlogloss")