"""Configurable threshold policies for binary classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score


@dataclass
class ThresholdResult:
    threshold: float
    policy: dict[str, Any]
    metrics: dict[str, float]


def _policy_value(policy: Any, key: str, default: Any) -> Any:
    if isinstance(policy, dict):
        return policy.get(key, default)
    return getattr(policy, key, default)


def normalize_policy(policy: Any, metric_name: str | None = None) -> dict[str, Any]:
    """Normalize dict/dataclass/None policy objects into a plain mapping."""
    if policy is None:
        raw: dict[str, Any] = {}
    elif hasattr(policy, "__dataclass_fields__"):
        raw = asdict(policy)
    elif isinstance(policy, dict):
        raw = dict(policy)
    else:
        raw = {
            "mode": getattr(policy, "mode", "auto"),
            "beta": getattr(policy, "beta", 1.0),
            "min_precision": getattr(policy, "min_precision", 0.80),
            "min_recall": getattr(policy, "min_recall", 0.80),
            "n_quantiles": getattr(policy, "n_quantiles", 200),
        }

    mode = str(raw.get("mode", "auto")).lower()
    if mode == "auto":
        mode = "f1" if metric_name in ("roc_auc", "auprc", None) else "disabled"

    return {
        "mode": mode,
        "beta": float(raw.get("beta", 1.0)),
        "min_precision": float(raw.get("min_precision", 0.80)),
        "min_recall": float(raw.get("min_recall", 0.80)),
        "n_quantiles": int(raw.get("n_quantiles", 200)),
    }


def _candidate_thresholds(y_proba: np.ndarray, n_quantiles: int) -> np.ndarray:
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.size == 0:
        return np.array([0.5])
    quantile_points = np.linspace(0, 100, max(1, n_quantiles) + 2)[1:-1]
    candidates = np.unique(np.percentile(y_proba, quantile_points))
    candidates = np.unique(np.concatenate([candidates, np.array([0.5])]))
    return candidates[(candidates >= 0) & (candidates <= 1)]


def tune_binary_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    policy: Any = None,
    metric_name: str | None = None,
) -> ThresholdResult:
    """Tune a binary decision threshold using a named, generic policy."""
    normalized = normalize_policy(policy, metric_name=metric_name)
    mode = normalized["mode"]
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    def score_at(threshold: float) -> dict[str, float]:
        pred = (y_proba >= threshold).astype(int)
        return {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "fbeta": float(fbeta_score(y_true, pred, beta=normalized["beta"], zero_division=0)),
            "support": float(pred.sum()),
        }

    if mode == "disabled":
        metrics = score_at(0.5)
        metrics["objective"] = metrics["f1"]
        return ThresholdResult(0.5, normalized, metrics)

    candidates = _candidate_thresholds(y_proba, normalized["n_quantiles"])
    rows = [(float(th), score_at(float(th))) for th in candidates]

    if mode == "f1":
        best_th, best_metrics = max(rows, key=lambda item: (item[1]["f1"], item[0]))
        best_metrics["objective"] = best_metrics["f1"]
    elif mode == "fbeta":
        best_th, best_metrics = max(rows, key=lambda item: (item[1]["fbeta"], item[0]))
        best_metrics["objective"] = best_metrics["fbeta"]
    elif mode == "precision_at_recall":
        feasible = [row for row in rows if row[1]["recall"] >= normalized["min_recall"]]
        if not feasible:
            feasible = rows
        best_th, best_metrics = max(feasible, key=lambda item: (item[1]["precision"], item[1]["recall"], item[0]))
        best_metrics["objective"] = best_metrics["precision"]
    elif mode == "recall_at_precision":
        feasible = [row for row in rows if row[1]["precision"] >= normalized["min_precision"]]
        if not feasible:
            feasible = rows
        best_th, best_metrics = max(feasible, key=lambda item: (item[1]["recall"], item[1]["precision"], item[0]))
        best_metrics["objective"] = best_metrics["recall"]
    else:
        raise ValueError(
            "threshold_policy.mode must be one of "
            "'auto', 'f1', 'fbeta', 'precision_at_recall', 'recall_at_precision', or 'disabled'"
        )

    return ThresholdResult(float(best_th), normalized, best_metrics)

