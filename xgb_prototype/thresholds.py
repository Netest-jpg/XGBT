"""Configurable threshold policies for binary classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np



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


def _vectorized_scores(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    candidates: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision/recall/f1/fbeta/support for all thresholds in one pass.

    Broadcasting shape: (n_candidates, n_samples) avoids any Python loop over
    thresholds and replaces ~200 individual sklearn calls with pure NumPy ops.
    Returns five 1-D arrays indexed by candidate position.
    """
    # preds[i, j] = True iff y_proba[j] >= candidates[i]
    preds = y_proba[np.newaxis, :] >= candidates[:, np.newaxis]  # (C, N)

    pos = y_true.astype(bool)
    tp = preds[:, pos].sum(axis=1).astype(float)    # (C,)
    fp = preds[:, ~pos].sum(axis=1).astype(float)   # (C,)
    fn = (~preds[:, pos]).sum(axis=1).astype(float)  # (C,)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

        denom_f1 = precision + recall
        f1       = np.where(denom_f1 > 0, 2 * precision * recall / denom_f1, 0.0)

        denom_fb = beta ** 2 * precision + recall
        fbeta    = np.where(denom_fb > 0, (1 + beta ** 2) * precision * recall / denom_fb, 0.0)

    support = preds.sum(axis=1).astype(float)
    return precision, recall, f1, fbeta, support

def _metrics_dict(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    fbeta: np.ndarray,
    support: np.ndarray,
    idx: int,
    objective_key: str,
) -> dict[str, float]:
    m = {
        "precision": float(precision[idx]),
        "recall":    float(recall[idx]),
        "f1":        float(f1[idx]),
        "fbeta":     float(fbeta[idx]),
        "support":   float(support[idx]),
    }
    m["objective"] = m[objective_key]
    return m


def tune_binary_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    policy: Any = None,
    metric_name: str | None = None,
) -> ThresholdResult:
    """Tune a binary decision threshold using a named, generic policy."""
    normalized = normalize_policy(policy, metric_name=metric_name)
    mode = normalized["mode"]
    beta = normalized["beta"]
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if mode == "disabled":
        # Single-threshold fast path — no vectorization needed.
        pred = (y_proba >= 0.5).astype(bool)
        pos = y_true.astype(bool)
        tp = float(pred[pos].sum());  fp = float(pred[~pos].sum());  fn = float((~pred)[pos].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        d_f1 = prec + rec
        f1v  = 2 * prec * rec / d_f1 if d_f1 else 0.0
        metrics = {"precision": prec, "recall": rec, "f1": f1v,
                   "fbeta": f1v, "support": float(pred.sum()), "objective": f1v}
        return ThresholdResult(0.5, normalized, metrics)

    candidates = _candidate_thresholds(y_proba, normalized["n_quantiles"])
    precision, recall, f1, fbeta, support = _vectorized_scores(y_true, y_proba, candidates, beta)

    if mode == "f1":
        # np.lexsort sorts ascending; last key is primary. Tie-break: prefer higher threshold.
        idx = int(np.lexsort((candidates, f1))[-1])
        obj_key = "f1"
    elif mode == "fbeta":
        idx = int(np.lexsort((candidates, fbeta))[-1])
        obj_key = "fbeta"
    elif mode == "precision_at_recall":
        mask = recall >= normalized["min_recall"]
        if not mask.any():
            mask = np.ones(len(candidates), dtype=bool)
        fi = np.where(mask)[0]
        idx = int(fi[np.lexsort((candidates[fi], recall[fi], precision[fi]))[-1]])
        obj_key = "precision"
    elif mode == "recall_at_precision":
        mask = precision >= normalized["min_precision"]
        if not mask.any():
            mask = np.ones(len(candidates), dtype=bool)
        fi = np.where(mask)[0]
        idx = int(fi[np.lexsort((candidates[fi], precision[fi], recall[fi]))[-1]])
        obj_key = "recall"
    else:
        raise ValueError(
            "threshold_policy.mode must be one of "
            "'auto', 'f1', 'fbeta', 'precision_at_recall', 'recall_at_precision', or 'disabled'"
        )

    return ThresholdResult(
        float(candidates[idx]),
        normalized,
        _metrics_dict(precision, recall, f1, fbeta, support, idx, obj_key),
    )