"""evaluation.py — Model evaluation, error analysis, threshold tuning."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from .settings import MODEL_OUTPUT_DIR, THRESHOLD_POLICY, PLOTS_ENABLED
from .metrics import MetricConfig
from .plots import plot_confusion_matrix, plot_pr_curve, plot_residuals, plot_roc_curve

log = logging.getLogger(__name__)


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    metric: MetricConfig,
    threshold: float = 0.5,
    X_test_proc: np.ndarray | None = None,
    log_transformed: bool = False,
) -> dict:
    """Score on held-out test set with the full extended metric suite."""
    from .metrics import compute_classification_metrics, compute_regression_metrics

    log.info("[7/9] Evaluation on held-out test set")
    model_step = pipeline.named_steps["model"] if X_test_proc is not None else None

    def _proba(X):
        if model_step is not None:
            return model_step.predict_proba(X_test_proc)
        return pipeline.predict_proba(X)

    def _predict_raw(X):
        if model_step is not None:
            raw = model_step.predict(X_test_proc)
            return np.expm1(raw) if log_transformed else raw
        if task == "classification" and metric.needs_proba:
            return (_proba(X)[:, 1] >= threshold).astype(int)
        raw = pipeline.predict(X)
        return np.expm1(raw) if log_transformed else raw

    y_test_eval = np.expm1(np.array(y_test)) if log_transformed else np.array(y_test)
    is_binary   = len(np.unique(y_test_eval)) == 2

    if task == "classification":
        y_proba_2d = _proba(X_test)                       # shape (n, n_classes)
        y_proba_1d = y_proba_2d[:, 1] if is_binary else None
        y_pred     = (y_proba_2d[:, 1] >= threshold).astype(int) if is_binary \
                     else np.argmax(y_proba_2d, axis=1)
        log.info("  Using threshold = %.4f", threshold)

        results = compute_classification_metrics(
            y_test_eval, y_pred,
            y_proba_2d if not is_binary else y_proba_1d,
            is_binary=is_binary,
        )

        log.info("\n%s", classification_report(y_test_eval, y_pred))
        log.info("Confusion Matrix:\n%s", confusion_matrix(y_test_eval, y_pred))

        # Log the full suite in a tidy table
        _CLF_ORDER = [
            "accuracy", "balanced_accuracy",
            "roc_auc", "auprc",
            "precision_micro", "precision_macro", "precision_weighted",
            "recall_micro",    "recall_macro",    "recall_weighted",
            "f1_micro",        "f1_macro",        "f1_weighted",
            "specificity", "hamming_loss",
            "log_loss", "brier_score", "ece",
        ]
        for k in _CLF_ORDER:
            if k in results:
                log.info("  %-26s: %.4f", k, results[k])

        if PLOTS_ENABLED and y_proba_1d is not None:
            plot_pr_curve(y_test_eval, y_proba_1d, threshold)
            plot_roc_curve(y_test_eval, y_proba_1d)
        if PLOTS_ENABLED:
            plot_confusion_matrix(y_test_eval, y_pred)

    else:
        y_pred  = _predict_raw(X_test)
        results = compute_regression_metrics(y_test_eval, y_pred)
        log.info(
            "  R²=%.4f  RMSE=%.4f  MAE=%.4f  MSE=%.4f  MedianAE=%.4f",
            results["r2"], results["rmse"], results["mae"],
            results["mse"], results["median_absolute_error"],
        )
        if PLOTS_ENABLED:
            plot_residuals(y_test_eval, y_pred)

    return results


def analyse_errors(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: MetricConfig,
    threshold: float,
    run_id: str,
    X_test_proc: np.ndarray | None = None,
    label_encoder=None,
) -> Path | None:
    """N5: Save FP/FN CSV with confidence scores for binary classification."""
    if not (metric.needs_proba and metric.name in ("auprc", "roc_auc")):
        return None

    log.info("  [N5] Error analysis — isolating FP / FN...")
    model_step = pipeline.named_steps["model"] if X_test_proc is not None else None
    y_proba = (model_step.predict_proba(X_test_proc)[:, 1]
               if model_step is not None
               else pipeline.predict_proba(X_test)[:, 1])

    y_pred = (y_proba >= threshold).astype(int)
    y_true = np.array(y_test)
    errors = y_pred != y_true
    if not errors.any():
        log.info("  [N5] No misclassifications — CSV skipped.")
        return None

    fp_mask = errors & (y_pred == 1)
    fn_mask = errors & (y_pred == 0)
    err_df  = X_test.copy().reset_index(drop=True)
    err_df["true_label"] = y_true
    err_df["pred_label"] = y_pred
    err_df["error_type"] = np.where(fp_mask, "FP", np.where(fn_mask, "FN", "OK"))
    err_df["confidence"] = np.where(y_pred == 1, y_proba, 1.0 - y_proba)
    err_df["margin"]     = np.abs(y_proba - threshold)
    err_df["raw_proba"]  = y_proba

    if label_encoder is not None:
        try:
            err_df["true_label"] = label_encoder.inverse_transform(err_df["true_label"])
            err_df["pred_label"] = label_encoder.inverse_transform(err_df["pred_label"])
        except Exception:
            pass

    misclf = err_df[err_df["error_type"] != "OK"].sort_values("confidence", ascending=False)
    n_fp, n_fn = int(fp_mask.sum()), int(fn_mask.sum())
    log.info("  [N5] FP: %d  FN: %d  total: %d / %d (%.1f%%)",
             n_fp, n_fn, n_fp + n_fn, len(y_true),
             100 * (n_fp + n_fn) / max(len(y_true), 1))

    out_path = MODEL_OUTPUT_DIR / f"error_analysis_{run_id}.csv"
    misclf.to_csv(out_path, index=False)
    log.info("  [N5] Error analysis saved → %s", out_path)
    return out_path


def tune_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: MetricConfig,
    n_quantiles: int = 200,
    X_val_proc: np.ndarray | None = None,
) -> float:
    """Tune decision threshold using the configured THRESHOLD_POLICY."""
    if not metric.needs_proba or metric.name not in ("auprc", "roc_auc"):
        return 0.5

    from .thresholds import tune_binary_threshold

    model_step = pipeline.named_steps["model"] if X_val_proc is not None else None
    y_proba = (model_step.predict_proba(X_val_proc)[:, 1]
               if model_step is not None
               else pipeline.predict_proba(X_val)[:, 1])

    policy = dict(THRESHOLD_POLICY)
    policy["n_quantiles"] = int(n_quantiles or policy.get("n_quantiles", 200))
    tuned = tune_binary_threshold(np.array(y_val), y_proba, policy=policy, metric_name=metric.name)

    default_f1 = float(f1_score(np.array(y_val), (y_proba >= 0.5).astype(int), zero_division=0))
    log.info(
        "  Threshold tuning → policy=%s  threshold=%.6f  objective=%.4f  "
        "(precision=%.4f, recall=%.4f, f1=%.4f, default-0.5 F1=%.4f)",
        tuned.policy["mode"], tuned.threshold, tuned.metrics.get("objective", 0.0),
        tuned.metrics.get("precision", 0.0), tuned.metrics.get("recall", 0.0),
        tuned.metrics.get("f1", 0.0), default_f1,
    )
    return tuned.threshold