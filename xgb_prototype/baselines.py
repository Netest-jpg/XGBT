"""Baseline model comparison for the generalized XGBoost prototype."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .thresholds import tune_binary_threshold

log = logging.getLogger(__name__)


def _classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    labels = np.unique(y_true)
    if len(labels) == 2:
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if y_proba is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            out["auprc"] = float(average_precision_score(y_true, y_proba))
    return out


def _regression_metrics(y_true, y_pred, log_transformed: bool = False) -> dict[str, float]:
    y_true_eval = np.expm1(y_true) if log_transformed else np.asarray(y_true)
    y_pred_eval = np.expm1(y_pred) if log_transformed else np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))),
        "mae": float(mean_absolute_error(y_true_eval, y_pred_eval)),
        "r2": float(r2_score(y_true_eval, y_pred_eval)),
    }


def _preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _row(model_name: str, selected_metric: str, threshold: float | None, metrics: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "selected_metric": selected_metric,
        "threshold": threshold,
    }
    row.update(metrics)
    return row


def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    task: str,
    metric: Any,
    random_state: int,
    output_dir: Path,
    run_id: str,
    threshold_policy: dict[str, Any] | None = None,
    log_transformed: bool = False,
    enabled: bool = True,
    include_dummy: bool = True,
    include_linear: bool = True,
    include_default_xgb: bool = True,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Fit cheap baselines and save a CSV comparison table."""
    if not enabled:
        log.info("[baseline] [skipped]")
        return [], None

    cat_cols = ohe_cat_cols + te_cat_cols
    prep = _preprocessor(num_cols, cat_cols)
    rows: list[dict[str, Any]] = []
    models: list[tuple[str, Any]] = []

    if task == "classification":
        if include_dummy:
            models.append(("dummy_most_frequent", DummyClassifier(strategy="most_frequent")))
        if include_linear:
            models.append((
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ))
        if include_default_xgb:
            xgb_params: dict[str, Any] = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_state,
                "eval_metric": metric.eval_metric,
                "tree_method": "hist",
            }
            if getattr(metric, "scale_pos_weight", None) is not None:
                xgb_params["scale_pos_weight"] = metric.scale_pos_weight
            models.append(("xgb_default", XGBClassifier(**xgb_params)))
    else:
        if include_dummy:
            models.append(("dummy_mean", DummyRegressor(strategy="mean")))
        if include_default_xgb:
            models.append((
                "xgb_default",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    eval_metric=metric.eval_metric,
                    tree_method="hist",
                ),
            ))

    for name, estimator in models:
        pipe = Pipeline([("preprocessor", prep), ("model", estimator)])
        try:
            pipe.fit(X_train, y_train)
            threshold = None
            y_proba_test = None
            if task == "classification":
                if getattr(metric, "needs_proba", False) and hasattr(pipe, "predict_proba"):
                    y_proba_val = pipe.predict_proba(X_val)[:, 1]
                    tuned = tune_binary_threshold(
                        np.asarray(y_val),
                        y_proba_val,
                        threshold_policy,
                        metric_name=metric.name,
                    )
                    threshold = tuned.threshold
                    y_proba_test = pipe.predict_proba(X_test)[:, 1]
                    y_pred = (y_proba_test >= threshold).astype(int)
                else:
                    y_pred = pipe.predict(X_test)
                metrics = _classification_metrics(y_test, y_pred, y_proba_test)
            else:
                y_pred = pipe.predict(X_test)
                metrics = _regression_metrics(y_test, y_pred, log_transformed=log_transformed)
            rows.append(_row(name, metric.name, threshold, metrics))
        except Exception as exc:
            rows.append({
                "model": name,
                "selected_metric": getattr(metric, "name", "unknown"),
                "threshold": None,
                "error": str(exc),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baseline_comparison_{run_id}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows, path