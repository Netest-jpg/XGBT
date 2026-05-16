"""Baseline model comparison for the generalized XGBoost prototype."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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


def _fit_and_evaluate(
    name: str,
    estimator: Any,
    # Pre-transformed arrays — avoids redundant preprocessing per model
    X_train_t: np.ndarray,
    X_val_t: np.ndarray,
    X_test_t: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    task: str,
    metric: Any,
    threshold_policy: dict[str, Any] | None,
    log_transformed: bool,
) -> dict[str, Any]:
    """Fit a single estimator on pre-transformed data and return a result row.

    Separated from the main loop so joblib can dispatch it to worker processes.
    All arguments must be picklable (numpy arrays, plain dicts, sklearn estimators).
    """
    try:
        estimator.fit(X_train_t, y_train)
        threshold = None
        y_proba_test = None

        if task == "classification":
            needs_proba = getattr(metric, "needs_proba", False)
            if needs_proba and hasattr(estimator, "predict_proba"):
                # Tune threshold on val, then apply once to test — no duplicate predict_proba call.
                y_proba_val = estimator.predict_proba(X_val_t)[:, 1]
                tuned = tune_binary_threshold(
                    np.asarray(y_val),
                    y_proba_val,
                    threshold_policy,
                    metric_name=metric.name,
                )
                threshold = tuned.threshold
                y_proba_test = estimator.predict_proba(X_test_t)[:, 1]
                y_pred = (y_proba_test >= threshold).astype(int)
            else:
                y_pred = estimator.predict(X_test_t)
            metrics = _classification_metrics(y_test, y_pred, y_proba_test)
        else:
            y_pred = estimator.predict(X_test_t)
            metrics = _regression_metrics(y_test, y_pred, log_transformed=log_transformed)

        return _row(name, metric.name, threshold, metrics)

    except Exception as exc:
        traceback.print_exc()
        return {
            "model": name,
            "selected_metric": getattr(metric, "name", "unknown"),
            "threshold": None,
            "error": str(exc),
        }


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
    n_jobs: int = -1,
    subsample_train: float | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Fit cheap baselines and save a Parquet comparison table.

    Performance improvements vs the original:
    - Preprocessor is fit once; all models receive pre-transformed arrays,
      avoiding redundant imputation, scaling, and one-hot encoding.
    - Models are trained in parallel via joblib (n_jobs=-1 uses all CPUs).
    - predict_proba is called only once per model: on val for threshold tuning,
      then on test for scoring — the val probabilities are not recomputed.
    - Results persisted as Parquet (snappy) — faster to write and read than CSV.
    - Optional subsample_train (0 < float ≤ 1) draws a stratified fraction of
      the training set for baselines, which is usually sufficient and speeds up
      slow estimators (LogisticRegression, XGBoost) on large datasets.

    Args:
        n_jobs: Number of parallel workers for the model loop.
            -1 = all CPUs (default). 1 = sequential (useful for debugging).
        subsample_train: If set, a stratified fraction of X_train/y_train is
            used for fitting baseline models. E.g. 0.2 uses 20% of rows.
            Has no effect on val/test evaluation.
    """
    if not enabled:
        log.info("[baseline] [skipped]")
        return [], None

    cat_cols = ohe_cat_cols + te_cat_cols

    # --- Optional training-set subsampling -----------------------------------
    if subsample_train is not None and 0.0 < subsample_train < 1.0:
        rng = np.random.RandomState(random_state)
        n_sample = max(1, int(len(X_train) * subsample_train))
        # Stratified sampling for classification, plain random for regression.
        if task == "classification":
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sample, random_state=random_state)
            idx, _ = next(sss.split(X_train, y_train))
        else:
            idx = rng.choice(len(X_train), size=n_sample, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        log.info("[baseline] subsampled training set to %d rows (%.0f%%)", n_sample, subsample_train * 100)

    # --- Fit preprocessor once for all models --------------------------------
    prep = _preprocessor(num_cols, cat_cols)
    X_train_t: np.ndarray = prep.fit_transform(X_train, y_train)
    X_val_t:   np.ndarray = prep.transform(X_val)
    X_test_t:  np.ndarray = prep.transform(X_test)

    y_train_a = np.asarray(y_train)
    y_val_a   = np.asarray(y_val)
    y_test_a  = np.asarray(y_test)

    # --- Assemble estimator list ---------------------------------------------
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
                    random_state=random_state,
                ),
            ))
        if include_default_xgb:
            xgb_params: dict[str, Any] = {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_state,
                "eval_metric": metric.eval_metric,
                "tree_method": "hist",
                "nthread": -1,
                "verbosity": 0,
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
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    eval_metric=metric.eval_metric,
                    tree_method="hist",
                    nthread=-1,
                    verbosity=0,
                ),
            ))

    # --- Parallel model training & evaluation --------------------------------
    # Use "loky" (default) backend — it spawns independent processes, so
    # XGBoost and sklearn GIL-heavy code both benefit. Fall back to n_jobs=1
    # automatically when only one model is present.
    effective_n_jobs = 1 if len(models) == 1 else n_jobs
    rows: list[dict[str, Any]] = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
        delayed(_fit_and_evaluate)(
            name, estimator,
            X_train_t, X_val_t, X_test_t,
            y_train_a, y_val_a, y_test_a,
            task, metric, threshold_policy, log_transformed,
        )
        for name, estimator in models
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baseline_comparison_{run_id}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return rows, path