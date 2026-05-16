"""pipeline.py — XGB callback, GPU helper, pipeline builder, Optuna tuning, ensemble."""
from __future__ import annotations
import importlib.metadata
import logging
import time
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, TargetEncoder
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from xgboost import callback as xgb_callback

from .settings import (
    CB_LOG_PERIOD, CALIBRATION_ENABLED, CV_FOLDS, CV_STRATEGY, EARLY_STOP_RNDS,
    ENSEMBLE_ENABLED, ENSEMBLE_TOP_K, N_ESTIMATORS_MAX, N_ESTIMATORS_MIN,
    N_TRIALS, NATIVE_XGB_CV_EARLY_STOP, NATIVE_XGB_CV_ROUNDS,
    OPTUNA_BUDGET_SECONDS, OPTUNA_TIMEOUT, PCA_MAX_COMPONENTS, PCA_VARIANCE,
    POWER_TRANSFORM, RANDOM_STATE, ROBUST_SCALER_COLS, SEARCH_BACKEND,
    SEARCH_SUBSAMPLE, SOBOL_ENABLED, SOBOL_MAX_EVALS, SOBOL_N_BASE_SAMPLES,
    TUNE_N_ESTIMATORS, USE_GPU, WIDE_SEARCH,
)
from .metrics import MetricConfig

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _as_xgb_matrix(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.astype(np.float32)
    arr = np.asarray(X)
    return arr.astype(np.float32, copy=False) if np.issubdtype(arr.dtype, np.floating) else arr


# ── N1: XGBoost iteration logging callback ────────────────────────────────────

class _IterationLogCallback(xgb_callback.TrainingCallback):
    """Log train/val metrics every CB_LOG_PERIOD boosting rounds."""

    def __init__(self, period: int = CB_LOG_PERIOD, label: str = "final") -> None:
        super().__init__()
        self.period, self.label = max(1, period), label
        self.history: list[dict] = []
        self._prev_val: float | None = None

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        # XGBoost renames every eval_set entry to validation_0, validation_1, ...
        # regardless of the name passed by the caller, so we cannot detect "train"
        # by name. Convention: if 2+ sets are present, the first is train, the last
        # is val. If only 1 set is present it is treated as val (Phase B blind refit).
        keys = list(evals_log.keys())

        if not keys:
            if epoch == 0:
                log.warning("  [%s] evals_log is empty — no eval_set was passed.", self.label)
            self.history.append({"round": epoch + 1, "train": float("nan"),
                                 "val": float("nan"), "label": self.label})
            return False

        def _last(key):
            for metric, vals in evals_log[key].items():
                if vals:
                    return key, metric, float(vals[-1])

        if len(keys) >= 2:
            tr_info  = _last(keys[0])   # first entry = train
            val_info = _last(keys[-1])  # last entry  = val
        else:
            tr_info  = None
            val_info = _last(keys[0])

        tr_val  = tr_info[2]  if tr_info  else float("nan")
        val_val = val_info[2] if val_info else float("nan")
        self.history.append({"round": epoch + 1, "train": tr_val, "val": val_val, "label": self.label})

        if (epoch + 1) % self.period == 0 or epoch == 0:
            delta = f" (Δval {val_val - self._prev_val:+.4f})" if self._prev_val is not None else ""
            if tr_info:
                tr_name  = f"{tr_info[0]}/{tr_info[1]}"
                val_name = f"{val_info[0]}/{val_info[1]}"
                log.info("  [%s] round %4d | %s=%.4f | %s=%.4f%s",
                         self.label, epoch + 1, tr_name, tr_val, val_name, val_val, delta)
            else:
                val_name = f"{val_info[0]}/{val_info[1]}" if val_info else "val/n/a"
                log.info("  [%s] round %4d | %s=%.4f%s",
                         self.label, epoch + 1, val_name, val_val, delta)
            self._prev_val = val_val
        return False


# ── V2: GPU helper ────────────────────────────────────────────────────────────

def _resolve_tree_method() -> tuple[str, str | None]:
    if not USE_GPU:
        return "hist", None

    gpu_available, detect_method = False, "unknown"
    for method, fn in [
        ("cupy",       lambda: __import__("cupy").cuda.runtime.getDeviceCount()),
        ("torch.cuda", lambda: __import__("torch").cuda.is_available() or (_ for _ in ()).throw(RuntimeError())),
    ]:
        try:
            fn(); gpu_available = True; detect_method = method; break
        except Exception:
            pass

    if not gpu_available:
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                gpu_available, detect_method = True, "nvidia-smi"
        except Exception:
            pass

    if not gpu_available:
        log.warning("[GPU] use_gpu=true but no CUDA GPU detected. Falling back to CPU.")
        return "hist", None

    from packaging.version import Version
    xgb_ver = Version(importlib.metadata.version("xgboost"))
    if xgb_ver >= Version("2.0.0"):
        log.info("[GPU] Detected via %s (XGBoost %s ≥ 2.0) → hist + device=cuda", detect_method, xgb_ver)
        return "hist", "cuda"
    log.info("[GPU] Detected via %s (XGBoost %s < 2.0) → gpu_hist", detect_method, xgb_ver)
    return "gpu_hist", None


@lru_cache(maxsize=1)
def _cached_tree_method() -> tuple[str, str | None]:
    return _resolve_tree_method()


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    task: str,
    metric: MetricConfig,
    params: dict | None = None,
    n_estimators: int = N_ESTIMATORS_MAX,
    early_stop: int = EARLY_STOP_RNDS,
    use_pca: bool = False,
    _power_transform: bool | None = None,
    _ct_n_jobs: int = -1,
) -> Pipeline:
    """Build preprocessor + XGBoost pipeline.

    Numerical features are split into two groups:
      robust_cols — columns in ROBUST_SCALER_COLS: RobustScaler only.
      num_cols    — all other numerical cols: impute → optional PowerTransformer → optional PCA.

    Internal kwargs prefixed with _ are not part of the public API.
      _power_transform : override POWER_TRANSFORM for this build only (None → global).
      _ct_n_jobs       : n_jobs for ColumnTransformer (-1 = all cores, 1 = no subprocess overhead).
    """
    params = params or {}
    tree_method, device = _cached_tree_method()
    apply_pt = POWER_TRANSFORM if _power_transform is None else _power_transform

    robust_in_num = [c for c in ROBUST_SCALER_COLS if c in num_cols]
    pca_cols      = [c for c in num_cols if c not in robust_in_num]

    def _num_steps():
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if apply_pt:
            steps.append(("power", PowerTransformer(method="yeo-johnson")))
        return steps

    transformers = []
    if robust_in_num:
        transformers.append(("robust", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
        ]), robust_in_num))

    if pca_cols:
        if use_pca:
            n_comp = PCA_MAX_COMPONENTS if PCA_MAX_COMPONENTS is not None else PCA_VARIANCE
            transformers.append(("num", Pipeline(_num_steps() + [
                ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE))
            ]), pca_cols))
        elif _ct_n_jobs == 1:
            transformers.append(("num", Pipeline(_num_steps()), pca_cols))
        else:
            for col in pca_cols:
                transformers.append((f"num_{col}".replace("__", "_"), Pipeline(_num_steps()), [col]))

    if ohe_cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ohe_cat_cols))
    if te_cat_cols:
        transformers.append(("te_cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(smooth="auto", random_state=RANDOM_STATE)),
        ]), te_cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=_ct_n_jobs)

    shared: dict[str, Any] = dict(
        n_estimators          = n_estimators,
        max_depth             = params.get("max_depth", 6),
        learning_rate         = params.get("learning_rate", 0.1),
        subsample             = params.get("subsample", 0.8),
        colsample_bytree      = params.get("colsample_bytree", 0.8),
        min_child_weight      = params.get("min_child_weight", 1),
        reg_alpha             = params.get("reg_alpha", 0.0),
        reg_lambda            = params.get("reg_lambda", 1.0),
        early_stopping_rounds = early_stop if early_stop > 0 else None,
        random_state          = RANDOM_STATE,
        eval_metric           = metric.eval_metric,
        tree_method           = tree_method,
    )
    if device is not None:
        shared["device"] = device
    if metric.scale_pos_weight is not None:
        shared["scale_pos_weight"] = metric.scale_pos_weight

    model: Any = XGBClassifier(**shared) if task == "classification" else XGBRegressor(**shared)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ── Optuna hyperparameter tuning ──────────────────────────────────────────────

def _xgb_model_params(
    task: str,
    metric: MetricConfig,
    params: dict[str, Any],
    n_classes: int | None = None,
) -> dict[str, Any]:
    if task == "classification":
        objective = "multi:softprob" if (n_classes and n_classes > 2) else "binary:logistic"
    else:
        objective = "reg:squarederror"
    tree_method, device = _cached_tree_method()
    shared = dict(
        objective        = objective,
        eval_metric      = metric.eval_metric,
        max_depth        = int(params.get("max_depth", 6)),
        learning_rate    = float(params.get("learning_rate", 0.1)),
        subsample        = float(params.get("subsample", 0.8)),
        colsample_bytree = float(params.get("colsample_bytree", 0.8)),
        min_child_weight = float(params.get("min_child_weight", 1)),
        reg_alpha        = float(params.get("reg_alpha", 0.0)),
        reg_lambda       = float(params.get("reg_lambda", 1.0)),
        nthread          = int(params.get("nthread", -1)),
        seed             = RANDOM_STATE,
        tree_method      = tree_method,
        verbosity        = 0,
    )
    if device is not None:
        shared["device"] = device
    if metric.scale_pos_weight is not None:
        shared["scale_pos_weight"] = metric.scale_pos_weight
    if task == "classification" and n_classes and n_classes > 2:
        shared["num_class"] = int(n_classes)
    return shared


def _score_booster(booster, dmat, y_true, task, metric) -> float:
    raw = booster.predict(dmat)
    if task == "classification":
        if raw.ndim == 2:
            y_pred  = np.argmax(raw, axis=1)
            y_proba = raw if metric.needs_proba else None
        else:
            y_proba = raw if metric.needs_proba else None
            y_pred  = (raw >= 0.5).astype(int)
    else:
        y_pred, y_proba = raw, None
    return float(metric.score(y_true, y_pred, y_proba))


def _train_xgb_score(
    X_train_proc, y_train_arr, X_val_proc, y_val_arr,
    task, metric, params, n_estimators,
    early_stopping_rounds=None,
) -> tuple[float, int]:
    n_classes  = len(np.unique(y_train_arr)) if task == "classification" else None
    xgb_params = _xgb_model_params(task, metric, params, n_classes=n_classes)
    dtrain     = xgb.DMatrix(X_train_proc, label=y_train_arr)
    dval       = xgb.DMatrix(X_val_proc,   label=y_val_arr)
    fit_kw: dict[str, Any] = {"verbose_eval": False}
    if early_stopping_rounds and early_stopping_rounds > 0:
        fit_kw["early_stopping_rounds"] = int(early_stopping_rounds)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=max(1, int(n_estimators)),
                        evals=[(dval, "validation")], **fit_kw)
    best_it = int(getattr(booster, "best_iteration", n_estimators - 1) or 0)
    return _score_booster(booster, dval, y_val_arr, task, metric), best_it


def _score_candidate(X_tr, y_tr, X_vl, y_vl, task, metric, params, n_estimators) -> float:
    score, _ = _train_xgb_score(X_tr, y_tr, X_vl, y_vl, task, metric, params, n_estimators)
    return score


def _run_sobol_sensitivity(
    X_train_proc, y_train_arr, X_val_proc, y_val_arr,
    task, metric, n_sub,
) -> dict[str, Any]:
    if not SOBOL_ENABLED:
        return {"enabled": False}
    try:
        from SALib.analyze import sobol as sobol_analyze
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        log.warning("[Sobol] SALib not installed — sensitivity analysis skipped.")
        return {"enabled": True, "status": "skipped", "reason": "SALib not installed"}

    problem = {
        "num_vars": 3,
        "names": ["max_depth", "learning_rate", "subsample"],
        "bounds": [[3, 10], [0.01, 0.30], [0.50, 1.00]],
    }
    try:
        samples = sobol_sample.sample(problem, SOBOL_N_BASE_SAMPLES, calc_second_order=False)
    except Exception:
        from SALib.sample import saltelli
        samples = saltelli.sample(problem, SOBOL_N_BASE_SAMPLES, calc_second_order=False)
    if len(samples) > SOBOL_MAX_EVALS:
        samples = samples[:SOBOL_MAX_EVALS]

    rng    = np.random.default_rng(RANDOM_STATE)
    n_rows = len(X_train_proc)
    n_eval = min(max(100, n_sub), n_rows)
    idx    = rng.choice(n_rows, size=n_eval, replace=False) if n_eval < n_rows else np.arange(n_rows)
    _base_params = dict(colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0)

    log.info("[Sobol] Analysing max_depth, learning_rate, subsample (%d evaluations)...", len(samples))
    scores = []
    for raw_depth, lr, subsample in samples:
        try:
            scores.append(_score_candidate(
                X_train_proc[idx], y_train_arr[idx], X_val_proc, y_val_arr,
                task, metric,
                {**_base_params, "max_depth": int(round(raw_depth)),
                 "learning_rate": float(lr), "subsample": float(subsample)},
                n_estimators=min(150, N_ESTIMATORS_MAX),
            ))
        except Exception as exc:
            log.debug("[Sobol] candidate failed: %s", exc)
            scores.append(np.nan)

    y      = np.asarray(scores, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 8:
        return {"enabled": True, "status": "skipped", "reason": "too few successful evaluations"}
    samples_used, y_used = samples[finite], y[finite]
    try:
        si   = sobol_analyze.analyze(problem, y_used, calc_second_order=False, print_to_console=False)
        rows = sorted([
            {"parameter": n, "first_order": float(s1), "total_order": float(st),
             "first_order_conf": float(s1c), "total_order_conf": float(stc)}
            for n, s1, st, s1c, stc in zip(
                problem["names"], si["S1"], si["ST"], si["S1_conf"], si["ST_conf"])
        ], key=lambda r: abs(r["total_order"]), reverse=True)
        log.info("[Sobol] Parameter impact ranking: %s", rows)
        return {"enabled": True, "status": "completed", "metric": metric.name,
                "direction": metric.direction, "evaluations": int(finite.sum()), "parameters": rows}
    except Exception as exc:
        corr_rows = sorted([
            {"parameter": n, "score_correlation":
             float(c) if np.isfinite(c := np.corrcoef(samples_used[:, i], y_used)[0, 1]) else 0.0}
            for i, n in enumerate(problem["names"])
        ], key=lambda r: abs(r["score_correlation"]), reverse=True)
        log.warning("[Sobol] Exact analysis unavailable (%s); using correlation fallback.", exc)
        return {"enabled": True, "status": "fallback", "metric": metric.name,
                "evaluations": int(finite.sum()), "parameters": corr_rows}


def _native_xgb_cv_search(
    X_train_proc, y_train_arr, task, metric,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0)
    n_classes  = len(np.unique(y_train_arr)) if task == "classification" else None
    xgb_params = _xgb_model_params(task, metric, params, n_classes=n_classes)
    folds      = CV_FOLDS if CV_FOLDS > 1 else 5
    log.info("[5/9] Native xgboost.cv search [ rounds=%d | folds=%d | metric=%s ]...",
             NATIVE_XGB_CV_ROUNDS, folds, metric.eval_metric)
    cv = xgb.cv(
        params=xgb_params,
        dtrain=xgb.DMatrix(X_train_proc, label=y_train_arr),
        num_boost_round=max(1, NATIVE_XGB_CV_ROUNDS),
        nfold=folds,
        stratified=task == "classification",
        seed=RANDOM_STATE,
        early_stopping_rounds=max(1, NATIVE_XGB_CV_EARLY_STOP),
        verbose_eval=False,
    )
    best_n     = int(len(cv))
    metric_cols = [c for c in cv.columns if c.endswith("-mean")]
    best_score  = float(cv[metric_cols[-1]].iloc[-1]) if metric_cols else float("nan")
    params["n_estimators"] = best_n
    log.info("  Native xgboost.cv best_n_estimators=%d score=%.6f", best_n, best_score)
    return params, {"backend": "native_xgb_cv", "best_n_estimators": best_n,
                    "best_cv_score": best_score, "cv_columns": list(cv.columns)}


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    task: str,
    metric: MetricConfig,
    use_pca: bool,
) -> tuple[dict, optuna.Study | None, dict[str, Any]]:
    """Tune hyperparameters. Returns (best_params, optuna_study_or_none, search_summary)."""
    # Pre-fit preprocessor once. PowerTransformer skipped on fast-path (scale-invariant for XGB)
    # unless PCA is active (PCA is scale-sensitive). _ct_n_jobs=1 avoids loky fork overhead.
    _apply_power = POWER_TRANSFORM and use_pca
    log.info("  Pre-fitting preprocessor on X_train (search fast-path: n_jobs=1, power_transform=%s)...",
             _apply_power)
    _prep_pipe   = build_pipeline(num_cols, ohe_cat_cols, te_cat_cols, task, metric,
                                  use_pca=use_pca, _power_transform=_apply_power, _ct_n_jobs=1)
    preprocessor = _prep_pipe.named_steps["preprocessor"]
    y_train_arr  = np.array(y_train)
    y_val_arr    = np.array(y_val)
    X_train_proc = _as_xgb_matrix(preprocessor.fit_transform(X_train, y_train_arr))
    X_val_proc   = _as_xgb_matrix(preprocessor.transform(X_val))
    n_rows       = len(X_train_proc)

    # CV strategy: >0 forces on, 0 forces off, -1 = auto (use CV if n_rows < 50k)
    _CV_AUTO_THRESHOLD = 50_000
    use_cv = (n_rows < _CV_AUTO_THRESHOLD) if CV_FOLDS < 0 else bool(CV_FOLDS)
    if use_cv:
        n_folds = CV_FOLDS if CV_FOLDS > 0 else 5
        cv_splitter = (TimeSeriesSplit(n_splits=n_folds) if CV_STRATEGY == "timeseries"
                       else StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE))
        log.info("  CV enabled: %s, %d folds (n_rows=%d)", cv_splitter.__class__.__name__, n_folds, n_rows)
    else:
        log.info("  CV disabled — fast-path subsample (n_rows=%d)", n_rows)

    # Subsample cap: fraction of rows capped at 50k to preserve fast-path intent
    _SUBSAMPLE_ROW_CAP = 50_000
    n_sub = min(max(int(n_rows * SEARCH_SUBSAMPLE), 100), _SUBSAMPLE_ROW_CAP)
    log.info("  Search subsample: %d / %d rows (%.1f%%)", n_sub, n_rows, 100 * n_sub / n_rows)

    search_summary: dict[str, Any] = {
        "backend": SEARCH_BACKEND,
        "sobol_sensitivity": _run_sobol_sensitivity(
            X_train_proc, y_train_arr, X_val_proc, y_val_arr, task, metric, n_sub),
    }

    if SEARCH_BACKEND in ("native_xgb_cv", "xgb_cv", "xgboost_cv"):
        best_params, native_summary = _native_xgb_cv_search(X_train_proc, y_train_arr, task, metric)
        search_summary.update(native_summary)
        return best_params, None, search_summary
    if SEARCH_BACKEND != "optuna":
        log.warning("  Unknown search.backend='%s'; falling back to Optuna.", SEARCH_BACKEND)
        search_summary["backend"] = "optuna"

    # Budget: explicit OPTUNA_TIMEOUT beats budget knob; no budget → bare N_TRIALS
    budget_sec = OPTUNA_BUDGET_SECONDS

    def _run_canary() -> float:
        t0  = time.perf_counter()
        _n  = min(n_sub, 5_000)
        _idx = np.random.default_rng(0).integers(0, n_rows, size=_n)
        _train_xgb_score(
            X_train_proc[_idx], y_train_arr[_idx], X_val_proc, y_val_arr,
            task, metric,
            dict(max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0),
            n_estimators=50, early_stopping_rounds=10,
        )
        return max(time.perf_counter() - t0, 0.1)

    if OPTUNA_TIMEOUT is not None:
        effective_timeout  = OPTUNA_TIMEOUT
        effective_n_trials = N_TRIALS
        log.info("  Budget: explicit timeout=%ds, n_trials=%d", effective_timeout, effective_n_trials)
    elif budget_sec is not None:
        canary_sec         = _run_canary()
        log.info("  Canary trial: %.2fs per trial", canary_sec)
        effective_timeout  = budget_sec
        effective_n_trials = max(10, int(budget_sec * 0.90 / canary_sec))
        log.info("  Budget=%ds → ~%d trials (%.1fs each)", budget_sec, effective_n_trials, canary_sec)
    else:
        effective_timeout  = None
        effective_n_trials = N_TRIALS
        log.info("  Budget: n_trials=%d, timeout=none", effective_n_trials)

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        if WIDE_SEARCH:
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            }
        else:
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 7),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            }

        if TUNE_N_ESTIMATORS:
            trial_n_est   = trial.suggest_int("n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX)
            trial_early_stop = None
        else:
            trial_n_est      = N_ESTIMATORS_MAX
            trial_early_stop = EARLY_STOP_RNDS

        if use_cv:
            # Build xgb params once per trial; each fold worker builds its own DMatrix locally
            _trial_xgb_params = _xgb_model_params(task, metric, {**params, "nthread": 1},
                                                   n_classes=_n_classes_cv)
            _trial_early_stop = int(trial_early_stop) if trial_early_stop else None

            def _fit_fold_np(X_tr_f, X_vl_f, y_tr_f, y_vl_f):
                fit_kw: dict = {"verbose_eval": False}
                if _trial_early_stop:
                    fit_kw["early_stopping_rounds"] = _trial_early_stop
                booster = xgb.train(_trial_xgb_params, xgb.DMatrix(X_tr_f, label=y_tr_f),
                                    num_boost_round=max(1, int(trial_n_est)),
                                    evals=[(xgb.DMatrix(X_vl_f, label=y_vl_f), "validation")],
                                    **fit_kw)
                return _score_booster(booster, xgb.DMatrix(X_vl_f, label=y_vl_f), y_vl_f, task, metric)

            fold_scores = joblib.Parallel(n_jobs=-1, backend="loky")(
                joblib.delayed(_fit_fold_np)(X_tr_f, X_vl_f, y_tr_f, y_vl_f)
                for X_tr_f, X_vl_f, y_tr_f, y_vl_f in _cv_fold_arrays
            )
            trial.report(float(np.mean(fold_scores)), step=n_folds - 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return float(np.mean(fold_scores))

        # Fast-path: stratified subsample, reuse precomputed _dval_search
        if task == "classification":
            rng = np.random.default_rng(trial.number)
            idx = np.concatenate([rng.choice(cls_idx, size=_class_counts[cls], replace=False)
                                  for cls, cls_idx in _class_indices.items()])
        else:
            idx = np.random.default_rng(trial.number).choice(n_rows, size=n_sub, replace=False)

        fit_kw: dict[str, Any] = {"verbose_eval": False}
        if trial_early_stop and trial_early_stop > 0:
            fit_kw["early_stopping_rounds"] = int(trial_early_stop)
        booster = xgb.train(_xgb_model_params(task, metric, params, n_classes=_n_classes),
                             xgb.DMatrix(X_train_proc[idx], label=y_train_arr[idx]),
                             num_boost_round=max(1, int(trial_n_est)),
                             evals=[(_dval_search, "validation")], **fit_kw)
        best_it = int(getattr(booster, "best_iteration", trial_n_est - 1) or 0)
        trial.set_user_attr("best_n_estimators", best_it + 1)
        score = _score_booster(booster, _dval_search, y_val_arr, task, metric)
        trial.report(score, step=best_it)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return score

    # Precompute objects shared across all trials
    _n_classes   = int(len(np.unique(y_train_arr))) if task == "classification" else None
    _dval_search = xgb.DMatrix(X_val_proc, label=y_val_arr)

    if task == "classification" and not use_cv:
        _class_indices = {cls: np.where(y_train_arr == cls)[0] for cls in np.unique(y_train_arr)}
        _class_counts  = {cls: max(1, int(round(n_sub * len(idxs) / n_rows)))
                          for cls, idxs in _class_indices.items()}
    else:
        _class_indices, _class_counts = {}, {}

    # Precompute CV numpy slices once (contiguous copies — DMatrix built per-worker to avoid IPC cost)
    if use_cv:
        _cv_fold_arrays = [
            (np.ascontiguousarray(X_train_proc[tr]), np.ascontiguousarray(X_train_proc[vl]),
             y_train_arr[tr], y_train_arr[vl])
            for tr, vl in cv_splitter.split(X_train_proc, y_train_arr)
        ]
        _n_classes_cv = int(len(np.unique(y_train_arr))) if task == "classification" else None
    else:
        _cv_fold_arrays, _n_classes_cv = [], None

    # Run study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    _nest_label = (f"tuned({N_ESTIMATORS_MIN}–{N_ESTIMATORS_MAX})" if TUNE_N_ESTIMATORS
                   else f"fixed@{N_ESTIMATORS_MAX}+earlystop")
    log.info("[5/9] Optuna search [ Trials : %d | Timeout : %s | Subsample rows : %d | "
             "Metric : %s | CV : %s | Wide : %s | n_estimators : %s | Parallel folds : %s ]...",
             effective_n_trials,
             f"{effective_timeout}s" if effective_timeout is not None else "none",
             n_sub, metric.name,
             n_folds if use_cv else "off",
             WIDE_SEARCH, _nest_label,
             "yes" if use_cv else "n/a")

    _running_best: list[float] = [float("-inf")]
    with tqdm(total=effective_n_trials, ncols=100, desc="Optuna search") as pbar:
        def _objective_with_progress(trial):
            result = objective(trial)
            if result > _running_best[0]:
                _running_best[0] = result
            pbar.set_postfix(best=f"{_running_best[0]:.6f}", trial=trial.number)
            pbar.update(1)
            return result

        study.optimize(_objective_with_progress, n_trials=effective_n_trials,
                       timeout=effective_timeout, n_jobs=1, show_progress_bar=False)

    pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    log.info("  Best %s=%.4f  params=%s  (complete=%d, pruned=%d)",
             metric.name, study.best_value, study.best_params, complete, pruned)
    search_summary.update({"backend": "optuna", "best_value": float(study.best_value),
                           "complete_trials": complete, "pruned_trials": pruned})
    return study.best_params, study, search_summary


# ── Top-K ensemble ────────────────────────────────────────────────────────────

def _fit_one(name: str, est, X, y):
    est.fit(X, y)
    return name, est


class _IdentityLabelEncoder:
    """Picklable stand-in for sklearn LabelEncoder when targets are already 0-based ints."""
    def __init__(self, classes: np.ndarray) -> None:
        self.classes_ = classes
    def transform(self, y):
        return np.asarray(y)


def build_top_k_ensemble(
    study: optuna.Study | None,
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    task: str,
    metric: MetricConfig,
    use_pca: bool,
    top_k: int,
    n_estimators: int,
    threshold: float,
    log_transformed: bool,
) -> tuple[Any | None, dict]:
    """Fit a soft-voting ensemble from the top-K completed Optuna trials."""
    _skipped = lambda reason, used=0: (None, {
        "enabled": True, "status": "skipped", "reason": reason,
        "top_k_requested": top_k, "trials_used": used,
    })

    if study is None:
        return _skipped("Optuna study unavailable for current search backend")

    complete_trials = sorted(
        [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None],
        key=lambda t: float(t.value),
        reverse=(metric.direction == "maximize"),
    )
    selected = complete_trials[:top_k]
    if len(selected) < 2:
        return _skipped("fewer than 2 completed Optuna trials", len(selected))

    n_est_global = max(1, int(n_estimators))
    named_estimators, trial_rows = [], []
    for idx, trial in enumerate(selected, 1):
        n_est = max(1, int(
            trial.user_attrs.get("best_n_estimators") or
            trial.params.get("n_estimators") or
            n_est_global
        ))
        est = build_pipeline(num_cols, ohe_cat_cols, te_cat_cols, task, metric,
                             params=trial.params, n_estimators=n_est, early_stop=0, use_pca=use_pca)
        named_estimators.append((f"trial_{trial.number}", est))
        trial_rows.append({"rank": idx, "trial_number": trial.number, "value": float(trial.value),
                           "n_estimators": n_est, "params": dict(trial.params)})

    log.info("[ensemble] Fitting %d members in parallel...", len(named_estimators))
    fitted_estimators = list(Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit_one)(name, est, X_trainval, y_trainval)
        for name, est in named_estimators
    ))

    ensemble: Any = (VotingClassifier(estimators=fitted_estimators, voting="soft", n_jobs=-1)
                     if task == "classification"
                     else VotingRegressor(estimators=fitted_estimators, n_jobs=-1))
    ensemble.estimators_       = [est for _, est in fitted_estimators]
    ensemble.named_estimators_ = dict(fitted_estimators)
    if task == "classification":
        classes = np.unique(y_trainval)
        ensemble.le_      = _IdentityLabelEncoder(classes)
        ensemble.classes_ = classes

    y_test_arr = np.asarray(y_test)
    if task == "classification":
        if metric.needs_proba:
            y_proba = ensemble.predict_proba(X_test)[:, 1]
            y_pred  = (y_proba >= threshold).astype(int)
            eval_metrics = {
                "selected_metric": float(metric.score(y_test_arr, y_pred, y_proba)),
                "auprc":           float(average_precision_score(y_test_arr, y_proba)),
                "roc_auc":         float(roc_auc_score(y_test_arr, y_proba)),
                "threshold":       float(threshold),
            }
        else:
            y_pred = ensemble.predict(X_test)
            eval_metrics = {"selected_metric": float(metric.score(y_test_arr, y_pred, None)),
                            "threshold": None}
    else:
        y_pred = ensemble.predict(X_test)
        if log_transformed:
            y_test_arr, y_pred = np.expm1(y_test_arr), np.expm1(y_pred)
        eval_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_arr, y_pred))),
            "mae":  float(mean_absolute_error(y_test_arr, y_pred)),
            "r2":   float(r2_score(y_test_arr, y_pred)),
        }

    return ensemble, {
        "enabled": True, "status": "fitted",
        "top_k_requested": top_k, "trials_used": len(fitted_estimators),
        "n_estimators_global_fallback": n_est_global,
        "member_trials": trial_rows, "eval_metrics": eval_metrics,
    }