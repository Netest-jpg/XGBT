"""pipeline.py — XGB callback, GPU helper, pipeline builder, Optuna tuning, ensemble."""
from __future__ import annotations
import time
import importlib.metadata
import logging
from functools import lru_cache
from typing import Any
from tqdm import tqdm
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, TargetEncoder
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import callback as xgb_callback

from .settings import (
    CB_LOG_PERIOD, CALIBRATION_ENABLED, CV_FOLDS, CV_STRATEGY, EARLY_STOP_RNDS,
    N_ESTIMATORS_MAX, N_ESTIMATORS_MIN, TUNE_N_ESTIMATORS,
    N_TRIALS, OPTUNA_BUDGET_SECONDS, OPTUNA_TIMEOUT,
    PCA_MAX_COMPONENTS, PCA_VARIANCE,
    RANDOM_STATE, SEARCH_SUBSAMPLE, USE_GPU, WIDE_SEARCH, ENSEMBLE_ENABLED, ENSEMBLE_TOP_K,
    POWER_TRANSFORM, ROBUST_SCALER_COLS,
    SEARCH_BACKEND, NATIVE_XGB_CV_ROUNDS, NATIVE_XGB_CV_EARLY_STOP,
    SOBOL_ENABLED, SOBOL_N_BASE_SAMPLES, SOBOL_MAX_EVALS,
)
from .metrics import MetricConfig

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _as_xgb_matrix(X) -> np.ndarray:
    """Return a compact numeric matrix for repeated XGBoost trial scoring."""
    if hasattr(X, "toarray"):
        return X.astype(np.float32)
    arr = np.asarray(X)
    return arr.astype(np.float32, copy=False) if np.issubdtype(arr.dtype, np.floating) else arr


# ── N1: XGBoost iteration logging callback ────────────────────────────────────

class _IterationLogCallback(xgb_callback.TrainingCallback):
    """N1: Log train/val metrics every CB_LOG_PERIOD boosting rounds."""

    def __init__(self, period: int = CB_LOG_PERIOD, label: str = "final") -> None:
        super().__init__()
        self.period   = max(1, period)
        self.label    = label
        self.history: list[dict] = []
        self._prev_val: float | None = None

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        train_sets = [k for k in evals_log if "train" in k.lower()]
        val_sets   = [k for k in evals_log if "train" not in k.lower()]

        def _last(ds):
            for ds_name in ds:
                for metric_name, vals in evals_log[ds_name].items():
                    if vals:
                        return ds_name, metric_name, float(vals[-1])
            return None

        tr_info  = _last(train_sets)
        val_info = _last(val_sets)
        tr_val   = tr_info[2]  if tr_info  else float("nan")
        val_val  = val_info[2] if val_info else float("nan")
        self.history.append({"round": epoch + 1, "train": tr_val, "val": val_val, "label": self.label})

        if (epoch + 1) % self.period == 0 or epoch == 0:
            delta = (f" (Δval {val_val - self._prev_val:+.4f})"
                     if self._prev_val is not None else "")
            tr_name  = f"{tr_info[0]}/{tr_info[1]}"   if tr_info  else "train/??"
            val_name = f"{val_info[0]}/{val_info[1]}" if val_info else "val/??"
            log.info("  [%s] round %4d | %s=%.4f | %s=%.4f%s",
                     self.label, epoch + 1, tr_name, tr_val, val_name, val_val, delta)
            self._prev_val = val_val
        return False


# ── V2: GPU helper ────────────────────────────────────────────────────────────

def _resolve_tree_method() -> tuple[str, str | None]:
    """V2: Resolve XGBoost tree_method/device based on USE_GPU and runtime availability."""
    if not USE_GPU:
        return "hist", None

    gpu_available  = False
    detect_method  = "unknown"

    for method, fn in [
        ("cupy",       lambda: __import__("cupy").cuda.runtime.getDeviceCount()),
        ("torch.cuda", lambda: __import__("torch").cuda.is_available() or (_ for _ in ()).throw(RuntimeError())),
    ]:
        try:
            fn()
            gpu_available = True
            detect_method = method
            break
        except Exception:
            pass

    if not gpu_available:
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                gpu_available = True
                detect_method = "nvidia-smi"
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
      robust_cols — columns in ROBUST_SCALER_COLS:
                    RobustScaler only. Outlier structure is preserved.
      num_cols    — all other numerical cols:
                    impute → optional PowerTransformer → optional PCA.

    Internal keyword arguments (prefixed with _) are not part of the public
    API and exist only for the search-phase fast path in tune_hyperparameters.
    Do not pass them from train.py or external callers.

      _power_transform : override POWER_TRANSFORM for this build only.
                         None (default) → respect the global setting.
      _ct_n_jobs       : n_jobs for ColumnTransformer.
                         -1 (default) → use all cores (final fit).
                         1            → no subprocess overhead (search pre-fit).
    """
    params      = params or {}
    tree_method, device = _cached_tree_method()

    # Resolve power-transform flag: caller override takes precedence over global.
    apply_power_transform = POWER_TRANSFORM if _power_transform is None else _power_transform

    # ── Split numerical cols into robust-scaled vs power-transformed ──────────
    robust_in_num = [c for c in ROBUST_SCALER_COLS if c in num_cols]
    pca_cols      = [c for c in num_cols if c not in robust_in_num]

    robust_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    ohe_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    te_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(smooth="auto", random_state=RANDOM_STATE)),
    ])

    transformers = []
    if robust_in_num:
        transformers.append(("robust", robust_transformer, robust_in_num))
    if pca_cols:
        if use_pca:
            # PCA is a joint decomposition — all cols must stay in one branch.
            pca_steps = [("imputer", SimpleImputer(strategy="median"))]
            if apply_power_transform:
                pca_steps.append(("power", PowerTransformer(method="yeo-johnson")))
            n_comp = PCA_MAX_COMPONENTS if PCA_MAX_COMPONENTS is not None else PCA_VARIANCE
            pca_steps.append(("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)))
            transformers.append(("num", Pipeline(pca_steps), pca_cols))
        elif _ct_n_jobs == 1:
            # Search pre-fit fast path: single branch eliminates loky fork/IPC
            # overhead. With n_jobs=1 there are no workers to parallelise across
            # anyway, so per-column branches only add scheduler overhead.
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if apply_power_transform:
                num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
            transformers.append(("num", Pipeline(num_steps), pca_cols))
        else:
            # Final fit (n_jobs=-1): one branch per column so ColumnTransformer
            # distributes work across cores. Fork cost is justified here because
            # each worker processes the full train+val set.
            def _col_pipe():
                steps = [("imputer", SimpleImputer(strategy="median"))]
                if apply_power_transform:
                    steps.append(("power", PowerTransformer(method="yeo-johnson")))
                return Pipeline(steps)
            for col in pca_cols:
                safe_name = f"num_{col}".replace("__", "_")
                transformers.append((safe_name, _col_pipe(), [col]))
    if ohe_cat_cols:
        transformers.append(("cat",    ohe_transformer, ohe_cat_cols))
    if te_cat_cols:
        transformers.append(("te_cat", te_transformer,  te_cat_cols))

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
    if task == "classification" and n_classes and n_classes > 2:
        objective = "multi:softprob"
    elif task == "classification":
        objective = "binary:logistic"
    else:
        objective = "reg:squarederror"
    tree_method, device = _cached_tree_method()
    shared = dict(
        objective=objective,
        eval_metric=metric.eval_metric,
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        min_child_weight=float(params.get("min_child_weight", 1)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        nthread=int(params.get("nthread", -1)),
        seed=RANDOM_STATE,
        tree_method=tree_method,
        verbosity=0,
    )
    if device is not None:
        shared["device"] = device
    if metric.scale_pos_weight is not None:
        shared["scale_pos_weight"] = metric.scale_pos_weight
    if task == "classification" and n_classes and n_classes > 2:
        shared["num_class"] = int(n_classes)
    return shared


def _score_booster(
    booster: xgb.Booster,
    dmat: xgb.DMatrix,
    y_true: np.ndarray,
    task: str,
    metric: MetricConfig,
) -> float:
    raw = booster.predict(dmat)
    if task == "classification":
        if raw.ndim == 2:
            y_pred = np.argmax(raw, axis=1)
            y_proba = raw if metric.needs_proba else None
        else:
            y_proba = raw if metric.needs_proba else None
            y_pred = (raw >= 0.5).astype(int)
    else:
        y_pred = raw
        y_proba = None
    return float(metric.score(y_true, y_pred, y_proba))


def _train_xgb_score(
    X_train_proc,
    y_train_arr: np.ndarray,
    X_val_proc,
    y_val_arr: np.ndarray,
    task: str,
    metric: MetricConfig,
    params: dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int | None = None,
) -> tuple[float, int]:
    n_classes = len(np.unique(y_train_arr)) if task == "classification" else None
    xgb_params = _xgb_model_params(task, metric, params, n_classes=n_classes)
    dtrain = xgb.DMatrix(X_train_proc, label=y_train_arr)
    dval = xgb.DMatrix(X_val_proc, label=y_val_arr)
    fit_kwargs: dict[str, Any] = {"verbose_eval": False}
    evals = [(dval, "validation")]
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=max(1, int(n_estimators)),
        evals=evals,
        **fit_kwargs,
    )
    best_iteration = int(getattr(booster, "best_iteration", n_estimators - 1) or 0)
    return _score_booster(booster, dval, y_val_arr, task, metric), best_iteration


def _score_candidate(
    X_train_proc,
    y_train_arr: np.ndarray,
    X_val_proc,
    y_val_arr: np.ndarray,
    task: str,
    metric: MetricConfig,
    params: dict[str, Any],
    n_estimators: int,
) -> float:
    score, _ = _train_xgb_score(
        X_train_proc, y_train_arr, X_val_proc, y_val_arr,
        task, metric, params, n_estimators,
    )
    return score


def _run_sobol_sensitivity(
    X_train_proc,
    y_train_arr: np.ndarray,
    X_val_proc,
    y_val_arr: np.ndarray,
    task: str,
    metric: MetricConfig,
    n_sub: int,
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

    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = len(X_train_proc)
    n_eval = min(max(100, n_sub), n_rows)
    idx = rng.choice(n_rows, size=n_eval, replace=False) if n_eval < n_rows else np.arange(n_rows)

    scores = []
    log.info("[Sobol] Analysing max_depth, learning_rate, subsample (%d evaluations)...", len(samples))
    for raw_depth, lr, subsample in samples:
        params = {
            "max_depth": int(round(raw_depth)),
            "learning_rate": float(lr),
            "subsample": float(subsample),
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        try:
            score = _score_candidate(
                X_train_proc[idx], y_train_arr[idx],
                X_val_proc, y_val_arr,
                task, metric, params,
                n_estimators=min(150, N_ESTIMATORS_MAX),
            )
            scores.append(score)
        except Exception as exc:
            log.debug("[Sobol] candidate failed: %s", exc)
            scores.append(np.nan)

    y = np.asarray(scores, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 8:
        return {"enabled": True, "status": "skipped", "reason": "too few successful evaluations"}
    samples_used = samples[finite]
    y_used = y[finite]
    try:
        si = sobol_analyze.analyze(problem, y_used, calc_second_order=False, print_to_console=False)
        rows = []
        for name, s1, st, s1_conf, st_conf in zip(
            problem["names"], si["S1"], si["ST"], si["S1_conf"], si["ST_conf"]
        ):
            rows.append({
                "parameter": name,
                "first_order": float(s1),
                "total_order": float(st),
                "first_order_conf": float(s1_conf),
                "total_order_conf": float(st_conf),
            })
        rows.sort(key=lambda r: abs(r["total_order"]), reverse=True)
        log.info("[Sobol] Parameter impact ranking: %s", rows)
        return {
            "enabled": True,
            "status": "completed",
            "metric": metric.name,
            "direction": metric.direction,
            "evaluations": int(finite.sum()),
            "parameters": rows,
        }
    except Exception as exc:
        # SALib requires the full Sobol design for exact indices. If users cap
        # evaluations too aggressively, keep a useful correlation fallback.
        corr_rows = []
        for i, name in enumerate(problem["names"]):
            corr = np.corrcoef(samples_used[:, i], y_used)[0, 1]
            corr_rows.append({"parameter": name, "score_correlation": float(corr) if np.isfinite(corr) else 0.0})
        corr_rows.sort(key=lambda r: abs(r["score_correlation"]), reverse=True)
        log.warning("[Sobol] Exact analysis unavailable (%s); using correlation fallback.", exc)
        return {
            "enabled": True,
            "status": "fallback",
            "metric": metric.name,
            "evaluations": int(finite.sum()),
            "parameters": corr_rows,
        }


def _native_xgb_cv_search(
    X_train_proc,
    y_train_arr: np.ndarray,
    task: str,
    metric: MetricConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    dtrain = xgb.DMatrix(X_train_proc, label=y_train_arr)
    n_classes = len(np.unique(y_train_arr)) if task == "classification" else None
    xgb_params = _xgb_model_params(task, metric, params, n_classes=n_classes)
    stratified = bool(task == "classification")
    folds = CV_FOLDS if CV_FOLDS > 1 else 5
    log.info(
        "[5/9] Native xgboost.cv search [ rounds=%d | folds=%d | metric=%s ]...",
        NATIVE_XGB_CV_ROUNDS, folds, metric.eval_metric,
    )
    cv = xgb.cv(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=max(1, NATIVE_XGB_CV_ROUNDS),
        nfold=folds,
        stratified=stratified,
        seed=RANDOM_STATE,
        early_stopping_rounds=max(1, NATIVE_XGB_CV_EARLY_STOP),
        verbose_eval=False,
    )
    best_n = int(len(cv))
    metric_cols = [c for c in cv.columns if c.endswith("-mean")]
    best_score = float(cv[metric_cols[-1]].iloc[-1]) if metric_cols else float("nan")
    params["n_estimators"] = best_n
    summary = {
        "backend": "native_xgb_cv",
        "best_n_estimators": best_n,
        "best_cv_score": best_score,
        "cv_columns": list(cv.columns),
    }
    log.info("  Native xgboost.cv best_n_estimators=%d score=%.6f", best_n, best_score)
    return params, summary

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
    import time

    # ── Pre-fit preprocessor once (search-phase fast path) ───────────────────
    # PowerTransformer (Yeo-Johnson) runs an MLE optimisation per column and
    # dominates pre-fit time on wide or large datasets. XGBoost's tree splits
    # are scale- and monotone-transform-invariant, so skipping it during the
    # search phase does not change which hyperparameter combinations rank best.
    #
    # The one exception is use_pca=True: PCA is sensitive to feature scale and
    # distribution shape, so PowerTransformer must be retained to keep the
    # search-phase PCA components comparable to those in the final model.
    #
    # _ct_n_jobs=1 avoids loky subprocess overhead: with a single consolidated
    # num branch (3-4 total branches) the fork/IPC cost exceeds the work itself
    # on large datasets. The final fit uses n_jobs=-1 as usual.
    _apply_power = POWER_TRANSFORM and use_pca
    log.info(
        "  Pre-fitting preprocessor on X_train "
        "(search fast-path: n_jobs=1, power_transform=%s)...",
        _apply_power,
    )
    _prep_pipe = build_pipeline(
        num_cols, ohe_cat_cols, te_cat_cols, task, metric,
        use_pca=use_pca,
        _power_transform=_apply_power,
        _ct_n_jobs=1,
    )
    preprocessor = _prep_pipe.named_steps["preprocessor"]
    y_train_arr  = np.array(y_train)
    y_val_arr    = np.array(y_val)
    X_train_proc = _as_xgb_matrix(preprocessor.fit_transform(X_train, y_train_arr))
    X_val_proc   = _as_xgb_matrix(preprocessor.transform(X_val))

    n_rows = len(X_train_proc)

    # ── CV vs. fast-path: data-driven default, manual override respected ──────
    # Manual override: CV_FOLDS > 0 forces CV on; CV_FOLDS == 0 forces it off.
    # Auto (CV_FOLDS < 0 or sentinel -1 in config): use CV only for small data.
    _CV_AUTO_THRESHOLD = 50_000
    if CV_FOLDS > 0:
        use_cv = True
    elif CV_FOLDS == 0:
        use_cv = False
    else:                           # CV_FOLDS = -1 → auto
        use_cv = n_rows < _CV_AUTO_THRESHOLD

    if use_cv:
        n_folds = CV_FOLDS if CV_FOLDS > 0 else 5   # auto → 5 folds
        cv_splitter = (
            TimeSeriesSplit(n_splits=n_folds)
            if CV_STRATEGY == "timeseries"
            else StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        )
        log.info("  CV enabled: %s, %d folds (n_rows=%d)",
                 cv_splitter.__class__.__name__, n_folds, n_rows)
    else:
        log.info("  CV disabled — fast-path subsample (n_rows=%d)", n_rows)

    # ── Subsample cap: absolute rows, not just a fraction ─────────────────────
    # SEARCH_SUBSAMPLE=0.6 on 500k rows = 300k — defeats the "fast path" purpose.
    # Cap at 50k rows regardless of fraction.
    _SUBSAMPLE_ROW_CAP = 50_000
    n_sub = min(max(int(n_rows * SEARCH_SUBSAMPLE), 100), _SUBSAMPLE_ROW_CAP)
    log.info("  Search subsample: %d / %d rows (%.1f%%)",
             n_sub, n_rows, 100 * n_sub / n_rows)

    search_summary: dict[str, Any] = {
        "backend": SEARCH_BACKEND,
        "sobol_sensitivity": _run_sobol_sensitivity(
            X_train_proc, y_train_arr, X_val_proc, y_val_arr,
            task, metric, n_sub,
        ),
    }

    if SEARCH_BACKEND in ("native_xgb_cv", "xgb_cv", "xgboost_cv"):
        best_params, native_summary = _native_xgb_cv_search(
            X_train_proc, y_train_arr, task, metric
        )
        search_summary.update(native_summary)
        return best_params, None, search_summary
    if SEARCH_BACKEND != "optuna":
        log.warning("  Unknown search.backend='%s'; falling back to Optuna.", SEARCH_BACKEND)
        search_summary["backend"] = "optuna"

    # ── Budget-driven n_trials / timeout ──────────────────────────────────────
    # Priority: explicit N_TRIALS / OPTUNA_TIMEOUT beat the budget knob.
    # Budget knob (OPTUNA_BUDGET_SECONDS) is the "easy" path for new users.
    budget_sec = OPTUNA_BUDGET_SECONDS   # may be None

    def _run_canary() -> float:
        """Time one trial to calibrate n_trials from the budget."""
        t0 = time.perf_counter()
        _n   = min(n_sub, 5_000)          # tiny slice — just need a timing signal
        _idx = np.random.default_rng(0).integers(0, n_rows, size=_n)
        _train_xgb_score(
            X_train_proc[_idx], y_train_arr[_idx],
            X_val_proc, y_val_arr,
            task, metric,
            {
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            n_estimators=50,
            early_stopping_rounds=10,
        )
        elapsed = time.perf_counter() - t0
        return max(elapsed, 0.1)     # floor to avoid division by zero

    # Derive timeout and n_trials
    if OPTUNA_TIMEOUT is not None:
        # Fully explicit override — respect it as-is
        effective_timeout = OPTUNA_TIMEOUT
        effective_n_trials = N_TRIALS
        log.info("  Budget: explicit timeout=%ds, n_trials=%d", effective_timeout, effective_n_trials)
    elif budget_sec is not None:
        canary_sec = _run_canary()
        log.info("  Canary trial: %.2fs per trial", canary_sec)
        # Reserve 10 % of the budget for TPE overhead / pruner book-keeping
        usable = budget_sec * 0.90
        derived_trials = max(10, int(usable / canary_sec))
        effective_timeout = budget_sec
        effective_n_trials = derived_trials
        log.info("  Budget=%ds → ~%d trials (%.1fs each)",
                 budget_sec, derived_trials, canary_sec)
    else:
        # No budget, no timeout — fall back to bare N_TRIALS, no wall-clock guard
        effective_timeout = None
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

        # ── n_estimators: tune directly or rely on early stopping ─────────────
        if TUNE_N_ESTIMATORS:
            trial_n_estimators = trial.suggest_int("n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX)
            trial_early_stop   = None
        else:
            trial_n_estimators = N_ESTIMATORS_MAX
            trial_early_stop   = EARLY_STOP_RNDS

        if use_cv:
            # ── Parallel CV folds ─────────────────────────────────────────────
            # xgb_params is the same for every fold in this trial — build once.
            # Each worker receives contiguous numpy slices (cheap to pickle) and
            # constructs its own DMatrix locally (no IPC serialisation of DMatrix).
            _trial_xgb_params = _xgb_model_params(
                task, metric, {**params, "nthread": 1}, n_classes=_n_classes_cv
            )
            _trial_early_stop = int(trial_early_stop) if trial_early_stop else None

            def _fit_fold_np(X_tr_f, X_vl_f, y_tr_f, y_vl_f):
                dtrain_f = xgb.DMatrix(X_tr_f, label=y_tr_f)
                dval_f   = xgb.DMatrix(X_vl_f, label=y_vl_f)
                fit_kwargs: dict = {"verbose_eval": False}
                if _trial_early_stop:
                    fit_kwargs["early_stopping_rounds"] = _trial_early_stop
                booster = xgb.train(
                    _trial_xgb_params, dtrain_f,
                    num_boost_round=max(1, int(trial_n_estimators)),
                    evals=[(dval_f, "validation")],
                    **fit_kwargs,
                )
                return _score_booster(booster, dval_f, y_vl_f, task, metric)

            fold_scores = joblib.Parallel(n_jobs=-1, backend="loky")(
                joblib.delayed(_fit_fold_np)(X_tr_f, X_vl_f, y_tr_f, y_vl_f)
                for X_tr_f, X_vl_f, y_tr_f, y_vl_f in _cv_fold_arrays
            )
            trial.report(float(np.mean(fold_scores)), step=n_folds - 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return float(np.mean(fold_scores))

        # Fast-path: stratified subsample using precomputed per-class indices.
        # Inlines xgb.train directly to reuse _dval_search (precomputed val
        # DMatrix) and _n_classes — bypasses _train_xgb_score which rebuilds
        # both from scratch every trial.
        if task == "classification":
            rng = np.random.default_rng(trial.number)
            idx = np.concatenate([
                rng.choice(cls_idx, size=_class_counts[cls], replace=False)
                for cls, cls_idx in _class_indices.items()
            ])
        else:
            idx = np.random.default_rng(trial.number).choice(n_rows, size=n_sub, replace=False)

        _trial_params = _xgb_model_params(task, metric, params, n_classes=_n_classes)
        dtrain = xgb.DMatrix(X_train_proc[idx], label=y_train_arr[idx])
        fit_kwargs: dict[str, Any] = {"verbose_eval": False}
        if trial_early_stop is not None and trial_early_stop > 0:
            fit_kwargs["early_stopping_rounds"] = int(trial_early_stop)
        booster = xgb.train(
            _trial_params, dtrain,
            num_boost_round=max(1, int(trial_n_estimators)),
            evals=[(_dval_search, "validation")],
            **fit_kwargs,
        )
        best_iteration = int(getattr(booster, "best_iteration", trial_n_estimators - 1) or 0)
        # +1: best_iteration is 0-based round index; n_estimators is a count.
        trial.set_user_attr("best_n_estimators", best_iteration + 1)
        score = _score_booster(booster, _dval_search, y_val_arr, task, metric)
        trial.report(score, step=best_iteration)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return score

    # ── Precompute objects shared across all trials ───────────────────────────
    # _dval_search : val DMatrix built once; _train_xgb_score rebuilds it every
    #                trial which is the single biggest redundant cost on fast-path.
    # _n_classes   : np.unique sort of full y_train_arr, hoisted out of per-trial.
    # _class_indices/_class_counts: per-class row lookup for cheap stratified
    #                sampling — replaces StratifiedShuffleSplit O(n_rows) scan.
    _n_classes   = int(len(np.unique(y_train_arr))) if task == "classification" else None
    _dval_search = xgb.DMatrix(X_val_proc, label=y_val_arr)

    if task == "classification" and not use_cv:
        _class_indices = {
            cls: np.where(y_train_arr == cls)[0]
            for cls in np.unique(y_train_arr)
        }
        _class_counts = {
            cls: max(1, int(round(n_sub * len(idxs) / n_rows)))
            for cls, idxs in _class_indices.items()
        }
    else:
        _class_indices = {}
        _class_counts  = {}

    # ── Precompute CV numpy slices once ──────────────────────────────────────
    # Store contiguous numpy copies, not DMatrix objects. DMatrix cannot be
    # efficiently pickled across loky workers — passing it through the IPC pipe
    # serialises the entire binary blob each trial, which is slower than letting
    # each worker build its own DMatrix locally from a small contiguous array.
    # Fancy-indexing X_train_proc[fold_idx] here copies once; inside the worker
    # it arrives as a contiguous array so DMatrix construction is fast.
    if use_cv:
        _cv_fold_arrays = [
            (np.ascontiguousarray(X_train_proc[tr]),
             np.ascontiguousarray(X_train_proc[vl]),
             y_train_arr[tr], y_train_arr[vl])
            for tr, vl in cv_splitter.split(X_train_proc, y_train_arr)
        ]
        # Hoist n_classes — doesn't change across trials
        _n_classes_cv = int(len(np.unique(y_train_arr))) if task == "classification" else None
    else:
        _cv_fold_arrays = []
        _n_classes_cv   = None

    # ── Run study ─────────────────────────────────────────────────────────────
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=10)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    _timeout_label = f"{effective_timeout}s" if effective_timeout is not None else "none"
    _nest_label    = f"tuned({N_ESTIMATORS_MIN}–{N_ESTIMATORS_MAX})" if TUNE_N_ESTIMATORS else f"fixed@{N_ESTIMATORS_MAX}+earlystop"
    log.info("[5/9] Optuna search [ Trials : %d | Timeout : %s | Subsample rows : %d | "
             "Metric : %s | CV : %s | Wide : %s | n_estimators : %s | Parallel folds : %s ]...",
             effective_n_trials,
             f"{effective_timeout}s" if effective_timeout is not None else "none",
             n_sub,
             metric.name,
             n_folds if use_cv else "off",
             WIDE_SEARCH,
             _nest_label,
             "yes" if use_cv else "n/a")
    # Track running best to avoid O(n²) completed-trial scan on each tqdm update
    _running_best: list[float] = [float("-inf")]
    with tqdm(total=effective_n_trials, ncols=100, desc="Optuna search") as pbar:
        def _objective_with_progress(trial):
            result = objective(trial)
            if result > _running_best[0]:
                _running_best[0] = result
            pbar.set_postfix(best=f"{_running_best[0]:.6f}", trial=trial.number)
            pbar.update(1)
            return result

        study.optimize(
            _objective_with_progress,
            n_trials=effective_n_trials,
            timeout=effective_timeout,
            n_jobs=1,
            show_progress_bar=False,
        )
    pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    log.info("  Best %s=%.4f  params=%s  (complete=%d, pruned=%d)",
             metric.name, study.best_value, study.best_params, complete, pruned)
    search_summary.update({
        "backend": "optuna",
        "best_value": float(study.best_value),
        "complete_trials": complete,
        "pruned_trials": pruned,
    })
    return study.best_params, study, search_summary

# ── Top-K ensemble (UPGRADE 11) ───────────────────────────────────────────────

from joblib import Parallel, delayed


def _fit_one(name: str, est, X, y):
    """Fit a single pipeline — runs in a worker process."""
    est.fit(X, y)
    return name, est


class _IdentityLabelEncoder:
    """Picklable stand-in for sklearn's LabelEncoder inside VotingClassifier.

    VotingClassifier.predict/predict_proba use ``self.le_.transform`` to map
    integer class indices back to original labels.  When the targets are
    already 0-based integers we can skip the real LabelEncoder and use this
    lightweight, fully-picklable replacement instead.
    """

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
    if study is None:
        log.info("[ensemble] [skipped] — Optuna study unavailable for current search backend")
        return None, {
            "enabled": True, "status": "skipped",
            "reason": "Optuna study unavailable for current search backend",
            "top_k_requested": top_k, "trials_used": 0,
        }

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete_trials.sort(
        key=lambda t: float(t.value),
        reverse=(metric.direction == "maximize"),
    )
    selected = complete_trials[:top_k]

    if len(selected) < 2:
        log.info("[ensemble] [skipped] — fewer than 2 completed Optuna trials")
        return None, {
            "enabled": True, "status": "skipped",
            "reason": "fewer than 2 completed Optuna trials",
            "top_k_requested": top_k, "trials_used": len(selected),
        }

    # Build unfitted pipelines — n_estimators resolved per member:
    #   1. early-stop path (TUNE_N_ESTIMATORS=False): use best_n_estimators stored
    #      in user_attrs by the objective (best_iteration + 1).
    #   2. tuned path (TUNE_N_ESTIMATORS=True): n_estimators lives in trial.params.
    #   3. fallback: global n_estimators argument.
    named_estimators = []
    trial_rows = []
    n_est_global = max(1, int(n_estimators))
    for idx, trial in enumerate(selected, 1):
        n_est_member = int(
            trial.user_attrs.get("best_n_estimators", None)
            or trial.params.get("n_estimators", None)
            or n_est_global
        )
        n_est_member = max(1, n_est_member)
        est = build_pipeline(
            num_cols, ohe_cat_cols, te_cat_cols, task, metric,
            params=trial.params, n_estimators=n_est_member,
            early_stop=0, use_pca=use_pca,
        )
        named_estimators.append((f"trial_{trial.number}", est))
        trial_rows.append({
            "rank": idx,
            "trial_number": trial.number,
            "value": float(trial.value),
            "n_estimators": n_est_member,
            "params": dict(trial.params),
        })

    # ── Parallel fit ──────────────────────────────────────────────────────────
    # Each member is independent — no need to fit them sequentially.
    # n_jobs=-1 uses all cores; prefer="threads" avoids pickling overhead when
    # pipelines contain objects that serialise poorly (swap to "processes" if
    # your estimators release the GIL and serialise cleanly).
    log.info("[ensemble] Fitting %d members in parallel...", len(named_estimators))
    fitted_pairs = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit_one)(name, est, X_trainval, y_trainval)
        for name, est in named_estimators
    )
    fitted_estimators = [(name, est) for name, est in fitted_pairs]

    # Wrap in a voting ensemble — pass pre_fit=True to skip internal refit
    # (sklearn ≥ 1.4 honours already-fitted estimators automatically)
    ensemble: Any = (
        VotingClassifier(estimators=fitted_estimators, voting="soft", n_jobs=-1)
        if task == "classification"
        else VotingRegressor(estimators=fitted_estimators, n_jobs=-1)
    )
    # Mark as fitted so sklearn won't refit internally on predict calls
    ensemble.estimators_ = [est for _, est in fitted_estimators]
    ensemble.named_estimators_ = dict(fitted_estimators)
    if task == "classification":
        # VotingClassifier also needs these attributes
        classes = np.unique(y_trainval)
        ensemble.le_      = _IdentityLabelEncoder(classes)
        ensemble.classes_ = classes

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_test_arr = np.asarray(y_test)   # cast once

    if task == "classification":
        if metric.needs_proba:
            y_proba     = ensemble.predict_proba(X_test)[:, 1]
            y_pred      = (y_proba >= threshold).astype(int)
            eval_metrics = {
                "selected_metric": float(metric.score(y_test_arr, y_pred, y_proba)),
                "auprc":           float(average_precision_score(y_test_arr, y_proba)),
                "roc_auc":         float(roc_auc_score(y_test_arr, y_proba)),
                "threshold":       float(threshold),
            }
        else:
            y_pred = ensemble.predict(X_test)
            eval_metrics = {
                "selected_metric": float(metric.score(y_test_arr, y_pred, None)),
                "threshold": None,
            }
    else:
        y_pred = ensemble.predict(X_test)
        if log_transformed:
            y_test_arr = np.expm1(y_test_arr)
            y_pred     = np.expm1(y_pred)
        eval_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_arr, y_pred))),
            "mae":  float(mean_absolute_error(y_test_arr, y_pred)),
            "r2":   float(r2_score(y_test_arr, y_pred)),
        }

    return ensemble, {
        "enabled": True, "status": "fitted",
        "top_k_requested": top_k, "trials_used": len(fitted_estimators),
        "n_estimators_global_fallback": n_est_global,
        "member_trials": trial_rows,
        "eval_metrics": eval_metrics,
    }