"""pipeline.py — XGB callback, GPU helper, pipeline builder, Optuna tuning, ensemble."""
from __future__ import annotations
import time
import importlib.metadata
import logging
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
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, TargetEncoder
from xgboost import XGBClassifier, XGBRegressor
from xgboost import callback as xgb_callback

from .settings import (
    CB_LOG_PERIOD, CALIBRATION_ENABLED, CV_FOLDS, CV_STRATEGY, EARLY_STOP_RNDS,
    N_ESTIMATORS_MAX, N_TRIALS, OPTUNA_BUDGET_SECONDS, OPTUNA_TIMEOUT,
    PCA_MAX_COMPONENTS, PCA_VARIANCE,
    RANDOM_STATE, SEARCH_SUBSAMPLE, USE_GPU, WIDE_SEARCH, ENSEMBLE_ENABLED, ENSEMBLE_TOP_K, POWER_TRANSFORM,
)
from .metrics import MetricConfig

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
) -> Pipeline:
    """Build preprocessor + XGBoost pipeline."""
    params      = params or {}
    tree_method, device = _resolve_tree_method()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if POWER_TRANSFORM:
        num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
    if use_pca:
        n_comp = PCA_MAX_COMPONENTS if PCA_MAX_COMPONENTS is not None else PCA_VARIANCE
        num_steps.append(("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)))

    num_transformer = Pipeline(num_steps)
    ohe_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    te_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(smooth="auto", random_state=RANDOM_STATE)),
    ])

    transformers = []
    if num_cols:     transformers.append(("num",    num_transformer, num_cols))
    if ohe_cat_cols: transformers.append(("cat",    ohe_transformer, ohe_cat_cols))
    if te_cat_cols:  transformers.append(("te_cat", te_transformer,  te_cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

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
) -> tuple[dict, optuna.Study]:
    """FIX 2 + U1/U8/U17: Optuna TPE search. Returns (best_params, study)."""
    import time

    # ── Pre-fit preprocessor once ─────────────────────────────────────────────
    log.info("  Pre-fitting preprocessor on X_train (once, loky backend)...")
    _prep_pipe = build_pipeline(num_cols, ohe_cat_cols, te_cat_cols, task, metric, use_pca=use_pca)
    preprocessor = _prep_pipe.named_steps["preprocessor"]
    with joblib.parallel_backend("loky", n_jobs=-1):
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_val_proc   = preprocessor.transform(X_val)
    y_train_arr = np.array(y_train)
    y_val_arr   = np.array(y_val)

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

    # ── Budget-driven n_trials / timeout ──────────────────────────────────────
    # Priority: explicit N_TRIALS / OPTUNA_TIMEOUT beat the budget knob.
    # Budget knob (OPTUNA_BUDGET_SECONDS) is the "easy" path for new users.
    budget_sec = OPTUNA_BUDGET_SECONDS   # may be None

    def _run_canary() -> float:
        """Time one trial to calibrate n_trials from the budget."""
        t0 = time.perf_counter()
        _mdl = XGBClassifier if task == "classification" else XGBRegressor
        _n   = min(n_sub, 5_000)          # tiny slice — just need a timing signal
        _idx = np.random.default_rng(0).integers(0, n_rows, size=_n)
        _m   = _mdl(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            early_stopping_rounds=10, random_state=RANDOM_STATE,
            eval_metric=metric.eval_metric, nthread=-1, tree_method="hist",
        )
        _m.fit(X_train_proc[_idx], y_train_arr[_idx],
               eval_set=[(X_val_proc, y_val_arr)], verbose=False)
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
        log.info("  Budget: n_trials=%d, no timeout", effective_n_trials)

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

        shared = dict(
            n_estimators=N_ESTIMATORS_MAX, early_stopping_rounds=EARLY_STOP_RNDS,
            random_state=RANDOM_STATE, eval_metric=metric.eval_metric,
            nthread=-1,   # ← was nthread=1; safe because Optuna n_jobs=1
            **params,
        )
        if metric.scale_pos_weight is not None:
            shared["scale_pos_weight"] = metric.scale_pos_weight

        if use_cv:
            fold_scores: list[float] = []
            for fold_idx, (tr_idx, vl_idx) in enumerate(cv_splitter.split(X_train_proc, y_train_arr)):
                X_tr_f, X_vl_f = X_train_proc[tr_idx], X_train_proc[vl_idx]
                y_tr_f, y_vl_f = y_train_arr[tr_idx],  y_train_arr[vl_idx]
                mdl = XGBClassifier(**shared) if task == "classification" else XGBRegressor(**shared)
                mdl.fit(X_tr_f, y_tr_f, eval_set=[(X_vl_f, y_vl_f)], verbose=False)
                y_pred  = mdl.predict(X_vl_f)
                y_proba = mdl.predict_proba(X_vl_f)[:, 1] if metric.needs_proba and task == "classification" else None
                fold_scores.append(metric.score(y_vl_f, y_pred, y_proba))
                trial.report(np.mean(fold_scores), step=fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return float(np.mean(fold_scores))

        # Fast-path: stratified subsample (capped at _SUBSAMPLE_ROW_CAP rows)
        if task == "classification":
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=trial.number)
            idx, _ = next(sss.split(X_train_proc, y_train_arr))
        else:
            idx = np.random.RandomState(trial.number).choice(n_rows, size=n_sub, replace=False)
        mdl = XGBClassifier(**shared) if task == "classification" else XGBRegressor(**shared)
        mdl.fit(X_train_proc[idx], y_train_arr[idx], eval_set=[(X_val_proc, y_val_arr)], verbose=False)
        y_pred  = mdl.predict(X_val_proc)
        y_proba = mdl.predict_proba(X_val_proc)[:, 1] if metric.needs_proba and task == "classification" else None
        score   = metric.score(y_val_arr, y_pred, y_proba)
        trial.report(score, step=mdl.best_iteration)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return score

    # ── Run study ─────────────────────────────────────────────────────────────
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=10)
    study   = optuna.create_study(direction=metric.direction, sampler=sampler, pruner=pruner)
    _timeout_label = f"{effective_timeout}s" if effective_timeout is not None else "none"
    log.info("[5/9] Optuna search (%d trials · %s timeout · %d subsample rows · "
             "metric=%s · cv=%s · wide=%s)...",
             effective_n_trials, _timeout_label, n_sub,
             metric.name, n_folds if use_cv else "off", WIDE_SEARCH)
    with tqdm(total=effective_n_trials, ncols=100, desc="Optuna search") as pbar:
        def _objective_with_progress(trial):
            result = objective(trial)
            completed = [t for t in study.trials if t.value is not None]
            best = max(t.value for t in completed) if completed else float("nan")
            pbar.set_postfix(best=f"{best:.6f}", trial=trial.number)
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
    return study.best_params, study

# ── Top-K ensemble (UPGRADE 11) ───────────────────────────────────────────────

def build_top_k_ensemble(
    study: optuna.Study,
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
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete_trials.sort(key=lambda t: float(t.value), reverse=(metric.direction == "maximize"))
    selected = complete_trials[:top_k]

    if len(selected) < 2:
        return None, {
            "enabled": True, "status": "skipped",
            "reason": "fewer than 2 completed Optuna trials",
            "top_k_requested": top_k, "trials_used": len(selected),
        }

    estimators = []
    trial_rows = []
    for idx, trial in enumerate(selected, 1):
        est = build_pipeline(num_cols, ohe_cat_cols, te_cat_cols, task, metric,
                             params=trial.params, n_estimators=max(1, int(n_estimators)),
                             early_stop=0, use_pca=use_pca)
        estimators.append((f"trial_{trial.number}", est))
        trial_rows.append({"rank": idx, "trial_number": trial.number,
                           "value": float(trial.value), "params": dict(trial.params)})

    ensemble: Any = (
        VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        if task == "classification"
        else VotingRegressor(estimators=estimators, n_jobs=-1)
    )
    log.info("[ensemble] Fitting top-%d soft-voting ensemble...", top_k)
    ensemble.fit(X_trainval, y_trainval)

    if task == "classification":
        if metric.needs_proba:
            y_proba = ensemble.predict_proba(X_test)[:, 1]
            y_pred  = (y_proba >= threshold).astype(int)
            eval_metrics = {
                "selected_metric": float(metric.score(np.array(y_test), y_pred, y_proba)),
                "auprc":           float(average_precision_score(y_test, y_proba)),
                "roc_auc":         float(roc_auc_score(y_test, y_proba)),
                "threshold":       float(threshold),
            }
        else:
            y_pred = ensemble.predict(X_test)
            eval_metrics = {"selected_metric": float(metric.score(np.array(y_test), y_pred, None)),
                            "threshold": None}
    else:
        y_pred      = ensemble.predict(X_test)
        y_test_eval = np.expm1(np.array(y_test)) if log_transformed else np.array(y_test)
        y_pred_eval = np.expm1(y_pred) if log_transformed else y_pred
        eval_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))),
            "mae":  float(mean_absolute_error(y_test_eval, y_pred_eval)),
            "r2":   float(r2_score(y_test_eval, y_pred_eval)),
        }

    return ensemble, {
        "enabled": True, "status": "fitted",
        "top_k_requested": top_k, "trials_used": len(estimators),
        "n_estimators_per_member": int(n_estimators),
        "member_trials": trial_rows, "eval_metrics": eval_metrics,
    }