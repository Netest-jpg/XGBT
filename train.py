"""
XGBoost Prototyping Template — v7
══════════════════════════════════
Orchestrator only. All logic lives in xgb_prototype/:

  deps.py        — N4  dependency version checker
  settings.py    — config loading, all globals, logging setup
  metrics.py     — MetricConfig, select_metric
  data.py        — load_data, clean_data, validate_data, validate_pandera,
                   DriftReport, detect_drift, maybe_log_transform
  features.py    — detect_feature_types, filter_low_variance,
                   generate_feature_interactions, select_features_rfecv
  pipeline.py    — _IterationLogCallback, _resolve_tree_method,
                   build_pipeline, tune_hyperparameters, build_top_k_ensemble
  evaluation.py  — evaluate, analyse_errors, tune_threshold
  plots.py       — all plot_* functions
  tracking.py    — _MLflowRun, write_requirements_lock,
                   register_model, train_summary
  inference.py   — PredictWrapper, ModelServer
  baselines.py   — evaluate_baselines
  drift_monitor.py — ContinuousDriftMonitor
  config.py      — load_config, to_plain_dict
  thresholds.py  — normalize_policy, tune_binary_threshold
"""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)
logging.getLogger("great_expectations").setLevel(logging.WARNING)
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Package imports ───────────────────────────────────────────────────────────

from xgb_prototype.deps import check_deps
from xgb_prototype.settings import (
    # task
    TASK, TARGET_COL, TEST_SIZE, RANDOM_STATE,
    # paths
    MODEL_OUTPUT_DIR, PLOT_OUTPUT_DIR,
    # optuna / model
    N_ESTIMATORS_MAX, N_ESTIMATORS_MIN, TUNE_N_ESTIMATORS,
    N_TRIALS, OPTUNA_TIMEOUT, WIDE_SEARCH,
    SEARCH_SUBSAMPLE, EARLY_STOP_RNDS, CV_FOLDS, CV_STRATEGY,
    SEARCH_BACKEND, SOBOL_ENABLED,
    # pca
    PCA_THRESHOLD, PCA_VARIANCE,
    # features
    FEATURE_SELECTION, CARDINALITY_LIMIT, TARGET_ENC_THRESHOLD,
    VARIANCE_THRESHOLD, INTERACTION_TOP_K, AUTO_FE_ENABLED, AUTO_FE_ENGINE,
    # drift
    DRIFT_ALPHA, DRIFT_WARN_ONLY,
    DRIFT_MONITOR_ENABLED, DRIFT_MONITOR_PERSISTENCE,
    DRIFT_MONITOR_MIN_RATIO, DRIFT_MONITOR_RETRAIN_RATIO,
    DRIFT_MONITOR_RETRAIN_SEVERITY,
    # misc model flags
    USE_GPU, PANDERA_VALIDATION, METRIC_NAME, CALIBRATION_ENABLED,
    CB_LOG_PERIOD, TARGET_LOG_TRANSFORM, OUTLIER_CONTAMINATION, PDP_TOP_N,
    UNCERTAINTY_ENABLED, UNCERTAINTY_ALPHA,
    UNCERTAINTY_QUANTILE_LOW, UNCERTAINTY_QUANTILE_HIGH,
    # threshold
    THRESHOLD_POLICY,
    # baselines
    BASELINES_ENABLED, BASELINE_INCLUDE_DUMMY,
    BASELINE_INCLUDE_LINEAR, BASELINE_INCLUDE_XGB,
    # ensemble
    ENSEMBLE_ENABLED, ENSEMBLE_TOP_K,
    # diagnostics
    PLOTS_ENABLED, OPTUNA_PLOTS_ENABLED, LEARNING_CURVE_ENABLED,
    PERM_IMPORTANCE_ENABLED, THRESHOLD_SWEEP_ENABLED,
    OUTLIER_REPORT_ENABLED, PDP_ENABLED, PCA_PLOTS_ENABLED,
    SHAP_ENABLED, CALIBRATION_CURVE_ENABLED, CORR_HEATMAP_ENABLED,
    # mlflow
    MLFLOW_URI, MLFLOW_EXPERIMENT,
    # typed config
    APP_CONFIG,
)
from xgb_prototype.metrics import select_metric
from xgb_prototype.data import (
    load_data, clean_data, validate_data, validate_pandera,
    detect_drift, maybe_log_transform, check_config, apply_pretransforms,
    automatic_missing_value_report,
)
from xgb_prototype.features import (
    detect_feature_types, filter_low_variance,
    generate_feature_interactions, select_features_rfecv,
    apply_auto_feature_engineering,
)
from xgb_prototype.pipeline import (
    _IterationLogCallback, build_pipeline,
    tune_hyperparameters, build_top_k_ensemble,
)
from xgb_prototype.evaluation import evaluate, analyse_errors, tune_threshold
from xgb_prototype.plots import (
    plot_feature_importance, plot_optuna_diagnostics, plot_learning_curve,
    plot_permutation_importance, plot_threshold_sweep, plot_outlier_report,
    plot_partial_dependence, plot_pca_diagnostics,
    plot_shap_summary, plot_shap_interactions,
    plot_calibration_curve, plot_correlation_heatmap,
)
from xgb_prototype.tracking import (
    _MLflowRun, write_requirements_lock, register_model, train_summary,
)
from xgb_prototype.inference import PredictWrapper, ModelServer  # noqa: F401 — re-exported for callers
from xgb_prototype.baselines import evaluate_baselines
from xgb_prototype.drift_monitor import ContinuousDriftMonitor
from xgb_prototype.config import to_plain_dict
from xgb_prototype.uncertainty import estimate_uncertainty

# ── Dependency check ──────────────────────────────────────────────────────────

check_deps()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    # ── Dynamic startup banner ───────────────────────────────────────────────
    def _banner() -> None:
        import sys, os
        try:
            _term_w = os.get_terminal_size().columns
        except OSError:
            _term_w = 62          # fallback for piped output / no tty
        W = max(40, min(60, _term_w - 2))

        def _top():     return "╔" + "═" * W + "╗"
        def _bot():     return "╚" + "═" * W + "╝"
        def _div():     return "╠" + "═" * W + "╣"
        def _row(t=""): return "║ " + t[:W - 2].ljust(W - 2) + " ║"

        def _section(label: str, items: list) -> list:
            out = [_div(), _row(f"  {label}")]
            for item in items:
                out.append(_row(f"      · {item}"))
            return out

        lines: list = []
        lines.append("")
        lines.append(_top())
        lines.append(_row())
        title = "XGBoost Training Pipeline"
        pad = (W - 2 - len(title) - 1) // 2
        lines.append(_row(" " * pad + title))
        lines.append(_row())

        # ── Search ────────────────────────────────────────────────────────
        cv_label  = f"CV {CV_FOLDS}-fold ({CV_STRATEGY})" if CV_FOLDS > 0 else "fast-path subsample"
        n_est_lbl = (f"n_estimators tuned [{N_ESTIMATORS_MIN}–{N_ESTIMATORS_MAX}]"
                     if TUNE_N_ESTIMATORS else f"early-stop @ {N_ESTIMATORS_MAX}")
        lines += _section("Search", [
            f"{N_TRIALS} trials",
            f"backend: {SEARCH_BACKEND}",
            cv_label,
            "wide search" if WIDE_SEARCH else "standard search",
            n_est_lbl,
            f"metric: {METRIC_NAME}",
        ])

        # ── Options ───────────────────────────────────────────────────────
        core: list = []
        if CALIBRATION_ENABLED:   core.append("calibration")
        if FEATURE_SELECTION:     core.append("RFECV feature selection")
        if AUTO_FE_ENABLED:       core.append(f"auto feature engineering ({AUTO_FE_ENGINE})")
        if SOBOL_ENABLED:         core.append("Sobol sensitivity")
        if UNCERTAINTY_ENABLED:   core.append("uncertainty estimation")
        if PANDERA_VALIDATION:    core.append("Pandera validation")
        if USE_GPU:               core.append("GPU acceleration")
        if TARGET_LOG_TRANSFORM:  core.append("log-transform target")
        if core:
            lines += _section("Options", core)

        # ── Baselines ─────────────────────────────────────────────────────
        if BASELINES_ENABLED:
            bl: list = []
            if BASELINE_INCLUDE_DUMMY:  bl.append("dummy")
            if BASELINE_INCLUDE_LINEAR: bl.append("logistic regression")
            if BASELINE_INCLUDE_XGB:    bl.append("default XGBoost")
            if bl:
                lines += _section("Baselines", bl)

        # ── Ensemble ──────────────────────────────────────────────────────
        if ENSEMBLE_ENABLED:
            lines += _section("Ensemble", [f"top-{ENSEMBLE_TOP_K} soft-vote"])

        # ── Diagnostics ───────────────────────────────────────────────────
        if PLOTS_ENABLED:
            diag: list = []
            if OPTUNA_PLOTS_ENABLED:    diag.append("Optuna plots")
            if LEARNING_CURVE_ENABLED:  diag.append("learning curve")
            if PERM_IMPORTANCE_ENABLED: diag.append("permutation importance")
            if THRESHOLD_SWEEP_ENABLED: diag.append("threshold sweep")
            if OUTLIER_REPORT_ENABLED:  diag.append("outlier report")
            if PDP_ENABLED:             diag.append(f"partial dependence (top {PDP_TOP_N})")
            if PCA_PLOTS_ENABLED:       diag.append("PCA plots")
            if SHAP_ENABLED:            diag.append("SHAP summary + interactions")
            if CALIBRATION_CURVE_ENABLED: diag.append("calibration curve")
            if CORR_HEATMAP_ENABLED:    diag.append("correlation heatmap")
            if diag:
                lines += _section("Diagnostics", diag)

        # ── Drift Monitor ─────────────────────────────────────────────────
        if DRIFT_MONITOR_ENABLED:
            lines += _section("Drift Monitor", [
                f"persistence = {DRIFT_MONITOR_PERSISTENCE} consecutive checks",
                f"retrain severity = {DRIFT_MONITOR_RETRAIN_SEVERITY}",
            ])

        # ── MLflow ────────────────────────────────────────────────────────
        if MLFLOW_URI:
            lines += _section("MLflow", [
                f"experiment: {MLFLOW_EXPERIMENT}",
                f"uri: {MLFLOW_URI}",
            ])

        lines.append(_bot())
        lines.append("")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    _banner()

    run_id    = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("  Run ID: %s_%s", timestamp, run_id)

    # UPGRADE 26: lock environment immediately so the file is always written
    # Fire-and-forget into a thread so pip freeze doesn't stall the critical path.
    _lock_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lock")
    _lock_future   = _lock_executor.submit(write_requirements_lock, run_id, MODEL_OUTPUT_DIR)

    with _MLflowRun(run_id, timestamp, MLFLOW_URI, MLFLOW_EXPERIMENT) as mlrun:

        # ── 1-3. Load, clean, validate ───────────────────────────────────────
        df = load_data()
        log.info("[0/9] Config validated — target='%s', task='%s'", TARGET_COL, TASK)
        check_config(df)
        df = clean_data(df)
        df = apply_pretransforms(df)
        missing_report = automatic_missing_value_report(
            df, TARGET_COL, output_dir=MODEL_OUTPUT_DIR, run_id=run_id
        )
        validate_data(df)
        validate_pandera(df, TARGET_COL)
        log.info("")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # UPGRADE 30: optional log-transform of regression target
        y, log_transformed = maybe_log_transform(y)

        le = None
        if TASK == "classification":
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y), name=TARGET_COL)

        # ── 3. Split ─────────────────────────────────────────────────────────
        log.info("[metric] Inspecting target distribution...")
        metric = select_metric(y, TASK)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y if TASK == "classification" else None,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE,
            stratify=y_train if TASK == "classification" else None,
        )
        log.info("[3/9] Split — train: %d, val: %d, test: %d",
                 len(X_train), len(X_val), len(X_test))
        log.info("")

        # ── 3b. PCA check ────────────────────────────────────────────────────
        # Feature types needed to evaluate PCA threshold
        num_cols, ohe_cat_cols, te_cat_cols = detect_feature_types(X)

        use_pca = len(num_cols) > PCA_THRESHOLD
        if use_pca:
            log.info(
                "[3b/9] PCA enabled (num_cols=%d > threshold=%d, target variance=%.0f%%)",
                len(num_cols), PCA_THRESHOLD, PCA_VARIANCE * 100,
            )
        else:
            log.info("[3b/9] PCA skipped (num_cols=%d ≤ threshold=%d)",
                     len(num_cols), PCA_THRESHOLD)
        log.info("")

        # ── 3c. Drift detection ──────────────────────────────────────────────
        drift_report = detect_drift(
            X_train, X_test, num_cols, ohe_cat_cols + te_cat_cols,
            y_train=y_train, y_test=y_test,
        )
        log.info("")

        # ── 3d. Act on drift (severity-tiered by p-value) ───────────────────
        drifted_cols: set[str] = set()
        if not DRIFT_WARN_ONLY:
            # Tier by p-value magnitude, not just pass/fail at alpha.
            # p < 0.01  → drop:         distribution too unreliable to train on
            # p < alpha → flag only:    detectable shift but signal likely intact
            DROP_THRESHOLD = 0.01
            drop_cols  = {c: p for c, p in drift_report.pvalues_numerical.items() if p < DROP_THRESHOLD}
            flag_cols  = {c: p for c, p in drift_report.pvalues_numerical.items()
                  if DROP_THRESHOLD <= p < DRIFT_ALPHA}
            drop_cols.update({c: p for c, p in drift_report.pvalues_categorical.items() if p < DROP_THRESHOLD})
            flag_cols.update({c: p for c, p in drift_report.pvalues_categorical.items()
                      if DROP_THRESHOLD <= p < DRIFT_ALPHA})

            if drop_cols:
                to_drop = [c for c in drop_cols if c in X_train.columns]
                X_train      = X_train.drop(columns=to_drop)
                X_val        = X_val.drop(columns=to_drop)
                X_test       = X_test.drop(columns=to_drop)
                num_cols     = [c for c in num_cols     if c not in drop_cols]
                ohe_cat_cols = [c for c in ohe_cat_cols if c not in drop_cols]
                te_cat_cols  = [c for c in te_cat_cols  if c not in drop_cols]
                drifted_cols = set(drop_cols)
                log.warning(
                    "[3d/9] DROPPED %d severely drifted feature(s) (p < %.2f): %s",
                    len(to_drop), DROP_THRESHOLD,
                    {c: f"{p:.4f}" for c, p in drop_cols.items()},
                )

            if flag_cols:
                log.warning(
                    "[3d/9] FLAGGED %d mildly drifted feature(s) (%.2f ≤ p < %.2f), retained: %s",
                    len(flag_cols), DROP_THRESHOLD, DRIFT_ALPHA,
                    {c: f"{p:.4f}" for c, p in flag_cols.items()},
                )
                drifted_cols |= set(flag_cols)


        # ── 4. Feature engineering ───────────────────────────────────────────
        num_cols, ohe_cat_cols, te_cat_cols = filter_low_variance(
            X_train, num_cols, ohe_cat_cols, te_cat_cols
        )

        X_train, X_val, X_test, num_cols = generate_feature_interactions(
            X_train, X_val, X_test, num_cols
        )

        X_train, X_val, X_test, num_cols = apply_auto_feature_engineering(
            X_train, X_val, X_test, y_train,
            num_cols, ohe_cat_cols, te_cat_cols,
        )

        if FEATURE_SELECTION:
            log.info("[RFECV] Feature selection enabled...")
            num_cols, ohe_cat_cols, te_cat_cols = select_features_rfecv(
                X_train, y_train, num_cols, ohe_cat_cols, te_cat_cols, TASK, metric
            )
        log.info("")

        feature_schema = {
            "num_cols":     num_cols,
            "ohe_cat_cols": ohe_cat_cols,
            "te_cat_cols":  te_cat_cols,
            "use_pca":      use_pca,
            "drifted_cols_dropped": sorted(drop_cols) if not DRIFT_WARN_ONLY else [],
            "drifted_cols_flagged": sorted(flag_cols)  if not DRIFT_WARN_ONLY else
                                    sorted(drift_report.drifted_numerical +
                                            drift_report.drifted_categorical),
        }

        # ── Baselines ────────────────────────────────────────────────────────
        log.info("[baseline] Comparing cheap baselines before Optuna tuning...")
        baseline_results, baseline_path = evaluate_baselines(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_cols, ohe_cat_cols, te_cat_cols,
            TASK, metric, RANDOM_STATE, MODEL_OUTPUT_DIR, run_id,
            threshold_policy=THRESHOLD_POLICY,
            log_transformed=log_transformed,
            enabled=BASELINES_ENABLED,
            include_dummy=BASELINE_INCLUDE_DUMMY,
            include_linear=BASELINE_INCLUDE_LINEAR,
            include_default_xgb=BASELINE_INCLUDE_XGB,
        )
        if baseline_path is not None:
            log.info("  Baseline comparison saved → %s", baseline_path)
        log.info("")
        # ── 5. Optuna tuning ─────────────────────────────────────────────────
        best_params, study, search_summary = tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, use_pca,
        )
        log.info("")

        if PLOTS_ENABLED and OPTUNA_PLOTS_ENABLED and study is not None:
            plot_optuna_diagnostics(study)
        elif PLOTS_ENABLED and OPTUNA_PLOTS_ENABLED:
            log.info("[optuna_plots] [skipped] — search backend did not produce an Optuna study")

        # ── Learning curve (before final refit) ──────────────────────────────
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])
        if PLOTS_ENABLED and LEARNING_CURVE_ENABLED:
            log.info("[LC] Computing learning curve...")
            _lc_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, best_params,
                n_estimators=min(100, N_ESTIMATORS_MAX), early_stop=0, use_pca=use_pca,
            )
            plot_learning_curve(_lc_pipeline, X_trainval, y_trainval, metric, TASK)

        # ── 6. Final fit ──────────────────────────────────────────────────────
        # tune_n_estimators=true:  n_estimators in best_params from Optuna,
        #                          fit directly, no early stopping.
        # tune_n_estimators=false: early stopping finds best_iteration,
        #                          then refit at exact depth (original behaviour).
        log.info("[6/9] Fitting final pipeline (train+val) with best params...")

        final_params = dict(best_params)

        # ── Phase A: fit on X_train only, with X_val as a true held-out eval_set.
        # This is the only place where an eval_set is valid — val was never seen by
        # this model, so the logged val metric is honest and can be used for both
        # convergence monitoring and threshold tuning.
        #
        # Bug fixed: the previous code called preprocessor.fit_transform(X_trainval)
        # and then passed X_val (a subset of X_trainval) as eval_set, causing the
        # model to be trained on val and evaluated on val simultaneously, producing
        # a spurious AUCPR=1.0000 from round 150 onward.

        if TUNE_N_ESTIMATORS:
            best_n = int(final_params.pop("n_estimators", N_ESTIMATORS_MAX))

            # Phase A — train-only fit with honest val eval_set (for threshold tuning
            # and calibration; X_val has never been seen by this preprocessor fit).
            # _power_transform=use_pca: PowerTransformer is only meaningful before PCA
            # (scale-sensitive); XGBoost is split-based and invariant to monotonic
            # feature transforms, so skip it when PCA is off. Matches tuning fast-path.
            # _ct_n_jobs=1: avoids loky worker spawn overhead for the preprocessor fit;
            # on a dense float matrix the parallelism cost exceeds the benefit.
            probe_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=best_n, early_stop=0, use_pca=use_pca,
                _power_transform=use_pca, _ct_n_jobs=1,
            )
            preprocessor  = probe_pipeline.named_steps["preprocessor"]
            X_train_proc  = preprocessor.fit_transform(X_train, y_train).astype(np.float32)
            X_val_proc    = preprocessor.transform(X_val).astype(np.float32)
            X_test_proc   = preprocessor.transform(X_test).astype(np.float32)
            _probe_cb     = _IterationLogCallback(period=CB_LOG_PERIOD, label="probe")
            probe_pipeline.named_steps["model"].set_params(callbacks=[_probe_cb])
            probe_pipeline.named_steps["model"].fit(
                X_train_proc, y_train,
                eval_set=[(X_train_proc, y_train), (X_val_proc, y_val)],
                verbose=False,
            )
            cb_history = _probe_cb.history
            # Preserve Phase A val/test transforms — these are the honest held-out
            # versions (preprocessor fit on X_train only) used for threshold tuning
            # and calibration downstream. Do NOT overwrite with tv_preprocessor transforms.
            X_val_proc_probe  = X_val_proc
            X_test_proc_probe = X_test_proc
            log.info("  Phase A (probe) complete — val eval_set is honest.")

            # Phase B — blind refit on train+val, fixed n_estimators, NO eval_set.
            # Preprocessor is refit on X_trainval so the final model's internal
            # feature transforms match the full training distribution, but we never
            # evaluate on val here — that would be leakage.
            # Same _power_transform / _ct_n_jobs rationale as Phase A above.
            refit_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=best_n, early_stop=0, use_pca=use_pca,
                _power_transform=use_pca, _ct_n_jobs=1,
            )
            tv_preprocessor = refit_pipeline.named_steps["preprocessor"]
            X_tv_proc       = tv_preprocessor.fit_transform(X_trainval, y_trainval).astype(np.float32)
            X_test_proc     = tv_preprocessor.transform(X_test).astype(np.float32)
            _refit_cb       = _IterationLogCallback(period=CB_LOG_PERIOD, label="final")
            refit_pipeline.named_steps["model"].set_params(callbacks=[_refit_cb])
            refit_pipeline.named_steps["model"].fit(X_tv_proc, y_trainval, verbose=False)
            # Use Phase A val transform for threshold tuning and calibration (no leakage).
            X_val_proc  = X_val_proc_probe
            log.info("  Phase B (train+val refit) complete — no eval_set, fixed n_estimators=%d.", best_n)

        else:
            # TUNE_N_ESTIMATORS=false: use early stopping to find best_iteration,
            # then refit at that depth. Early stopping requires a held-out eval_set,
            # so we use X_train → X_val here (same honest-val fix as above).

            # Phase A — early-stopping probe on X_train only.
            # _power_transform=use_pca / _ct_n_jobs=1: same rationale as TUNE_N_ESTIMATORS branch.
            probe_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=N_ESTIMATORS_MAX, early_stop=30, use_pca=use_pca,
                _power_transform=use_pca, _ct_n_jobs=1,
            )
            preprocessor  = probe_pipeline.named_steps["preprocessor"]
            X_train_proc  = preprocessor.fit_transform(X_train, y_train).astype(np.float32)
            X_val_proc    = preprocessor.transform(X_val).astype(np.float32)
            X_test_proc   = preprocessor.transform(X_test).astype(np.float32)
            _es_model = probe_pipeline.named_steps["model"]
            _es_model.set_params(callbacks=[_IterationLogCallback(period=CB_LOG_PERIOD, label="es-probe")])
            _es_model.fit(
                X_train_proc, y_train,
                eval_set=[(X_train_proc, y_train), (X_val_proc, y_val)],
                verbose=False,
            )
            best_n = probe_pipeline.named_steps["model"].best_iteration
            log.info("  Early stopping → best n_estimators = %d (max was %d)", best_n, N_ESTIMATORS_MAX)

            # Preserve Phase A val/test transforms (preprocessor fit on X_train only).
            X_val_proc_probe  = X_val_proc
            X_test_proc_probe = X_test_proc
            # Phase B — refit on train+val at confirmed best_n, no eval_set.
            # _power_transform=use_pca / _ct_n_jobs=1: same rationale as TUNE_N_ESTIMATORS branch.
            refit_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=best_n, early_stop=0, use_pca=use_pca,
                _power_transform=use_pca, _ct_n_jobs=1,
            )
            tv_preprocessor = refit_pipeline.named_steps["preprocessor"]
            X_tv_proc       = tv_preprocessor.fit_transform(X_trainval, y_trainval).astype(np.float32)
            X_test_proc     = tv_preprocessor.transform(X_test).astype(np.float32)
            log.info("  Refitting final model at best_iteration=%d (train+val, no eval_set)...", best_n)
            _refit_cb    = _IterationLogCallback(period=CB_LOG_PERIOD, label="refit")
            refit_pipeline.named_steps["model"].set_params(callbacks=[_refit_cb])
            refit_pipeline.named_steps["model"].fit(X_tv_proc, y_trainval, verbose=False)
            # Use Phase A val transform for threshold tuning and calibration (no leakage).
            X_val_proc  = X_val_proc_probe
            cb_history = _refit_cb.history

        final_pipeline = refit_pipeline

        # ── 6b. Threshold tuning ─────────────────────────────────────────────
        # X_val_proc here is the Phase A transform (preprocessor fit on X_train only),
        # so val rows were never seen during preprocessing — truly held-out.
        log.info("[6b/9] Tuning decision threshold on val set...")
        best_threshold = tune_threshold(
            final_pipeline, X_val, y_val, metric, X_val_proc=X_val_proc
        )
        log.info("")

        # ── 6c. Calibration ──────────────────────────────────────────────────
        if TASK == "classification" and CALIBRATION_ENABLED:
            log.info("[6c/9] Calibrating classifier (CalibratedClassifierCV prefit)...")
            raw_xgb = refit_pipeline.named_steps["model"]

            import sklearn as _sklearn
            _sk_ver = tuple(int(x) for x in _sklearn.__version__.split(".")[:2])
            if _sk_ver >= (1, 6):
                try:
                    from sklearn.frozen import FrozenEstimator
                    calibrated = CalibratedClassifierCV(FrozenEstimator(raw_xgb))
                except ImportError:
                    calibrated = CalibratedClassifierCV(estimator=raw_xgb, cv="prefit")
            else:
                calibrated = CalibratedClassifierCV(estimator=raw_xgb, cv="prefit")

            calibrated.fit(X_val_proc, np.array(y_val))
            refit_pipeline.steps[-1] = ("model", calibrated)
            final_pipeline = refit_pipeline
            log.info("  Calibration done.")
            log.info("")

        # ── 7. Evaluate ──────────────────────────────────────────────────────
        eval_metrics = evaluate(
            final_pipeline, X_test, y_test, TASK, metric,
            threshold=best_threshold, X_test_proc=X_test_proc,
            log_transformed=log_transformed,
        )
        log.info("")

        # ── Ensemble + error analysis + uncertainty — all independent after final fit ─
        # Submit all three concurrently; resolve them after plots finish.
        _post_fit_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="post_fit")

        ensemble_path: Path | None = None
        ensemble_summary: dict = {
            "enabled": ENSEMBLE_ENABLED, "status": "disabled",
            "top_k_requested": ENSEMBLE_TOP_K,
        }
        if ENSEMBLE_ENABLED:
            _ens_future = _post_fit_pool.submit(
                build_top_k_ensemble,
                study=study,
                X_trainval=X_trainval, y_trainval=y_trainval,
                X_test=X_test, y_test=y_test,
                num_cols=num_cols, ohe_cat_cols=ohe_cat_cols, te_cat_cols=te_cat_cols,
                task=TASK, metric=metric, use_pca=use_pca,
                top_k=ENSEMBLE_TOP_K, n_estimators=best_n,
                threshold=best_threshold, log_transformed=log_transformed,
            )
        else:
            _ens_future = None
            log.info("[ensemble] [skipped]")

        # ── N5: error analysis  (runs concurrently with ensemble + uncertainty) ──
        _err_future = _post_fit_pool.submit(
            analyse_errors,
            final_pipeline, X_test, y_test, metric, best_threshold, run_id,
            X_test_proc=X_test_proc, label_encoder=le,
        )

        _unc_future = _post_fit_pool.submit(
            estimate_uncertainty,
            final_pipeline,
            X_trainval, y_trainval,
            X_val, y_val,
            X_test, y_test,
            TASK,
            MODEL_OUTPUT_DIR,
            run_id,
            alpha=UNCERTAINTY_ALPHA,
            quantile_low=UNCERTAINTY_QUANTILE_LOW,
            quantile_high=UNCERTAINTY_QUANTILE_HIGH,
            enabled=UNCERTAINTY_ENABLED,
        )

        # ── 8. Plots (parallel) ───────────────────────────────────────────────
        if PLOTS_ENABLED:
            _plot_tasks: list = []
            _plot_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="plot")

            _plot_tasks.append(_plot_pool.submit(
                plot_feature_importance,
                final_pipeline, num_cols, ohe_cat_cols, te_cat_cols, use_pca,
            ))
            if PERM_IMPORTANCE_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_permutation_importance,
                    final_pipeline, X_test, y_test,
                    num_cols, ohe_cat_cols, te_cat_cols,
                    metric, TASK, X_test_proc=X_test_proc,
                ))
            else:
                log.info("[perm_importance] [skipped]")
            if THRESHOLD_SWEEP_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_threshold_sweep,
                    final_pipeline, X_val, y_val, metric, X_val_proc=X_val_proc,
                ))
            else:
                log.info("[threshold_sweep] [skipped]")
            if OUTLIER_REPORT_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_outlier_report,
                    final_pipeline, X_train, X_test, y_test, num_cols, use_pca,
                ))
            else:
                log.info("[outlier_report] [skipped]")
            if PDP_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_partial_dependence,
                    final_pipeline, X_train, y_train,
                    num_cols, ohe_cat_cols, te_cat_cols, use_pca, TASK,
                ))
            else:
                log.info("[partial_dependence] [skipped]")
            if PCA_PLOTS_ENABLED and use_pca:
                _plot_tasks.append(_plot_pool.submit(
                    plot_pca_diagnostics, final_pipeline, X_train, y_train, TASK,
                ))
            elif PCA_PLOTS_ENABLED:
                log.info("[pca_plots] [skipped] — PCA not active")
            else:
                log.info("[pca_plots] [skipped]")
            if SHAP_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_shap_summary,
                    final_pipeline, X_test,
                    num_cols, ohe_cat_cols, te_cat_cols, use_pca, TASK,
                ))
                _plot_tasks.append(_plot_pool.submit(
                    plot_shap_interactions,
                    final_pipeline, X_test,
                    num_cols, ohe_cat_cols, te_cat_cols, use_pca,
                ))
            else:
                log.info("[shap] [skipped]")
            if CALIBRATION_CURVE_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_calibration_curve, final_pipeline, X_val, y_val, TASK,
                ))
            else:
                log.info("[calibration_curve] [skipped]")
            if CORR_HEATMAP_ENABLED:
                _plot_tasks.append(_plot_pool.submit(
                    plot_correlation_heatmap, X_train, num_cols,
                ))
            else:
                log.info("[corr_heatmap] [skipped]")

            # Drain plot futures; surface any exceptions without aborting the run.
            for _f in as_completed(_plot_tasks):
                try:
                    _f.result()
                except Exception as _exc:
                    log.warning("[plots] A plot task failed: %s", _exc)
            _plot_pool.shutdown(wait=False)
        else:
            log.info("[8/9] Plots [skipped]")

        # ── Resolve post-fit futures (error analysis, uncertainty, ensemble) ──
        error_csv       = _err_future.result()
        uncertainty_report = _unc_future.result()

        if _ens_future is not None:
            ensemble_model, ensemble_summary = _ens_future.result()
            if ensemble_model is not None:
                ensemble_path = MODEL_OUTPUT_DIR / f"ensemble_{timestamp}_{run_id}.joblib"
                joblib.dump(
                    {
                        "ensemble": ensemble_model, "summary": ensemble_summary,
                        "threshold": best_threshold, "metric": metric,
                        "feature_schema": feature_schema,
                        "run_id": run_id, "timestamp": timestamp,
                    },
                    ensemble_path,
                )
                log.info("  Top-K ensemble saved → %s", ensemble_path)

        _post_fit_pool.shutdown(wait=False)

        # ── Drift monitor reference ───────────────────────────────────────────
        known_categories: dict[str, set] = {
            col: set(X_train[col].dropna().unique())
            for col in ohe_cat_cols + te_cat_cols
        }

        drift_monitor = None
        holdout_monitor_report = None
        if DRIFT_MONITOR_ENABLED:
            ref_predictions = None
            ref_proba = None
            holdout_predictions = None
            holdout_proba = None
            try:
                monitor_model = final_pipeline.named_steps["model"]
                X_train_proc_for_monitor = final_pipeline.named_steps["preprocessor"].transform(X_train)
                if TASK == "classification" and hasattr(monitor_model, "predict_proba"):
                    ref_proba = monitor_model.predict_proba(X_train_proc_for_monitor)
                    holdout_proba = monitor_model.predict_proba(X_test_proc)
                ref_predictions = monitor_model.predict(X_train_proc_for_monitor)
                holdout_predictions = monitor_model.predict(X_test_proc)
            except Exception as exc:
                log.debug("  Drift monitor prediction references skipped: %s", exc)
            drift_monitor = ContinuousDriftMonitor.from_reference_data(
                X_train,
                num_cols=num_cols,
                cat_cols=ohe_cat_cols + te_cat_cols,
                alpha=DRIFT_ALPHA,
                persistence=DRIFT_MONITOR_PERSISTENCE,
                min_feature_drift_ratio=DRIFT_MONITOR_MIN_RATIO,
                retrain_feature_ratio=DRIFT_MONITOR_RETRAIN_RATIO,
                retrain_severity=DRIFT_MONITOR_RETRAIN_SEVERITY,
                random_state=RANDOM_STATE,
                y_ref=y_train,
                predictions_ref=ref_predictions,
                prediction_proba_ref=ref_proba,
            )
            holdout_monitor_report = drift_monitor.check(
                X_test,
                y_new=y_test,
                predictions=holdout_predictions,
                prediction_proba=holdout_proba,
            ).to_dict()
            log.info(
                "  Drift monitor reference captured → %d monitored feature(s)",
                drift_monitor.feature_count,
            )
        else:
            log.info("[drift_monitor] [skipped]")

        # ── 9. Save artifact ─────────────────────────────────────────────────
        # Resolve the requirements lock now — it's had the entire run to finish.
        lock_path = _lock_future.result()
        _lock_executor.shutdown(wait=False)
        config_snapshot = dict(
            task=TASK, target_col=TARGET_COL, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, cv_folds=CV_FOLDS, cv_strategy=CV_STRATEGY,
            search_backend=SEARCH_BACKEND,
            data_path=str(PLOT_OUTPUT_DIR.parent / "creditcard.csv"),  # resolved at runtime
            n_trials=N_TRIALS, optuna_timeout=OPTUNA_TIMEOUT,
            n_estimators_max=N_ESTIMATORS_MAX, wide_search=WIDE_SEARCH,
            pca_threshold=PCA_THRESHOLD, pca_variance=PCA_VARIANCE,
            cardinality_limit=CARDINALITY_LIMIT,
            drift_alpha=DRIFT_ALPHA, drift_warn_only=DRIFT_WARN_ONLY,
            feature_selection=FEATURE_SELECTION,
            target_encoding_threshold=TARGET_ENC_THRESHOLD,
            outlier_contamination=OUTLIER_CONTAMINATION,
            pdp_top_n=PDP_TOP_N,
            target_log_transform=TARGET_LOG_TRANSFORM,
            interaction_top_k=INTERACTION_TOP_K,
            auto_feature_engineering=dict(enabled=AUTO_FE_ENABLED, engine=AUTO_FE_ENGINE),
            metric=METRIC_NAME,
            use_gpu=USE_GPU,
            pandera_validation=PANDERA_VALIDATION,
            callback_log_period=CB_LOG_PERIOD,
            variance_threshold=VARIANCE_THRESHOLD,
            calibration_enabled=CALIBRATION_ENABLED,
            uncertainty=dict(
                enabled=UNCERTAINTY_ENABLED,
                alpha=UNCERTAINTY_ALPHA,
                quantile_alpha_low=UNCERTAINTY_QUANTILE_LOW,
                quantile_alpha_high=UNCERTAINTY_QUANTILE_HIGH,
            ),
            threshold_policy=THRESHOLD_POLICY,
            baselines=dict(
                enabled=BASELINES_ENABLED,
                include_dummy=BASELINE_INCLUDE_DUMMY,
                include_linear=BASELINE_INCLUDE_LINEAR,
                include_default_xgb=BASELINE_INCLUDE_XGB,
            ),
            ensemble=dict(enabled=ENSEMBLE_ENABLED, top_k=ENSEMBLE_TOP_K),
            drift_monitor=dict(
                enabled=DRIFT_MONITOR_ENABLED,
                persistence=DRIFT_MONITOR_PERSISTENCE,
                min_feature_drift_ratio=DRIFT_MONITOR_MIN_RATIO,
                retrain_feature_ratio=DRIFT_MONITOR_RETRAIN_RATIO,
                retrain_severity=DRIFT_MONITOR_RETRAIN_SEVERITY,
            ),
            diagnostics=dict(
                plots_enabled=PLOTS_ENABLED,
                optuna_plots=OPTUNA_PLOTS_ENABLED,
                learning_curve=LEARNING_CURVE_ENABLED,
                permutation_importance=PERM_IMPORTANCE_ENABLED,
                threshold_sweep=THRESHOLD_SWEEP_ENABLED,
                outlier_report=OUTLIER_REPORT_ENABLED,
                partial_dependence=PDP_ENABLED,
                pca_plots=PCA_PLOTS_ENABLED,
            ),
        )
        if APP_CONFIG is not None:
            try:
                config_snapshot["typed_config"] = to_plain_dict(APP_CONFIG)
            except Exception:
                pass

        artifact_name = f"model_{timestamp}_{run_id}.joblib"
        model_path    = MODEL_OUTPUT_DIR / artifact_name
        search_summary_path = MODEL_OUTPUT_DIR / f"search_summary_{run_id}.json"
        try:
            import json as _json
            search_summary_path.write_text(_json.dumps(search_summary, indent=2))
        except Exception as exc:
            log.debug("  Search summary write skipped: %s", exc)
        artifact_paths = {
            "model":                str(model_path),
            "ensemble":             str(ensemble_path) if ensemble_path is not None else None,
            "requirements_lock":    str(lock_path),
            "baseline_comparison":  str(baseline_path) if baseline_path is not None else None,
            "error_analysis":       str(error_csv) if error_csv is not None else None,
            "missing_value_report": missing_report.output_csv,
            "uncertainty_report":   uncertainty_report.output_csv,
            "search_summary":       str(search_summary_path),
            "plots_dir":            str(PLOT_OUTPUT_DIR),
        }

        log.info("[9/9] Saving to '%s'...", model_path)
        joblib.dump(
            {
                "pipeline":          final_pipeline,
                "best_params":       best_params,
                "best_n_estimators": best_n,
                "best_threshold":    best_threshold,
                "num_cols":          num_cols,
                "ohe_cat_cols":      ohe_cat_cols,
                "te_cat_cols":       te_cat_cols,
                "cat_cols":          ohe_cat_cols + te_cat_cols,  # backward compat
                "use_pca":           use_pca,
                "metric":            metric,
                "label_encoder":     le,
                "known_categories":  known_categories,
                "eval_metrics":      eval_metrics,
                "baseline_results":  baseline_results,
                "ensemble_summary":  ensemble_summary,
                "drift_report":      drift_report.to_dict(),
                "drift_monitor":     drift_monitor,
                "holdout_monitor_report": holdout_monitor_report,
                "missing_value_report": missing_report.to_dict(),
                "uncertainty_report": uncertainty_report.to_dict(),
                "search_summary":    search_summary,
                "threshold_policy":  THRESHOLD_POLICY,
                "feature_schema":    feature_schema,
                "artifact_paths":    artifact_paths,
                "config":            config_snapshot,
                "run_id":            run_id,
                "timestamp":         timestamp,
                "log_transformed":   log_transformed,
                "cb_history":        cb_history,
                "task":              TASK,
            },
            model_path,
        )

        # ── Summary + registry ───────────────────────────────────────────────
        train_summary(
            run_id, timestamp, model_path, eval_metrics, best_params,
            best_n, best_threshold, drift_report,
            output_dir=MODEL_OUTPUT_DIR,
            plot_output_dir=PLOT_OUTPUT_DIR,
            task=TASK,
            target_col=TARGET_COL,
            baseline_results=baseline_results,
            ensemble_summary=ensemble_summary,
            threshold_policy=THRESHOLD_POLICY,
            feature_schema=feature_schema,
            artifact_paths=artifact_paths,
            search_summary=search_summary,
            missing_report=missing_report.to_dict(),
            uncertainty_report=uncertainty_report.to_dict(),
        )

        register_model(
            run_id, timestamp, model_path, eval_metrics, best_params,
            best_n, best_threshold, TASK, config_snapshot,
            output_dir=MODEL_OUTPUT_DIR,
            baseline_results=baseline_results,
            ensemble_summary=ensemble_summary,
            threshold_policy=THRESHOLD_POLICY,
        )

        # ── MLflow logging ───────────────────────────────────────────────────
        mlrun.log_config(config_snapshot)
        mlrun.log_params(best_params)
        mlrun.log_params({"best_n_estimators": best_n, "best_threshold": best_threshold})
        mlrun.log_metrics(eval_metrics)
        mlrun.log_artifact(model_path)
        if ensemble_path is not None:
            mlrun.log_artifact(ensemble_path)
        mlrun.log_artifact(lock_path)
        if baseline_path is not None:
            mlrun.log_artifact(baseline_path)
        if error_csv is not None:
            mlrun.log_artifact(error_csv)
        mlrun.log_artifact(search_summary_path)
        if missing_report.output_csv is not None:
            mlrun.log_artifact(missing_report.output_csv)
        if uncertainty_report.output_csv is not None:
            mlrun.log_artifact(uncertainty_report.output_csv)
        for html in PLOT_OUTPUT_DIR.glob("*.html"):
            mlrun.log_artifact(html)
        for png in PLOT_OUTPUT_DIR.glob("*.png"):
            mlrun.log_artifact(png)

        log.info("\n  Load later:")
        log.info("  from xgb_prototype.inference import PredictWrapper")
        log.info("  artifact = joblib.load('%s')", model_path)
        log.info("  model = PredictWrapper(artifact)")
        log.info("  labels = model.predict(new_df)                # original labels")
        log.info("  probas = model.predict_proba(new_df)          # calibrated probabilities")
        log.info("  preds  = model.predict_with_threshold(new_df) # tuned threshold")


if __name__ == "__main__":
    main()