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
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True  # This is the "sledgehammer" that overrides other configs
)
logging.getLogger("great_expectations").setLevel(logging.WARNING)
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    # pca
    PCA_THRESHOLD, PCA_VARIANCE,
    # features
    FEATURE_SELECTION, CARDINALITY_LIMIT, TARGET_ENC_THRESHOLD,
    VARIANCE_THRESHOLD, INTERACTION_TOP_K,
    # drift
    DRIFT_ALPHA, DRIFT_WARN_ONLY,
    DRIFT_MONITOR_ENABLED, DRIFT_MONITOR_PERSISTENCE,
    DRIFT_MONITOR_MIN_RATIO, DRIFT_MONITOR_RETRAIN_RATIO,
    DRIFT_MONITOR_RETRAIN_SEVERITY,
    # misc model flags
    USE_GPU, PANDERA_VALIDATION, METRIC_NAME, CALIBRATION_ENABLED,
    CB_LOG_PERIOD, TARGET_LOG_TRANSFORM, OUTLIER_CONTAMINATION, PDP_TOP_N,
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
    # mlflow
    MLFLOW_URI, MLFLOW_EXPERIMENT,
    # typed config
    APP_CONFIG,
)
from xgb_prototype.metrics import select_metric
from xgb_prototype.data import (
    load_data, clean_data, validate_data, validate_pandera,
    detect_drift, maybe_log_transform, check_config, apply_pretransforms,
)
from xgb_prototype.features import (
    detect_feature_types, filter_low_variance,
    generate_feature_interactions, select_features_rfecv,
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
)
from xgb_prototype.tracking import (
    _MLflowRun, write_requirements_lock, register_model, train_summary,
)
from xgb_prototype.inference import PredictWrapper, ModelServer  # noqa: F401 — re-exported for callers
from xgb_prototype.baselines import evaluate_baselines
from xgb_prototype.drift_monitor import ContinuousDriftMonitor
from xgb_prototype.config import to_plain_dict

# ── Dependency check ──────────────────────────────────────────────────────────

check_deps()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info(" XGBoost v7 · Optuna+CV · Calibration · SHAP · Plotly · Versioned")
    log.info("=" * 60)

    run_id    = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("  Run ID: %s_%s", timestamp, run_id)

    # UPGRADE 26: lock environment immediately so the file is always written
    lock_path = write_requirements_lock(run_id, MODEL_OUTPUT_DIR)

    with _MLflowRun(run_id, timestamp, MLFLOW_URI, MLFLOW_EXPERIMENT) as mlrun:

        # ── 1-3. Load, clean, validate ───────────────────────────────────────
        df = load_data()
        check_config(df)
        df = clean_data(df)
        df = apply_pretransforms(df)
        validate_data(df)
        validate_pandera(df, TARGET_COL)

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # UPGRADE 30: optional log-transform of regression target
        y, log_transformed = maybe_log_transform(y)

        le = None
        if TASK == "classification":
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y), name=TARGET_COL)

        # ── 4. Feature detection ─────────────────────────────────────────────
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

        log.info("[metric] Inspecting target distribution...")
        metric = select_metric(y, TASK)

        # ── Three-way split ──────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y if TASK == "classification" else None,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE,
            stratify=y_train if TASK == "classification" else None,
        )
        log.info("  Split — train: %d, val: %d, test: %d",
                 len(X_train), len(X_val), len(X_test))

        # ── Drift, variance filter, interactions, RFECV ──────────────────────
        drift_report = detect_drift(X_train, X_test, num_cols, ohe_cat_cols + te_cat_cols)

        num_cols, ohe_cat_cols, te_cat_cols = filter_low_variance(
            X_train, num_cols, ohe_cat_cols, te_cat_cols
        )

        X_train, X_val, X_test, num_cols = generate_feature_interactions(
            X_train, X_val, X_test, num_cols
        )

        if FEATURE_SELECTION:
            log.info("[RFECV] Feature selection enabled...")
            num_cols, ohe_cat_cols, te_cat_cols = select_features_rfecv(
                X_train, y_train, num_cols, ohe_cat_cols, te_cat_cols, TASK, metric
            )

        feature_schema = {
            "num_cols":     num_cols,
            "ohe_cat_cols": ohe_cat_cols,
            "te_cat_cols":  te_cat_cols,
            "use_pca":      use_pca,
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

        # ── 5. Optuna tuning ─────────────────────────────────────────────────
        best_params, study = tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, use_pca,
        )

        if PLOTS_ENABLED and OPTUNA_PLOTS_ENABLED:
            plot_optuna_diagnostics(study)

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

        if TUNE_N_ESTIMATORS:
            best_n = int(final_params.pop("n_estimators", N_ESTIMATORS_MAX))
            log.info("  tune_n_estimators=true → using Optuna best n_estimators=%d", best_n)
            refit_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=best_n, early_stop=0, use_pca=use_pca,
            )
            preprocessor  = refit_pipeline.named_steps["preprocessor"]
            X_tv_proc     = preprocessor.fit_transform(X_trainval, y_trainval)
            X_val_proc    = preprocessor.transform(X_val)
            X_test_proc   = preprocessor.transform(X_test)
            _refit_cb     = _IterationLogCallback(period=CB_LOG_PERIOD, label="final")
            refit_pipeline.named_steps["model"].set_params(callbacks=[_refit_cb])
            refit_pipeline.named_steps["model"].fit(
                X_tv_proc, y_trainval,
                eval_set=[(X_val_proc, y_val)],
                verbose=False,
            )
            cb_history = _refit_cb.history
        else:
            final_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=N_ESTIMATORS_MAX, early_stop=30, use_pca=use_pca,
            )
            preprocessor  = final_pipeline.named_steps["preprocessor"]
            X_tv_proc     = preprocessor.fit_transform(X_trainval, y_trainval)
            X_val_proc    = preprocessor.transform(X_val)
            X_test_proc   = preprocessor.transform(X_test)

            _es_model = final_pipeline.named_steps["model"]
            _es_model.set_params(callbacks=[_IterationLogCallback(period=CB_LOG_PERIOD, label="es-fit")])
            _es_model.fit(
                X_tv_proc, y_trainval,
                eval_set=[(X_val_proc, y_val)],
                verbose=False,
            )
            best_n = final_pipeline.named_steps["model"].best_iteration
            log.info("  Early stopping → best n_estimators = %d (max was %d)", best_n, N_ESTIMATORS_MAX)

            log.info("  Refitting final model at best_iteration=%d...", best_n)
            refit_pipeline = build_pipeline(
                num_cols, ohe_cat_cols, te_cat_cols, TASK, metric, final_params,
                n_estimators=best_n, early_stop=0, use_pca=use_pca,
            )
            step_names = [name for name, _ in refit_pipeline.steps]
            refit_pipeline.steps[step_names.index("preprocessor")] = ("preprocessor", preprocessor)
            _refit_cb    = _IterationLogCallback(period=CB_LOG_PERIOD, label="refit")
            _refit_model = refit_pipeline.named_steps["model"]
            _refit_model.set_params(callbacks=[_refit_cb])
            _refit_model.fit(X_tv_proc, y_trainval, verbose=False)
            cb_history = _refit_cb.history
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
            log.info("  Calibration done.")

        final_pipeline = refit_pipeline

        # ── 6b. Threshold tuning ─────────────────────────────────────────────
        log.info("[6b/9] Tuning decision threshold on val set...")
        best_threshold = tune_threshold(
            final_pipeline, X_val, y_val, metric, X_val_proc=X_val_proc
        )

        # ── 7. Evaluate ──────────────────────────────────────────────────────
        eval_metrics = evaluate(
            final_pipeline, X_test, y_test, TASK, metric,
            threshold=best_threshold, X_test_proc=X_test_proc,
            log_transformed=log_transformed,
        )

        # ── Ensemble ─────────────────────────────────────────────────────────
        ensemble_path: Path | None = None
        ensemble_summary: dict = {
            "enabled": ENSEMBLE_ENABLED, "status": "disabled",
            "top_k_requested": ENSEMBLE_TOP_K,
        }
        if ENSEMBLE_ENABLED:
            ensemble_model, ensemble_summary = build_top_k_ensemble(
                study=study,
                X_trainval=X_trainval, y_trainval=y_trainval,
                X_test=X_test, y_test=y_test,
                num_cols=num_cols, ohe_cat_cols=ohe_cat_cols, te_cat_cols=te_cat_cols,
                task=TASK, metric=metric, use_pca=use_pca,
                top_k=ENSEMBLE_TOP_K, n_estimators=best_n,
                threshold=best_threshold, log_transformed=log_transformed,
            )
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

        # ── N5: error analysis ────────────────────────────────────────────────
        error_csv = analyse_errors(
            final_pipeline, X_test, y_test, metric, best_threshold, run_id,
            X_test_proc=X_test_proc, label_encoder=le,
        )

        # ── 8. Plots ─────────────────────────────────────────────────────────
        if PLOTS_ENABLED:
            plot_feature_importance(
                final_pipeline, num_cols, ohe_cat_cols, te_cat_cols, use_pca
            )
        if PLOTS_ENABLED and PERM_IMPORTANCE_ENABLED:
            plot_permutation_importance(
                final_pipeline, X_test, y_test,
                num_cols, ohe_cat_cols, te_cat_cols,
                metric, TASK, X_test_proc=X_test_proc,
            )
        if PLOTS_ENABLED and THRESHOLD_SWEEP_ENABLED:
            plot_threshold_sweep(
                final_pipeline, X_val, y_val, metric, X_val_proc=X_val_proc
            )
        if PLOTS_ENABLED and OUTLIER_REPORT_ENABLED:
            plot_outlier_report(
                final_pipeline, X_train, X_test, y_test, num_cols, use_pca
            )
        if PLOTS_ENABLED and PDP_ENABLED:
            plot_partial_dependence(
                final_pipeline, X_train, y_train,
                num_cols, ohe_cat_cols, te_cat_cols, use_pca, TASK,
            )
        if PLOTS_ENABLED and PCA_PLOTS_ENABLED and use_pca:
            plot_pca_diagnostics(final_pipeline, X_train, y_train, TASK)

        # ── Drift monitor reference ───────────────────────────────────────────
        known_categories: dict[str, set] = {
            col: set(X_train[col].dropna().unique())
            for col in ohe_cat_cols + te_cat_cols
        }

        drift_monitor = None
        if DRIFT_MONITOR_ENABLED:
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
            )
            log.info(
                "  Drift monitor reference captured → %d monitored feature(s)",
                drift_monitor.feature_count,
            )

        # ── 9. Save artifact ─────────────────────────────────────────────────
        config_snapshot = dict(
            task=TASK, target_col=TARGET_COL, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, cv_folds=CV_FOLDS, cv_strategy=CV_STRATEGY,
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
            metric=METRIC_NAME,
            use_gpu=USE_GPU,
            pandera_validation=PANDERA_VALIDATION,
            callback_log_period=CB_LOG_PERIOD,
            variance_threshold=VARIANCE_THRESHOLD,
            calibration_enabled=CALIBRATION_ENABLED,
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
        artifact_paths = {
            "model":                str(model_path),
            "ensemble":             str(ensemble_path) if ensemble_path is not None else None,
            "requirements_lock":    str(lock_path),
            "baseline_comparison":  str(baseline_path) if baseline_path is not None else None,
            "error_analysis":       str(error_csv) if error_csv is not None else None,
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