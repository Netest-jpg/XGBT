"""settings.py — All runtime constants loaded from config.yaml via OmegaConf."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _resolve_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.yaml")
    args, _ = parser.parse_known_args()
    return Path(args.config)


try:
    from omegaconf import OmegaConf
    _cfg_path = _resolve_config_path()
    _cfg = OmegaConf.load(_cfg_path) if _cfg_path.exists() else OmegaConf.create({})
except ImportError:
    OmegaConf = None  # type: ignore
    _cfg_path = _resolve_config_path()
    _cfg = {}

try:
    from xgb_prototype.config import load_config
    APP_CONFIG = load_config(_cfg_path)
except Exception:
    APP_CONFIG = None

try:
    from thresholds import normalize_policy
except ImportError:
    def normalize_policy(p, metric_name=None):  # type: ignore
        return {"mode": "f1", "beta": 1.0, "min_precision": 0.80, "min_recall": 0.80, "n_quantiles": 200}


def _c(key: str, default):
    try:
        return OmegaConf.select(_cfg, key, default=default)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ── Logging ───────────────────────────────────────────────────────────────────
_log_level_str = _c("log_level", "INFO")
_log_level = getattr(logging, _log_level_str.upper(), logging.INFO)
_log_file   = _c("log_file", None)

_handlers: list[logging.Handler] = [logging.StreamHandler()]
if _log_file:
    _handlers.append(logging.FileHandler(_log_file))

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [ %(levelname)s ]\t%(message)s",
    datefmt="%H:%M:%S",
    handlers=_handlers,
    force=True,
)
# Suppress noisy great_expectations startup logs
logging.getLogger("great_expectations").setLevel(logging.WARNING)
# ── Core ──────────────────────────────────────────────────────────────────────
TASK              = _c("task",           "classification")
TARGET_COL        = _c("target_col",     "Class")
TEST_SIZE         = _c("test_size",      0.2)
RANDOM_STATE      = _c("random_state",   42)
MODEL_OUTPUT_DIR  = Path(_c("model_output_dir", "models"))
PLOT_OUTPUT_DIR   = Path(_c("plot_output_dir",  "plots"))
DATA_PATH         = _c("data_path",      "creditcard.csv")

_csv_chunk_size_cfg  = _c("csv_chunk_size", None)
CSV_CHUNK_SIZE       = None if _csv_chunk_size_cfg in (None, "null", 0) else int(_csv_chunk_size_cfg)
CSV_CHUNK_LOG_EVERY  = max(1, int(_c("csv_chunk_log_every", 10)))

# ── CV / search ───────────────────────────────────────────────────────────────
CV_FOLDS          = int(_c("cv_folds",    -1))   # -1=auto, 0=force off, N>0=force N folds
CV_STRATEGY       = _c("cv_strategy",    "stratified")
WIDE_SEARCH       = bool(_c("wide_search", False))

N_TRIALS          = _c("n_trials",       50 if WIDE_SEARCH else 30)
_timeout_cfg      = _c("optuna_timeout", None)
OPTUNA_TIMEOUT    = None if _timeout_cfg is None else int(_timeout_cfg)
_budget_cfg       = _c("optuna_budget_seconds", None)
OPTUNA_BUDGET_SECONDS = None if _budget_cfg is None else int(_budget_cfg)
SEARCH_SUBSAMPLE  = _c("search_subsample",   0.6)
N_ESTIMATORS_MAX  = _c("n_estimators_max",   500)
N_ESTIMATORS_MIN  = int(_c("n_estimators_min",  100))
TUNE_N_ESTIMATORS = bool(_c("tune_n_estimators", True))
EARLY_STOP_RNDS   = _c("early_stop_rnds",    20)

# ── PCA / imbalance ───────────────────────────────────────────────────────────
PCA_THRESHOLD     = _c("pca_threshold",      10)
PCA_VARIANCE      = _c("pca_variance",       0.95)
PCA_MAX_COMPONENTS = _c("pca_max_components", None)
IMBALANCE_THRESHOLD = _c("imbalance_threshold", 0.15)
CARDINALITY_LIMIT   = _c("cardinality_limit",   20)

# ── Drift ─────────────────────────────────────────────────────────────────────
DRIFT_ALPHA       = float(_c("drift_alpha",     0.05))
DRIFT_WARN_ONLY   = bool(_c("drift_warn_only",  True))

# ── Features / encoding ───────────────────────────────────────────────────────
FEATURE_SELECTION    = bool(_c("feature_selection",          False))
TARGET_ENC_THRESHOLD = int(_c("target_encoding_threshold",   50))
VARIANCE_THRESHOLD   = float(_c("variance_threshold",        0.0))
INTERACTION_TOP_K    = int(_c("interaction_top_k",           10))

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_URI        = _c("mlflow_tracking_uri",  None)
MLFLOW_EXPERIMENT = _c("mlflow_experiment",    "xgb_prototype")

# ── Misc ──────────────────────────────────────────────────────────────────────
OUTLIER_CONTAMINATION = float(_c("outlier_contamination", 0.05))
PDP_TOP_N             = int(_c("pdp_top_n",               5))
TARGET_LOG_TRANSFORM  = bool(_c("target_log_transform",   False))
CB_LOG_PERIOD         = int(_c("callback_log_period",     50))
USE_GPU               = _env_bool("USE_GPU", bool(_c("use_gpu", False)))
RASTER_FORMAT         = os.getenv("RASTER_FORMAT", str(_c("raster_format", "png"))).strip().lower()
if RASTER_FORMAT not in {"png", "jpeg", "webp"}:
    logging.getLogger(__name__).warning(
        "Unsupported RASTER_FORMAT=%r; using png. Supported: png, jpeg, webp.",
        RASTER_FORMAT,
    )
    RASTER_FORMAT = "png"
PANDERA_VALIDATION    = bool(_c("pandera_validation",     True))
METRIC_NAME           = str(_c("metric",                  "auto")).lower()
CALIBRATION_ENABLED   = bool(_c("calibration_enabled",   True))
POWER_TRANSFORM         = bool(_c("power_transform", True))
ROBUST_SCALER_COLS      = list(_c("robust_scaler_cols", ["Amount", "Time"]))
PRETRANSFORM_LOG1P_COLS = list(_c("pretransform_log1p_cols", []))
PRETRANSFORM_DROP_COLS  = list(_c("pretransform_drop_cols",  []))

THRESHOLD_POLICY = normalize_policy(
    {
        "mode":          _c("threshold_policy.mode",          "auto"),
        "beta":          _c("threshold_policy.beta",          1.0),
        "min_precision": _c("threshold_policy.min_precision", 0.80),
        "min_recall":    _c("threshold_policy.min_recall",    0.80),
        "n_quantiles":   _c("threshold_policy.n_quantiles",   200),
    }
)

# ── Baselines ─────────────────────────────────────────────────────────────────
BASELINES_ENABLED       = bool(_c("baselines.enabled",             True))
BASELINE_INCLUDE_DUMMY  = bool(_c("baselines.include_dummy",       True))
BASELINE_INCLUDE_LINEAR = bool(_c("baselines.include_linear",      True))
BASELINE_INCLUDE_XGB    = bool(_c("baselines.include_default_xgb", True))

# ── Ensemble ──────────────────────────────────────────────────────────────────
ENSEMBLE_ENABLED = bool(_c("ensemble.enabled", False))
ENSEMBLE_TOP_K   = max(1, int(_c("ensemble.top_k", 3)))

# ── Auto feature engineering ─────────────────────────────────────────────────
AUTO_FE_ENABLED       = bool(_c("auto_feature_engineering.enabled", False))
AUTO_FE_ENGINE        = str(_c("auto_feature_engineering.engine", "featuretools")).lower()
AUTO_FE_MAX_FEATURES  = max(1, int(_c("auto_feature_engineering.max_features", 25)))
AUTO_FE_MAX_DEPTH     = max(1, int(_c("auto_feature_engineering.max_depth", 1)))
AUTO_FE_ENTITY_ID_COL = _c("auto_feature_engineering.entity_id_col", None)
AUTO_FE_TIME_INDEX_COL = _c("auto_feature_engineering.time_index_col", None)
AUTO_FE_TSFRESH_COLUMN_ID = _c("auto_feature_engineering.tsfresh_column_id", None)
AUTO_FE_TSFRESH_COLUMN_SORT = _c("auto_feature_engineering.tsfresh_column_sort", None)

# ── Search backend / sensitivity ─────────────────────────────────────────────
SEARCH_BACKEND = str(_c("search.backend", _c("search_backend", "optuna"))).lower()
NATIVE_XGB_CV_ROUNDS = int(_c("search.native_xgb_cv_rounds", N_ESTIMATORS_MAX))
NATIVE_XGB_CV_EARLY_STOP = int(_c("search.native_xgb_cv_early_stop", EARLY_STOP_RNDS))
SOBOL_ENABLED = bool(_c("sobol_sensitivity.enabled", False))
SOBOL_N_BASE_SAMPLES = max(2, int(_c("sobol_sensitivity.n_base_samples", 16)))
SOBOL_MAX_EVALS = max(8, int(_c("sobol_sensitivity.max_evals", 128)))

# ── Uncertainty estimation ───────────────────────────────────────────────────
UNCERTAINTY_ENABLED = bool(_c("uncertainty.enabled", False))
UNCERTAINTY_ALPHA = float(_c("uncertainty.alpha", 0.10))
UNCERTAINTY_QUANTILE_LOW = float(_c("uncertainty.quantile_alpha_low", 0.05))
UNCERTAINTY_QUANTILE_HIGH = float(_c("uncertainty.quantile_alpha_high", 0.95))

# ── Drift monitor ─────────────────────────────────────────────────────────────
DRIFT_MONITOR_ENABLED         = bool(_c("drift_monitor.enabled",                True))
DRIFT_MONITOR_PERSISTENCE     = max(1, int(_c("drift_monitor.persistence",      3)))
DRIFT_MONITOR_MIN_RATIO       = float(_c("drift_monitor.min_feature_drift_ratio", 0.10))
DRIFT_MONITOR_RETRAIN_RATIO   = float(_c("drift_monitor.retrain_feature_ratio", 0.25))
DRIFT_MONITOR_RETRAIN_SEVERITY = str(_c("drift_monitor.retrain_severity",       "high")).lower()

# ── Diagnostics / plots ───────────────────────────────────────────────────────
PLOTS_ENABLED            = bool(_c("diagnostics.plots_enabled",          True))
OPTUNA_PLOTS_ENABLED     = bool(_c("diagnostics.optuna_plots",           True))
LEARNING_CURVE_ENABLED   = bool(_c("diagnostics.learning_curve",         True))
PERM_IMPORTANCE_ENABLED  = bool(_c("diagnostics.permutation_importance", True))
THRESHOLD_SWEEP_ENABLED  = bool(_c("diagnostics.threshold_sweep",        True))
OUTLIER_REPORT_ENABLED   = bool(_c("diagnostics.outlier_report",         True))
PDP_ENABLED              = bool(_c("diagnostics.partial_dependence",     True))
PCA_PLOTS_ENABLED        = bool(_c("diagnostics.pca_plots",              True))
SHAP_ENABLED             = bool(_c("diagnostics.shap",                   True))
CALIBRATION_CURVE_ENABLED = bool(_c("diagnostics.calibration_curve",    True))
CORR_HEATMAP_ENABLED     = bool(_c("diagnostics.corr_heatmap",           True))

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
