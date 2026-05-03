"""data.py — Data loading, cleaning, GX validation, Pandera schema, drift detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import great_expectations as gx
logging.getLogger("great_expectations").setLevel(logging.WARNING)
from scipy import stats as scipy_stats

try:
    import pandera as pa
    _PANDERA_AVAILABLE = True
except ImportError:
    _PANDERA_AVAILABLE = False

from .settings import (
    DATA_PATH, CSV_CHUNK_SIZE, CSV_CHUNK_LOG_EVERY,
    TARGET_COL, TASK, TARGET_LOG_TRANSFORM,
    DRIFT_ALPHA, DRIFT_WARN_ONLY, PANDERA_VALIDATION,
    TEST_SIZE, CV_FOLDS, CV_STRATEGY, N_TRIALS, SEARCH_SUBSAMPLE,
    N_ESTIMATORS_MAX, EARLY_STOP_RNDS, PCA_VARIANCE, IMBALANCE_THRESHOLD,
    CARDINALITY_LIMIT, OUTLIER_CONTAMINATION, PDP_TOP_N, CB_LOG_PERIOD,
    VARIANCE_THRESHOLD, TARGET_ENC_THRESHOLD,
)

log = logging.getLogger(__name__)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """UPGRADE 15: Glob-aware multi-format loader (csv/parquet/json/xlsx)."""
    import glob as _glob

    path_str = str(DATA_PATH)
    paths    = sorted(_glob.glob(path_str))
    if not paths:
        raise FileNotFoundError(
            f"data_path='{path_str}' matched no files. Check config.yaml → data_path."
        )

    def _read_csv(p: str) -> pd.DataFrame:
        if CSV_CHUNK_SIZE is None:
            return pd.read_csv(p)
        log.info("[load] Streaming CSV '%s' in chunks of %d row(s)...", p, CSV_CHUNK_SIZE)
        chunks: list[pd.DataFrame] = []
        total_rows = 0
        for chunk_idx, chunk in enumerate(pd.read_csv(p, chunksize=CSV_CHUNK_SIZE), 1):
            chunks.append(chunk)
            total_rows += len(chunk)
            if chunk_idx == 1 or chunk_idx % CSV_CHUNK_LOG_EVERY == 0:
                log.info("[load]   chunk %d read → %d cumulative row(s)", chunk_idx, total_rows)
        if not chunks:
            return pd.DataFrame()
        log.info("[load] Completed chunked CSV '%s' → %d chunk(s), %d row(s)",
                 p, len(chunks), total_rows)
        return pd.concat(chunks, ignore_index=True)

    def _read_one(p: str) -> pd.DataFrame:
        ext = Path(p).suffix.lower()
        if ext == ".parquet":              return pd.read_parquet(p)
        if ext in (".json", ".jsonl"):     return pd.read_json(p, lines=(ext == ".jsonl"))
        if ext in (".xlsx", ".xls"):       return pd.read_excel(p)
        return _read_csv(p)

    frames = [_read_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    log.info("[load] Read %d file(s) → %d rows × %d cols", len(paths), *df.shape)
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    log.info("[1/9] Cleaning data... shape before: %s", df.shape)
    df = df.dropna(how="all").drop_duplicates()
    df = df.replace([-999, -9999, "N/A", "n/a", "NA", "none", "None", ""], np.nan)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    date_cols = [
        c for c in df.columns
        if ("date" in c.lower() or "time" in c.lower()) and df[c].dtype == "object"
    ]
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.5:
            df[col] = parsed
            df[f"{col}_year"]           = parsed.dt.year
            df[f"{col}_month"]          = parsed.dt.month
            df[f"{col}_dayofweek"]      = parsed.dt.dayofweek
            df[f"{col}_hour"]           = parsed.dt.hour
            month_angle = 2 * np.pi * (parsed.dt.month.astype(float) - 1) / 12
            dow_angle   = 2 * np.pi * parsed.dt.dayofweek.astype(float) / 7
            hour_angle  = 2 * np.pi * parsed.dt.hour.astype(float) / 24
            df[f"{col}_month_sin"]      = np.sin(month_angle)
            df[f"{col}_month_cos"]      = np.cos(month_angle)
            df[f"{col}_dayofweek_sin"]  = np.sin(dow_angle)
            df[f"{col}_dayofweek_cos"]  = np.cos(dow_angle)
            df[f"{col}_hour_sin"]       = np.sin(hour_angle)
            df[f"{col}_hour_cos"]       = np.cos(hour_angle)
            df = df.drop(columns=[col])
            log.info("  Parsed date column: '%s' → year/month/dayofweek/hour + sin/cos", col)

    log.info("  Shape after: %s", df.shape)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        log.info("  Missing values:\n%s", missing)
    return df


# ── Great Expectations validation ─────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> None:
    log.info("[2/9] Validating data (Great Expectations)...")
    context = gx.get_context()
    try:
        data_source = context.data_sources.add_pandas("pandas")
    except Exception:
        data_source = context.data_sources.get("pandas")
    try:
        data_asset = data_source.add_dataframe_asset(name="train_data")
    except Exception:
        data_asset = data_source.get_asset("train_data")
    batch_def = data_asset.add_batch_definition_whole_dataframe("full_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    rules = []  # ← add Great Expectations expectations here
    failed = []
    for rule in rules:
        result = batch.validate(rule)
        label  = getattr(rule, "column", "table-level")
        status = "✓" if result.success else "✗"
        log.info("  %s %s [%s]", status, rule.__class__.__name__, label)
        if not result.success:
            failed.append(rule)
    if failed:
        raise ValueError(f"\n  Validation failed for {len(failed)} rule(s). Fix data before training.")
    log.info("  All validations passed!")


# ── Pandera schema validation (V1) ────────────────────────────────────────────

def _infer_pandera_schema(df: pd.DataFrame, target_col: str):
    if not _PANDERA_AVAILABLE:
        log.warning("[Pandera] pandera not installed — schema validation skipped.")
        return None

    columns: dict = {}
    for col in df.columns:
        series    = df[col]
        is_target = (col == target_col)
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 10:
                lo = float(np.percentile(clean, 0.1))
                hi = float(np.percentile(clean, 99.9))
                def _sigfig(x, n=4):
                    if x == 0: return 0.0
                    mag = 10 ** (n - 1 - int(np.floor(np.log10(abs(x)))))
                    return round(x * mag) / mag
                lo_r, hi_r = _sigfig(lo), _sigfig(hi)
                checks = []
                if lo_r < hi_r:
                    checks.append(pa.Check(
                        lambda s, lo=lo_r, hi=hi_r: s.between(lo, hi),
                        element_wise=False,
                        error=f"Values outside inferred range [{lo_r}, {hi_r}]",
                        raise_warning=True,
                    ))
                columns[col] = pa.Column(dtype=float, checks=checks,
                                         nullable=not is_target, coerce=True, required=True)
            else:
                columns[col] = pa.Column(dtype=float, nullable=not is_target,
                                         coerce=True, required=True)
        elif pd.api.types.is_object_dtype(series) or hasattr(series, "cat"):
            columns[col] = pa.Column(dtype=object, nullable=not is_target,
                                     coerce=False, required=True)
        else:
            columns[col] = pa.Column(nullable=True, required=True)

    return pa.DataFrameSchema(columns=columns, index=None, coerce=True, strict=False)


def validate_pandera(df: pd.DataFrame, target_col: str) -> None:
    """V1: Pandera schema validation with inferred column rules."""
    if not PANDERA_VALIDATION:
        log.info("[Pandera] pandera_validation=false — skipped.")
        return
    if not _PANDERA_AVAILABLE:
        log.warning("[Pandera] pandera not installed — skipped.")
        return

    log.info("[3a/9] Pandera schema validation...")
    schema = _infer_pandera_schema(df, target_col)
    if schema is None:
        return

    for col_name, col_schema in schema.columns.items():
        dtype_str = str(col_schema.dtype) if col_schema.dtype is not None else "any"
        null_str  = "nullable" if col_schema.nullable else "non-null"
        log.info("  %-30s  dtype=%-10s  %-10s  checks=%d",
                 col_name, dtype_str, null_str, len(col_schema.checks))

    try:
        schema.validate(df, lazy=True)
        log.info("  [Pandera] All schema checks passed.")
    except pa.errors.SchemaErrors as exc:
        err_df     = exc.failure_cases
        n_errors   = len(err_df)
        col_counts = (err_df["schema_context"].value_counts().to_dict()
                      if "schema_context" in err_df.columns else {})
        log.warning("  [Pandera] %d schema violation(s) detected (warnings only). "
                    "Review before production use.", n_errors)
        for ctx, cnt in col_counts.items():
            log.warning("    %-35s  %d violation(s)", ctx, cnt)
        if "check_id" in err_df.columns:
            target_null_rows = err_df[
                (err_df.get("schema_context", pd.Series()) == target_col) &
                (err_df.get("check_id", pd.Series()).str.contains("not_nullable", na=False))
            ]
            if not target_null_rows.empty:
                raise ValueError(
                    f"[Pandera] Target column '{target_col}' contains "
                    f"{len(target_null_rows)} null value(s). Drop or impute before training."
                ) from exc
    except Exception as exc:
        log.warning("  [Pandera] Unexpected validation error (skipping): %s", exc)


# ── Config validation (N2) ────────────────────────────────────────────────────

def check_config(df: pd.DataFrame) -> None:
    """N2: Hard errors + range warnings for all config constants."""
    errors: list[str] = []
    warnings_: list[str] = []

    if TARGET_COL not in df.columns:
        errors.append(f"TARGET_COL '{TARGET_COL}' not found. Available: {df.columns.tolist()}")
    if TASK not in ("classification", "regression"):
        errors.append(f"TASK must be 'classification' or 'regression', got '{TASK}'")
    if CV_STRATEGY not in ("stratified", "timeseries"):
        errors.append(f"cv_strategy must be 'stratified' or 'timeseries', got '{CV_STRATEGY}'")

    if errors:
        raise ValueError("\n  CONFIG ERRORS:\n  " + "\n  ".join(errors))

    def _warn(cond: bool, msg: str) -> None:
        if cond:
            warnings_.append(msg)

    _warn(not (0.05 < TEST_SIZE <= 0.40),       f"test_size={TEST_SIZE} outside (0.05, 0.40]")
    _warn(CV_FOLDS != 0 and not (2 <= CV_FOLDS <= 20), f"cv_folds={CV_FOLDS} outside [2, 20]")
    _warn(not (5 <= N_TRIALS <= 500),           f"n_trials={N_TRIALS} outside [5, 500]")
    _warn(not (0.10 < SEARCH_SUBSAMPLE <= 1.0), f"search_subsample={SEARCH_SUBSAMPLE} outside (0.10, 1.0]")
    _warn(not (50 <= N_ESTIMATORS_MAX <= 5000), f"n_estimators_max={N_ESTIMATORS_MAX} outside [50, 5000]")
    _warn(not (5 <= EARLY_STOP_RNDS <= 200),    f"early_stop_rnds={EARLY_STOP_RNDS} outside [5, 200]")
    _warn(not (0.5 < PCA_VARIANCE <= 1.0),      f"pca_variance={PCA_VARIANCE} outside (0.5, 1.0]")
    _warn(not (0.0 < IMBALANCE_THRESHOLD <= 0.5), f"imbalance_threshold={IMBALANCE_THRESHOLD} outside (0.0, 0.5]")
    _warn(not (2 <= CARDINALITY_LIMIT <= 500),  f"cardinality_limit={CARDINALITY_LIMIT} outside [2, 500]")
    _warn(not (0.0 < DRIFT_ALPHA <= 0.5),       f"drift_alpha={DRIFT_ALPHA} outside (0.0, 0.5]")
    _warn(not (0.0 < OUTLIER_CONTAMINATION < 0.5), f"outlier_contamination={OUTLIER_CONTAMINATION} outside (0.0, 0.5)")
    _warn(not (1 <= PDP_TOP_N <= 30),           f"pdp_top_n={PDP_TOP_N} outside [1, 30]")
    _warn(not (1 <= CB_LOG_PERIOD <= 500),      f"callback_log_period={CB_LOG_PERIOD} outside [1, 500]")
    _warn(not (0.0 <= VARIANCE_THRESHOLD < 1.0), f"variance_threshold={VARIANCE_THRESHOLD} outside [0.0, 1.0)")
    _warn(not (2 <= TARGET_ENC_THRESHOLD <= 10_000), f"target_encoding_threshold={TARGET_ENC_THRESHOLD} outside [2, 10000]")

    if TASK == "regression" and not TARGET_LOG_TRANSFORM and TARGET_COL in df.columns:
        y_vals = df[TARGET_COL].dropna()
        if len(y_vals) > 10 and abs(y_vals.skew()) > 2.0:
            _warn(True, f"target '{TARGET_COL}' skewness={abs(y_vals.skew()):.2f} > 2.0 — "
                  "consider target_log_transform: true in config.yaml.")

    for w in warnings_:
        log.warning("  [CONFIG WARNING] %s", w)
    if warnings_:
        log.warning("  %d config warning(s) above. Training continues.", len(warnings_))
    log.info("Config validated — target='%s', task='%s', warnings=%d",
             TARGET_COL, TASK, len(warnings_))


# ── Drift detection ───────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    drifted_numerical:   list[str]        = field(default_factory=list)
    drifted_categorical: list[str]        = field(default_factory=list)
    pvalues_numerical:   dict[str, float] = field(default_factory=dict)
    pvalues_categorical: dict[str, float] = field(default_factory=dict)

    @property
    def any_drift(self) -> bool:
        return bool(self.drifted_numerical or self.drifted_categorical)

    def to_dict(self) -> dict:
        return {
            "drifted_numerical":   self.drifted_numerical,
            "drifted_categorical": self.drifted_categorical,
            "pvalues_numerical":   self.pvalues_numerical,
            "pvalues_categorical": self.pvalues_categorical,
        }


def detect_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> DriftReport:
    """UPGRADE 12: KS test (numerical) + chi-squared (categorical) drift detection."""
    log.info("[3c/9] Drift detection (α=%.3f)...", DRIFT_ALPHA)
    report = DriftReport()

    for col in num_cols:
        if col not in X_train.columns or col not in X_test.columns:
            continue
        tr = X_train[col].dropna().values
        te = X_test[col].dropna().values
        if len(tr) == 0 or len(te) == 0:
            continue
        _, pval = scipy_stats.ks_2samp(tr, te)
        report.pvalues_numerical[col] = float(pval)
        if pval < DRIFT_ALPHA:
            report.drifted_numerical.append(col)
            log.warning("  DRIFT detected in '%s' (KS p=%.4f < %.3f)", col, pval, DRIFT_ALPHA)

    for col in cat_cols:
        if col not in X_train.columns or col not in X_test.columns:
            continue
        all_cats = set(X_train[col].dropna().unique()) | set(X_test[col].dropna().unique())
        if not all_cats:
            continue
        cats = sorted(all_cats)
        tr_counts = np.array([(X_train[col] == c).sum() for c in cats], dtype=float)
        te_counts = np.array([(X_test[col]  == c).sum() for c in cats], dtype=float)
        if tr_counts.sum() == 0 or te_counts.sum() == 0:
            continue
        expected = tr_counts / tr_counts.sum() * te_counts.sum()
        mask = expected > 0
        if mask.sum() < 2:
            continue
        _, pval = scipy_stats.chisquare(te_counts[mask], f_exp=expected[mask])
        report.pvalues_categorical[col] = float(pval)
        if pval < DRIFT_ALPHA:
            report.drifted_categorical.append(col)
            log.warning("  DRIFT detected in '%s' (χ² p=%.4f < %.3f)", col, pval, DRIFT_ALPHA)

    total = len(report.drifted_numerical) + len(report.drifted_categorical)
    if total == 0:
        log.info("  No drift detected.")
    else:
        msg = f"  {total} feature(s) show drift vs test set."
        if DRIFT_WARN_ONLY:
            log.warning(msg)
        else:
            raise ValueError(msg + " Set drift_warn_only=true to proceed anyway.")
    return report


# ── Log-transform (UPGRADE 30) ────────────────────────────────────────────────

def maybe_log_transform(y: pd.Series) -> tuple[pd.Series, bool]:
    """U30: log1p-transform regression target when configured and y > 0."""
    if TASK != "regression" or not TARGET_LOG_TRANSFORM:
        return y, False
    if y.min() <= 0:
        log.warning("  target_log_transform=true but target has non-positive values "
                    "(min=%.4g). Skipping.", y.min())
        return y, False
    log.info("  UPGRADE 30: log1p-transforming target (min=%.4g → log1p=%.4g)",
             float(y.min()), float(np.log1p(y.min())))
    return np.log1p(y), True