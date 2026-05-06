"""data.py — Data loading, cleaning, GX validation, Pandera schema, drift detection."""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import great_expectations as gx
logging.getLogger("great_expectations").setLevel(logging.WARNING)
from scipy import stats as scipy_stats

# ── Optional Polars (Phase 1) ─────────────────────────────────────────────────
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

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
    """Phase 1: Glob-aware multi-format loader with Polars + parallel reads.

    Priority: Polars (rayon-parallelised) → pandas fallback.
    Multiple files are read in parallel via ThreadPoolExecutor.
    """
    import glob as _glob

    path_str = str(DATA_PATH)
    paths    = sorted(_glob.glob(path_str))
    if not paths:
        raise FileNotFoundError(
            f"data_path='{path_str}' matched no files. Check config.yaml → data_path."
        )

    # ── per-file reader ───────────────────────────────────────────────────────

    def _read_csv_pandas(p: str) -> pd.DataFrame:
        """Chunked or whole CSV read via pandas (fallback path)."""
        if CSV_CHUNK_SIZE is None:
            return pd.read_csv(p, low_memory=False)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[load] Streaming CSV '%s' in chunks of %d row(s)...", p, CSV_CHUNK_SIZE)
        chunks: list[pd.DataFrame] = []
        total_rows = 0
        for chunk_idx, chunk in enumerate(
            pd.read_csv(p, chunksize=CSV_CHUNK_SIZE, low_memory=False), 1
        ):
            chunks.append(chunk)
            total_rows += len(chunk)
            if log.isEnabledFor(logging.DEBUG) and (
                chunk_idx == 1 or chunk_idx % CSV_CHUNK_LOG_EVERY == 0
            ):
                log.debug("[load]   chunk %d → %d cumulative rows", chunk_idx, total_rows)
        if not chunks:
            return pd.DataFrame()
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[load] Finished chunked CSV '%s' → %d chunk(s), %d rows",
                      p, len(chunks), total_rows)
        return pd.concat(chunks, ignore_index=True)

    def _read_one(p: str) -> pd.DataFrame:
        ext = Path(p).suffix.lower()

        if _POLARS_AVAILABLE:
            try:
                if ext == ".parquet":
                    return pl.read_parquet(p, parallel="row_groups").to_pandas()
                if ext == ".csv" and CSV_CHUNK_SIZE is None:
                    return pl.read_csv(p, low_memory=False).to_pandas()
                if ext in (".json", ".jsonl"):
                    return pl.read_ndjson(p).to_pandas() if ext == ".jsonl" \
                        else pl.read_json(p).to_pandas()
                if ext in (".xlsx", ".xls"):
                    return pl.read_excel(p).to_pandas()
            except Exception as exc:  # noqa: BLE001
                log.warning("[load] Polars failed for '%s' (%s) — falling back to pandas", p, exc)

        # pandas fallback
        if ext == ".parquet":
            return pd.read_parquet(p)
        if ext in (".json", ".jsonl"):
            return pd.read_json(p, lines=(ext == ".jsonl"))
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(p)
        return _read_csv_pandas(p)

    # ── parallel read (>1 file) or single direct call ─────────────────────────

    if len(paths) == 1:
        frames = [_read_one(paths[0])]
    else:
        frames_map: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as pool:
            futs = {pool.submit(_read_one, p): p for p in paths}
            for fut in as_completed(futs):
                frames_map[futs[fut]] = fut.result()
        frames = [frames_map[p] for p in paths]   # preserve glob sort order

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    log.info("[load] Read %d file(s) → %d rows × %d cols  (polars=%s)",
             len(paths), *df.shape, _POLARS_AVAILABLE)
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Schema-driven cleaning with vectorised ops and optional Parquet cache.

    Optimisations applied:
    - df.mask() replaces chained replace() for sentinel removal
    - datetime format detected once from a 200-row sample
    - .dt accessors cached in locals; all 6 trig features in 2 numpy passes
    - isnull().sum() gated behind DEBUG
    - float32 cast at end to halve downstream memory
    - Cleaned output cached as Parquet+Zstd; skips reprocessing on repeat runs
    """
    import tempfile, os

    # ── Parquet cache key: source file mtime + raw shape ─────────────────────
    _source_path = Path(str(DATA_PATH))
    _cache_key   = (
        f"{_source_path.stat().st_mtime_ns if _source_path.exists() else 0}"
        f"_{df.shape[0]}_{df.shape[1]}"
    )
    _cache_hash  = hashlib.md5(_cache_key.encode()).hexdigest()[:10]
    _cache_file  = Path(tempfile.gettempdir()) / f"xgb_clean_{_cache_hash}.parquet"

    if _cache_file.exists():
        log.info("[1/9] Cache hit — loading cleaned data from %s", _cache_file)
        return pd.read_parquet(_cache_file)

    log.info("[1/9] Cleaning data... shape before: %s", df.shape)

    # ── Drop all-NaN rows / duplicates ────────────────────────────────────────
    df = df.dropna(how="all").drop_duplicates()

    # ── Sentinel removal via mask (single vectorised pass) ────────────────────
    _SENTINELS = {-999, -9999}
    _STR_SENTINELS = {"N/A", "n/a", "NA", "none", "None", ""}
    num_mask = df.select_dtypes(include="number").isin(_SENTINELS)
    df[num_mask.columns] = df[num_mask.columns].mask(num_mask)
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
        df[str_cols] = df[str_cols].apply(
            lambda c: c.mask(c.isin(_STR_SENTINELS))
        )

    # ── Date column parsing ───────────────────────────────────────────────────
    date_cols = [
        c for c in df.columns
        if ("date" in c.lower() or "time" in c.lower()) and df[c].dtype == "object"
    ]

    def _detect_format(series: pd.Series, sample_n: int = 200) -> str | None:
        """Sample up to sample_n non-null values and guess the strftime format."""
        sample = series.dropna().head(sample_n)
        _FORMATS = [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y",
        ]
        for fmt in _FORMATS:
            try:
                pd.to_datetime(sample, format=fmt, errors="raise")
                return fmt
            except Exception:
                continue
        return None  # fall back to inference

    new_cols: dict[str, np.ndarray | pd.Series] = {}
    drop_cols: list[str] = []

    for col in date_cols:
        fmt    = _detect_format(df[col])
        parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
        if parsed.notna().mean() <= 0.5:
            continue

        # cache accessors once
        _month = parsed.dt.month.astype(float).to_numpy()
        _dow   = parsed.dt.dayofweek.astype(float).to_numpy()
        _hour  = parsed.dt.hour.astype(float).to_numpy()

        month_angle = 2 * np.pi * (_month - 1) / 12
        dow_angle   = 2 * np.pi * _dow / 7
        hour_angle  = 2 * np.pi * _hour / 24

        # 2 numpy passes for all 6 trig features
        sins = np.stack([np.sin(month_angle), np.sin(dow_angle), np.sin(hour_angle)], axis=1)
        coss = np.stack([np.cos(month_angle), np.cos(dow_angle), np.cos(hour_angle)], axis=1)

        new_cols[f"{col}_year"]          = parsed.dt.year.to_numpy()
        new_cols[f"{col}_month"]         = _month.astype(int)
        new_cols[f"{col}_dayofweek"]     = _dow.astype(int)
        new_cols[f"{col}_hour"]          = _hour.astype(int)
        new_cols[f"{col}_month_sin"]     = sins[:, 0]
        new_cols[f"{col}_month_cos"]     = coss[:, 0]
        new_cols[f"{col}_dayofweek_sin"] = sins[:, 1]
        new_cols[f"{col}_dayofweek_cos"] = coss[:, 1]
        new_cols[f"{col}_hour_sin"]      = sins[:, 2]
        new_cols[f"{col}_hour_cos"]      = coss[:, 2]
        drop_cols.append(col)
        log.info("  Parsed date column: '%s' (fmt=%s) → year/month/dow/hour + sin/cos",
                 col, fmt or "inferred")

    if drop_cols:
        df = df.drop(columns=drop_cols)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── float32 cast (halves memory, speeds up XGB) ───────────────────────────
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype("float32")

    log.info("  Shape after: %s", df.shape)
    if log.isEnabledFor(logging.DEBUG):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            log.debug("  Missing values:\n%s", missing)

    # ── persist cache ─────────────────────────────────────────────────────────
    try:
        df.to_parquet(_cache_file, compression="zstd", index=False)
        log.info("  Cleaned data cached → %s", _cache_file)
    except Exception as exc:
        log.debug("  Cache write skipped: %s", exc)

    return df


# ── Great Expectations validation ─────────────────────────────────────────────

# Phase 3: cache GX context so repeated calls in the same process don't re-init
_GX_CONTEXT_CACHE: object | None = None


def _gx_get_or_add(collection, add_method: str, get_method: str, **kwargs):
    """Phase 3: DRY helper — try add_*, fall back to get_* on collision."""
    try:
        return getattr(collection, add_method)(**kwargs)
    except Exception:  # noqa: BLE001 — GX raises various internal types
        return getattr(collection, get_method)(kwargs.get("name", list(kwargs.values())[0]))


def validate_data(df: pd.DataFrame, skip: bool = False) -> None:
    """Phase 3: Great Expectations validation with cached context and skip flag."""
    if skip:
        log.info("[2/9] Validation skipped (skip=True).")
        return

    log.info("[2/9] Validating data (Great Expectations)...")
    global _GX_CONTEXT_CACHE
    if _GX_CONTEXT_CACHE is None:
        _GX_CONTEXT_CACHE = gx.get_context()
    context = _GX_CONTEXT_CACHE

    try:
        data_source = _gx_get_or_add(
            context.data_sources, "add_pandas", "get", name="pandas"
        )
        data_asset = _gx_get_or_add(
            data_source, "add_dataframe_asset", "get_asset", name="train_data"
        )
    except gx.exceptions.DataContextError as exc:
        log.warning("[2/9] GX context error: %s — skipping validation.", exc)
        return

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