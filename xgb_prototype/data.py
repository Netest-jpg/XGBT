"""data.py — Data loading, cleaning, GX validation, Pandera schema, drift detection."""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import great_expectations as gx
logging.getLogger("great_expectations").setLevel(logging.WARNING)
from scipy import stats as scipy_stats

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
    PRETRANSFORM_LOG1P_COLS, PRETRANSFORM_DROP_COLS,
)

log = logging.getLogger(__name__)


# ── Missing-value report ──────────────────────────────────────────────────────

@dataclass
class MissingValueReport:
    total_missing: int
    missing_cells_pct: float
    rows_with_missing: int
    rows_with_missing_pct: float
    columns: list[dict]
    output_csv: str | None = None
    output_json: str | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def automatic_missing_value_report(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> MissingValueReport:
    """Create a structured missing-value report and optionally persist it."""
    missing_mask   = df.isna()
    missing_by_col = missing_mask.sum().sort_values(ascending=False)
    rows_total     = max(len(df), 1)
    total_cells    = max(df.shape[0] * df.shape[1], 1)
    rows_with_missing = int(missing_mask.any(axis=1).sum())

    column_rows = [
        {
            "column": str(col), "missing_count": int(n),
            "missing_pct": float(n / rows_total),
            "is_target": bool(col == target_col),
            "dtype": str(df[col].dtype),
        }
        for col, n in missing_by_col.items() if int(n) > 0
    ]

    report = MissingValueReport(
        total_missing=int(missing_by_col.sum()),
        missing_cells_pct=float(missing_by_col.sum() / total_cells),
        rows_with_missing=rows_with_missing,
        rows_with_missing_pct=float(rows_with_missing / rows_total),
        columns=column_rows,
    )

    if output_dir is not None and run_id is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path  = output_dir / f"missing_values_{run_id}.csv"
        json_path = output_dir / f"missing_values_{run_id}.json"
        pd.DataFrame(column_rows).to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
        report.output_csv, report.output_json = str(csv_path), str(json_path)

    if report.total_missing:
        log.warning("  [missing] %d missing value(s), %.2f%% of cells; %d/%d row(s) affected.",
                    report.total_missing, 100 * report.missing_cells_pct,
                    report.rows_with_missing, len(df))
        if column_rows:
            log.warning("  [missing] Top columns: %s", column_rows[:5])
    else:
        log.info("  [missing] No missing values detected.")
    return report


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Glob-aware multi-format loader with Polars + parallel reads."""
    import glob as _glob
    paths = sorted(_glob.glob(str(DATA_PATH)))
    if not paths:
        raise FileNotFoundError(
            f"data_path='{DATA_PATH}' matched no files. Check config.yaml → data_path."
        )

    def _read_csv_pandas(p: str) -> pd.DataFrame:
        if CSV_CHUNK_SIZE is None:
            return pd.read_csv(p, low_memory=False)
        log.debug("[load] Streaming CSV '%s' in chunks of %d row(s)...", p, CSV_CHUNK_SIZE)
        chunks, total_rows = [], 0
        for i, chunk in enumerate(pd.read_csv(p, chunksize=CSV_CHUNK_SIZE, low_memory=False), 1):
            chunks.append(chunk)
            total_rows += len(chunk)
            if log.isEnabledFor(logging.DEBUG) and (i == 1 or i % CSV_CHUNK_LOG_EVERY == 0):
                log.debug("[load]   chunk %d → %d cumulative rows", i, total_rows)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def _read_one(p: str) -> pd.DataFrame:
        ext = Path(p).suffix.lower()
        if _POLARS_AVAILABLE:
            try:
                if ext == ".parquet":
                    return pl.read_parquet(p, parallel="row_groups").to_pandas()
                if ext == ".csv" and CSV_CHUNK_SIZE is None:
                    return pl.read_csv(p, low_memory=False).to_pandas()
                if ext in (".json", ".jsonl"):
                    return (pl.read_ndjson if ext == ".jsonl" else pl.read_json)(p).to_pandas()
                if ext in (".xlsx", ".xls"):
                    return pl.read_excel(p).to_pandas()
            except Exception as exc:
                log.warning("[load] Polars failed for '%s' (%s) — falling back to pandas", p, exc)
        if ext == ".parquet":
            return pd.read_parquet(p)
        if ext in (".json", ".jsonl"):
            return pd.read_json(p, lines=(ext == ".jsonl"))
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(p)
        return _read_csv_pandas(p)

    if len(paths) == 1:
        frames = [_read_one(paths[0])]
    else:
        results: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as pool:
            futs = {pool.submit(_read_one, p): p for p in paths}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        frames = [results[p] for p in paths]

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    log.info("[load] Read %d file(s) → %d rows × %d cols  (polars=%s)",
             len(paths), *df.shape, _POLARS_AVAILABLE)
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Schema-driven cleaning with vectorised ops and optional Parquet cache."""
    src = Path(str(DATA_PATH))
    pretransform_sig = (
        "log1p=" + ",".join(sorted(PRETRANSFORM_LOG1P_COLS))
        + "|drop=" + ",".join(sorted(PRETRANSFORM_DROP_COLS))
    )
    cache_key  = f"{src.stat().st_mtime_ns if src.exists() else 0}_{df.shape[0]}_{df.shape[1]}_{pretransform_sig}"
    cache_file = Path(tempfile.gettempdir()) / f"xgb_clean_{hashlib.md5(cache_key.encode()).hexdigest()[:10]}.parquet"

    if cache_file.exists():
        log.info("[1/9] Cache hit — loading cleaned data from %s", cache_file)
        return pd.read_parquet(cache_file)

    log.info("[1/9] Cleaning data... shape before: %s", df.shape)
    df = df.dropna(how="all").drop_duplicates()

    # Sentinel removal
    _SENTINELS, _STR_SENTINELS = {-999, -9999}, {"N/A", "n/a", "NA", "none", "None", ""}
    num_mask = df.select_dtypes(include="number").isin(_SENTINELS)
    df[num_mask.columns] = df[num_mask.columns].mask(num_mask)
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip().mask(c.str.strip().isin(_STR_SENTINELS)))

    # Date parsing
    date_cols = [c for c in df.columns
                 if ("date" in c.lower() or "time" in c.lower()) and df[c].dtype == "object"]

    _FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]

    def _detect_format(s: pd.Series) -> str | None:
        sample = s.dropna().head(200)
        for fmt in _FORMATS:
            try:
                pd.to_datetime(sample, format=fmt, errors="raise")
                return fmt
            except Exception:
                continue
        return None

    new_cols: dict = {}
    drop_cols: list[str] = []
    for col in date_cols:
        parsed = pd.to_datetime(df[col], format=_detect_format(df[col]), errors="coerce")
        if parsed.notna().mean() <= 0.5:
            continue
        m, dow, h = parsed.dt.month.to_numpy(float), parsed.dt.dayofweek.to_numpy(float), parsed.dt.hour.to_numpy(float)
        sins = np.stack([np.sin(2*np.pi*m/12), np.sin(2*np.pi*dow/7), np.sin(2*np.pi*h/24)], axis=1)
        coss = np.stack([np.cos(2*np.pi*m/12), np.cos(2*np.pi*dow/7), np.cos(2*np.pi*h/24)], axis=1)
        new_cols.update({
            f"{col}_year": parsed.dt.year.to_numpy(), f"{col}_month": m.astype(int),
            f"{col}_dayofweek": dow.astype(int),      f"{col}_hour": h.astype(int),
            f"{col}_month_sin": sins[:,0],   f"{col}_month_cos": coss[:,0],
            f"{col}_dayofweek_sin": sins[:,1], f"{col}_dayofweek_cos": coss[:,1],
            f"{col}_hour_sin": sins[:,2],    f"{col}_hour_cos": coss[:,2],
        })
        drop_cols.append(col)
        log.info("  Parsed date column: '%s' (fmt=%s) → year/month/dow/hour + sin/cos",
                 col, _detect_format(df[col]) or "inferred")

    if drop_cols:
        df = df.drop(columns=drop_cols)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype("float32")

    log.info("  Shape after: %s", df.shape)
    if log.isEnabledFor(logging.DEBUG):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            log.debug("  Missing values:\n%s", missing)

    try:
        df.to_parquet(cache_file, compression="zstd", index=False)
        log.info("  Cleaned data cached → %s", cache_file)
    except Exception as exc:
        log.debug("  Cache write skipped: %s", exc)
    return df


def apply_pretransforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply config-driven column-level transforms before the main pipeline."""
    for col in PRETRANSFORM_LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            log.info("  [pretransform] log1p applied to '%s'", col)
        else:
            log.warning("  [pretransform] log1p requested for '%s' but column not found", col)
    drop = [c for c in PRETRANSFORM_DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        log.info("  [pretransform] dropped column(s): %s", drop)
    return df


# ── Cache management ──────────────────────────────────────────────────────────

def clear_cache(verbose: bool = True) -> int:
    """Delete all xgb_clean_*.parquet files from the system temp directory."""
    tmp   = Path(tempfile.gettempdir())
    files = list(tmp.glob("xgb_clean_*.parquet"))
    for f in files:
        try:
            f.unlink()
            if verbose:
                log.info("[cache] Deleted stale cache file: %s", f)
        except OSError as exc:
            log.warning("[cache] Could not delete %s: %s", f, exc)
    if verbose:
        if files:
            log.info("[cache] Cleared %d cache file(s) from %s", len(files), tmp)
        else:
            log.info("[cache] No cache files found in %s — nothing to clear.", tmp)
    return len(files)


def cache_info() -> list[dict]:
    """Return metadata about existing xgb_clean_*.parquet cache files."""
    tmp, now = Path(tempfile.gettempdir()), time.time()
    infos = []
    for f in sorted(tmp.glob("xgb_clean_*.parquet")):
        stat    = f.stat()
        age_min = (now - stat.st_mtime) / 60
        infos.append({"path": str(f), "size_mb": round(stat.st_size / 1e6, 2), "age_min": round(age_min, 1)})
        log.info("[cache] %s  (%.1f MB, %.0f min old)", f.name, infos[-1]["size_mb"], age_min)
    if not infos:
        log.info("[cache] No xgb_clean_*.parquet files found in %s", tmp)
    return infos


# ── Great Expectations validation ─────────────────────────────────────────────

_GX_CONTEXT_CACHE: object | None = None


def _gx_get_or_add(collection, add_method: str, get_method: str, **kwargs):
    try:
        return getattr(collection, add_method)(**kwargs)
    except Exception:
        return getattr(collection, get_method)(kwargs.get("name", list(kwargs.values())[0]))


def validate_data(df: pd.DataFrame, skip: bool = False) -> None:
    """Great Expectations validation with cached context and skip flag."""
    if skip:
        log.info("[2/9] Validation skipped (skip=True).")
        return
    log.info("[2/9] Validating data (Great Expectations)...")
    global _GX_CONTEXT_CACHE
    if _GX_CONTEXT_CACHE is None:
        _GX_CONTEXT_CACHE = gx.get_context()
    context = _GX_CONTEXT_CACHE

    try:
        data_source = _gx_get_or_add(context.data_sources, "add_pandas", "get", name="pandas")
        data_asset  = _gx_get_or_add(data_source, "add_dataframe_asset", "get_asset", name="train_data")
    except gx.exceptions.DataContextError as exc:
        log.warning("[2/9] GX context error: %s — skipping validation.", exc)
        return

    batch  = data_asset.add_batch_definition_whole_dataframe("full_batch").get_batch(
        batch_parameters={"dataframe": df}
    )
    rules, failed = [], []
    for rule in rules:
        result = batch.validate(rule)
        label  = getattr(rule, "column", "table-level")
        log.info("  %s %s [%s]", "✓" if result.success else "✗", rule.__class__.__name__, label)
        if not result.success:
            failed.append(rule)
    if failed:
        raise ValueError(f"\n  Validation failed for {len(failed)} rule(s). Fix data before training.")
    log.info("  All validations passed!")


# ── Pandera schema validation ─────────────────────────────────────────────────

def _sigfig(x: float, n: int = 4) -> float:
    if x == 0:
        return 0.0
    mag = 10 ** (n - 1 - int(np.floor(np.log10(abs(x)))))
    return round(x * mag) / mag


def _infer_pandera_schema(df: pd.DataFrame, target_col: str):
    if not _PANDERA_AVAILABLE:
        log.warning("[Pandera] pandera not installed — schema validation skipped.")
        return None
    columns: dict = {}
    for col in df.columns:
        series, is_target = df[col], (col == target_col)
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 10:
                lo_r, hi_r = _sigfig(float(np.percentile(clean, 0.1))), _sigfig(float(np.percentile(clean, 99.9)))
                checks = ([pa.Check(lambda s, lo=lo_r, hi=hi_r: s.between(lo, hi),
                                    element_wise=False,
                                    error=f"Values outside inferred range [{lo_r}, {hi_r}]",
                                    raise_warning=True)]
                          if lo_r < hi_r else [])
                columns[col] = pa.Column(dtype=float, checks=checks, nullable=not is_target, coerce=True, required=True)
            else:
                columns[col] = pa.Column(dtype=float, nullable=not is_target, coerce=True, required=True)
        elif pd.api.types.is_object_dtype(series) or hasattr(series, "cat"):
            columns[col] = pa.Column(dtype=object, nullable=not is_target, coerce=False, required=True)
        else:
            columns[col] = pa.Column(nullable=True, required=True)
    return pa.DataFrameSchema(columns=columns, index=None, coerce=True, strict=False)


def validate_pandera(df: pd.DataFrame, target_col: str) -> None:
    """Pandera schema validation with inferred column rules."""
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
        log.info("  %-30s  dtype=%-10s  %-10s  checks=%d",
                 col_name, dtype_str, "nullable" if col_schema.nullable else "non-null",
                 len(col_schema.checks))

    try:
        schema.validate(df, lazy=True)
        log.info("  [Pandera] All schema checks passed.")
    except pa.errors.SchemaErrors as exc:
        err_df     = exc.failure_cases
        n_errors   = len(err_df)
        col_counts = err_df["schema_context"].value_counts().to_dict() if "schema_context" in err_df.columns else {}
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


# ── Config validation ─────────────────────────────────────────────────────────

def check_config(df: pd.DataFrame) -> None:
    """Hard errors + range warnings for all config constants."""
    errors: list[str] = []
    if TARGET_COL not in df.columns:
        errors.append(f"TARGET_COL '{TARGET_COL}' not found. Available: {df.columns.tolist()}")
    if TASK not in ("classification", "regression"):
        errors.append(f"TASK must be 'classification' or 'regression', got '{TASK}'")
    if CV_STRATEGY not in ("stratified", "timeseries"):
        errors.append(f"cv_strategy must be 'stratified' or 'timeseries', got '{CV_STRATEGY}'")
    if errors:
        raise ValueError("\n  CONFIG ERRORS:\n  " + "\n  ".join(errors))

    range_checks = [
        (not (0.05 < TEST_SIZE <= 0.40),                    f"test_size={TEST_SIZE} outside (0.05, 0.40]"),
        (CV_FOLDS != 0 and not (2 <= CV_FOLDS <= 20),       f"cv_folds={CV_FOLDS} outside [2, 20]"),
        (not (5 <= N_TRIALS <= 500),                        f"n_trials={N_TRIALS} outside [5, 500]"),
        (not (0.10 < SEARCH_SUBSAMPLE <= 1.0),              f"search_subsample={SEARCH_SUBSAMPLE} outside (0.10, 1.0]"),
        (not (50 <= N_ESTIMATORS_MAX <= 5000),              f"n_estimators_max={N_ESTIMATORS_MAX} outside [50, 5000]"),
        (not (5 <= EARLY_STOP_RNDS <= 200),                 f"early_stop_rnds={EARLY_STOP_RNDS} outside [5, 200]"),
        (not (0.5 < PCA_VARIANCE <= 1.0),                   f"pca_variance={PCA_VARIANCE} outside (0.5, 1.0]"),
        (not (0.0 < IMBALANCE_THRESHOLD <= 0.5),            f"imbalance_threshold={IMBALANCE_THRESHOLD} outside (0.0, 0.5]"),
        (not (2 <= CARDINALITY_LIMIT <= 500),               f"cardinality_limit={CARDINALITY_LIMIT} outside [2, 500]"),
        (not (0.0 < DRIFT_ALPHA <= 0.5),                    f"drift_alpha={DRIFT_ALPHA} outside (0.0, 0.5]"),
        (not (0.0 < OUTLIER_CONTAMINATION < 0.5),           f"outlier_contamination={OUTLIER_CONTAMINATION} outside (0.0, 0.5)"),
        (not (1 <= PDP_TOP_N <= 30),                        f"pdp_top_n={PDP_TOP_N} outside [1, 30]"),
        (not (1 <= CB_LOG_PERIOD <= 500),                   f"callback_log_period={CB_LOG_PERIOD} outside [1, 500]"),
        (not (0.0 <= VARIANCE_THRESHOLD < 1.0),             f"variance_threshold={VARIANCE_THRESHOLD} outside [0.0, 1.0)"),
        (not (2 <= TARGET_ENC_THRESHOLD <= 10_000),         f"target_encoding_threshold={TARGET_ENC_THRESHOLD} outside [2, 10000]"),
    ]
    if TASK == "regression" and not TARGET_LOG_TRANSFORM and TARGET_COL in df.columns:
        y_vals = df[TARGET_COL].dropna()
        if len(y_vals) > 10 and abs(y_vals.skew()) > 2.0:
            range_checks.append((True, f"target '{TARGET_COL}' skewness={abs(y_vals.skew()):.2f} > 2.0 — "
                                  "consider target_log_transform: true in config.yaml."))

    warnings_ = [msg for cond, msg in range_checks if cond]
    for w in warnings_:
        log.warning("  [CONFIG WARNING] %s", w)
    if warnings_:
        log.warning("  %d config warning(s) above. Training continues.", len(warnings_))
    log.info("Config validated — target='%s', task='%s', warnings=%d", TARGET_COL, TASK, len(warnings_))


# ── Drift detection ───────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    drifted_numerical:     list[str]        = field(default_factory=list)
    drifted_categorical:   list[str]        = field(default_factory=list)
    label_drift:           dict             = field(default_factory=dict)
    data_quality_drift:    dict             = field(default_factory=dict)
    serving_training_skew: dict             = field(default_factory=dict)
    novel_class_emergence: dict             = field(default_factory=dict)
    segment_drift:         dict             = field(default_factory=dict)
    pvalues_numerical:     dict[str, float] = field(default_factory=dict)
    pvalues_categorical:   dict[str, float] = field(default_factory=dict)

    @property
    def any_drift(self) -> bool:
        return bool(
            self.drifted_numerical or self.drifted_categorical
            or self.label_drift.get("drift_detected")
            or self.data_quality_drift.get("drift_detected")
            or self.serving_training_skew.get("skew_detected")
            or self.novel_class_emergence.get("novel_classes")
            or self.segment_drift.get("drifted_segments")
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _distribution_chisquare(ref: pd.Series, new: pd.Series, alpha: float) -> dict:
    ref_vc = ref.dropna().astype(str).value_counts()
    new_vc = new.dropna().astype(str).value_counts()
    all_vals = sorted(set(ref_vc.index) | set(new_vc.index))
    if len(all_vals) < 2 or ref_vc.sum() == 0 or new_vc.sum() == 0:
        return {"drift_detected": False, "pvalue": None, "reason": "insufficient categories"}
    ref_c = ref_vc.reindex(all_vals, fill_value=0).to_numpy(float)
    new_c = new_vc.reindex(all_vals, fill_value=0).to_numpy(float)
    _, pval = scipy_stats.chisquare(new_c, f_exp=(ref_c + 1e-6) / (ref_c + 1e-6).sum() * new_c.sum())
    return {
        "drift_detected": bool(pval < alpha),
        "pvalue": float(pval),
        "reference_distribution": dict(zip(all_vals, ref_c.astype(int))),
        "new_distribution":       dict(zip(all_vals, new_c.astype(int))),
    }


def _data_quality_drift(X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float) -> dict:
    ref_miss = X_ref.isna().mean()
    new_miss = X_new.isna().mean()
    changed_missing = {
        col: {"reference_missing_pct": float(ref_miss.get(col, 0.0)),
              "new_missing_pct":       float(new_miss.get(col, 0.0)),
              "delta":                 float(new_miss.get(col, 0.0) - ref_miss.get(col, 0.0))}
        for col in sorted(set(X_ref.columns) & set(X_new.columns))
        if abs(new_miss.get(col, 0.0) - ref_miss.get(col, 0.0)) >= max(0.05, alpha)
    }
    all_null_new   = [str(c) for c in X_new.columns if X_new[c].isna().all()]
    constant_new   = [str(c) for c in X_new.columns if X_new[c].nunique(dropna=True) <= 1]
    dup_delta      = float(X_new.duplicated().mean() - X_ref.duplicated().mean())
    return {
        "drift_detected":           bool(changed_missing or all_null_new or abs(dup_delta) >= max(0.05, alpha)),
        "missing_rate_changes":     changed_missing,
        "all_null_columns_new":     all_null_new,
        "constant_columns_new":     constant_new,
        "duplicate_rate_reference": float(X_ref.duplicated().mean()),
        "duplicate_rate_new":       float(X_new.duplicated().mean()),
        "duplicate_rate_delta":     dup_delta,
    }


def _serving_training_skew(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> dict:
    ref_cols, new_cols = set(X_ref.columns), set(X_new.columns)
    return {
        "skew_detected":  bool((ref_cols - new_cols) or (new_cols - ref_cols) or
                               any(str(X_ref[c].dtype) != str(X_new[c].dtype) for c in ref_cols & new_cols)),
        "missing_columns": sorted(ref_cols - new_cols),
        "extra_columns":   sorted(new_cols - ref_cols),
        "dtype_changes":   {str(c): {"reference": str(X_ref[c].dtype), "new": str(X_new[c].dtype)}
                            for c in sorted(ref_cols & new_cols)
                            if str(X_ref[c].dtype) != str(X_new[c].dtype)},
    }


def _segment_drift(X_ref: pd.DataFrame, X_new: pd.DataFrame,
                   num_cols: list[str], segment_cols: list[str], alpha: float) -> dict:
    drifted_segments: list[dict] = []
    for seg_col in segment_cols:
        if seg_col not in X_ref.columns or seg_col not in X_new.columns:
            continue
        for seg in sorted(set(X_ref[seg_col].dropna().unique()) & set(X_new[seg_col].dropna().unique()))[:25]:
            ref_seg = X_ref[X_ref[seg_col] == seg]
            new_seg = X_new[X_new[seg_col] == seg]
            if len(ref_seg) < 20 or len(new_seg) < 20:
                continue
            drifted_features = []
            for col in num_cols[:50]:
                if col not in ref_seg or col not in new_seg:
                    continue
                rv, nv = ref_seg[col].dropna().values, new_seg[col].dropna().values
                if len(rv) < 10 or len(nv) < 10:
                    continue
                _, pval = scipy_stats.ks_2samp(rv, nv)
                if pval < alpha:
                    drifted_features.append({"feature": str(col), "pvalue": float(pval)})
            if drifted_features:
                drifted_segments.append({"segment_column": str(seg_col), "segment_value": str(seg),
                                         "drifted_features": drifted_features[:10]})
    return {"drifted_segments": drifted_segments}


def detect_drift(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    num_cols: list[str], cat_cols: list[str],
    y_train: pd.Series | None = None, y_test: pd.Series | None = None,
    segment_cols: list[str] | None = None,
) -> DriftReport:
    """KS/χ² feature drift plus label, quality, skew, novel-class, and segment checks."""
    log.info("[3c/9] Drift detection (α=%.3f)...", DRIFT_ALPHA)
    report = DriftReport()
    segment_cols = segment_cols or []

    report.data_quality_drift  = _data_quality_drift(X_train, X_test, DRIFT_ALPHA)
    report.serving_training_skew = _serving_training_skew(X_train, X_test)

    for col in num_cols:
        if col not in X_train.columns or col not in X_test.columns:
            continue
        tr, te = X_train[col].dropna().values, X_test[col].dropna().values
        if not len(tr) or not len(te):
            continue
        _, pval = scipy_stats.ks_2samp(tr, te)
        report.pvalues_numerical[col] = float(pval)
        if pval < DRIFT_ALPHA:
            report.drifted_numerical.append(col)
            log.warning("  DRIFT detected in '%s' (KS p=%.4f < %.3f)", col, pval, DRIFT_ALPHA)

    for col in cat_cols:
        if col not in X_train.columns or col not in X_test.columns:
            continue
        tr_vc = X_train[col].dropna().value_counts()
        te_vc = X_test[col].dropna().value_counts()
        all_cats = sorted(set(tr_vc.index) | set(te_vc.index))
        if not all_cats:
            continue
        tr_c = tr_vc.reindex(all_cats, fill_value=0).to_numpy(float)
        te_c = te_vc.reindex(all_cats, fill_value=0).to_numpy(float)
        if not tr_c.sum() or not te_c.sum():
            continue
        mask = (tr_c > 0) & (te_c > 0)
        if mask.sum() < 2:
            continue
        new_in_test = [all_cats[i] for i, v in enumerate(tr_c) if v == 0 and te_c[i] > 0]
        if new_in_test:
            log.warning("  '%s': %d unseen-in-train category(s) excluded from χ² test: %s",
                        col, len(new_in_test), new_in_test[:5])
        tr_m, te_m = tr_c[mask], te_c[mask]
        _, pval = scipy_stats.chisquare(te_m, f_exp=tr_m / tr_m.sum() * te_m.sum())
        report.pvalues_categorical[col] = float(pval)
        if pval < DRIFT_ALPHA:
            report.drifted_categorical.append(col)
            log.warning("  DRIFT detected in '%s' (χ² p=%.4f < %.3f)", col, pval, DRIFT_ALPHA)

    if y_train is not None and y_test is not None:
        report.label_drift = _distribution_chisquare(y_train, y_test, DRIFT_ALPHA)
        train_labels = set(pd.Series(y_train).dropna().astype(str).unique())
        test_labels  = set(pd.Series(y_test).dropna().astype(str).unique())
        novel = sorted(test_labels - train_labels)
        report.novel_class_emergence = {
            "novel_classes": novel,
            "reference_class_count": len(train_labels),
            "new_class_count": len(test_labels),
        }
        if report.label_drift.get("drift_detected"):
            log.warning("  LABEL DRIFT detected (χ² p=%.4f < %.3f)",
                        report.label_drift.get("pvalue"), DRIFT_ALPHA)
        if novel:
            log.warning("  NOVEL CLASS emergence detected: %s", novel)

    if segment_cols:
        report.segment_drift = _segment_drift(X_train, X_test, num_cols, segment_cols, DRIFT_ALPHA)
        if report.segment_drift.get("drifted_segments"):
            log.warning("  SEGMENT DRIFT detected in %d segment(s)",
                        len(report.segment_drift["drifted_segments"]))

    if report.data_quality_drift.get("drift_detected"):
        log.warning("  DATA QUALITY DRIFT detected: %s", report.data_quality_drift)
    if report.serving_training_skew.get("skew_detected"):
        log.warning("  SERVING/TRAINING SKEW detected: %s", report.serving_training_skew)

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


# ── Log-transform ─────────────────────────────────────────────────────────────

def maybe_log_transform(y: pd.Series) -> tuple[pd.Series, bool]:
    """log1p-transform regression target when configured and y > 0."""
    if TASK != "regression" or not TARGET_LOG_TRANSFORM:
        return y, False
    if y.min() <= 0:
        log.warning("  target_log_transform=true but target has non-positive values "
                    "(min=%.4g). Skipping.", y.min())
        return y, False
    log.info("  UPGRADE 30: log1p-transforming target (min=%.4g → log1p=%.4g)",
             float(y.min()), float(np.log1p(y.min())))
    return np.log1p(y), True