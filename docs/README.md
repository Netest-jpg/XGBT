# XGBoost Prototype — Complete Reference

> A generalized, reproducible XGBoost training prototype for tabular classification and regression. Configured entirely through `config.yaml` — new datasets require config changes, not code edits.

---

## Table of Contents

0. [Setup](#0-setup)
1. [Project Overview](#1-project-overview)
2. [Project Layout](#2-project-layout)
3. [Quick Start](#3-quick-start)
4. [Using This Template With a New Dataset](#4-using-this-template-with-a-new-dataset)
5. [config.yaml — Complete Reference](#5-configyaml--complete-reference)
6. [Training Process — Step by Step](#6-training-process--step-by-step)
7. [Package Modules](#7-package-modules-xgb_prototype)
8. [Inference and Serving](#8-inference-and-serving)
9. [Artifacts and Outputs](#9-artifacts-and-outputs)
10. [Diagnostic Plots](#10-diagnostic-plots)
11. [Test Suite](#11-test-suite)
12. [Architecture and Design Decisions](#12-architecture-and-design-decisions)
13. [Known Limitations](#13-known-limitations)
14. [Development Notes](#14-development-notes)

---

## 0. Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and sync:

```bash
git clone https://github.com/Netest-jpg/XGBT.git
cd XGBT
uv sync --extra dev
```

Hard dependencies (enforced at startup): `xgboost ≥ 1.7`, `scikit-learn ≥ 1.3`, `numpy ≥ 1.23`, `pandas ≥ 1.5`.

Optional extras:

```bash
pip install mlflow          # experiment tracking
pip install pandera         # schema validation
pip install category_encoders  # TargetEncoder for high-cardinality columns
```

---

## 1. Project Overview

This is a generalized XGBoost training prototype. It supports classification and regression, runs Optuna hyperparameter tuning, compares cheap baselines before committing to a full search, writes versioned model artifacts, generates a full diagnostic plot suite, and exposes stable inference wrappers for both local and production use.

The sample dataset is `creditcard.csv`, but the pipeline is not fraud-specific. The same code runs across tabular datasets by changing `config.yaml`.

---

## 2. Project Layout

```text
.
├── config.yaml                  # Main runtime configuration
├── creditcard.csv               # Example dataset
├── pyproject.toml               # Dependencies, dev extras, console script
├── train.py                     # Entrypoint launcher
├── uv.lock                      # Locked dependency resolution
├── docs/
└── xgb_prototype/
    ├── __init__.py              # Package exports
    ├── baselines.py             # Dummy, linear, and default-XGBoost comparisons
    ├── config.py                # Typed dataclass config loader
    ├── data.py                  # Data loading, cleaning, drift detection
    ├── deps.py                  # Hard/soft dependency version checks
    ├── drift_monitor.py         # Continuous production drift monitoring
    ├── evaluation.py            # Test-set evaluation, error analysis, threshold tuning
    ├── features.py              # Type inference, variance filter, interactions, RFECV
    ├── inference.py             # PredictWrapper and ModelServer
    ├── metrics.py               # MetricConfig dataclass and auto metric selection
    ├── pipeline.py              # sklearn Pipeline construction and Optuna tuning
    ├── plots.py                 # All diagnostic plot functions
    ├── settings.py              # Module-level constants loaded from config.yaml
    ├── thresholds.py            # Binary threshold tuning policies
    ├── tracking.py              # MLflow, requirements lock, registry, run summary
    └── tests/
        ├── test_config.py
        ├── test_data_loading_and_temporal.py
        ├── test_drift_monitor.py
        ├── test_feature_inference.py
        ├── test_serving.py
        ├── test_smoke_training.py
        └── test_thresholds.py
```

Generated outputs:

```text
models/     # Joblib artifacts, run JSON, registry, requirements locks, CSVs
plots/      # Plotly HTML and PNG diagnostics
```

---

## 3. Quick Start

```bash
# Sync environment
uv sync --extra dev

# Run training
uv run python train.py

# Point to a custom config
uv run python train.py --config path/to/config.yaml

# Fast test suite
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q

# End-to-end smoke test (launches actual training)
RUN_XGB_PROTOTYPE_SMOKE=1 uv run python -m pytest tests/test_smoke_training.py -q
```

---

## 4. Using This Template With a New Dataset

### Step 0 — Prepare your data

One row per training example, one target column, a header row. Supported formats: `.csv`, `.parquet`, `.json`, `.jsonl`, `.xlsx`, `.xls`, or a glob like `data/monthly_*.csv`.

```csv
age,income,plan_type,tenure_months,churned
34,72000,pro,18,0
52,105000,business,44,1
```

### Step 1 — Edit the three required config values

```yaml
task: classification        # or: regression
target_col: churned         # must match exactly
data_path: data/my_data.csv
```

### Step 2 — Run a fast draft first

Confirm the dataset loads and the pipeline trains before committing to a full search:

```yaml
cv_folds: 0
n_trials: 5
n_estimators_max: 100
diagnostics:
  plots_enabled: false
```

```bash
uv run python train.py
```

### Step 3 — Move to a fuller run

```yaml
cv_folds: 5
n_trials: 30
n_estimators_max: 300
diagnostics:
  plots_enabled: true
```

### After training, check

```text
models/baseline_comparison_<run_id>.csv   # Is tuned XGBoost beating simpler models?
models/error_analysis_<run_id>.csv        # What high-confidence mistakes did it make?
plots/                                    # Feature importance, PR curve, threshold sweep, etc.
```

If the tuned model barely beats the dummy baseline: check for data leakage, target quality, or whether the task is learnable from available columns. If logistic regression beats XGBoost, the relationship may be mostly linear or the search needs more trials.

### Common problems

| Problem | Fix |
|---------|-----|
| Target column not found | Check `target_col` capitalization |
| Data path matched no files | Use an absolute path; verify glob pattern |
| Too slow | Set `cv_folds: 0`, lower `n_trials`, disable plots |
| Memory issues | Disable permutation importance, lower `target_encoding_threshold` |
| Poor classification threshold | Try `threshold_policy.mode: f1` or `fbeta` |
| Regression target skew | Set `target_log_transform: true` if target is strictly positive |

### Dataset size tips

Large datasets:
```yaml
cv_folds: 0
search_subsample: 0.3
n_trials: 10
diagnostics:
  plots_enabled: false
```

Small datasets:
```yaml
test_size: 0.2
cv_folds: 5
n_trials: 20
n_estimators_max: 150
```

Time-series-like data:
```yaml
cv_strategy: timeseries
```

---

## 5. `config.yaml` — Complete Reference

### Task

| Key | Default | Description |
|-----|---------|-------------|
| `task` | `classification` | `classification` or `regression` |
| `target_col` | `Class` | Target column name — must match exactly |
| `test_size` | `0.2` | Fraction held out for final evaluation |
| `random_state` | `42` | Seed for splits and model behavior |

### Data loading

```yaml
data_path: creditcard.csv       # path, or glob like data/batch_*.csv
csv_chunk_size: null            # null = read normally; integer = stream in chunks
csv_chunk_log_every: 10         # log every N chunks when chunked loading is active
```

`csv_chunk_size` improves progress logging for large files but does not enable true out-of-core training — the full DataFrame is still materialized before training.

### Logging

```yaml
log_level: INFO     # DEBUG | INFO | WARNING | ERROR
log_file: null      # null = console only; path = console + file
```

### Cross-validation

```yaml
cv_folds: 5             # -1 = auto (CV if n_rows < 50k), 0 = fast-path subsample, N = force N folds
cv_strategy: stratified # stratified | timeseries
```

### Optuna search

```yaml
n_trials: 30
optuna_timeout: null            # null = run all trials; integer = max seconds
optuna_budget_seconds: null     # unified budget knob; drives timeout + n_trials
search_subsample: 0.6           # fraction of training rows per trial (fast-path only; hard-capped at 50k)
n_estimators_max: 500
n_estimators_min: 100
tune_n_estimators: true         # true = Optuna tunes n_estimators; false = early stopping sets it
early_stop_rnds: 20
wide_search: false              # true = broader hyperparameter ranges + more trials
```

### Metric selection

```yaml
metric: auto                # auto | roc_auc | auprc | macro_f1 | weighted_f1 | r2
imbalance_threshold: 0.15   # minority fraction below this → AUPRC preferred over ROC-AUC
```

| Value | When to use |
|-------|-------------|
| `auto` | Recommended — code picks based on task and class balance |
| `roc_auc` | Balanced binary classification |
| `auprc` | Highly imbalanced binary classification |
| `macro_f1` | Multiclass where all classes matter equally |
| `weighted_f1` | Multiclass with class imbalance |
| `r2` | Regression |

### Feature engineering

```yaml
cardinality_limit: 20               # integer columns with ≤ N unique values → categorical (OHE)
target_encoding_threshold: 50       # object columns with > N unique values → TargetEncoder
variance_threshold: 0.0             # drop numerical columns at/below this variance
interaction_top_k: 10               # top-K correlated numerical pairs → product features
feature_selection: true             # RFECV before Optuna; can be slow on wide datasets
robust_scaler_cols: ["Amount", "Time"]   # these columns use RobustScaler; all others use PowerTransformer
pretransform_log1p_cols: []         # log1p applied before PowerTransformer
pretransform_drop_cols: []          # dropped before any feature work
```

### PCA

```yaml
pca_threshold: 999      # enable PCA when numerical column count exceeds this value
pca_variance: 0.95      # target explained variance
pca_max_components: null
```

Set `pca_threshold: 1000` to effectively disable PCA on most datasets.

### Threshold policy (binary classification only)

```yaml
threshold_policy:
  mode: auto        # auto | f1 | fbeta | precision_at_recall | recall_at_precision | disabled
  beta: 1.0         # for fbeta mode
  min_precision: 0.80
  min_recall: 0.80
  n_quantiles: 200
```

Examples:

```yaml
# Balanced — maximize F1
threshold_policy:
  mode: f1

# Recall-sensitive (e.g. medical screening)
threshold_policy:
  mode: fbeta
  beta: 2.0

# Maintain precision above 90%
threshold_policy:
  mode: recall_at_precision
  min_precision: 0.90
```

### Baselines

```yaml
baselines:
  enabled: false
  include_dummy: true
  include_linear: true       # LogisticRegression for classification
  include_default_xgb: true
```

### Ensemble

```yaml
ensemble:
  enabled: false
  top_k: 3      # number of completed Optuna trials to soft-vote
```

When enabled, a soft-voting ensemble is fit from the top-K completed trials and saved alongside the best single model.

### Diagnostics

```yaml
diagnostics:
  plots_enabled: false    # master switch; false skips all plot generation
  optuna_plots: true
  learning_curve: true
  permutation_importance: true
  threshold_sweep: true
  outlier_report: true
  partial_dependence: true
  pca_plots: true
```

### Runtime

```yaml
use_gpu: false              # true attempts CUDA; falls back to CPU with a warning if unavailable
pandera_validation: false   # infer and run Pandera schema checks
calibration_enabled: true   # CalibratedClassifierCV on val set after final refit
callback_log_period: 50     # log XGBoost progress every N rounds
```

### Drift detection (train/test)

```yaml
drift_alpha: 0.05       # p-value threshold
drift_warn_only: true   # false = raise ValueError on detected drift
```

### Continuous drift monitor (production)

```yaml
drift_monitor:
  enabled: false
  persistence: 3                    # consecutive drifting checks before alerting
  min_feature_drift_ratio: 0.10
  retrain_feature_ratio: 0.25       # drifting fraction that triggers retrain recommendation
  retrain_severity: high            # none | low | medium | high
```

### MLflow

```yaml
mlflow_tracking_uri: null       # null = disabled; e.g. "http://localhost:5000"
mlflow_experiment: xgb_prototype
```

### Regression-specific

```yaml
target_log_transform: false     # log1p(y) before training; requires y > 0
```

### Recommended starting configs

Fast draft:
```yaml
task: classification
target_col: target
data_path: data/my_data.csv
cv_folds: 0
n_trials: 5
n_estimators_max: 100
metric: auto
diagnostics:
  plots_enabled: false
```

Full run:
```yaml
task: classification
target_col: target
data_path: data/my_data.csv
cv_folds: 5
n_trials: 50
n_estimators_max: 500
metric: auto
diagnostics:
  plots_enabled: true
```

Regression:
```yaml
task: regression
target_col: target_value
data_path: data/regression_data.csv
metric: auto
target_log_transform: false
cv_folds: 5
n_trials: 50
```

---

## 6. Training Process — Step by Step

### Pipeline flow

```text
load_data()
  └─ clean_data()
      └─ apply_pretransforms()
          └─ validate_data()           [Great Expectations scaffold]
              └─ validate_pandera()    [Pandera schema]
                  └─ detect_feature_types()
                      └─ detect_drift()      [KS / χ²]
                          └─ filter_low_variance()
                              └─ generate_feature_interactions()
                                  └─ [optional: select_features_rfecv()]
                                      └─ evaluate_baselines()
                                          └─ tune_hyperparameters()  [Optuna]
                                              └─ final fit + early stop + refit
                                                  └─ CalibratedClassifierCV
                                                      └─ tune_threshold()
                                                          └─ evaluate()
                                                              └─ analyse_errors()
                                                                  └─ plots
                                                                      └─ joblib.dump()
                                                                          └─ register_model()
                                                                              └─ mlflow.log_*()
```

Data flows in one direction. Nothing downstream re-touches upstream data — the key design principle that prevents leakage.

### Stage reference

| Log label | Stage | What happens |
|-----------|-------|--------------|
| Bootstrap | — | Run ID (`uuid4().hex[:8]`), timestamp, requirements lock written immediately |
| `[0/9]` | Config validation | Hard errors (missing target, bad task) caught before any data work |
| — | Load + clean | CSV/Parquet/JSON/JSONL/Excel/glob; dedup, sentinels, date parsing, cyclical encoding |
| — | Validation | Great Expectations scaffold (empty by default); Pandera schema inferred from training data |
| `[3/9]` | Split | Stratified train/val/test (68/12/20%); val handles early stopping, calibration, and threshold tuning only |
| `[3b/9]` | PCA decision | Enabled when `len(num_cols) > pca_threshold` |
| `[3c/9]` | Drift detection | KS test (numerical), chi-squared (categorical) — train vs test |
| `[4/9]` | Feature engineering | Low-variance filter, interaction terms, optional RFECV |
| `[baseline]` | Baselines | Dummy, logistic regression, default XGBoost scored before Optuna |
| `[5/9]` | Optuna tuning | TPE sampler, MedianPruner; CV or subsample mode |
| `[6/9]` | Final fit | Fit with early stopping → record `best_n` → refit at exactly `best_n` |
| `[6b/9]` | Threshold tuning | Quantile-candidate search on val set predictions |
| `[6c/9]` | Calibration | `CalibratedClassifierCV` on val set; handles sklearn ≥ 1.6 `FrozenEstimator` |
| `[7/9]` | Evaluation | Test set used only here — never seen during tuning or calibration |
| — | Error analysis | FP/FN CSV sorted by confidence (binary classification) |
| `[8/9]` | Plots | Feature importance, PR curve, ROC, confusion matrix, threshold sweep, PDP, etc. |
| `[9/9]` | Artifact + registry | `joblib.dump` full dict payload; `model_registry.json` updated; run JSON written |
| — | MLflow | Skipped silently when `mlflow_tracking_uri: null` |

### Key design decisions

**Three-way split.** The test set is held out completely for final evaluation. The val set handles early stopping, calibration, and threshold tuning. Using val for both calibration and reported metrics would give optimistically biased results.

**Refit at `best_n`.** Early stopping internally trains up to `N_ESTIMATORS_MAX` trees but only uses `best_n`. Refitting at exactly `best_n` eliminates unused trees, shrinks the artifact, and speeds up inference.

**Preprocessor reuse.** The preprocessor is fitted once on `train+val`. The same fitted instance is reused for the refit step — no redundant work and no possibility of leakage from refit data touching the transformer.

**Artifact before MLflow.** The joblib artifact is written to disk before any MLflow calls. MLflow failures never cause the artifact to be lost.

**Cyclical encoding for dates.** `sin(2π × x / period)` + `cos(2π × x / period)` encodes time as points on the unit circle so December is near January and hour 23 is near hour 0. Both sin and cos are required because sin alone is not uniquely invertible.

**AUPRC for imbalanced binary.** ROC-AUC is optimistic on highly imbalanced data — a classifier that never predicts positive can still score ~0.5. AUPRC measures the precision-recall tradeoff where the positive class is rare and matters.

---

## 7. Package Modules (`xgb_prototype/`)

### `config.py`

Typed dataclass representations of `config.yaml`. Loads YAML via OmegaConf while staying compatible with older flat configs. Missing nested sections receive defaults; unknown keys are ignored.

Main dataclasses: `TrainingConfig`, `ThresholdPolicyConfig`, `BaselineConfig`, `DiagnosticsConfig`, `EnsembleConfig`, `DriftMonitorConfig`.
Main functions: `load_config(path)`, `to_plain_dict(config)`.

```python
from xgb_prototype.config import load_config
cfg = load_config("config.yaml")
```

### `settings.py`

Loads `config.yaml` at import time via OmegaConf and exposes all runtime constants as module-level `ALL_CAPS` names. The `_c(key, default)` helper wraps `OmegaConf.select` so every read is fault-tolerant when OmegaConf is absent. Output directories are created at import time so they always exist before any write, even when the module is imported for inference only.

### `deps.py`

Checks hard and soft dependency versions at import time. Hard failures raise `ImportError` with an explicit `pip install` fix message. Soft failures log a warning. The check is skipped gracefully if `packaging` is not installed.

### `data.py`

Multi-format data loading (CSV/Parquet/JSON/JSONL/Excel/glob), chunked CSV reading, deduplication, sentinel replacement, date parsing with cyclical encoding, Great Expectations scaffold validation, Pandera schema inference and checking, train/test drift detection (KS + chi-squared), pretransform hooks, and optional log-transform for regression targets.

### `features.py`

Feature type inference (numerical / OHE-categorical / TargetEncoder-categorical), low-variance filtering via `VarianceThreshold`, pairwise interaction feature generation (top-K by Pearson |r|), and optional RFECV feature selection using a lightweight XGBoost (cv=3, 100 trees, depth=3).

### `pipeline.py`

Builds the sklearn `Pipeline`:

```text
"num" branch  → SimpleImputer(median) → [RobustScaler or PowerTransformer(yeo-johnson)] → [PCA]
"cat" branch  → SimpleImputer(mode)   → OneHotEncoder(handle_unknown="ignore")
"te"  branch  → TargetEncoder
                   └─ XGBClassifier / XGBRegressor
```

Also contains `tune_hyperparameters` (Optuna TPE sampler + MedianPruner, CV or subsample mode), `build_top_k_ensemble` (soft-voting from top Optuna trials), and `_IterationLogCallback` for periodic round logging.

### `metrics.py`

`MetricConfig` dataclass (name, direction, `needs_proba`, `scale_pos_weight`, `eval_metric`) and `select_metric()` for automatic selection. Binary: ROC-AUC when balanced, AUPRC when imbalanced. Multiclass: macro-F1 when balanced, weighted-F1 when skewed.

### `thresholds.py`

Generic binary threshold tuning. Builds quantile-spaced candidate thresholds from predicted probabilities and scores each under a named policy (F1, Fβ, precision-at-recall, recall-at-precision, or disabled).

```python
from xgb_prototype.thresholds import tune_binary_threshold

result = tune_binary_threshold(
    y_true, y_proba,
    policy={"mode": "fbeta", "beta": 2.0},
    metric_name="auprc",
)
print(result.threshold, result.metrics)
```

### `evaluation.py`

`evaluate()` — full test-set scoring (classification report, confusion matrix, AUPRC, ROC-AUC for classification; RMSE, MAE, R² for regression).
`tune_threshold()` — val-set threshold search using the configured policy.
`analyse_errors()` — FP/FN CSV with confidence and margin scores for binary classification.

### `baselines.py`

Fits cheap baselines (dummy, logistic regression, default XGBoost) before Optuna and saves a comparison CSV to `models/baseline_comparison_<run_id>.csv`. Baseline errors are caught and written as error rows rather than crashing the run.

### `plots.py`

All diagnostic plot functions. Produces interactive Plotly HTML files and static PNGs in `PLOT_OUTPUT_DIR`. Respects the `diagnostics.*` flags from config — unset flags skip specific plot families without failing the run.

### `tracking.py`

`_MLflowRun` — context manager wrapping an MLflow run; no-ops gracefully when MLflow is absent or the URI is unset.
`write_requirements_lock` — captures all installed package versions at startup to `requirements_<run_id>.txt`.
`register_model` — appends run metadata to `model_registry.json`.
`train_summary` — human-readable log table and machine-readable JSON run report.

### `drift_monitor.py`

`ContinuousDriftMonitor` for production-time drift monitoring. Compares incoming batches against training reference distributions (KS for numerical, chi-squared for categorical). Alerts only after `persistence` consecutive drifting checks to reduce false positives from single anomalous batches. Embedded in the `.joblib` artifact for deployment use.

### `inference.py`

`PredictWrapper` — thin local wrapper providing LabelEncoder decoding, unseen-category warnings (OHE silently outputs all-zeros for these), and log-untransform for regression targets.

`ModelServer` — production wrapper adding input validation (missing/extra column reporting), a standardised JSON response envelope, and descriptive error messages on failure. Designed to wrap in FastAPI, Flask, or similar.

---

## 8. Inference and Serving

### Local / notebook

```python
import joblib
from xgb_prototype.inference import PredictWrapper

artifact = joblib.load("models/model_<timestamp>_<run_id>.joblib")
model = PredictWrapper(artifact)

labels = model.predict(new_df)                           # decoded class labels
probas = model.predict_proba(new_df)                     # calibrated probabilities
preds  = model.predict_with_threshold(new_df)            # apply tuned threshold
custom = model.predict_with_threshold(new_df, threshold=0.3)
```

### Production API

```python
from xgb_prototype.inference import ModelServer

server = ModelServer(artifact)

response      = server.predict(new_df)        # returns dict
json_response = server.predict_json(new_df)   # returns JSON string
metadata      = server.info()                 # model version, features, eval metrics
```

Input validation: coerces `dict → DataFrame`, validates non-empty input, raises `ValueError` with an explicit list of missing columns, warns and ignores extra columns.

Response envelope:

```json
{
  "model_version": "20260503_123206_5983ae22",
  "task": "classification",
  "metric": "auprc",
  "threshold": 0.3142,
  "n_rows": 128,
  "predictions": [0, 1, 0],
  "probabilities": [[0.91, 0.09], "..."],
  "positive_proba": [0.09, "..."],
  "warnings": []
}
```

### Continuous drift monitoring (post-deployment)

The `ContinuousDriftMonitor` is embedded in the `.joblib` artifact and ready to use after loading:

```python
monitor = artifact["drift_monitor"]
result  = monitor.check(new_batch_df)

print(result.severity)                # "none" | "low" | "medium" | "high"
print(result.alert)                   # True after N consecutive drifting batches
print(result.retraining_recommended)
print(result.recommendation)         # human-readable action string
```

---

## 9. Artifacts and Outputs

### Model artifact

```text
models/model_<timestamp>_<run_id>.joblib
```

The artifact is a Python dict:

| Key | Type | Description |
|-----|------|-------------|
| `pipeline` | `sklearn.Pipeline` | Fitted preprocessor + model |
| `best_params` | `dict` | Best Optuna hyperparameters |
| `best_n_estimators` | `int` | Trees used after early stopping refit |
| `best_threshold` | `float` | Tuned decision threshold |
| `num_cols` | `list[str]` | Numerical column names |
| `ohe_cat_cols` | `list[str]` | OHE-encoded categorical columns |
| `te_cat_cols` | `list[str]` | Target-encoded categorical columns |
| `cat_cols` | `list[str]` | Union of ohe + te (backward compat) |
| `use_pca` | `bool` | Whether PCA was applied |
| `metric` | `MetricConfig` | Metric used for tuning |
| `label_encoder` | `LabelEncoder\|None` | For decoding integer labels |
| `known_categories` | `dict[str, set]` | Training-time category sets per column |
| `eval_metrics` | `dict` | Test-set evaluation results |
| `baseline_results` | `list[dict]` | Baseline model comparison rows |
| `ensemble_summary` | `dict` | Top-K ensemble results |
| `drift_report` | `dict` | Train/test drift statistics |
| `drift_monitor` | `ContinuousDriftMonitor\|None` | Embedded production drift monitor |
| `threshold_policy` | `dict` | Threshold policy config snapshot |
| `feature_schema` | `dict` | Column lists + `use_pca` flag |
| `artifact_paths` | `dict` | Paths to all saved files from this run |
| `config` | `dict` | Full config snapshot |
| `run_id` | `str` | 8-character hex run identifier |
| `timestamp` | `str` | `%Y%m%d_%H%M%S` |
| `log_transformed` | `bool` | Whether target was log1p-transformed |
| `cb_history` | `list[dict]` | Per-round train/val metrics from callback |

### Other output files

```text
models/run_<timestamp>_<run_id>.json              # Machine-readable run report
models/model_registry.json                        # Persistent registry of all runs
models/requirements_<run_id>.txt                  # Pinned environment snapshot
models/baseline_comparison_<run_id>.csv           # Baseline model scores
models/error_analysis_<run_id>.csv                # FP/FN analysis (binary classification only)
models/ensemble_<timestamp>_<run_id>.joblib       # Top-K ensemble (if enabled)
```

### Model registry

`model_registry.json` is a JSON array, newest run last. Each run is marked `production_ready` or `needs_review` based on conservative defaults:

- Classification: ROC-AUC ≥ 0.80 **and** AUPRC ≥ 0.50 → `production_ready`
- Regression: R² ≥ 0.70 → `production_ready`

These thresholds should be adjusted per use case.

### Error analysis CSV

For binary classification, misclassified test samples are saved with: `true_label`, `pred_label`, `error_type` (FP or FN), `confidence`, `margin` (|probability − threshold|), `raw_proba`, and all original feature values. Sorted by confidence descending — high-confidence mistakes appear first.

---

## 10. Diagnostic Plots

All plots are interactive Plotly HTML files in `plots/`, except `pca_2d.png` which is a static matplotlib PNG. HTML files embed Plotly via CDN to keep file sizes small.

| File | What it shows |
|------|---------------|
| `pr_curve.html` | Precision-recall curve with AUPRC and operating point |
| `roc_curve.html` | ROC curve with AUC |
| `confusion_matrix.html` | Heatmap of TP/FP/TN/FN |
| `residuals.html` | Predicted vs actual + residual distribution (regression) |
| `feature_importance.html` | XGBoost gain importance, top 20 features |
| `permutation_importance.html` | Model-agnostic importance on test set |
| `learning_curve.html` | Train vs val score as a function of training set size |
| `threshold_sweep.html` | Precision, recall, F1 across threshold values |
| `optuna_history.html` | Optuna optimization history |
| `optuna_importance.html` | Hyperparameter importance from Optuna |
| `pca_scree.html` | Explained variance per PCA component |
| `pca_2d.png` | 2D scatter of first two PCs, colored by label |
| `pca_3d.html` | Interactive 3D scatter of first three PCs (downsampled to 10k pts) |
| `corr_heatmap.html` | Feature correlation matrix |
| `calibration_curve.html` | Reliability diagram: calibrated vs uncalibrated probabilities |
| `outlier_report.html` | IsolationForest anomaly scores |
| `pdp_*.html` | Partial dependence + ICE plots for top-N features by gain |

---

## 11. Test Suite

### Running tests

```bash
# Full fast suite
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q

# Single file
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest tests/test_thresholds.py -q

# Opt-in smoke test (launches actual training on a tiny synthetic dataset)
RUN_XGB_PROTOTYPE_SMOKE=1 uv run python -m pytest tests/test_smoke_training.py -q
```

### What each file covers

**`test_config.py`** — config loading, backward compatibility with flat YAML, nested section parsing, `to_plain_dict()` serialization.

**`test_thresholds.py`** — `fbeta` choosing a recall-heavy threshold, `disabled` returning 0.5, `auto` normalizing to `f1` for AUPRC. Tests `normalize_policy` and `tune_binary_threshold`.

**`test_feature_inference.py`** — float columns → numerical, low-cardinality integers → OHE, object columns → categorical, imbalanced binary → AUPRC, `scale_pos_weight` computation. Tests `detect_feature_types` and `select_metric`.

**`test_serving.py`** — `ModelServer` with a fake pipeline and minimal artifact dict; validates response envelope shape and that missing required columns raise `ValueError`.

**`test_smoke_training.py`** — end-to-end subprocess run on a tiny synthetic dataset with minimal config. Confirms the entrypoint writes a `.joblib` artifact and a run JSON. Skipped by default.

**`test_data_loading_and_temporal.py`** — data loading and datetime feature generation.

**`test_drift_monitor.py`** — `ContinuousDriftMonitor` behavior, alert persistence, severity thresholds.

### Good testing patterns

Use small synthetic DataFrames:
```python
df = pd.DataFrame({
    "amount": [10.0, 20.0, 30.0],
    "merchant": ["a", "b", "a"],
    "target": [0, 1, 0],
})
```

Use `tmp_path` for config files and output directories. Use fake pipelines for serving tests so they stay fast. Only use the smoke test to verify the full training command.

Not yet deeply covered: full Optuna search quality, all plot generation functions, MLflow integration, Pandera failure modes, GPU fallback, regression end-to-end, multiclass end-to-end.

### When to add tests

Add or update tests when changing: config keys or defaults, metric selection logic, feature inference rules, threshold policy behavior, artifact structure, serving response shape, model registry format, entrypoint behavior, or preprocessing logic.

---

## 12. Architecture and Design Decisions

### Module bootstrapping

`parse_known_args()` instead of `parse_args()` so the `--config` parser silently ignores unrecognized arguments from pytest, Jupyter, or other callers. The `_c(key, default)` helper wraps `OmegaConf.select` to make every config read fault-tolerant when OmegaConf is absent. `MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` is called at import time so output directories always exist before any write, even when the module is imported for inference only.

### Data loading

`sorted(glob.glob(path))` ensures deterministic concatenation order (alphabetical) when multiple files match a glob. Chunked CSV reading streams batches into a list and concatenates at the end — avoids single-chunk peak RAM during the read, but still materializes the full DataFrame before training begins.

### Data cleaning

`dropna(how="all")` only drops rows where every value is NaN; partial missingness is kept because it may carry signal. Sentinel replacement covers `-999, -9999, "N/A", "n/a", "NA", "none", "None", ""` — common survey and legacy system conventions. Date detection looks for `"date"` or `"time"` in the column name and a >50% parse success rate to avoid false positives; raw date columns are replaced with year, month, day-of-week, hour, and cyclical sin/cos pairs.

### Feature type detection

All-NaN columns are dropped before building column lists. Integer columns with ≤ `cardinality_limit` unique values are routed to OHE (e.g. `num_rooms` with values 1–8 is better as binary features than assuming a linear relationship). Object columns above `target_encoding_threshold` cardinality use `TargetEncoder` — one-hot encoding thousands of unique values would create an unmanageable feature space.

### Pipeline construction

`PowerTransformer(yeo-johnson)` instead of `StandardScaler`: XGBoost doesn't require feature scaling, but Yeo-Johnson brings skewed features closer to Gaussian, which benefits PCA and interaction feature quality. `SimpleImputer(median)` is robust to outliers in skewed distributions. `OneHotEncoder(handle_unknown="ignore")` outputs all-zeros for unseen categories at inference rather than raising — essential for production. `nthread=1` on XGBoost prevents the model from spawning its own thread pool inside each Optuna trial; outer trial-level parallelism handles concurrency.

### Calibration

XGBoost's raw `predict_proba` is not well-calibrated — a score of 0.8 does not reliably mean 80% empirical probability. `cv="prefit"` (sklearn < 1.6) and `FrozenEstimator` (sklearn ≥ 1.6) both treat the XGBoost model as already fitted and train only the isotonic/sigmoid layer on top, using the val set the model has not seen during tree construction.

### Drift detection (train/test)

`detect_drift` compares train and test splits from the same dataset. On random splits this mostly measures sampling variance. The intended use is when train and test come from different time periods or sources — interpret cautiously on standard random splits. The `ContinuousDriftMonitor` handles production-time monitoring separately and independently.

### Interaction features

Pearson correlation is computed on training data only → upper triangle of unique pairs (since `corr(A,B) == corr(B,A)`) → sort by |r| descending → top-K with |r| ≥ 0.05 → add `A × B` product features. The 0.05 |r| floor is heuristic and avoids adding pure noise; it misses non-linear relationships (Spearman or mutual information would be more robust).

### GPU resolution

Three-probe strategy in order: `cupy` import → `torch.cuda.is_available()` → `nvidia-smi` subprocess. First success stops the chain. XGBoost 2.0 uses `device='cuda'` and deprecated `tree_method='gpu_hist'`; a version check ensures compatibility with both 1.x and 2.x. If `use_gpu: true` but no GPU is detected, training falls back to CPU with a warning rather than raising.

---

## 13. Known Limitations

**Chunked CSV: no true streaming.** `load_data` still materializes the full DataFrame in memory. For datasets larger than available RAM, a Dask-based approach would be required.

**Interaction features: multiplicative only.** No polynomial features, division-based interactions, or interactions involving categoricals. Pearson correlation misses non-linear relationships.

**RFECV and OHE expansion.** RFECV's support mask operates on the post-OHE feature space. A categorical column can only be eliminated if all its dummies are eliminated together, which rarely happens in practice.

**Calibration on small val sets.** If the val set is small (< 500 samples), isotonic calibration will have high variance and may worsen probability estimates. Consider `cv=5` instead of `cv="prefit"` for small datasets.

**Threshold tuning on val set.** If val is not representative of production (e.g. temporal shift), the tuned threshold may be suboptimal. For time-sensitive applications, val should be a chronologically later window.

**Ensemble uses the same `best_n` for all members.** Each Optuna trial's optimal tree count may differ; a more correct ensemble would re-determine `best_n` per member.

**PCA trigger is count-based.** A condition-number-based trigger would be more principled than a raw column count threshold.

**`scale_pos_weight` is fixed.** Computed once from the global class ratio; does not reflect asymmetric misclassification costs (e.g. false negatives being 10× more costly than false positives).

**Pandera schema inferred from first run.** If training data has errors, those errors are baked into the schema. Ideal: infer once from a known-good reference batch and version-control it.

**MLflow failures are silent after startup.** When `mlflow_tracking_uri` is set but tracking calls fail mid-run, they are silently skipped after a debug log. A startup connectivity check would surface problems earlier.

---

## 14. Development Notes

- `train.py` at the project root is the entrypoint; keep it as a thin launcher.
- New reusable behavior belongs under `xgb_prototype/`.
- Prefer config-driven behavior over hard-coded dataset assumptions. If a change only applies to one dataset, use a config option or validation rule — don't hard-code it.
- Keep artifact dict keys backward-compatible: add new keys rather than renaming existing ones.
- Add tests when changing config keys or defaults, metric selection logic, feature inference rules, threshold policy behavior, artifact structure, serving response shape, or preprocessing logic.
- Generated directories (`models/`, `plots/`, `.pytest_cache/`, `__pycache__/`) are not source — do not commit them.
- The package should stay general-purpose. Dataset-specific logic belongs in config or per-project validation rules.