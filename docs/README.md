# XGBoost Prototype — Complete Reference

> A generalized, reproducible XGBoost training prototype for tabular classification and regression. Configured through `config.yaml`. New datasets require only config changes, not code edits.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Layout](#2-project-layout)
3. [Quick Start](#3-quick-start)
4. [Using This Template With a New CSV](#4-using-this-template-with-a-new-csv)
5. [config.yaml — Complete Reference](#5-configyaml--complete-reference)
6. [Training Process — Step by Step](#6-training-process--step-by-step)
7. [Package Modules (`xgb_prototype/`)](#7-package-modules-xgb_prototype)
8. [Inference and Serving](#8-inference-and-serving)
9. [Artifacts and Outputs](#9-artifacts-and-outputs)
10. [Diagnostic Plots](#10-diagnostic-plots)
11. [Test Suite](#11-test-suite)
12. [Deep-Dive: Architecture and Design Decisions](#12-deep-dive-architecture-and-design-decisions)
13. [Known Limitations](#13-known-limitations)

---

## 1. Project Overview

This project is a generalized XGBoost training prototype. It is configured through `config.yaml`, supports classification and regression, runs Optuna hyperparameter tuning, compares simple baselines, writes versioned model artifacts, generates diagnostics, and exposes stable inference wrappers.

The current sample dataset is `creditcard.csv`, but the code is intentionally not fraud-specific. The same pipeline works across many tabular datasets by changing configuration values rather than editing code.

---

## 2. Project Layout

```text
.
├── config.yaml                  # Main runtime configuration
├── creditcard.csv               # Example tabular dataset
├── pyproject.toml               # Dependencies, dev extras, console script
├── train.py                     # Backward-compatible launcher
├── uv.lock                      # Locked dependency resolution
├── docs/                        # Documentation files
├── xgb_prototype/
│   ├── __init__.py              # Package exports
│   ├── baselines.py             # Dummy, linear, and default-XGBoost comparisons
│   ├── config.py                # Typed dataclass config loader
│   ├── data.py                  # Data loading and cleaning
│   ├── deps.py                  # Dependency version checks
│   ├── drift_monitor.py         # Continuous drift monitoring
│   ├── evaluation.py            # Test-set evaluation
│   ├── features.py              # Feature type inference and interactions
│   ├── inference.py             # PredictWrapper and ModelServer
│   ├── metrics.py               # MetricConfig and metric selection
│   ├── pipeline.py              # sklearn Pipeline construction
│   ├── plots.py                 # Diagnostic plot generation
│   ├── serving.py               # Stable serving imports
│   ├── settings.py              # Module-level config constants
│   ├── thresholds.py            # Binary threshold tuning policies
│   ├── tracking.py              # MLflow integration
│   └── train.py                 # Main training orchestration
└── tests/
    ├── test_config.py
    ├── test_data_loading_and_temporal.py
    ├── test_drift_monitor.py
    ├── test_feature_inference.py
    ├── test_serving.py
    ├── test_smoke_training.py
    └── test_thresholds.py
```

Generated outputs go to:

```text
models/     # Joblib artifacts, run JSON, registry, locks, CSVs
plots/      # Plotly HTML and PNG diagnostics
```

---

## 3. Quick Start

Install or sync the project environment:

```bash
uv sync --extra dev
```

Run training with the backward-compatible launcher:

```bash
python train.py
```

Run the fast tests:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q
```

Run the opt-in tiny training smoke test:

```bash
RUN_XGB_PROTOTYPE_SMOKE=1 uv run python -m pytest tests/test_smoke_training.py -q
```

---

## 4. Using This Template With a New CSV

### Step 1 — Prepare your CSV

Your CSV should have one row per training example, one target column, any number of feature columns, and a header row with column names.

```csv
age,income,plan_type,tenure_months,churned
34,72000,pro,18,0
52,105000,business,44,1
```

### Step 2 — Put the CSV somewhere predictable

Recommended layout:

```text
my_project/
├── data/
│   └── my_new_dataset.csv
├── config.yaml
└── xgb_prototype/
```

### Step 3 — Edit the three required config values in `config.yaml`

```yaml
task: classification        # or regression
target_col: your_target
data_path: data/my_new_dataset.csv
```

### Step 4 — Run a fast draft first

Confirm the dataset loads and the pipeline trains before tuning:

```yaml
cv_folds: 0
n_trials: 5
n_estimators_max: 100
diagnostics:
  plots_enabled: false
```

```bash
python train.py
```

### Step 5 — Move to a better run

```yaml
cv_folds: 5
n_trials: 30
n_estimators_max: 300
diagnostics:
  plots_enabled: true
```

### Task type identification

Classification targets: binary flags (`0/1`), yes/no, multi-class labels like `low/medium/high`.

Regression targets: continuous numeric values like `price`, `revenue`, `delivery_time_minutes`.

### After training, check

```text
models/baseline_comparison_<run_id>.csv   # Is tuned XGBoost beating simpler models?
models/error_analysis_<run_id>.csv        # What high-confidence mistakes did it make?
plots/                                    # Feature importance, PR curve, threshold sweep, etc.
```

If the tuned model barely beats the dummy baseline: check data leakage, target quality, feature usefulness, or whether the task is learnable from available columns. If logistic regression beats XGBoost: the relationship may be mostly linear, or the search needs more trials.

### Common problems

| Problem | Fix |
|---------|-----|
| Target column not found | Check `target_col` capitalization |
| Data path matched no files | Use an absolute path; verify glob pattern |
| Too slow | Set `cv_folds: 0`, lower `n_trials`, disable diagnostics |
| Memory issues | Disable PCA plots and permutation importance, lower `target_encoding_threshold` |
| Bad classification threshold | Try `threshold_policy.mode: f1` or `fbeta` |
| Regression target skew | Try `target_log_transform: true` if target is positive |

### Large dataset tips

```yaml
cv_folds: 0
search_subsample: 0.3
n_trials: 10
diagnostics:
  plots_enabled: false
```

### Small dataset tips

```yaml
test_size: 0.2
cv_folds: 5
n_trials: 20
n_estimators_max: 150
```

### Time-series-like data

```yaml
cv_strategy: timeseries
```

---

## 5. `config.yaml` — Complete Reference

`config.yaml` controls the training run. For most new datasets, you only need to change this file.



### Task settings

| Key | Description |
|-----|-------------|
| `task` | `classification` or `regression` |
| `target_col` | Target column name, must match exactly |
| `test_size` | Fraction held out for final evaluation (default `0.2`) |
| `random_state` | Seed for reproducible splits and model behavior |

### Path settings

```yaml
model_output_dir: models
plot_output_dir: plots
data_path: creditcard.csv
csv_chunk_size: null          # null = read normally; integer = stream in chunks
csv_chunk_log_every: 10
```

Supported `data_path` formats: `.csv`, `.parquet`, `.json`, `.jsonl`, `.xlsx`, `.xls`, glob patterns like `data/monthly_*.csv`.

`csv_chunk_size` improves CSV ingestion progress logging for large files but does not enable true out-of-core training — the dataframe is still materialized before training begins.

### Logging settings

```yaml
log_level: INFO             # DEBUG | INFO | WARNING | ERROR
log_file: null              # null = console only; path = console + file
```
`log_level`
- Use `INFO` for normal training.
- Use `DEBUG` when diagnosing a problem.

`log_file`
- `null` logs to the console only.
- A path such as `train.log` writes logs to a file too.

### Cross-validation settings

```yaml
cv_folds: 5                 # 0 = faster validation-split path
cv_strategy: stratified     # stratified | timeseries
```
`cv_folds`: Number of folds used during Optuna tuning. Use `cv_folds: 0` for a quick first run on a large dataset. Use `cv_folds: 5` for a reliable search.

### Optuna search settings

```yaml
n_trials: 30        # number of hyperparameter trials.

optuna_timeout: null        # null = run until n_trials; integer = max seconds

search_subsample: 0.6       # used when cv_folds: 0; fraction of training data sampled for faster tuning.

n_estimators_max: 300       # maximum boosting rounds; early stopping can stop before this number.

early_stop_rnds: 20       # stop after this many rounds without validation improvement.

wide_search: false          # true = expanded search ranges
```

Fast draft run:
```yaml
n_trials: 5
cv_folds: 0
n_estimators_max: 100
```

More serious run:
```yaml
n_trials: 50
cv_folds: 5
n_estimators_max: 500
```

### PCA settings

```yaml
pca_threshold: 10           # enable PCA when numerical column count exceeds this
pca_variance: 0.95          # target explained variance
pca_max_components: null    # optional hard limit for number of PCA components
```

To effectively disable PCA unless there are many numerical features:
```yaml
pca_threshold: 1000
```

### Imbalance and metric settings

```yaml
imbalance_threshold: 0.15   # minority class ratio below this → use AUPRC
metric: auto
```

Supported `metric` values:

| Value | Use when |
|-------|----------|
| `auto` | Default — code chooses sensible metric |
| `roc_auc` | Balanced binary classification |
| `auprc` | Very imbalanced binary classification |
| `macro_f1` | Multiclass where all classes matter equally |
| `weighted_f1` | Multiclass with class imbalance |
| `r2` | Regression |

### Feature inference settings

```yaml
cardinality_limit: 20               # integer columns with ≤ this many unique values → categorical

target_encoding_threshold: 50       # object/category columns above this cardinality → TargetEncoder
```

Datetime columns detected during cleaning generate cyclical features: month, day-of-week, and hour sine/cosine pairs.

### Drift settings

```yaml
drift_alpha: 0.05
drift_warn_only: true       # false = stop run on detected drift
```

### Feature selection and feature generation

```yaml
feature_selection: false    # enables RFECV — can be slow
variance_threshold: 0.0     # 0.0 drops only perfectly constant columns
interaction_top_k: 10       # number of correlated numerical pairs → product features
```

To make training faster:
```yaml
feature_selection: false
interaction_top_k: 0
```

### MLflow settings

```yaml
mlflow_tracking_uri: null               # null = disabled
mlflow_experiment: xgb_prototype
```

### Diagnostics settings

```yaml
diagnostics:
  plots_enabled: true
  optuna_plots: true
  learning_curve: true
  permutation_importance: true
  threshold_sweep: true
  outlier_report: true
  partial_dependence: true
  pca_plots: true
```

`plots_enabled` is a master switch. Individual families can also be toggled.

### Threshold policy settings

Applies to binary classification only.

```yaml
threshold_policy:
  mode: auto                # auto | disabled | f1 | fbeta | precision_at_recall | recall_at_precision
  beta: 1.0
  min_precision: 0.80
  min_recall: 0.80
  n_quantiles: 200
```

Examples:

```yaml
# Balanced classification
threshold_policy:
  mode: f1

# Recall-sensitive (e.g. medical screening)
threshold_policy:
  mode: fbeta
  beta: 2.0

# Keep precision above 90%
threshold_policy:
  mode: recall_at_precision
  min_precision: 0.90
```

### Baseline settings

```yaml
baselines:
  enabled: true
  include_dummy: true
  include_linear: true
  include_default_xgb: true
```

### Top-K ensemble settings

```yaml
ensemble:
  enabled: false
  top_k: 3
```

When enabled, fits a soft-voting ensemble from the top completed Optuna trials and saves it alongside the best single model.

### Runtime settings

```yaml
use_gpu: false              # true attempts CUDA; falls back to CPU if unavailable
pandera_validation: true
callback_log_period: 50     # log XGBoost iteration progress every N rounds
calibration_enabled: true
```

### Continuous drift monitor settings

```yaml
drift_monitor:
  enabled: true
  persistence: 3                    # consecutive drifting checks before alerting
  min_feature_drift_ratio: 0.10     # minimum drifting feature fraction for a check to count
  retrain_feature_ratio: 0.25       # drifting feature fraction triggering retrain recommendation
  retrain_severity: high
```

### Regression-specific setting

```yaml
target_log_transform: false   # true when task=regression, target strictly positive and heavily skewed
```

### Recommended starting configs

Fast classification draft:
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

Full classification run:
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

Regression run:
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

### How to start training

Run command:
```bash
python train.py
```

Both run the same `main()` function in `xgb_prototype/train.py`. The root `train.py` is a thin launcher for backward compatibility.

### High-level pipeline

```text
load_data()
  └─ clean_data()
      └─ validate_data()           [Great Expectations]
          └─ validate_pandera()    [Pandera]
              └─ detect_feature_types()
                  └─ detect_drift()  [KS / χ²]
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

Data flows in one direction. Nothing downstream re-touches upstream data — this is the key design principle that prevents leakage.

### Stage reference

| Stage | What happens |
|-------|--------------|
| Bootstrap | Run ID (`uuid4().hex[:8]`), timestamp, requirements lock written immediately |
| Config load | OmegaConf + typed dataclasses from `config.py` |
| Dependency check | Hard requirements raise `ImportError`; soft requirements warn |
| Data load | CSV/Parquet/JSON/JSONL/Excel/glob, optional chunked reading |
| Config validation | Hard errors (missing target, bad task) before any data work begins |
| Data cleaning | Dedup, sentinel values, string stripping, date parsing, cyclical encoding |
| Great Expectations | Scaffold validation — empty by default, add rules per project |
| Pandera validation | Schema inferred from training data; range checks warn, null target hard-fails |
| Feature type detection | num / OHE-categorical / TargetEncoder-categorical split |
| PCA decision | Enabled when `len(num_cols) > pca_threshold` |
| Metric selection | Auto or explicit; `scale_pos_weight` computed for imbalanced binary |
| Train/val/test split | Stratified for classification; 15% of non-test → val set |
| Drift detection | KS test (numerical), chi-squared (categorical) |
| Low-variance filtering | Drops constant/near-constant numerical columns |
| Interaction features | Product of top-K correlated numerical pairs, train correlations only |
| Optional RFECV | Lightweight XGBoost; expensive, off by default |
| Baseline comparison | Dummy, logistic regression, default XGBoost |
| Optuna tuning | TPE sampler, MedianPruner; CV or subsample mode |
| Final fit | Fit with early stopping → record `best_n` → refit at exactly `best_n` |
| Calibration | `CalibratedClassifierCV` on val set; handles sklearn 1.6 `FrozenEstimator` |
| Threshold tuning | Quantile-candidate search on val set predictions |
| Held-out evaluation | Test set used only here — never seen during tuning or calibration |
| Error analysis | FP/FN CSV sorted by confidence, for binary classification |
| Plot generation | Feature importance, PR curve, ROC, confusion matrix, threshold sweep, etc. |
| Artifact writing | `joblib.dump` with full dict payload |
| Registry update | Appended to `models/model_registry.json`; marked `production_ready` or `needs_review` |
| MLflow logging | Skipped when `mlflow_tracking_uri: null` |

### Key design details

**Three-way split:** The test set is held out completely and used only for final evaluation. The val set handles early stopping, calibration, and threshold tuning. If val influenced the threshold and was also used to report metrics, results would be optimistically biased.

**Final fit refit:** With early stopping, XGBoost internally trains up to `N_ESTIMATORS_MAX` trees but only uses the best `best_n`. A refit at exactly `best_n` eliminates unused trees, shrinking the artifact and speeding up inference.

**Preprocessor reuse:** The preprocessor is fitted on `train+val` before early stopping. The same fitted preprocessor is reused for the refit — no re-fitting happens, avoiding redundant computation and any possibility of val-set leakage into the preprocessor.

**`nthread=1` on XGBoost:** Prevents XGBoost from spawning its own thread pool inside each Optuna trial. Without this, trial-level and tree-level parallelism fight for threads and cause slowdowns.

**Cyclical encoding for dates:** `sin(2π × x / period)` + `cos(2π × x / period)` encodes time components as points on the unit circle. December is then near January, hour 23 near hour 0. Both sin and cos are needed because sin alone is not uniquely invertible.

**AUPRC for imbalanced binary:** ROC-AUC is optimistic for highly imbalanced data — a classifier predicting "never positive" can still score ~0.5. AUPRC measures the precision-recall tradeoff, which is more meaningful when the positive class is rare.

**Artifact written before MLflow:** The joblib artifact is saved first. MLflow failures never cause the artifact to be lost.

---

## 7. Package Modules (`xgb_prototype/`)

### `__init__.py`

Re-exports the main typed config helpers:

```python
from xgb_prototype import load_config
cfg = load_config("config.yaml")
```

### `config.py`

Typed dataclass representations of `config.yaml`. Loads YAML while remaining compatible with older flat configs. Converts config dataclasses to plain dicts for JSON artifacts.

Main dataclasses: `ThresholdPolicyConfig`, `BaselineConfig`, `DiagnosticsConfig`, `TrainingConfig`.
Main functions: `load_config(path)`, `to_plain_dict(config)`.

Missing nested sections receive defaults. Unknown keys are ignored rather than breaking the run.

### `thresholds.py`

Generic binary classification threshold tuning. Keeps threshold logic configurable instead of hard-coding domain-specific behavior.

Main dataclass: `ThresholdResult`.
Main functions: `normalize_policy(policy, metric_name)`, `tune_binary_threshold(y_true, y_proba, policy, metric_name)`.

Algorithm: build candidate thresholds from prediction-probability quantiles (`n_quantiles=200`) → score each candidate → select best per policy.

```python
from xgb_prototype.thresholds import tune_binary_threshold

result = tune_binary_threshold(
    y_true, y_proba,
    {"mode": "fbeta", "beta": 2.0},
    metric_name="auprc",
)
print(result.threshold)
```

### `baselines.py`

Trains and evaluates cheap baseline models before the tuned XGBoost. Produces a comparison table.

Main function: `evaluate_baselines(...)`.

Classification baselines: `dummy_most_frequent`, `logistic_regression`, `xgb_default`.
Regression baselines: `dummy_mean`, `xgb_default`.

Output: `models/baseline_comparison_<run_id>.csv`. Baseline errors are captured into result rows rather than crashing the run.

### `drift_monitor.py`

Reusable continuous drift monitor comparing future data batches against training reference distributions. Raises alerts only when drift persists across multiple checks.

Main classes: `ContinuousDriftMonitor`, `DriftCheckResult`.

Uses KS tests for numerical features and chi-squared for categorical. The monitor is embedded in the model artifact and can be used post-deployment:

```python
artifact = joblib.load("models/model_<timestamp>_<run_id>.joblib")
monitor = artifact["drift_monitor"]
result = monitor.check(new_batch_df)

print(result.alert)
print(result.retraining_recommended)
print(result.recommendation)
```

### `serving.py`

Stable public inference imports. Provides a fixed import path independent of internal refactoring.

```python
from xgb_prototype.serving import PredictWrapper, ModelServer
```

### `train.py`

Main training orchestration. Contains `main()`, pipeline construction, Optuna tuning, evaluation, artifact writing, registry updates, and the `PredictWrapper` / `ModelServer` classes.

Edit this file when changing the core training sequence, preprocessing, model construction, Optuna tuning, final artifact contents, or diagnostics.

---

## 8. Inference and Serving

### `PredictWrapper`

The raw `pipeline.predict()` returns integer-encoded labels when a `LabelEncoder` was used. `PredictWrapper` inverts this transparently and warns when unseen categories are encountered at inference time (OHE will encode them as all-zeros).

```python
import joblib
from xgb_prototype.serving import PredictWrapper

artifact = joblib.load("models/model_<timestamp>_<run_id>.joblib")
model = PredictWrapper(artifact)

labels = model.predict(new_df)
probas = model.predict_proba(new_df)
preds  = model.predict_with_threshold(new_df)          # uses tuned threshold
custom = model.predict_with_threshold(new_df, threshold=0.3)
```

### `ModelServer`

`PredictWrapper` plus input validation and a JSON response envelope. Designed for wrapping in FastAPI, Flask, etc.

```python
import joblib
from xgb_prototype.serving import ModelServer

artifact = joblib.load("models/model_<timestamp>_<run_id>.joblib")
server = ModelServer(artifact)

response      = server.predict(new_df)
json_response = server.predict_json(new_df)
metadata      = server.info()
```

Input validation: coerces `dict` → `DataFrame`, checks for empty input, reports exactly which columns are missing (raises `ValueError`) or extra (warns and ignores). Missing required columns are always a bug; extra columns (audit fields, metadata) are common and harmless.

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
| `known_categories` | `dict[str, set]` | Training-time category sets |
| `eval_metrics` | `dict` | Test-set evaluation results |
| `baseline_results` | `list[dict]` | Baseline model comparison |
| `ensemble_summary` | `dict` | Top-K ensemble results |
| `drift_report` | `dict` | Train/test drift statistics |
| `drift_monitor` | `ContinuousDriftMonitor\|None` | Production drift monitor |
| `threshold_policy` | `dict` | Threshold policy config |
| `feature_schema` | `dict` | Column lists + use_pca flag |
| `artifact_paths` | `dict` | Paths to all saved files |
| `config` | `dict` | Full config snapshot |
| `run_id` | `str` | 8-character hex run identifier |
| `timestamp` | `str` | `%Y%m%d_%H%M%S` |
| `log_transformed` | `bool` | Whether target was log1p-transformed |
| `cb_history` | `list[dict]` | Per-round train/val metrics |

### Other output files

```text
models/run_<timestamp>_<run_id>.json              # Run summary
models/model_registry.json                        # Persistent registry of all runs
models/requirements_<run_id>.txt                  # Installed package versions at run time
models/baseline_comparison_<run_id>.csv           # Baseline model scores
models/error_analysis_<run_id>.csv                # FP/FN analysis (binary classification only)
```

### Model registry

`model_registry.json` is a JSON array, newest run last. Each run is marked `production_ready` or `needs_review` based on conservative thresholds:
- Classification: ROC-AUC ≥ 0.80 AND AUPRC ≥ 0.50 → `production_ready`
- Regression: R² ≥ 0.70 → `production_ready`

These thresholds are defaults and should be tuned per use case.

### Error analysis CSV

For binary classification, misclassified test samples are saved with columns: `true_label`, `pred_label`, `error_type` (FP or FN), `confidence`, `margin` (|probability − threshold|), `raw_proba`, and all original feature values. Sorted by confidence descending — the most actionable errors (high-confidence mistakes) appear first.

---

## 10. Diagnostic Plots

All plots are interactive Plotly HTML files in `plots/`, except `pca_2d.png` which is a static matplotlib PNG. Each HTML file embeds Plotly via CDN to keep file sizes small.

| Plot file | What it shows |
|-----------|---------------|
| `pr_curve.html` | Precision-recall curve with AUPRC and operating point |
| `roc_curve.html` | ROC curve with AUC |
| `confusion_matrix.html` | Heatmap of TP/FP/TN/FN |
| `residuals.html` | Predicted vs actual + residual distribution (regression) |
| `feature_importance.html` | XGBoost gain importance, top 20 features |
| `permutation_importance.html` | Model-agnostic permutation importance on test set |
| `learning_curve.html` | Train vs val score as a function of training set size |
| `threshold_sweep.html` | Precision, recall, F1 as a function of threshold |
| `optuna_history.html` | Optuna optimization history |
| `optuna_importance.html` | Hyperparameter importance from Optuna |
| `pca_scree.html` | Explained variance per PCA component |
| `pca_2d.png` | 2D scatter of first two PCs, colored by label |
| `pca_3d.html` | Interactive 3D scatter of first three PCs (downsampled to 10,000 pts) |
| `corr_heatmap.html` | Feature correlation matrix |
| `shap_summary.html` | SHAP beeswarm plot |
| `shap_interactions.html` | SHAP interaction values |
| `calibration_curve.html` | Reliability diagram (calibrated vs uncalibrated) |
| `outlier_report.html` | IsolationForest anomaly scores |
| `pdp_*.html` | Partial dependence + ICE plots |

---

## 11. Test Suite

### How to run

Fast test suite (runs on every save):
```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q
```

Single file:
```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest tests/test_thresholds.py -q
```

Opt-in training smoke test:
```bash
RUN_XGB_PROTOTYPE_SMOKE=1 uv run python -m pytest tests/test_smoke_training.py -q
```

### What each file covers

**`test_config.py`** — config loading, backward compatibility with flat YAML files, nested section parsing, `to_plain_dict()` serialization.

**`test_thresholds.py`** — `fbeta` policy choosing a recall-heavy threshold, `disabled` returning 0.5, `auto` normalizing to `f1` for AUPRC. Tests `normalize_policy` and `tune_binary_threshold`.

**`test_feature_inference.py`** — float columns → numerical, low-cardinality integers → OHE categorical, object columns → categorical, imbalanced binary → AUPRC, `scale_pos_weight` computation. Tests `detect_feature_types` and `select_metric`.

**`test_serving.py`** — `ModelServer` with a fake pipeline and minimal artifact dict; validates response envelope shape and that missing required columns raise `ValueError`.

**`test_smoke_training.py`** — end-to-end subprocess run on a tiny synthetic dataset with minimal config. Confirms the package entrypoint writes a `.joblib` artifact and a run JSON. Skipped by default because it launches an actual training run.

**`test_data_loading_and_temporal.py`** — data loading and date/time feature generation.

**`test_drift_monitor.py`** — `ContinuousDriftMonitor` behavior.

### Good testing patterns for this project

Use small synthetic DataFrames for unit tests:
```python
df = pd.DataFrame({
    "amount": [10.0, 20.0, 30.0],
    "merchant": ["a", "b", "a"],
    "target": [0, 1, 0],
})
```

Use `tmp_path` for config files and output folders. Use fake pipelines for serving tests so they stay fast. Only use the smoke test when verifying the whole training command still works.

### What is not yet deeply covered

Full Optuna search quality, all plot generation functions, MLflow integration, Great Expectations rules, Pandera failure modes, GPU fallback behavior, RFECV behavior, target encoding on high-cardinality columns, regression end-to-end training, multiclass end-to-end training.

### When to add tests

Add or update tests when changing: config keys or defaults, metric selection behavior, feature inference rules, threshold policy behavior, artifact structure, serving response shape, model registry format, train entrypoint behavior, preprocessing logic.

---

## 12. Deep-Dive: Architecture and Design Decisions

This section explains the *why* behind non-obvious implementation choices.

### Module bootstrapping

`parse_known_args()` is used instead of `parse_args()` so the `--config` parser silently ignores unrecognized arguments from pytest, Jupyter, or other tools that call the script as a module.

The `_c(key, default)` helper wraps `OmegaConf.select` to make every config read fault-tolerant. If OmegaConf is absent, it returns the Python literal default without raising.

Module-level constants are read once at import time. They use all-caps naming to signal they are not modified after initialization.

`CSV_CHUNK_SIZE = None if _csv_chunk_size_cfg in (None, "null", 0) else int(...)` — YAML `null`, Python `None`, and integer `0` all mean "don't chunk," and OmegaConf may return the string `"null"` for unset keys.

`MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` is called at module load time so output directories always exist before any function tries to write to them, even when the module is imported for inference only.

### Data loading (`load_data`)

`sorted(glob.glob(path))` ensures deterministic concatenation order (alphabetical by filename) when multiple files match a glob. Without sorting, the order would depend on filesystem metadata.

Chunked CSV reading streams batches into a list and concatenates at the end. This avoids single-chunk peak RAM during the read but still materializes the full DataFrame before training.

### Data cleaning (`clean_data`)

`dropna(how="all")` only drops rows where every value is NaN — rows with partial missingness are kept because they may carry useful signal (missingness itself can be informative). `drop_duplicates()` runs after because all-NaN rows cannot contain useful duplicates.

Sentinel replacement covers `-999, -9999, "N/A", "n/a", "NA", "none", "None", ""` — common survey and legacy system conventions. Domain-specific sentinels (e.g. `99`, `9999`) need to be added per project.

Date auto-detection looks for `"date"` or `"time"` in the column name (case-insensitive) and a `> 0.5` parse success rate to avoid false positives. Once parsed, the raw column is dropped and replaced with year, month, day-of-week, hour, and cyclical sin/cos pairs for month, day-of-week, and hour.

### Feature type detection (`detect_feature_types`)

All-NaN columns are dropped before building the column lists because they provide zero information and cause sklearn transformers to produce NaN-contaminated outputs.

Integer columns with ≤ `cardinality_limit` unique values are treated as categorical (OHE). An integer like `num_rooms` (values 1–8) is better as OHE binary features than assuming a linear 1→8 relationship. Object columns above `target_encoding_threshold` cardinality use `TargetEncoder` — one-hot encoding 10,000 unique values would create 10,000 features.

### Pipeline construction (`build_pipeline`)

```
"num" branch  → SimpleImputer(median) → PowerTransformer(yeo-johnson) [→ PCA if use_pca]
"cat" branch  → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
"te"  branch  → TargetEncoder
```

`PowerTransformer` instead of `StandardScaler`: XGBoost doesn't require feature scaling, but Yeo-Johnson brings skewed features closer to Gaussian, which helps PCA and improves the quality of interaction features.

`SimpleImputer(strategy="median")`: robust to outliers in skewed financial data. Mean imputation would be pulled toward extremes.

`OneHotEncoder(handle_unknown="ignore")`: outputs all-zeros for unseen categories at inference rather than raising. Essential for production. `sparse_output=False` returns a dense array, required by XGBoost.

`nthread=1`: prevents XGBoost from spawning its own thread pool inside each Optuna trial. The outer `study.optimize(n_jobs=-1)` handles trial-level parallelism; XGBoost's thread pool would cause oversubscription.

`n_jobs=-1` on `ColumnTransformer`: parallelises the num/cat/te branches across CPU cores, the dominant source of speedup during Optuna trials.

### Optuna tuning (`tune_hyperparameters`)

Two modes: `cv_folds > 0` uses cross-validation (slower, lower variance); `cv_folds = 0` uses a stratified subsample of the training data (faster, noisier). The subsample mode is sufficient for ranking parameter combinations — it does not need to measure exact performance.

TPE sampler with `n_startup_trials=10`: the first 10 trials use random search to seed the TPE model, preventing it from getting trapped in a bad early configuration.

MedianPruner with `n_warmup_steps=10`: terminates trials performing significantly below the median. The 10-step warmup lets each trial produce a meaningful early score before pruning decisions.

`n_jobs=-1` for subsample mode, `n_jobs=1` for CV mode: parallel trials with inner CV fold parallelism causes thread contention.

### Final fit / refit (`main` steps 6–6b)

1. Train on `train+val` with early stopping, using val as the eval set. Record `best_n`.
2. Build a fresh pipeline with `n_estimators=best_n` and `early_stop=0`.
3. Reuse the already-fitted preprocessor (no redundant fitting, no leakage).
4. Refit on `train+val` at exactly `best_n` — eliminates unused trees, shrinks the artifact, speeds up inference.

### Calibration (`CalibratedClassifierCV`)

XGBoost's raw `predict_proba` is not well-calibrated — a score of 0.8 does not reliably mean 80% probability. Calibration adjusts probabilities to better reflect empirical frequencies.

`cv="prefit"` (sklearn < 1.6) / `FrozenEstimator` (sklearn ≥ 1.6): in both cases the XGBoost model is treated as already fitted and only the sigmoid/isotonic layer is fitted on top. Calibration is fit on the val set, which the model has not seen during tree construction.

### Threshold tuning (`tune_threshold`)

The search evaluates `n_quantiles` candidates spaced as quantiles of the predicted probability distribution. This is faster than a dense grid over [0, 1] and focuses candidates where the actual probability mass is.

The default F1 at threshold 0.5 is logged alongside the tuned threshold so the practitioner can see exactly how much threshold tuning helped.

### Drift detection

`detect_drift` compares train vs test splits from the *same* dataset. For datasets randomly split, this measures natural sampling variance and will rarely flag drift. The intended use is when train and test come from different time periods or data sources. Interpret results cautiously on standard random splits.

The `ContinuousDriftMonitor` (separate from `detect_drift`) handles production-time drift monitoring on incoming batches.

### Interaction features (`generate_feature_interactions`)

Compute Pearson correlation on training data only → upper triangle of unique pairs → sort by |r| descending → select top-K with |r| ≥ 0.05 → add `A × B` features.

Only the upper triangle is used because `corr(A,B) == corr(B,A)`. Products are computed on original (potentially NaN) columns so missingness is inherited naturally. The 0.05 |r| floor avoids adding pure noise interactions, though this threshold is heuristic.

### GPU resolution (`_resolve_tree_method`)

Three-probe strategy: `cupy` → `torch.cuda.is_available()` → `nvidia-smi` subprocess. Tries in order; first success stops the chain. Robust across different CUDA stack configurations.

XGBoost 2.0 introduced `device='cuda'` and deprecated `tree_method='gpu_hist'`. The version check ensures compatibility with both 1.x and 2.x.

Soft fallback: if `use_gpu: true` but no GPU is detected, training falls back to CPU with a warning rather than raising.

---

## 13. Known Limitations

**Single-file monolith (partially addressed):** `train.py` was previously ~3,800 lines. The codebase has been refactored into sub-modules (`data.py`, `features.py`, `metrics.py`, `pipeline.py`, `plots.py`, `evaluation.py`, `inference.py`, `tracking.py`, etc.), improving testability and discoverability.

**Chunked CSV: no true streaming.** `load_data` concatenates all chunks into memory. For datasets larger than available RAM, chunked reading still runs out of memory. True out-of-core training would require Dask or a fundamentally different approach.

**Interaction features: multiplicative pairwise only.** No polynomial features, division-based features, or interactions involving categoricals. Pearson correlation is used to select pairs, which misses non-linear relationships (Spearman or mutual information would be more robust).

**RFECV and OHE expansion.** RFECV's support mask operates on the post-OHE feature space. In practice, RFECV can only eliminate an original categorical column if *all* of its one-hot dummies are eliminated together, which is unlikely.

**Calibration on a small val set.** If the val set is small (< 500 samples), calibration will have high variance and may worsen probability estimates. For small datasets, consider `cv=5` for calibration instead of `cv="prefit"`.

**Threshold tuning on val set.** If the val set is not representative of production data (e.g. temporal shift), the tuned threshold may be suboptimal. For time-sensitive applications, the val set should be a chronologically later window.

**Ensemble uses same `best_n` for all members.** Each Optuna trial's optimal tree count may differ; a more correct ensemble would re-determine `best_n` per member.

**PCA trigger is count-based.** A dataset with 11 numerical features (above the default threshold of 10) uses PCA even if features are uncorrelated. A condition-number-based trigger would be more principled.

**Great Expectations is a scaffold.** No expectations are configured by default — the GE integration always passes without user customization.

**`scale_pos_weight` is fixed.** Computed once from the global training set ratio. Does not reflect asymmetric misclassification costs (e.g. false negatives being 10× more costly than false positives). Cost-sensitive weighting requires domain knowledge.

**Pandera schema inferred from first run.** If the training data has errors (e.g. corrupted batch), those errors are baked into the schema. Ideal setup: infer once from a known-good reference batch and version it.

**MLflow has no startup warning.** When `MLFLOW_URI` is set but MLflow initialization fails, tracking is silently skipped. A warning on startup would be safer.

---

## Development Notes

- Keep `train.py` at the project root as a compatibility wrapper.
- Put new reusable behavior under `xgb_prototype/`.
- Prefer config-driven behavior over hard-coded dataset assumptions.
- Keep artifact keys backward-compatible when possible.
- Add tests for new config keys, metrics, threshold policies, and serving behavior.
- Generated directories (`models/`, `plots/`, `.pytest_cache/`, `__pycache__/`) are not source code.
- The package should stay general-purpose. If a change only applies to one dataset, prefer a config option or validation rule over hard-coding it.