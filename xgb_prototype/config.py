"""Typed configuration helpers for the generalized XGBoost prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - train.py already treats OmegaConf as optional
    OmegaConf = None


@dataclass
class ThresholdPolicyConfig:
    mode: str = "auto"
    beta: float = 1.0
    min_precision: float = 0.80
    min_recall: float = 0.80
    n_quantiles: int = 200


@dataclass
class BaselineConfig:
    enabled: bool = True
    include_dummy: bool = True
    include_linear: bool = True
    include_default_xgb: bool = True


@dataclass
class DiagnosticsConfig:
    plots_enabled: bool = True
    optuna_plots: bool = True
    learning_curve: bool = True
    permutation_importance: bool = True
    threshold_sweep: bool = True
    outlier_report: bool = True
    partial_dependence: bool = True
    pca_plots: bool = True
    shap: bool = True
    calibration_curve: bool = True
    corr_heatmap: bool = True


@dataclass
class EnsembleConfig:
    enabled: bool = False
    top_k: int = 3


@dataclass
class AutoFeatureEngineeringConfig:
    enabled: bool = False
    engine: str = "featuretools"
    max_features: int = 25
    max_depth: int = 1
    entity_id_col: str | None = None
    time_index_col: str | None = None
    tsfresh_column_id: str | None = None
    tsfresh_column_sort: str | None = None


@dataclass
class SearchConfig:
    backend: str = "optuna"
    native_xgb_cv_rounds: int = 500
    native_xgb_cv_early_stop: int = 20


@dataclass
class SobolSensitivityConfig:
    enabled: bool = False
    n_base_samples: int = 16
    max_evals: int = 128


@dataclass
class UncertaintyConfig:
    enabled: bool = False
    alpha: float = 0.10
    quantile_alpha_low: float = 0.05
    quantile_alpha_high: float = 0.95


@dataclass
class DriftMonitorConfig:
    enabled: bool = True
    persistence: int = 3
    min_feature_drift_ratio: float = 0.10
    retrain_feature_ratio: float = 0.25
    retrain_severity: str = "high"


@dataclass
class TrainingConfig:
    task: str = "classification"
    target_col: str = "Class"
    test_size: float = 0.2
    random_state: int = 42
    data_path: str = "creditcard.csv"
    csv_chunk_size: int | None = None
    csv_chunk_log_every: int = 10
    model_output_dir: str = "models"
    plot_output_dir: str = "plots"
    log_level: str = "INFO"
    log_file: str | None = None
    cv_folds: int = -1
    cv_strategy: str = "stratified"
    n_trials: int = 30
    optuna_timeout: int | None = None
    optuna_budget_seconds: int | None = None  # single budget knob; drives timeout + n_trials
    search_subsample: float = 0.6
    n_estimators_max: int = 500
    n_estimators_min: int = 100
    tune_n_estimators: bool = True
    early_stop_rnds: int = 20
    robust_scaler_cols: list[str] = field(default_factory=lambda: ["Amount", "Time"])
    pca_threshold: int = 10
    pca_variance: float = 0.95
    pca_max_components: int | float | None = None
    imbalance_threshold: float = 0.15
    cardinality_limit: int = 20
    wide_search: bool = False
    drift_alpha: float = 0.05
    drift_warn_only: bool = True
    feature_selection: bool = False
    target_encoding_threshold: int = 50
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "xgb_prototype"
    outlier_contamination: float = 0.05
    pdp_top_n: int = 5
    target_log_transform: bool = False
    callback_log_period: int = 50
    variance_threshold: float = 0.0
    use_gpu: bool = False
    pandera_validation: bool = True
    interaction_top_k: int = 10
    calibration_enabled: bool = True
    metric: str = "auto"
    threshold_policy: ThresholdPolicyConfig = field(default_factory=ThresholdPolicyConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    auto_feature_engineering: AutoFeatureEngineeringConfig = field(default_factory=AutoFeatureEngineeringConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    sobol_sensitivity: SobolSensitivityConfig = field(default_factory=SobolSensitivityConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    drift_monitor: DriftMonitorConfig = field(default_factory=DriftMonitorConfig)
    pretransform_log1p_cols: list[str] = field(default_factory=list)
    pretransform_drop_cols:  list[str] = field(default_factory=list)


def _section(cls, data: dict[str, Any], key: str):
    """
    Extract a nested config section from a raw dictionary and instantiate it as
    a dataclass, handling three YAML edge cases:
      - Key absent from file   → defaults to {}.
      - Key explicitly null    → treated as {}.
      - Key is not a mapping   → raises TypeError.

    Unknown keys from the YAML are silently ignored, keeping older or richer
    config files compatible with a narrower dataclass definition.

    Args:
        cls:  A dataclass type whose __dataclass_fields__ defines accepted keys.
        data: The top-level config dictionary (already deserialised from YAML).
        key:  The string key whose value should be used to populate `cls`.

    Returns:
        An instance of `cls` initialised from the matching sub-dict.

    Raises:
        TypeError: If the value at `data[key]` is not a dict (and not None).
    """
    raw = data.get(key, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"config section '{key}' must be a mapping")
    return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def load_config(path: str | Path = "config.yaml") -> TrainingConfig:
    """
    Load a YAML config file into a nested TrainingConfig dataclass tree,
    tolerating older flat config files that pre-date the nested layout.

    Flow
    ----

    ::

        ┌─────────────────────────────────┐
        │  load_config(path)              │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ OmegaConf available?       │
        │                            │
        │  No ──► return             │
        │         TrainingConfig()   │
        │                            │
        │  Yes ──► continue          │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ cfg_path.exists()?         │
        │                            │
        │  No  ──► data = {}         │
        │                            │
        │  Yes ──► OmegaConf.load()  │
        │          + to_container()  │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Validate root is a dict    │
        │ (None → {}, non-dict →     │
        │  TypeError)                │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │ Build nested section dataclasses via       │
        │ _section() for each sub-config key         │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │ Collect remaining top-level scalar fields  │
        │ (TrainingConfig fields minus nested keys)  │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Merge scalars + sections   │
        │ ──► TrainingConfig(**kwargs)│
        └────────────────────────────┘

    Args:
        path: Path to the YAML config file. Defaults to ``"config.yaml"`` in
              the current working directory.

    Returns:
        A fully-populated :class:`TrainingConfig` instance. Any key absent
        from the file falls back to the dataclass field's default value.

    Raises:
        TypeError: If the YAML root (or any expected section) is not a mapping.
    """
    if OmegaConf is None:
        return TrainingConfig()

    cfg_path = Path(path)
    data = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) if cfg_path.exists() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping, got {type(data).__name__}")

    nested = {
        "threshold_policy":         _section(ThresholdPolicyConfig,        data, "threshold_policy"),
        "baselines":                _section(BaselineConfig,               data, "baselines"),
        "diagnostics":              _section(DiagnosticsConfig,            data, "diagnostics"),
        "ensemble":                 _section(EnsembleConfig,               data, "ensemble"),
        "auto_feature_engineering": _section(AutoFeatureEngineeringConfig, data, "auto_feature_engineering"),
        "search":                   _section(SearchConfig,                 data, "search"),
        "sobol_sensitivity":        _section(SobolSensitivityConfig,       data, "sobol_sensitivity"),
        "uncertainty":              _section(UncertaintyConfig,            data, "uncertainty"),
        "drift_monitor":            _section(DriftMonitorConfig,           data, "drift_monitor"),
    }
    top_level_fields = set(TrainingConfig.__dataclass_fields__) - set(nested)
    kwargs = {k: v for k, v in data.items() if k in top_level_fields}
    kwargs.update(nested)
    return TrainingConfig(**kwargs)


def to_plain_dict(config: TrainingConfig) -> dict[str, Any]:
    """Return a JSON-friendly dataclass snapshot."""
    return asdict(config)