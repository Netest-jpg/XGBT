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


@dataclass
class EnsembleConfig:
    enabled: bool = False
    top_k: int = 3


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
    n_estimators_max: int = 300
    early_stop_rnds: int = 20
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
    drift_monitor: DriftMonitorConfig = field(default_factory=DriftMonitorConfig)
    pretransform_log1p_cols: list[str] = field(default_factory=list)
    pretransform_drop_cols:  list[str] = field(default_factory=list)


def _section(cls, data: dict[str, Any], key: str):
    raw = data.get(key, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"config section '{key}' must be a mapping")
    return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def load_config(path: str | Path = "config.yaml") -> TrainingConfig:
    """Load a YAML config into dataclasses, tolerating older flat config files."""
    if OmegaConf is None:
        return TrainingConfig()
    cfg_path = Path(path)
    data = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) if cfg_path.exists() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping, got {type(data).__name__}")

    nested = {
        "threshold_policy": _section(ThresholdPolicyConfig, data, "threshold_policy"),
        "baselines": _section(BaselineConfig, data, "baselines"),
        "diagnostics": _section(DiagnosticsConfig, data, "diagnostics"),
        "ensemble": _section(EnsembleConfig, data, "ensemble"),
        "drift_monitor": _section(DriftMonitorConfig, data, "drift_monitor"),
    }
    top_level_fields = set(TrainingConfig.__dataclass_fields__) - set(nested)
    kwargs = {k: v for k, v in data.items() if k in top_level_fields}
    kwargs.update(nested)
    return TrainingConfig(**kwargs)


def to_plain_dict(config: TrainingConfig) -> dict[str, Any]:
    """Return a JSON-friendly dataclass snapshot."""
    return asdict(config)
