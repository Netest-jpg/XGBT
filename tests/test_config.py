from xgb_prototype.config import load_config, to_plain_dict


def test_load_config_accepts_old_flat_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "task: regression",
            "target_col: y",
            "n_trials: 3",
            "threshold_policy:",
            "  mode: fbeta",
            "  beta: 2.0",
            "baselines:",
            "  enabled: false",
            "search:",
            "  backend: native_xgb_cv",
            "sobol_sensitivity:",
            "  enabled: true",
            "uncertainty:",
            "  alpha: 0.2",
            "auto_feature_engineering:",
            "  enabled: true",
            "  engine: tsfresh",
        ])
    )

    cfg = load_config(cfg_path)

    assert cfg.task == "regression"
    assert cfg.target_col == "y"
    assert cfg.n_trials == 3
    assert cfg.threshold_policy.mode == "fbeta"
    assert cfg.threshold_policy.beta == 2.0
    assert cfg.baselines.enabled is False
    assert cfg.search.backend == "native_xgb_cv"
    assert cfg.sobol_sensitivity.enabled is True
    assert cfg.uncertainty.alpha == 0.2
    assert cfg.auto_feature_engineering.engine == "tsfresh"
    assert to_plain_dict(cfg)["threshold_policy"]["mode"] == "fbeta"
