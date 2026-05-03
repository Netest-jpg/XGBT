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
        ])
    )

    cfg = load_config(cfg_path)

    assert cfg.task == "regression"
    assert cfg.target_col == "y"
    assert cfg.n_trials == 3
    assert cfg.threshold_policy.mode == "fbeta"
    assert cfg.threshold_policy.beta == 2.0
    assert cfg.baselines.enabled is False
    assert to_plain_dict(cfg)["threshold_policy"]["mode"] == "fbeta"

