import numpy as np

from xgb_prototype.thresholds import normalize_policy, tune_binary_threshold


def test_fbeta_policy_prefers_recall_heavy_threshold():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.05, 0.20, 0.45, 0.40, 0.90])

    result = tune_binary_threshold(
        y_true,
        y_proba,
        {"mode": "fbeta", "beta": 2.0, "n_quantiles": 20},
        metric_name="auprc",
    )

    assert 0.0 <= result.threshold <= 1.0
    assert result.policy["mode"] == "fbeta"
    assert result.metrics["recall"] == 1.0


def test_disabled_policy_returns_default_threshold():
    result = tune_binary_threshold(
        np.array([0, 1]),
        np.array([0.49, 0.51]),
        {"mode": "disabled"},
        metric_name="roc_auc",
    )

    assert result.threshold == 0.5


def test_auto_policy_normalizes_for_binary_metric():
    assert normalize_policy({"mode": "auto"}, metric_name="auprc")["mode"] == "f1"

