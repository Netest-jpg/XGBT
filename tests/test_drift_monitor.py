import pandas as pd

from xgb_prototype.drift_monitor import ContinuousDriftMonitor


def test_drift_monitor_alerts_after_persistent_drift():
    ref = pd.DataFrame({
        "amount": [1.0, 1.1, 0.9, 1.2, 1.0, 0.95],
        "segment": ["a", "a", "a", "b", "b", "b"],
    })
    new = pd.DataFrame({
        "amount": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "segment": ["z", "z", "z", "z", "z", "z"],
    })

    monitor = ContinuousDriftMonitor.from_reference_data(
        ref,
        num_cols=["amount"],
        cat_cols=["segment"],
        alpha=0.05,
        persistence=2,
        min_feature_drift_ratio=0.10,
        retrain_feature_ratio=0.50,
    )

    first = monitor.check(new)
    second = monitor.check(new)

    assert first.alert is False
    assert second.alert is True
    assert second.retraining_recommended is True
    assert "amount" in second.drifted_features

