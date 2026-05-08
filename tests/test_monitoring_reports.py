import json

import numpy as np
import pandas as pd

from xgb_prototype.data import automatic_missing_value_report
from xgb_prototype.drift_monitor import ContinuousDriftMonitor
from xgb_prototype.uncertainty import estimate_uncertainty


def test_missing_value_report_writes_structured_outputs(tmp_path):
    df = pd.DataFrame({
        "amount": [1.0, None, 3.0],
        "merchant": ["a", "b", None],
        "target": [0, 1, 0],
    })

    report = automatic_missing_value_report(df, "target", tmp_path, "abc123")

    assert report.total_missing == 2
    assert report.rows_with_missing == 2
    assert {row["column"] for row in report.columns} == {"amount", "merchant"}
    assert report.output_csv is not None
    assert report.output_json is not None
    assert json.loads((tmp_path / "missing_values_abc123.json").read_text())["total_missing"] == 2


def test_drift_monitor_reports_label_prediction_quality_and_skew_channels():
    ref = pd.DataFrame({
        "amount": [1.0, 1.1, 0.9, 1.2, 1.0, 0.95],
        "segment": ["a", "a", "a", "b", "b", "b"],
    })
    new = pd.DataFrame({
        "amount": [10.0, 11.0, np.nan, 13.0, 14.0, 15.0],
        "segment": ["a", "a", "a", "a", "a", "a"],
        "extra": [1, 1, 1, 1, 1, 1],
    })

    monitor = ContinuousDriftMonitor.from_reference_data(
        ref,
        num_cols=["amount"],
        cat_cols=["segment"],
        y_ref=pd.Series([0, 0, 0, 1, 1, 1]),
        prediction_proba_ref=np.array([0.1, 0.2, 0.25, 0.8, 0.85, 0.9]),
        alpha=0.20,
        persistence=1,
    )

    result = monitor.check(
        new,
        y_new=pd.Series([1, 1, 1, 1, 1, 2]),
        prediction_proba=np.array([0.8, 0.9, 0.95, 0.8, 0.9, 0.95]),
    )

    assert result.alert is True
    assert result.serving_training_skew["skew_detected"] is True
    assert result.novel_class_emergence["novel_classes"] == ["2"]
    assert result.prediction_drift["checked"] is True
    assert result.data_quality_drift["checked"] is True


class _FakeClassifier:
    def predict_proba(self, X):
        base = np.linspace(0.2, 0.8, len(X))
        return np.column_stack([1 - base, base])


def test_uncertainty_classification_report(tmp_path):
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])

    report = estimate_uncertainty(
        _FakeClassifier(),
        X,
        y,
        X,
        y,
        X,
        y,
        task="classification",
        output_dir=tmp_path,
        run_id="clf",
        enabled=True,
    )

    assert report.status == "completed"
    assert report.summary["method"] == "split_conformal_classification"
    assert report.output_csv is not None
