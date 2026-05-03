from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xgb_prototype import ModelServer


class FakePipeline:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile(np.array([[0.25, 0.75]]), (len(X), 1))


def _artifact():
    return {
        "pipeline": FakePipeline(),
        "num_cols": ["amount"],
        "ohe_cat_cols": ["merchant"],
        "te_cat_cols": [],
        "metric": SimpleNamespace(name="auprc"),
        "label_encoder": None,
        "best_threshold": 0.5,
        "run_id": "abc123",
        "timestamp": "20260101_000000",
        "task": "classification",
        "known_categories": {"merchant": {"a"}},
        "eval_metrics": {"auprc": 0.8},
    }


def test_model_server_predicts_with_standard_envelope():
    server = ModelServer(_artifact())

    response = server.predict(pd.DataFrame({"amount": [10.0], "merchant": ["a"]}))

    assert response["model_version"] == "20260101_000000_abc123"
    assert response["predictions"] == [1]
    assert response["positive_proba"] == [0.75]


def test_model_server_rejects_missing_columns():
    server = ModelServer(_artifact())

    with pytest.raises(ValueError, match="Missing required column"):
        server.predict(pd.DataFrame({"amount": [10.0]}))

