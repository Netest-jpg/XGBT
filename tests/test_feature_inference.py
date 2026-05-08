import pandas as pd

from xgb_prototype import detect_feature_types, select_metric


def test_detect_feature_types_routes_numeric_low_cardinality_and_high_cardinality_text():
    df = pd.DataFrame({
        "continuous": [0.1, 0.2, 0.3, 0.4],
        "low_card_int": [1, 1, 2, 2],
        "text": ["a", "b", "c", "d"],
    })

    num_cols, ohe_cols, te_cols = detect_feature_types(df)

    assert "continuous" in num_cols
    assert "low_card_int" in ohe_cols
    assert "text" in ohe_cols
    assert te_cols == []


def test_select_metric_uses_auprc_for_imbalanced_binary_target():
    y = pd.Series([0] * 100 + [1] * 2)

    metric = select_metric(y, "classification")

    assert metric.name == "auprc"
    assert metric.needs_proba is True
    assert metric.scale_pos_weight == 50

