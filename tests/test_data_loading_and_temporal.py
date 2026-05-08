import numpy as np
import pandas as pd

import xgb_prototype.settings as settings
from xgb_prototype.data import clean_data, load_data
import xgb_prototype.data as data_module

def test_chunked_csv_loading_reads_all_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(5), "target": [0, 1, 0, 1, 0]}).to_csv(csv_path, index=False)

    monkeypatch.setattr(data_module, "DATA_PATH", str(csv_path))
    monkeypatch.setattr(data_module, "CSV_CHUNK_SIZE", 2)
    monkeypatch.setattr(data_module, "CSV_CHUNK_LOG_EVERY", 1)

    df = load_data()

    assert df.shape == (5, 2)
    assert df["x"].tolist() == [0, 1, 2, 3, 4]


def test_clean_data_adds_cyclical_datetime_features(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    df = pd.DataFrame({
        "event_time": ["2026-01-05 00:00:00", "2026-07-05 12:00:00"],
        "target": [0, 1],
    })

    cleaned = clean_data(df)

    assert "event_time" not in cleaned.columns
    for suffix in [
        "month_sin", "month_cos",
        "dayofweek_sin", "dayofweek_cos",
        "hour_sin", "hour_cos",
    ]:
        assert f"event_time_{suffix}" in cleaned.columns
    assert np.isclose(cleaned.loc[0, "event_time_hour_sin"], 0.0)
    assert np.isclose(cleaned.loc[1, "event_time_hour_cos"], -1.0)