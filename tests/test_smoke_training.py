import os
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_XGB_PROTOTYPE_SMOKE") != "1",
    reason="opt-in smoke test; set RUN_XGB_PROTOTYPE_SMOKE=1 to run training",
)
def test_tiny_training_smoke(tmp_path):
    data_path = tmp_path / "tiny.csv"
    pd.DataFrame({
        "x1": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
        "x2": ["a", "a", "b", "b", "c", "c", "a", "b", "c", "a", "b", "c"],
        "target": [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    }).to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
task: classification
target_col: target
data_path: {data_path}
model_output_dir: {tmp_path / "models"}
plot_output_dir: {tmp_path / "plots"}
test_size: 0.25
cv_folds: 0
n_trials: 1
n_estimators_max: 8
early_stop_rnds: 2
pandera_validation: false
calibration_enabled: false
baselines:
  enabled: true
diagnostics:
  plots_enabled: false
"""
    )

    result = subprocess.run(
        [sys.executable, "-m", "xgb_prototype.train", "--config", str(cfg_path)],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert list((tmp_path / "models").glob("model_*.joblib"))
    assert list((tmp_path / "models").glob("run_*.json"))

