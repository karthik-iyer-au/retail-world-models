"""The whole pipeline on the official M5 test period. Skipped without the data."""
import json

import pytest
import yaml

from rwm.experiments.run import run
from rwm.utils.paths import DATA_RAW, REPO_ROOT

needs_files = pytest.mark.skipif(
    not (DATA_RAW / "m5" / "sales_test_evaluation.csv").exists(), reason="M5 files not present"
)


@needs_files
def test_full_pipeline_seasonal_naive_scores_the_published_value(tmp_path):
    config = yaml.safe_load((REPO_ROOT / "configs/experiments/m5_test_seasonal_naive.yaml").read_text())
    metrics = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())
    assert metrics["wrmsse"] == pytest.approx(0.847, abs=5e-4)
