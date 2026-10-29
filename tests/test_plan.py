"""The job plan shared by the job runner and the cloud launcher."""
from pathlib import Path

import pytest
import yaml

from rwm.experiments import plan
from rwm.utils.paths import REPO_ROOT

JOBS = sorted((REPO_ROOT / "configs" / "jobs").glob("*.yaml"))


@pytest.mark.parametrize("job_file", JOBS, ids=lambda p: p.stem)
def test_every_job_expands(job_file: Path):
    job = yaml.safe_load(job_file.read_text())
    runs = plan.expand(job)
    assert len(runs) == sum(len(r.get("seeds", [None])) for r in job["runs"])
    for r in runs:
        assert plan.machine(r["config"]) in {"cpu", "gpu"}
        assert plan.memory_gb(r["config"]) >= 16


def test_seeds_rename_runs_in_order():
    job = {"runs": [{"config": "configs/experiments/frat_dev_lightgbm.yaml", "seeds": [1, 2]}]}
    names = [r["config"]["name"] for r in plan.expand(job)]
    assert names == ["frat_dev_lightgbm_seed1", "frat_dev_lightgbm_seed2"]


def _model(name, **params):
    return {"model": {"name": name, "params": params}, "dataset": {"name": "dominicks"}}


def test_processor_models_run_without_a_gpu():
    assert plan.machine(_model("lightgbm_direct")) == "cpu"
    assert plan.machine(_model("seed_average", model="lightgbm_direct", params={})) == "cpu"
    assert plan.machine(_model("seasonal_naive")) == "cpu"


def test_deep_models_and_wrappers_holding_one_get_a_gpu():
    assert plan.machine(_model("state_model")) == "gpu"
    assert plan.machine(_model("chronos")) == "gpu"
    assert plan.machine(_model("seed_average", model="state_model", params={})) == "gpu"
    both = _model("combination", models=[{"name": "state_model", "params": {}},
                                         {"name": "lightgbm_direct", "params": {}}])
    assert plan.machine(both) == "gpu"


def test_large_data_sets_get_more_memory():
    assert plan.memory_gb({"dataset": {"name": "m5"}}) == 32
    assert plan.memory_gb({"dataset": {"name": "dominicks"}}) == 16
