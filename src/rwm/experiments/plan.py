"""What a job file asks for, worked out without running anything.

Shared by the job runner (`rwm.experiments.job`) and the cloud launcher
(`cloud/modal_app.py`), so both read a job file the same way.

  expand(job)        every run a job asks for: config file, seed and full config
  machine(config)    "cpu" for models that never use a GPU, "gpu" otherwise
  memory_gb(config)  memory a run of this config is given on a cloud machine
"""
from pathlib import Path

import yaml

from rwm.experiments.run import with_seed
from rwm.utils.paths import REPO_ROOT

# Models that run on the processor only. Every other model, and any wrapper
# (seed_average, combination) holding one, is given a GPU.
CPU_MODELS = {"lightgbm_direct", "seasonal_naive", "recent_average"}

# Memory per dataset on a cloud machine, in GB. M5 and Favorita hold tens of
# millions of rows; the others fit easily in less. A world-model run on M5 peaked
# at 11.1 GB (measured 28 October 2026 with cloud/modal_app.py's peak_memory);
# Favorita is not yet measured.
MEMORY_GB = {"m5": 32, "favorita": 64}
DEFAULT_MEMORY_GB = 16


def expand(job: dict) -> list[dict]:
    """One entry per run, in the job file's order: {"config_file", "seed", "config"}."""
    runs = []
    for r in job["runs"]:
        base = yaml.safe_load((REPO_ROOT / r["config"]).read_text())
        for seed in r.get("seeds", [None]):
            runs.append({"config_file": r["config"], "seed": seed, "config": with_seed(base, seed)})
    return runs


def model_names(model: dict) -> set[str]:
    """The model's name and the names of any models it wraps: a combination
    lists them as {name, params}, a seed average names one under `model`."""
    names = {model["name"]}

    def walk(value):
        if isinstance(value, dict):
            for key, v in value.items():
                if key in ("name", "model") and isinstance(v, str):
                    names.add(v)
                else:
                    walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(model.get("params", {}))
    return names


def machine(config: dict) -> str:
    return "cpu" if model_names(config["model"]) <= CPU_MODELS else "gpu"


def memory_gb(config: dict) -> int:
    return MEMORY_GB.get(config["dataset"]["name"], DEFAULT_MEMORY_GB)


def load(job_file: str | Path) -> dict:
    return yaml.safe_load((REPO_ROOT / job_file).read_text())
