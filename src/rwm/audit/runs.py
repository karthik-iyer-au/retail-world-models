"""Rebuild the training and test frames of a recorded run, and load its saved
models, so that the audit reads exactly what each run fitted and forecast.

`windows` follows `rwm.experiments.run.run` step by step (optional columns,
held-out final periods, rolling windows); `tests/test_audit_runs.py` checks that
a model refitted on these frames gives the run's recorded forecasts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data import load_dataset
from rwm.data.calendar import add_calendar, add_season
from rwm.data.crowding import add_crowding
from rwm.data.roles import add_roles
from rwm.data.schema import DATE
from rwm.evaluation.splits import rolling_origins
from rwm.forecaster import _MODELS


def windows(config: dict):
    """Yields (origin, train, test) for every window of a run config."""
    ds = load_dataset(config["dataset"]["name"], **config["dataset"].get("params", {}))
    ev, panel, extras = config["evaluation"], ds.panel, config["dataset"]
    if "calendar" in extras:
        add_calendar(panel, **extras["calendar"])
    if extras.get("season"):
        add_season(panel, **(extras["season"] if isinstance(extras["season"], dict) else {}))
    if "crowding" in extras:
        add_crowding(panel, **extras["crowding"])
    if "roles" in extras:
        every = np.sort(panel[DATE].unique())
        back = ev.get("holdout_periods", 0) + ev["horizon"] + (ev["n_origins"] - 1) * (ev.get("step") or ev["horizon"])
        add_roles(panel, until=every[-back - 1], **(extras["roles"] or {}))
    dates = np.sort(panel[DATE].unique())
    if ev.get("holdout_periods"):
        dates = dates[: -ev["holdout_periods"]]
        panel = panel[panel[DATE] <= dates[-1]].reset_index(drop=True)
    for sp in rolling_origins(dates, ev["horizon"], ev["n_origins"], ev.get("step")):
        test = panel[panel[DATE].isin(sp.test_dates)].reset_index(drop=True)
        train = panel[panel[DATE] <= sp.train_end]
        yield sp.origin, train, test


def model_params(config: dict) -> dict:
    """The state model's own settings (inside seed_average when averaged)."""
    p = config["model"].get("params", {})
    return p.get("params", p) if config["model"]["name"] == "seed_average" else p


def load_checkpoint(run_dir: Path, origin: int, device: str | None = None):
    config = json.loads((Path(run_dir) / "manifest.json").read_text())["config"]
    cls = _MODELS[config["model"]["name"]]
    return cls.load(Path(run_dir) / f"checkpoint_origin{origin}.pt", device=device)


def plan_levers(config: dict) -> list[str]:
    """Plan inputs switched off for 'nothing planned': the model's levers, or
    every extra column except cost (cost is not part of the plan)."""
    p = model_params(config)
    return list(p.get("levers") or [c for c in p.get("extra", []) if c != "cost"])
