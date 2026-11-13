"""Configs for the 52-week test (docs/PAPER1_REVISION_PLAN.md, 13 November 2026).

  python scripts/make_long_horizon_configs.py

From each final config (configs/experiments/final_<key>_{world_model,lightgbm}.yaml) writes
configs/experiments/h52_<key>_<model>_<setup>_w<window>.yaml, changing only the distance ahead:

  blocks   the model fitted to forecast 13 weeks (91 days on M5), forecasting 52 weeks (364 days) in four blocks
           through rwm.blocks
  direct   the model fitted to forecast all 52 weeks (364 days) at once

Each window is its own config so that each runs in one job: w1 ends where the final runs' data end, w0 starts 26 weeks
(182 days) earlier. The held-out final periods stay out. A world model's `validation_periods` is raised to the
horizon where it is shorter, as the model requires. M5's direct world model is not written (see the plan).
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["dominicks_ana", "dominicks_bjc", "dominicks_che", "dominicks_fre", "dominicks_sdr", "dominicks_sna", "m5"]


def inner_params(model: dict) -> dict:
    return model["params"]["params"] if model["name"] == "seed_average" else model["params"]


def make(base: dict, key: str, which: str, setup: str, window: int) -> dict:
    daily = key == "m5"
    long, block, gap = (364, 91, 182) if daily else (52, 13, 26)
    h = block if setup in ("blocks", "rollon") else long
    cfg = copy.deepcopy(base)
    p = inner_params(cfg["model"])
    p["horizon"] = h
    if p.get("validation_periods") and p["validation_periods"] < h:
        p["validation_periods"] = h
    if setup == "blocks":
        cfg["model"] = {"name": "blocks", "params": {"model": cfg["model"]["name"], "block": block,
                                                     "params": cfg["model"]["params"]}}
    elif setup == "byblock":  # added 14 November 2026: one LightGBM per block of 13 weeks (rwm.by_block)
        cfg["model"] = {"name": "by_block", "params": {"model": cfg["model"]["name"], "block": block, "blocks": 4,
                                                       "params": cfg["model"]["params"]}}
    elif setup == "rollon":  # added 14 November 2026: fitted for 13 weeks, the state rolled on for 52 (rwm.roll_on)
        cfg["model"] = {"name": "roll_on", "params": {"model": cfg["model"]["name"], "params": cfg["model"]["params"]}}
    ev = cfg["evaluation"]
    ev["horizon"], ev["n_origins"] = long, 1
    ev.pop("step", None)
    ev["holdout_periods"] = ev.get("holdout_periods", 0) + (gap if window == 0 else 0)
    cfg["name"] = f"h52_{key}_{which}_{setup}_w{window}"
    return cfg


OTHERS = {  # added 14 November 2026: the other models of paper 1's model table, each forecasting all 52 weeks its own
    # way, with the settings of paper 1's rolling runs (configs/experiments/rev_<key>_rolling_<model>.yaml)
    "seasonal_naive": lambda key: {"name": "seasonal_naive", "params": {"season": 364 if key == "m5" else 52}},
    "chronos": lambda key: _model_of(f"rev_{key}_rolling_chronos"),
    "timesfm_mean": lambda key: _model_of(f"rev_{key}_rolling_timesfm_mean"),
    "timesfm_plan": lambda key: _model_of(f"rev_{key}_rolling_timesfm_plan"),
    "timestral_plan": lambda key: _model_of(f"rev_{key}_rolling_timestral_plan"),
    "tft": lambda key: None if key == "m5" else _model_of(f"rev_{key}_rolling_tft"),
}


def _model_of(name: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/experiments/{name}.yaml").read_text())["model"]


def others() -> int:
    """Each model as paper 1 ran it, on the final runs' data and windows, forecasting 52 weeks (364 days) at once.
    The Temporal Fusion Transformer was not run on M5 in paper 1 and is not here."""
    n = 0
    for key in KEYS:
        base = yaml.safe_load((ROOT / f"configs/experiments/final_{key}_lightgbm.yaml").read_text())
        long, gap = (364, 182) if key == "m5" else (52, 26)
        for which, get in OTHERS.items():
            model = get(key)
            if model is None:
                continue
            model = copy.deepcopy(model)
            if "horizon" in model["params"]:
                model["params"]["horizon"] = long
            for window in (0, 1):
                cfg = copy.deepcopy(base)
                cfg["model"] = model
                ev = cfg["evaluation"]
                ev["horizon"], ev["n_origins"] = long, 1
                ev.pop("step", None)
                ev["holdout_periods"] = ev.get("holdout_periods", 0) + (gap if window == 0 else 0)
                cfg["name"] = f"h52_{key}_{which}_direct_w{window}"
                head = ("# 52-week test, other models (plan entry of 14 November 2026): the model as paper 1 ran it, "
                        "forecasting all 52 weeks at once; written by scripts/make_long_horizon_configs.py.\n")
                (ROOT / f"configs/experiments/{cfg['name']}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
                n += 1
    return n


def main():
    n = 0
    for key in KEYS:
        for which in ("world_model", "lightgbm"):
            src = ROOT / f"configs/experiments/final_{key}_{which}.yaml"
            base = yaml.safe_load(src.read_text())
            for setup in ("blocks", "direct", "rollon", "byblock"):
                if setup == "rollon" and which != "world_model":
                    continue
                if setup == "byblock" and which != "lightgbm":
                    continue
                if key == "m5" and which == "world_model" and setup == "direct":
                    continue
                for window in (0, 1):
                    cfg = make(base, key, which, setup, window)
                    head = (f"# 52-week test (plan entry of 13 November 2026): {src.name} with only the distance ahead "
                            f"changed ({setup}, window {window}); written by scripts/make_long_horizon_configs.py.\n")
                    (ROOT / f"configs/experiments/{cfg['name']}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
                    n += 1
    n += others()
    print(n, "configs")


if __name__ == "__main__":
    main()
