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
    elif setup == "rollon":  # added 14 November 2026: fitted for 13 weeks, the state rolled on for 52 (rwm.roll_on)
        cfg["model"] = {"name": "roll_on", "params": {"model": cfg["model"]["name"], "params": cfg["model"]["params"]}}
    ev = cfg["evaluation"]
    ev["horizon"], ev["n_origins"] = long, 1
    ev.pop("step", None)
    ev["holdout_periods"] = ev.get("holdout_periods", 0) + (gap if window == 0 else 0)
    cfg["name"] = f"h52_{key}_{which}_{setup}_w{window}"
    return cfg


def main():
    n = 0
    for key in KEYS:
        for which in ("world_model", "lightgbm"):
            src = ROOT / f"configs/experiments/final_{key}_{which}.yaml"
            base = yaml.safe_load(src.read_text())
            for setup in ("blocks", "direct", "rollon"):
                if setup == "rollon" and which != "world_model":
                    continue
                if key == "m5" and which == "world_model" and setup == "direct":
                    continue
                for window in (0, 1):
                    cfg = make(base, key, which, setup, window)
                    head = (f"# 52-week test (plan entry of 13 November 2026): {src.name} with only the distance ahead "
                            f"changed ({setup}, window {window}); written by scripts/make_long_horizon_configs.py.\n")
                    (ROOT / f"configs/experiments/{cfg['name']}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
                    n += 1
    print(n, "configs")


if __name__ == "__main__":
    main()
