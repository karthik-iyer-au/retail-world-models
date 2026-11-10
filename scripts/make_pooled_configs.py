"""Study U part (b) configs (docs/PAPER1_REVISION_PLAN.md, entry of 11 November 2026): each data set's
final world model with pooled pretraining, and its final LightGBM fitted on the same pooled data.

  python scripts/make_pooled_configs.py m5 frat dominicks_ana ...

Reads configs/experiments/final_<data>_{world_model,lightgbm}.yaml, writes
configs/experiments/pooled_<data>_{world_model,lightgbm}.yaml and prints their paths.
"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "configs" / "experiments"
DOM = [f"dominicks_{c}" for c in ("ana", "bjc", "che", "fre", "sdr", "sna")]
# where each data set's dataset block is read from (the final configs, and Favorita's rolling config)
SOURCE_CFG = {**{d: f"final_{d}_lightgbm" for d in DOM}, "frat": "final_frat_lightgbm",
              "m5": "final_m5_lightgbm", "favorita": "avg5_favorita_rolling_lightgbm"}
POOLS = {**{d: [x for x in DOM if x != d] + ["frat"] for d in DOM}, "frat": DOM, "m5": ["favorita"], "favorita": ["m5"]}


def _load(name):
    return yaml.safe_load((EXP / f"{name}.yaml").read_text())


def sources(target: str) -> list[dict]:
    out = []
    for s in POOLS[target]:
        block = copy.deepcopy(_load(SOURCE_CFG[s])["dataset"])
        block["tag"] = s
        if s.startswith("dominicks") and target.startswith("dominicks"):
            block["same_retailer"] = True  # shares the target's stores and weeks
        out.append(block)
    return out


def _inner(model: dict) -> dict:
    return model["params"]["params"] if model["name"] == "seed_average" else model["params"]


def write(target: str) -> list[str]:
    paths = []
    wm = _load(f"final_{target}_world_model")
    p = _inner(wm["model"])
    steps = p.get("steps", 30000)
    p.update(pretrain_steps=3 * steps, finetune="low_lr", encoder_lr_scale=0.1, latent_weight=0.5,
             pretrain_on=sources(target))
    wm["name"] = f"pooled_{target}_world_model"
    lg = _load(f"final_{target}_lightgbm")
    inner = lg["model"]
    if inner["name"] == "seed_average":  # the five copies each fitted on the pooled data
        inner["params"] = {**inner["params"], "model": "pooled",
                           "params": {"model": inner["params"]["model"], "params": inner["params"].get("params", {}),
                                      "pool_with": sources(target)}}
    else:
        lg["model"] = {"name": "pooled", "params": {"model": inner["name"], "params": inner.get("params", {}),
                                                    "pool_with": sources(target)}}
    lg["name"] = f"pooled_{target}_lightgbm"
    for cfg in (wm, lg):
        path = EXP / f"{cfg['name']}.yaml"
        path.write_text("# Study U part (b), pooled (plan entry of 11 November 2026); written by scripts/make_pooled_configs.py.\n"
                        + yaml.safe_dump(cfg, sort_keys=False))
        paths.append(str(path.relative_to(ROOT)))
    return paths


if __name__ == "__main__":
    print(" ".join(p for t in sys.argv[1:] for p in write(t)))
