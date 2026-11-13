"""LightGBM without the item label on the kept-out-item tests (plan entry of 13 November 2026).

  python scripts/run_lightgbm_noid.py --keys dominicks_che ... --out <folder>

For each key, reads configs/experiments/ftransfer_<key>_item_lightgbm_{kept_out,trained_on_all}.yaml, removes item_id
from LightGBM's categorical inputs and nothing else, renames the run ftransfer_<key>_item_lightgbm_noid_<which>, and
runs it with rwm.experiments.run.
"""
import argparse
import copy
from pathlib import Path

import yaml

from rwm.experiments.run import run

ROOT = Path(__file__).resolve().parents[1]


def noid(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    inner = out["model"]["params"]["params"] if out["model"]["name"] == "seed_average" else out["model"]["params"]
    inner["categorical"] = [c for c in inner.get("categorical", []) if c != "item_id"]
    out["name"] = out["name"].replace("_item_lightgbm_", "_item_lightgbm_noid_")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--which", nargs="+", default=["kept_out", "trained_on_all"])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    for key in a.keys:
        for which in a.which:
            cfg = yaml.safe_load((ROOT / f"configs/experiments/ftransfer_{key}_item_lightgbm_{which}.yaml").read_text())
            d = run(noid(cfg), out_root=Path(a.out))
            print(f"wrote {d}", flush=True)


if __name__ == "__main__":
    main()
