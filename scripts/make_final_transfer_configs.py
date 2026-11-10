"""The transfer test and the new-retailer test rerun on the final settings
(docs/PAPER1_REVISION_PLAN.md, entry of 10 November 2026).

1. Transfer (studies I and L, rerun): a fifth of stores, or of items, kept out of training in every
   rolling window, read back from their own history and forecast, against a comparison run trained on
   everything and scored on the same fifth; from each data set's final configs
   (configs/experiments/final_<data>_<model>.yaml).
2. New retailer with settings carried over (study D, rerun): Favorita's rolling configs with every model
   setting taken from M5's final configs (the other daily source, as study D did), except the settings
   that name the data set's own columns (categorical, extra, levers), which stay Favorita's.

  python scripts/make_final_transfer_configs.py m5 dominicks_che dominicks_sna   # transfer, these data sets
  python scripts/make_final_transfer_configs.py --new-retailer                    # study D rerun

Writes configs/experiments/ftransfer_*.yaml and fnewretailer_*.yaml and prints their paths.
"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "configs" / "experiments"
COLUMNS = ("categorical", "extra", "levers")


def _load(name: str) -> dict:
    return yaml.safe_load((EXP / f"{name}.yaml").read_text())


def _inner(cfg: dict) -> dict:
    m = cfg["model"]
    return m["params"]["params"] if m["name"] == "seed_average" else m["params"]


def _write(name: str, header: str, cfg: dict) -> str:
    cfg = {**cfg, "name": name}
    path = EXP / f"{name}.yaml"
    path.write_text(f"# {header}; written by scripts/make_final_transfer_configs.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
    return str(path.relative_to(ROOT))


def transfer(data: str) -> list[str]:
    out = []
    for model in ("world_model", "lightgbm"):
        src = f"final_{data}_{model}"
        for by in ("store_id", "item_id"):
            for form, all_ in (("kept_out", False), ("trained_on_all", True)):
                cfg = copy.deepcopy(_load(src))
                cfg["evaluation"]["transfer"] = {"by": by, "share": 0.2, "seed": 0, "trained_on_all": all_}
                p = _inner(cfg)
                pooled = model == "lightgbm" and by == "store_id" and p.get("group_by")
                if pooled:  # one model per store cannot forecast a store it never saw: one model, store as an input
                    p["group_by"] = None
                    p["categorical"] = list(p.get("categorical", [])) + ["store_id"]
                name = f"ftransfer_{data}_{by.split('_')[0]}_{model}{'_pooled' if pooled else ''}_{form}"
                out.append(_write(name, f"Transfer rerun (plan entry of 10 November 2026) from {src}", cfg))
    return out


def new_retailer() -> list[str]:
    out = []
    for model in ("world_model", "lightgbm"):
        fav = _load(f"avg5_favorita_rolling_{model}")
        m5 = _inner(_load(f"final_m5_{model}"))
        cfg = copy.deepcopy(fav)
        p = _inner(cfg)
        keep = {k: p[k] for k in COLUMNS if k in p}
        new = {k: copy.deepcopy(v) for k, v in m5.items() if k not in COLUMNS and k != "seed"}
        p.clear()
        p.update({**new, **keep})
        out.append(_write(f"fnewretailer_favorita_{model}",
                          f"New retailer, settings carried over (plan entry of 10 November 2026): avg5_favorita_rolling_{model} "
                          f"with every setting of final_m5_{model} except the column lists", cfg))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = new_retailer() if "--new-retailer" in args else [p for d in args for p in transfer(d)]
    print(" ".join(paths))
