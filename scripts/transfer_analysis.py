"""Study I result (docs/PAPER1_REVISION_PLAN.md, entry of 27 October 2026): the
cost of not training on a store or item, for each model, from the saved runs.

  python scripts/transfer_analysis.py <folder holding the study I run folders>

For each data set, kind (store or item), model and window: the kept-out run's
score on the kept-out fifth minus the comparison run's score on the same fifth
(WRMSSE, so a positive cost means accuracy lost), averaged over seeds where a
model has several. Then the world model's cost against LightGBM's, paired by
window, with a 90% bootstrap interval over windows. Writes
results/paper1_revision/tables/transfer.csv and transfer_vs_lightgbm.csv.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper1_revision" / "tables"
NAME = re.compile(r"transfer_(?P<data>m5|dominicks_[a-z]+)_(?P<kind>store|item)_(?P<model>world_model|lightgbm_pooled|lightgbm|seasonal_naive)"
                  r"_(?P<form>kept_out|trained_on_all)(?:_seed(?P<seed>\d+))?$")


def runs(folder: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(folder.iterdir()):
        m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else None
        if m is None or not (d / "metrics.json").exists():
            continue
        got = NAME.match(m["config"]["name"])
        if not got:
            continue
        for s in json.loads((d / "metrics.json").read_text())["splits"]:
            rows.append({**got.groupdict(), "seed": int(got["seed"] or 0), "window": s["test_start"],
                         "kept_out": s["kept_out"], "trained_on": s["trained_on"],
                         "kept_out_forecast_to_actual": s["kept_out_forecast_to_actual"]})
    return pd.DataFrame(rows)


def costs(r: pd.DataFrame) -> pd.DataFrame:
    keys = ["data", "kind", "model", "seed", "window"]
    out = r[r["form"] == "kept_out"].set_index(keys)
    ref = r[r["form"] == "trained_on_all"].set_index(keys)["kept_out"]
    if "seasonal_naive" in set(r["model"]):  # learns nothing: its kept-out run is its own comparison
        naive = out.xs("seasonal_naive", level="model", drop_level=False)["kept_out"]
        ref = pd.concat([ref, naive])
    out = out.assign(comparison=ref.reindex(out.index)).dropna(subset=["comparison"])
    out["cost"] = out["kept_out"] - out["comparison"]
    return out.groupby(["data", "kind", "model", "window"])[["kept_out", "comparison", "cost", "kept_out_forecast_to_actual"]].mean().reset_index()


def paired(c: pd.DataFrame, reps: int = 10000, seed: int = 0) -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(seed)
    for (data, kind), g in c.groupby(["data", "kind"]):
        trees = "lightgbm_pooled" if kind == "store" else "lightgbm"
        w = g.pivot(index="window", columns="model", values="cost")
        if not {"world_model", trees} <= set(w.columns):
            continue
        d = (w["world_model"] - w[trees]).dropna().to_numpy()
        boot = rng.choice(d, (reps, len(d))).mean(axis=1)
        lo, hi = np.quantile(boot, [0.05, 0.95])
        rows.append({"data": data, "kind": kind, "windows": len(d), "world_model_cost": w["world_model"].mean(),
                     "lightgbm_cost": w[trees].mean(), "difference": d.mean(), "low_90": lo, "high_90": hi,
                     "world_model_loses_less": hi < 0})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    c = costs(runs(Path(argv[0])))
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT / "transfer.csv", index=False)
    p = paired(c)
    p.to_csv(OUT / "transfer_vs_lightgbm.csv", index=False)
    print(c.groupby(["data", "kind", "model"])[["kept_out", "comparison", "cost", "kept_out_forecast_to_actual"]].mean().round(4).to_string())
    print()
    print(p.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
