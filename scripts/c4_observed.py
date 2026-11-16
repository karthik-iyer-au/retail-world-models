"""Paper 2: C4 counted only within the prices each item had in each store in training (plan amendment of
16 November 2026, docs/PAPER2_PLAN.md).

  python scripts/c4_observed.py --results <folder of final runs> --out <folder> [--pattern final_*_world_model_*]

For every run folder matching the pattern and every window with a saved model, the planned price is cut by 5, 10,
20 and 30% in turn (as C4 of 13 October). A step to a deeper cut counts only where the deeper price is no lower than
the lowest price the series had in the window's training data. Writes c4_observed.csv: run, window, steps, steps
kept, share of kept steps where the deeper cut lowers the forecast, and C4 on all steps (a check against the audit).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.audit.runs import load_checkpoint, windows
from rwm.data.schema import PRICE, SERIES, UNITS

CUTS = (0.05, 0.10, 0.20, 0.30)
TOL = 1e-3


def c4_observed(model, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    plan = test.drop(columns=[UNITS], errors="ignore")
    low = train[train[PRICE] > 0].groupby(SERIES)[PRICE].min()
    floor = plan[SERIES].map(low).to_numpy()
    prev = model.predict(plan)
    n = kept = worse_all = worse_kept = 0
    for c in CUTS:
        price = plan[PRICE].to_numpy() * (1 - c)
        cur = model.predict(plan.assign(**{PRICE: price}))
        drop = cur < prev - TOL * np.maximum(prev, 1e-9)
        inside = ~np.isnan(floor) & (price >= floor - 1e-9)
        n += len(cur)
        kept += int(inside.sum())
        worse_all += int(drop.sum())
        worse_kept += int((drop & inside).sum())
        prev = cur
    return {"steps": n, "kept": kept, "share_kept": kept / max(n, 1),
            "c4_all": worse_all / max(n, 1), "c4_observed": worse_kept / kept if kept else float("nan")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pattern", default="*final_*_world_model_*")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(Path(a.results).glob(a.pattern)):
        if "_piece_" in d.name or not (d / "manifest.json").exists():
            continue
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        print(d.name, flush=True)
        for origin, train, test in windows(cfg):
            if not (d / f"checkpoint_origin{origin}.pt").exists():
                continue
            rec = c4_observed(load_checkpoint(d, origin), train, test)
            rows.append({"run": d.name, "config": cfg["name"], "origin": origin, **rec})
            print(f"  window {origin}: kept {rec['share_kept']:.2f}, C4 all {rec['c4_all']:.3f}, "
                  f"within observed {rec['c4_observed']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(out / "c4_observed.csv", index=False)


if __name__ == "__main__":
    main()
