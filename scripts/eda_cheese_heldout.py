"""Why the world model's cheese forecasts ran away in the held-out window (3 November 2026).

Category total forecast against actual for each held-out week, and the series behind
the worst week, with their price, promotion and cost against their own history.
Finding: one item (UPC 2100061223) in six stores; the raw file gives it a price of
32 to 85 dollars in week 323 (normal 1.29, OK flag 1), and the forecasts after that
week run to a million units. See docs/PAPER1_REVISION_PLAN.md, entry of 3 November.

  RESULTS_DIR=... python scripts/eda_cheese_heldout.py
"""
import sys, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from rwm.data import load_dataset
import os
S = os.environ.get("RESULTS_DIR", "cheho") + "/"  # run folders fetched from rwm-results research/avg5_world_model_dominicks_che
ds = load_dataset("dominicks", categories=["che"]); p = ds.panel
runs = {"lr_half s1": "*lr_half_s1*", "start s1": "*start_s1*", "start s2": "*start_s2*"}
for label, g in runs.items():
    f = pd.read_csv(glob.glob(S + g + "/forecasts.csv")[0], parse_dates=["date"], dtype={"series_id": str})
    h = f[f.origin == f.origin.max()]
    wk = h.groupby("date")[["units", "forecast"]].sum()
    wk["ratio"] = wk.forecast / wk.units
    print(f"\n== {label}: held-out weeks, category total forecast / actual")
    print(wk.round(0).assign(ratio=wk.ratio.round(2)).to_string())
    if label != "start s2":
        bad = wk.ratio.idxmax()
        x = h[h.date == bad].copy(); x["over"] = x.forecast - x.units
        top = x.nlargest(8, "over")
        print(f"   worst week {bad.date()}: top series by over-forecast (share of total over {top.over.sum()/x.over.sum():.0%})")
        for _, r in top.iterrows():
            s = p[p.series_id == r.series_id]; hist = s[s.date < h.date.min()]
            now = s[s.date == bad]
            print(f"   {r.series_id} fc {r.forecast:9.0f} actual {r.units:5.0f} | hist max units {hist.units.max():5.0f} mean {hist.units.mean():6.1f} | "
                  f"price now {now.price.iloc[0] if len(now) else np.nan:.2f} hist min {hist.price.min():.2f} | promo now {now.promo.iloc[0] if len(now) else np.nan} "
                  f"| cost now {now.cost.iloc[0] if len(now) else np.nan:.2f} hist min {hist.cost.min():.2f} | weeks of history {len(hist)}")
