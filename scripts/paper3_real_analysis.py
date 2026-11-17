"""Paper 3, amendment 5: summary of the real-data check (scripts/paper3_real.py).

  python scripts/paper3_real_analysis.py --real results/paper3/real --out results/paper3

Writes real_summary.csv (per data set, group and version: absolute error over sales and forecast over sales, median
over the four origins) and real_pairs.csv (paired comparisons: difference in absolute error over sales, pooled over
origins, with a 95% interval from 2,000 bootstrap draws of series, and the number of origins in which the first
model has the lower error).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("wm_shape", "wm_d32_s2000_h26"), ("wm_rr", "wm_d32_s2000_h26"), ("lgb_shape", "lgb_l31_r500_m20"),
         ("lgb_rr", "lgb_l31_r500_m20"), ("loglinear_shape", "loglinear"), ("loglinear_rr", "loglinear"),
         ("wm_shape", "loglinear_shape"), ("wm_shape", "lgb_shape"), ("wm_shape", "lgb_l31_r500_m20")]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = []
    for f in glob.glob(f"{a.real}/**/*.json", recursive=True):
        r = json.loads(Path(f).read_text())
        for g in ("deep", "regular_move", "rest", "all"):
            if g in r:
                rows.append({"data": r["data"], "origin": r["origin"], "version": r["version"], "group": g, **r[g]})
    s = pd.DataFrame(rows)
    summ = s.groupby(["data", "group", "version"]).agg(abs_error=("abs_error", "median"), bias=("bias", "median"),
                                                       rows=("rows", "sum"), series=("series", "max")).reset_index()
    summ.to_csv(Path(a.out) / "real_summary.csv", index=False)

    frames = []
    for f in glob.glob(f"{a.real}/**/*.csv.gz", recursive=True):
        _, data, origin, version = Path(f).name.replace(".csv.gz", "").split("_", 3)
        frames.append(pd.read_csv(f).assign(data=data, origin=origin, version=version))
    d = pd.concat(frames)
    d["ae"] = (d["pred"] - d["units"]).abs()
    rng = np.random.default_rng(0)
    out = []
    for (data, group), x in d.groupby(["data", "group"]):
        if group == "rest":
            continue
        piv = x.pivot_table(index=["origin", "series_id", "date"], columns="version", values=["ae", "units"],
                            aggfunc="first")
        for m1, m2 in PAIRS:
            if m1 not in piv["ae"] or m2 not in piv["ae"]:
                continue
            ae, u = piv["ae"][[m1, m2]].dropna(), piv["units"][m1]
            u = u.loc[ae.index]
            won = sum(int(g[m1].sum() < g[m2].sum()) for _, g in ae.groupby(level="origin"))
            per = pd.DataFrame({"d": ae[m1] - ae[m2], "u": u}).groupby(level="series_id").sum()
            n = len(per)
            idx = rng.integers(0, n, size=(2000, n))
            boot = per["d"].to_numpy()[idx].sum(1) / per["u"].to_numpy()[idx].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append({"data": data, "group": group, "a": m1, "b": m2, "diff": per["d"].sum() / per["u"].sum(),
                        "lo": lo, "hi": hi, "origins_a_ahead": won, "origins": ae.index.get_level_values(0).nunique(),
                        "series": n})
    pd.DataFrame(out).to_csv(Path(a.out) / "real_pairs.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
