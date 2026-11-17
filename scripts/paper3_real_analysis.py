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


def frat_groups() -> pd.DataFrame:
    """Amendment 6: per Frat item-week, the flags needed to split deep cuts into pure price cuts (shelf tag only, no
    circular, no display) and the rest."""
    from rwm.data import load_dataset
    d = load_dataset("breakfast_at_the_frat").panel
    return pd.DataFrame({"series_id": d["store_id"].astype(str) + "_" + d["item_id"].astype(str),
                         "date": pd.to_datetime(d["date"]), "tpr_only": d["tpr_only"].astype(int),
                         "regular": d["base_price"].astype(float)})


def amend6(d: pd.DataFrame) -> pd.DataFrame:
    """Adds the groups of amendment 6 to the Frat rows: deep_pure (a deep cut with a shelf tag only) and clean_move
    (a regular-price move of 10% or more that holds in every held-out week, after four steady weeks before)."""
    f = d[d.data == "frat"].copy()
    g = frat_groups()
    f["date"] = pd.to_datetime(f["date"])
    f = f.merge(g, on=["series_id", "date"], how="left")
    f.loc[(f.group == "deep") & (f.tpr_only == 1), "group"] = "deep_pure"
    dates = np.sort(g["date"].unique())
    clean = []
    for o, x in f.groupby("origin"):
        start = dates[int(o.lstrip("o")) - 1]
        before = g[(g.date < start) & (g.date >= dates[max(int(o.lstrip("o")) - 5, 0)])]
        steady = before.groupby("series_id")["regular"].agg(lambda r: r.nunique() == 1 and len(r) >= 4)
        last = before.groupby("series_id")["regular"].last()
        held = x.groupby("series_id")["regular"].agg(lambda r: r.nunique() == 1)
        move = (x.groupby("series_id")["regular"].first() / last.reindex(held.index) - 1).abs() >= 0.10
        ok = held & move & steady.reindex(held.index).fillna(False).astype(bool)
        clean += [(o, s_) for s_ in ok[ok].index]
    key = set(clean)
    f.loc[[(o, s_) in key for o, s_ in zip(f.origin, f.series_id)], "group"] = "clean_move"
    return pd.concat([d[d.data != "frat"], f.drop(columns=["tpr_only", "regular"])])


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
                tag = Path(f).name.split("_")[2]  # real_<data>_[d20_]o<origin>_<version>.json
                data = f"{r['data']}_{tag}" if tag.startswith("d") else r["data"]
                rows.append({"data": data, "origin": r["origin"], "version": r["version"], "group": g, **r[g]})
    s = pd.DataFrame(rows)
    summ = s.groupby(["data", "group", "version"]).agg(abs_error=("abs_error", "median"), bias=("bias", "median"),
                                                       rows=("rows", "sum"), series=("series", "max")).reset_index()
    summ.to_csv(Path(a.out) / "real_summary.csv", index=False)

    frames = []
    for f in glob.glob(f"{a.real}/**/*.csv.gz", recursive=True):
        parts = Path(f).name.replace(".csv.gz", "").split("_")
        data = parts[1]
        thr = parts[2] if parts[2].startswith("d") else "d30"
        origin = next(x for x in parts[2:4] if x.startswith("o"))
        version = "_".join(parts[parts.index(origin) + 1:])
        frames.append(pd.read_csv(f).assign(data=data if data == "m5" else f"frat_{thr}", origin=origin,
                                            version=version))
    d = pd.concat(frames)
    d["data"] = d["data"].replace({"frat_d30": "frat"})
    fr = d[d.data.str.startswith("frat")]
    d = pd.concat([d[~d.data.str.startswith("frat")]] + [amend6(x.assign(data="frat")).assign(data=k)
                                                         for k, x in fr.groupby("data")])
    d["ae"] = (d["pred"] - d["units"]).abs()
    rng = np.random.default_rng(0)
    out = []
    groups = d.groupby(["data", "group", "version", "origin"]).apply(
        lambda x: pd.Series({"rows": len(x), "abs_error": (x.pred - x.units).abs().sum() / x.units.sum(),
                             "bias": x.pred.sum() / x.units.sum()})).reset_index()
    groups.groupby(["data", "group", "version"]).agg(abs_error=("abs_error", "median"), bias=("bias", "median"),
                                                     rows=("rows", "sum"), origins=("origin", "nunique")).reset_index() \
        .to_csv(Path(a.out) / "real_groups.csv", index=False)
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
