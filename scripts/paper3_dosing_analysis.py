"""Paper 3, amendment 8: summary of the dosing world (scripts/paper3_dosing.py).

  python scripts/paper3_dosing_analysis.py --bench results/paper3/dosing --out results/paper3

Writes dosing_versions.csv (one row per world and version), dosing_by_model.csv (medians over the 12 worlds),
dosing_link.csv (rank correlation over the 25 versions of forecast error with error on new doses, by strength, for all
versions and within each model class, mean over seeds, 2,000 bootstrap draws of versions) and dosing_pairs.csv (each
model with the rules against itself without, worlds won, two-sided sign test, median paired difference with a 95%
interval from a bootstrap over worlds). The error on new doses is that of the main study, |log(predicted change /
true change)| capped at a twentyfold miss, which a wrong sign also counts as.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper3_analysis import _sign_p, _spearman_rows  # noqa: E402

PLANS = ["dose_50", "dose_60", "steady_20", "off_pattern", "no_dose"]
PAIRS = [("wm_shape", "wm_d32_s2000_h26"), ("lgb_shape", "lgb_l31_r500_m20"), ("loglinear_shape", "loglinear"),
         ("wm_shape", "loglinear_shape"), ("wm_shape", "lgb_shape")]


def load(bench: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{bench}/**/dose_*.json", recursive=True):
        r = json.loads(Path(f).read_text())
        row = {"strength": r["strength"], "seed": r["seed"], "version": r["version"], "rmsse": r["rmsse"],
               "model": "world model" if r["version"].startswith("wm") else
               "LightGBM" if r["version"].startswith("lgb") else "log-linear"}
        for p in PLANS:
            row[f"ratio_{p}"] = r[p]["change_ratio"]
        rows.append(row)
    d = pd.DataFrame(rows).drop_duplicates(["strength", "seed", "version"])
    cap = np.log(20)
    for p in PLANS:
        ratio = d[f"ratio_{p}"]
        d[f"err_{p}"] = np.abs(np.log(ratio.where(ratio > 0))).clip(upper=cap).fillna(cap)
    d["err_mean"] = d[[f"err_{p}" for p in PLANS]].mean(axis=1)
    d["grid"] = d["version"].str.fullmatch(r"wm_d\d+_s\d+_h\d+|lgb_l\d+_r\d+_m\d+|loglinear")
    return d


def link(g: pd.DataFrame, draws: int = 2000) -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for s, x in g.groupby("strength"):
        worlds = [w.reset_index(drop=True) for _, w in x.groupby("seed")]
        est = np.mean([w["rmsse"].corr(w["err_mean"], method="spearman") for w in worlds])
        boot = np.nanmean([_spearman_rows(w["rmsse"].to_numpy()[i], w["err_mean"].to_numpy()[i])
                           for w in worlds for i in [rng.integers(0, len(w), size=(draws, len(w)))]], axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out.append({"strength": s, "spearman": est, "lo": lo, "hi": hi, "versions": len(x)})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = load(a.bench)
    out = Path(a.out)
    d.sort_values(["strength", "seed", "version"]).to_csv(out / "dosing_versions.csv", index=False)
    cols = ["rmsse", "err_mean"] + [f"ratio_{p}" for p in PLANS]
    d.groupby("version")[cols].median().to_csv(out / "dosing_by_model.csv")
    g = d[d["grid"]]
    L = [dict(r, versions_of="all") for r in link(g)]
    for name, gg in g[g.model != "log-linear"].groupby("model"):
        L += [dict(r, versions_of=name) for r in link(gg)]
    pd.DataFrame(L).to_csv(out / "dosing_link.csv", index=False)
    w = d.pivot_table(index=["strength", "seed"], columns="version", values="err_mean")
    rng = np.random.default_rng(0)
    rows = []
    for m1, m2 in PAIRS:
        diff = (w[m1] - w[m2]).dropna().to_numpy()
        n = len(diff)
        k = int((diff < 0).sum())
        boot = np.median(diff[rng.integers(0, n, size=(5000, n))], axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"a": m1, "b": m2, "worlds": n, "a_ahead": k, "sign_p": _sign_p(k, n),
                     "median_diff": float(np.median(diff)), "lo": lo, "hi": hi,
                     "a_median": float(w[m1].median()), "b_median": float(w[m2].median())})
    pd.DataFrame(rows).to_csv(out / "dosing_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.groupby("version")[cols].median().round(2).to_string())
    print(pd.DataFrame(L).round(2).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
