"""Paper 3: does forecast accuracy predict intervention accuracy and decision loss? (docs/PAPER3_PLAN.md)

  python scripts/paper3_analysis.py --bench results/paper3/bench --out results/paper3

Reads every <world>_<version>.json of scripts/paper3_benchmark.py. Writes:
  versions.csv      one row per world and version: forecast accuracy and every intervention and decision measure
  by_model.csv      per planning strength and model class: medians over seeds and versions
  link.csv          Spearman correlation between forecast accuracy (item-by-store RMSSE, lower is better) and each
                    intervention measure (signed so that higher is worse), within each world over the 25 versions,
                    then the mean over the three seeds of each planning strength; 95% interval from 2,000 bootstrap
                    draws of versions within worlds.
Intervention error is |log(predicted change / true change)|: 0 when the predicted change in sales is right, large
when it is too big, too small or of the wrong sign (capped at log 20, a twentyfold miss; a wrong sign counts as the cap).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PLANS = ["deeper_50", "deeper_60", "regular_-10", "regular_-5", "regular_+5", "regular_+10", "off_pattern", "no_plan"]


def load(bench: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{bench}/**/*.json", recursive=True):
        r = json.loads(Path(f).read_text())
        row = {"shape": r.get("shape", "loglinear"), "explore": r.get("explore", 0.0), "confounding": r["confounding"],
               "seed": r["seed"], "version": r["version"],
               "model": {"state_model": "world model", "lightgbm_direct": "LightGBM",
                         "split_state_model": "world model + log-linear",
                         "split_lightgbm_direct": "LightGBM + log-linear"}.get(r["kind"], "log-linear"),
               "rmsse": r["accuracy"]["item_by_store_rmsse"], "promo_choice": r["promo_choice"],
               "regular_loss": r["regular_choice_loss"], "stock_observed": r["observed"]["stock"]}
        for p in PLANS:
            row[f"ratio_{p}"] = r[p]["change_ratio"]
            row[f"rank_{p}"] = r[p]["change_rank"]
            row[f"stock_{p}"] = r[p]["stock"]
        rows.append(row)
    d = pd.DataFrame(rows).drop_duplicates(["shape", "explore", "confounding", "seed", "version"])
    amended = ("wm_jepa", "wm_latent", "wm_overshoot", "wm_adajepa", "wm_rr", "lgb_rr", "loglinear_rr", "wm_ll", "lgb_ll", "wm_shape", "loglinear_shape")
    d["grid"] = ~d["version"].isin(amended) & (d["explore"] == 0)  # the 25 versions of the fixed plan
    for p in PLANS:
        ratio = d[f"ratio_{p}"]
        # capped at a twentyfold miss, which a wrong sign also counts as, so the measure does not move as more
        # versions are added
        cap = np.log(20)
        d[f"err_{p}"] = np.abs(np.log(ratio.where(ratio > 0))).clip(upper=cap).fillna(cap)
    d["err_mean"] = d[[f"err_{p}" for p in PLANS]].mean(axis=1)
    d["stock_mean"] = d[[f"stock_{p}" for p in PLANS]].mean(axis=1)
    d["promo_loss"] = 1 - d["promo_choice"]
    return d


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row pair of x and y (draws x versions)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(axis=1) / den


def link(d: pd.DataFrame, measures: list[str], draws: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for (shape, conf), g in d.groupby(["shape", "confounding"]):
        worlds = [w.reset_index(drop=True) for _, w in g.groupby("seed")]
        idx = [rng.integers(0, len(w), size=(draws, len(w))) for w in worlds]
        for m in measures:
            est = np.mean([w["rmsse"].corr(w[m], method="spearman") for w in worlds])
            boot = np.nanmean([_spearman_rows(w["rmsse"].to_numpy()[i], w[m].to_numpy()[i])
                               for w, i in zip(worlds, idx)], axis=0)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            out.append({"shape": shape, "confounding": conf, "measure": m, "spearman": est, "lo": lo, "hi": hi,
                        "versions": len(g)})
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = load(a.bench)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.sort_values(["shape", "explore", "confounding", "seed", "version"]).to_csv(out / "versions.csv", index=False)
    g = d[d["grid"]]
    cols = ["rmsse", "stock_observed", "err_mean", "stock_mean", "promo_loss", "regular_loss"] + \
        [f"err_{p}" for p in PLANS] + [f"ratio_{p}" for p in PLANS]
    g.groupby(["shape", "confounding", "model"])[cols].median().to_csv(out / "by_model.csv")
    measures = ["err_mean", "stock_mean", "promo_loss", "regular_loss"] + [f"err_{p}" for p in PLANS]
    L = link(g, measures).assign(versions_of="all")  # the test fixed in the plan
    # added after the plan, labelled as such: the same link within each model class
    for name, gg in g[g.model != "log-linear"].groupby("model"):
        L = pd.concat([L, link(gg, measures).assign(versions_of=name)], ignore_index=True)
    L.to_csv(out / "link.csv", index=False)
    # amendment 1: each changed version against its unchanged counterpart in the same world (median of the
    # paired difference over the three seeds; negative is better)
    pairs = {"wm_jepa": "wm_d32_s2000_h26", "wm_latent": "wm_d32_s2000_h26", "wm_overshoot": "wm_d32_s2000_h26",
             "wm_adajepa": "wm_d32_s2000_h26", "wm_rr": "wm_d32_s2000_h26", "lgb_rr": "lgb_l31_r500_m20",
             "loglinear_rr": "loglinear"}
    # amendment 2: the split model against the forecaster alone (with and without the recorded regular price)
    pairs2 = [("wm_ll", "wm_rr"), ("wm_ll", "wm_d32_s2000_h26"), ("lgb_ll", "lgb_rr"), ("lgb_ll", "lgb_l31_r500_m20"),
              ("wm_shape", "wm_rr"), ("wm_shape", "wm_d32_s2000_h26"), ("wm_shape", "loglinear_shape"),
              ("loglinear_shape", "loglinear_rr")]
    rows = []
    key = ["shape", "confounding", "seed"]
    plain = d[d.explore == 0]
    for v, base in list(pairs.items()) + pairs2:
        a_, b_ = plain[plain.version == v].set_index(key), plain[plain.version == base].set_index(key)
        both = a_.index.intersection(b_.index)
        if not len(both):
            continue
        diff = (a_.loc[both, cols] - b_.loc[both, cols]).groupby(level=["shape", "confounding"]).median()
        rows.append(diff.assign(version=v, against=base).reset_index())
    if rows:
        pd.concat(rows).to_csv(out / "amended.csv", index=False)
    # amendment 2: test and learn, each version in the world with test plans against the same world without
    tl = d[d.explore > 0]
    rows = []
    for v in tl.version.unique():
        a_, b_ = tl[tl.version == v].set_index(key), plain[plain.version == v].set_index(key)
        both = a_.index.intersection(b_.index)
        if len(both):
            diff = (a_.loc[both, cols] - b_.loc[both, cols]).groupby(level=["shape", "confounding"]).median()
            rows.append(diff.assign(version=v).reset_index())
    if rows:
        pd.concat(rows).to_csv(out / "test_and_learn.csv", index=False)
        tl.groupby(["shape", "confounding", "version"])[cols].median().to_csv(out / "test_and_learn_levels.csv")
    pd.set_option("display.width", 200)
    print(g.groupby(["shape", "confounding", "model"])[["rmsse", "err_mean", "stock_mean", "promo_loss", "regular_loss"]]
          .median().round(3))
    print(L[L.measure.isin(["err_mean", "stock_mean", "promo_loss", "regular_loss"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
