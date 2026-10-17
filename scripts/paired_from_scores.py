"""Paired comparison of two models over rolling windows, from saved per-window scores.

  python scripts/paired_from_scores.py OUT.csv SCORES.csv [SCORES.csv ...] --pairs a:b [a:b ...]

For each data set and level: the mean over windows of A's WRMSSE minus B's, with
a 95% interval from 5,000 resamples of windows, the rule the paper uses to say
one model beats another (scripts/revision_tables.py `paired`, same resampling).
The per-window item bootstrap needs the saved forecasts and is not computed here.

Used for the Timestral against TimesFM test of study F, which the plan asked for
and the Colab analysis did not include.
"""
import argparse

import numpy as np
import pandas as pd


def paired(sc: pd.DataFrame, a: str, b: str, boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    rows = []
    r = sc[sc["set"] == "rolling"]
    wa = r[r.model == a].set_index(["key", "level", "origin"]).wrmsse
    wb = r[r.model == b].set_index(["key", "level", "origin"]).wrmsse
    d = (wa - wb).dropna().rename("diff").reset_index()
    for i, ((key, lv), g) in enumerate(d.groupby(["key", "level"])):
        x = g["diff"].to_numpy()
        rng = np.random.default_rng(seed)
        means = x[rng.integers(0, len(x), size=(boot, len(x)))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        rows.append({"key": key, "set": "rolling", "a": a, "b": b, "level": lv, "windows": len(x),
                     "mean_diff": x.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("scores", nargs="+")
    p.add_argument("--pairs", nargs="+", required=True)
    a = p.parse_args(argv)
    sc = pd.concat([pd.read_csv(f) for f in a.scores], ignore_index=True)
    sc = sc.drop_duplicates(["key", "set", "origin", "model", "level"])
    out = pd.concat([paired(sc, *pair.split(":")) for pair in a.pairs], ignore_index=True)
    out.to_csv(a.out, index=False, float_format="%.5g")
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
