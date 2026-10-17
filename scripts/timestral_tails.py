"""Timestral's quantile sums against actual sales (study F check, 17 October 2026).

  python scripts/timestral_tails.py RESULTS_DIR OUT.csv

Reads forecasts.csv of the latest rev_{key}_rolling_{model} run under
RESULTS_DIR (on Colab, the results folder on Drive) and writes, per window,
the sum over all series of the point forecast, the median, the 90% and the
99% quantile over the sum of actual sales, and the share of the point
forecast's excess over sales held by the top 1% of series by excess.

results/paper1_revision/tables/studyF_tails.csv was transcribed from the
console output of this computation, run on Colab on the saved forecasts
(about 390 MB per Dominick's run, kept outside the repository).
"""
import argparse
import glob

import pandas as pd

RUNS = [("dominicks_ana", "timestral_plan"), ("dominicks_ana", "timestral_mean"),
        ("dominicks_sdr", "timestral_plan"), ("dominicks_sdr", "timestral_mean"),
        ("frat", "timestral_plan"), ("frat", "timestral_mean")]


def tails(f: pd.DataFrame) -> pd.DataFrame:
    s = f.groupby("origin")[["units", "forecast", "q0.5", "q0.9", "q0.99"]].sum()
    out = pd.DataFrame({
        "forecast_ratio": s["forecast"] / s["units"],
        "median_ratio": s["q0.5"] / s["units"],
        "q90_ratio": s["q0.9"] / s["units"],
        "q99_ratio": s["q0.99"] / s["units"],
    })
    g = f.groupby(["origin", "series_id"])[["units", "forecast"]].sum()
    g["excess"] = g["forecast"] - g["units"]
    share = {}
    for o, h in g.groupby(level=0):
        h = h.sort_values("excess", ascending=False)
        top = h["excess"].head(max(1, len(h) // 100)).sum()
        share[o] = top / max(h["excess"].sum(), 1.0)
    out["top1pct_share_of_excess"] = pd.Series(share)
    return out.reset_index()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("out")
    a = p.parse_args(argv)
    rows = []
    for key, model in RUNS:
        runs = sorted(glob.glob(f"{a.results}/*rev_{key}_rolling_{model}_*"))
        if not runs:
            print(f"{key} {model}: no run found")
            continue
        t = tails(pd.read_csv(f"{runs[-1]}/forecasts.csv")).assign(key=key, model=model)
        rows.append(t)
        print(key, model)
        print(t.round(2).to_string(index=False))
    out = pd.concat(rows, ignore_index=True)
    cols = ["key", "model", "origin", "forecast_ratio", "median_ratio", "q90_ratio", "q99_ratio",
            "top1pct_share_of_excess"]
    out[cols].round(2).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
