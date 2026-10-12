"""Analyses of saved runs that need no training, run on Colab where the
forecast files are.

  python scripts/after_confirm.py --results <results folder>

1. Lift model, April confirmation window, seed 2 against seed 1: which series
   seed 2 forecast far above sales, and what those series sold before.
2. World model against LightGBM on the March and April confirmation windows
   (three seeds each), with scripts/compare_forecasts.py.
3. World model against LightGBM on the Frat final test (three seeds each),
   to explain the drop from the development windows.

Writes tables to <results>/analysis/ and prints them.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def runs(results: Path, config: str, seeds=(1, 2, 3)) -> list[str]:
    """Latest finished run folder for each seed of a config."""
    out = []
    for s in seeds:
        found = sorted(p for p in results.glob(f"*_{config}_seed{s}_*")
                       if (p / "forecasts.csv").exists() and not p.name.endswith("_partial"))
        if found:
            out.append(str(found[-1]))
        else:
            print(f"missing: {config} seed {s}")
    return out


def lift_blowup(results: Path, out: Path) -> None:
    s1, s2 = runs(results, "m5_confirm_apr_lift", (1,)), runs(results, "m5_confirm_apr_lift", (2,))
    if not (s1 and s2):
        return
    a = pd.read_csv(Path(s2[0]) / "forecasts.csv", usecols=["series_id", "date", "units", "forecast"])
    b = pd.read_csv(Path(s1[0]) / "forecasts.csv", usecols=["series_id", "date", "forecast"])
    d = a.merge(b, on=["series_id", "date"], suffixes=("_seed2", "_seed1"))
    g = d.groupby("series_id").agg(units=("units", "sum"), seed2=("forecast_seed2", "sum"),
                                   seed1=("forecast_seed1", "sum"), seed2_max_day=("forecast_seed2", "max"))
    g["excess_seed2"] = g["seed2"] - g["units"]
    g = g.sort_values("excess_seed2", ascending=False)
    total_excess = g["seed2"].sum() - g["units"].sum()
    print("\n== Lift model, April window: seed 2 against seed 1 ==")
    print(f"series: {len(g)}; units {g['units'].sum():,.0f}; seed 2 {g['seed2'].sum():,.0f}; seed 1 {g['seed1'].sum():,.0f}")
    for k in (5, 20, 100):
        share = (g["excess_seed2"].head(k).sum() / total_excess) if total_excess else float("nan")
        print(f"top {k} series hold {100 * share:.0f}% of seed 2's excess units")
    ratio = (g["seed2"] + 1) / (g["seed1"] + 1)
    print(f"series where seed 2 is over 10x seed 1: {(ratio > 10).sum()}; over 100x: {(ratio > 100).sum()}")
    print(g.head(20).round(1).to_string())
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "lift_apr_seed2_vs_seed1_by_series.csv")


def compare(a: list[str], b: list[str], names: list[str], out: Path) -> None:
    if not (a and b):
        return
    print(f"\n== {names[0]} against {names[1]}: {out.name} ==", flush=True)
    subprocess.run([sys.executable, str(HERE / "compare_forecasts.py"), "--a", *a, "--b", *b,
                    "--names", *names, "--out", str(out)], check=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    args = p.parse_args(argv)
    results = Path(args.results)
    analysis = results / "analysis"
    lift_blowup(results, analysis / "lift_apr_seed2")
    for w in ("mar", "apr"):
        compare(runs(results, f"m5_confirm_{w}_world_model_v1"), runs(results, f"m5_confirm_{w}_lightgbm"),
                ["World model", "LightGBM"], analysis / f"m5_confirm_{w}_wm_vs_lgb")
    compare(runs(results, "frat_final_world_model"), runs(results, "frat_final_lightgbm"),
            ["World model", "LightGBM"], analysis / "frat_final_wm_vs_lgb")


if __name__ == "__main__":
    main()
