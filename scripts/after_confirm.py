"""Analyses of saved runs that need no training, run on Colab where the
forecast files are.

  python scripts/after_confirm.py --results <results folder>

1. Lift model, April confirmation window, seed 2 against seed 1: which series
   seed 2 forecast far above sales, and what those series sold before.
2. World model against LightGBM on the March and April confirmation windows
   (three seeds each), with scripts/compare_forecasts.py.
3. World model against LightGBM on the Frat final test (three seeds each),
   to explain the drop from the development windows.
5. The plan read-back probe (scripts/plan_readback.py) on seed 1 of the world
   model on the March, April and test windows, and on the lift model
   (no roll-forward) for comparison.

4. What every model forecast, in April, for the five series where lift seed 2
   failed, and the April world-model-against-LightGBM tables with pairs banded
   by their sales in March (free of the pull that banding by the scored sales
   carries).
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


def readback(results: Path, out: Path) -> None:
    configs = HERE.parent / "configs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    for config in ("m5_confirm_mar_world_model_v1", "m5_confirm_apr_world_model_v1",
                   "m5_test_world_model_roll_norm", "m5_confirm_apr_lift"):
        found = runs(results, config, (1,))
        if not found:
            continue
        print(f"\n== Plan read-back: {config} ==", flush=True)
        subprocess.run([sys.executable, str(HERE / "plan_readback.py"), "--config", str(configs / f"{config}.yaml"),
                        "--run", found[0], "--out", str(out / f"{config}.json")], check=False)


FAILED = ["HOUSEHOLD_1_146_WI_2", "HOUSEHOLD_1_146_CA_3", "HOUSEHOLD_1_146_TX_3", "HOUSEHOLD_1_146_CA_2", "FOODS_3_443_WI_1"]


def failed_series(results: Path, out: Path) -> None:
    """Units sold and forecast over the April window for the five series, by model (seeds averaged)."""
    rows = {}
    for label, config in (("world model", "m5_confirm_apr_world_model_v1"), ("LightGBM", "m5_confirm_apr_lightgbm"),
                          ("lift", "m5_confirm_apr_lift"), ("TimesFM mean", "m5_confirm_apr_timesfm_mean"),
                          ("graph", "m5_confirm_apr_graph")):
        found = runs(results, config, (1, 2, 3)) or runs(results, config, (1,))
        sums = []
        for f in found:
            d = pd.read_csv(Path(f) / "forecasts.csv", usecols=["series_id", "units", "forecast"])
            d = d[d["series_id"].isin(FAILED)].groupby("series_id")[["units", "forecast"]].sum()
            sums.append(d)
            rows.setdefault("sold", d["units"])
        if sums:
            for i, d in enumerate(sums, 1):
                rows[f"{label} seed {i}"] = d["forecast"]
    if rows:
        t = pd.DataFrame(rows).round(1)
        print("\n== April window: five series where lift seed 2 failed ==")
        print(t.to_string())
        out.mkdir(parents=True, exist_ok=True)
        t.to_csv(out / "april_failed_series_by_model.csv")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--skip-readback", action="store_true")
    args = p.parse_args(argv)
    results = Path(args.results)
    analysis = results / "analysis"
    lift_blowup(results, analysis / "lift_apr_seed2")
    for w in ("mar", "apr"):
        compare(runs(results, f"m5_confirm_{w}_world_model_v1"), runs(results, f"m5_confirm_{w}_lightgbm"),
                ["World model", "LightGBM"], analysis / f"m5_confirm_{w}_wm_vs_lgb")
    compare(runs(results, "frat_final_world_model"), runs(results, "frat_final_lightgbm"),
            ["World model", "LightGBM"], analysis / "frat_final_wm_vs_lgb")
    failed_series(results, analysis / "lift_apr_seed2")
    march = runs(results, "m5_confirm_mar_world_model_v1", (1,))
    a, b = runs(results, "m5_confirm_apr_world_model_v1"), runs(results, "m5_confirm_apr_lightgbm")
    if march and a and b:
        print("\n== April, banded by March sales ==", flush=True)
        subprocess.run([sys.executable, str(HERE / "compare_forecasts.py"), "--a", *a, "--b", *b,
                        "--names", "World model", "LightGBM", "--band-from", march[0],
                        "--out", str(analysis / "m5_confirm_apr_wm_vs_lgb_banded_by_march")], check=False)
    if "--skip-readback" not in sys.argv:
        readback(results, analysis / "plan_readback")


if __name__ == "__main__":
    main()
