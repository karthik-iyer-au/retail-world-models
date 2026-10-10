"""Where two models' forecasts differ: bias and error by store, department,
sales band, weekday and week ahead, from the saved forecast files alone.

  python scripts/compare_forecasts.py --a <run folder> [<run folder> ...] \
      --b <run folder> [<run folder> ...] --names "World model" LightGBM \
      [--out <folder>]

Each run folder holds forecasts.csv (series_id, date, units, forecast, origin).
Several folders for one model (one per seed) are averaged into one forecast.
M5 series ids are "<item>_<store>", e.g. FOODS_3_090_CA_1, so state, store,
category and department are read from the id. Other datasets get the store
and item split only.

Writes one CSV per table to --out and prints the tables.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(folders: list[str]) -> pd.DataFrame:
    """Forecasts of one model, averaged over its runs (seeds)."""
    parts = []
    for f in folders:
        d = pd.read_csv(Path(f) / "forecasts.csv", usecols=["series_id", "date", "units", "forecast"])
        parts.append(d.set_index(["series_id", "date"]))
    first = parts[0]
    fc = np.mean([p["forecast"].reindex(first.index).to_numpy() for p in parts], axis=0)
    return pd.DataFrame({"units": first["units"].to_numpy(), "forecast": fc}, index=first.index)


def describe(ids: pd.Series) -> pd.DataFrame:
    """Store, state, item, department and category from M5-style series ids."""
    parts = ids.str.rsplit("_", n=2, expand=True)
    out = pd.DataFrame({"item": parts[0], "store": parts[1] + "_" + parts[2]})
    out["state"] = parts[1]
    item = out["item"].str.split("_", expand=True)
    if item.shape[1] >= 3:
        out["category"] = item[0]
        out["department"] = item[0] + "_" + item[1]
    return out


def table(df: pd.DataFrame, key: str, names: list[str]) -> pd.DataFrame:
    a, b = names
    g = df.groupby(key, observed=True)
    out = pd.DataFrame({
        "actual units": g["units"].sum(),
        f"{a} units": g["fa"].sum(),
        f"{b} units": g["fb"].sum(),
    })
    for n, col in ((a, "fa"), (b, "fb")):
        out[f"{n} bias %"] = 100 * (out[f"{n} units"] / out["actual units"] - 1)
    # Error of the summed forecast for the group, day by day: what a buyer
    # planning the whole group would see. Errors that cancel inside the group
    # do not count here.
    daily = df.groupby([key, "date"], observed=True)[["units", "fa", "fb"]].sum()
    for n, col in ((a, "fa"), (b, "fb")):
        e = (daily[col] - daily["units"]).abs().groupby(level=0, observed=True).sum()
        out[f"{n} summed error %"] = 100 * e / out["actual units"]
    # Error item by item, added up: what the shelves see.
    for n, col in ((a, "fa"), (b, "fb")):
        out[f"{n} item error %"] = 100 * df.assign(e=(df[col] - df["units"]).abs()).groupby(key, observed=True)["e"].sum() / out["actual units"]
    # Share of item-store pairs in the group where model a is closer over the window.
    per = df.assign(ea=(df["fa"] - df["units"]).abs(), eb=(df["fb"] - df["units"]).abs())
    per = per.groupby(["series_id", key], observed=True)[["ea", "eb"]].sum()
    closer = (per["ea"] < per["eb"]).groupby(level=1, observed=True).mean()
    out[f"pairs where {a} is closer %"] = 100 * closer
    return out.round(1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a", nargs="+", required=True)
    p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--names", nargs=2, default=["A", "B"])
    p.add_argument("--out", default=None)
    p.add_argument("--band-from", default=None,
                   help="a run folder whose forecasts.csv holds the sales just before this window; "
                        "pairs are put in sales bands by those sales instead of by sales in this window")
    args = p.parse_args(argv)

    A, B = load(args.a), load(args.b)
    df = pd.DataFrame({"units": A["units"], "fa": A["forecast"], "fb": B["forecast"].reindex(A.index)}).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = pd.concat([df, describe(df["series_id"])], axis=1)
    df["all"] = "all"
    df["weekday"] = df["date"].dt.day_name()
    first = df["date"].min()
    df["week ahead"] = "week " + (1 + (df["date"] - first).dt.days // 7).astype(str)
    # Sales band: each pair's average daily sales over the window it was scored on,
    # or, with --band-from, over an earlier window (free of the pull towards
    # under-forecasting that bands set by the scored sales themselves carry).
    if args.band_from:
        before = pd.read_csv(Path(args.band_from) / "forecasts.csv", usecols=["series_id", "units"])
        rate = df["series_id"].map(before.groupby("series_id")["units"].mean()).fillna(0)
    else:
        rate = df.groupby("series_id")["units"].transform("mean")
    df["sales band"] = pd.cut(rate, [-0.1, 0, 0.5, 1, 3, 10, np.inf],
                              labels=["no sales", "under 0.5 a day", "0.5 to 1", "1 to 3", "3 to 10", "over 10"])
    df["day had sales"] = np.where(df["units"] > 0, "sold", "sold nothing")

    keys = ["all", "state", "store", "category", "department", "sales band", "day had sales", "weekday", "week ahead"]
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for k in keys:
        if k not in df:
            continue
        t = table(df, k, args.names)
        print(f"\n### By {k}\n")
        print(t.to_string())
        if out:
            t.to_csv(out / f"by_{k.replace(' ', '_')}.csv")

    # Company total day by day: level against shape.
    tot = df.groupby("date")[["units", "fa", "fb"]].sum()
    tot.columns = ["actual units", f"{args.names[0]} units", f"{args.names[1]} units"]
    print("\n### Company total, day by day\n")
    print(tot.round(0).to_string())
    if out:
        tot.to_csv(out / "total_by_day.csv")


if __name__ == "__main__":
    main()
