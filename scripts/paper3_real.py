"""Paper 3, amendment 5: a real-data check on Breakfast at the Frat and M5 (docs/PAPER3_PLAN.md).

  python scripts/paper3_real.py --out <folder> --data frat --origin 125 [--versions wm_d32_s2000_h26 ...]

Fits each version on the weeks before the origin and forecasts the 8 weeks from it. On the Frat, item-weeks with a
cut deeper than 30% are removed from training. Writes <out>/real_<data>_o<origin>_<version>.json (measures by group)
and .csv.gz (the held-out rows with the forecast); finished versions are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper3_benchmark import RR, LogLinear, Split, versions  # noqa: E402

from rwm.data import load_dataset  # noqa: E402
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS  # noqa: E402
from rwm.forecaster import build_model  # noqa: E402
from rwm.utils.paths import DATA_RAW  # noqa: E402

HORIZON = 8
DEEP = 0.30
MOVE = {"frat": 0.10, "m5": 0.05, "cj": 0.10}
PURE = "_pure"  # cj only: no mailer and no display in any store (not given to the models)
MODELS = ["wm_d32_s2000_h26", "wm_rr", "wm_shape", "lgb_l31_r500_m20", "lgb_rr", "lgb_shape", "loglinear",
          "loglinear_rr", "loglinear_shape"]


def frat() -> pd.DataFrame:
    d = load_dataset("breakfast_at_the_frat").panel
    out = pd.DataFrame({
        SERIES: d["store_id"].astype(str) + "_" + d["item_id"].astype(str),  # store_item: LogLinear reads the item
        STORE: d["store_id"].astype(str), ITEM: d["item_id"].astype(str), DATE: pd.to_datetime(d[DATE]),
        UNITS: d[UNITS].astype(float), PRICE: d[PRICE].astype(float), "promo": d["promo"].astype(int),
        "display": d["display"].astype(int), RR: d["base_price"].astype(float)})
    out = out[(out[PRICE] > 0) & (out[RR] > 0)]
    return out.sort_values([SERIES, DATE]).reset_index(drop=True)


def constructed_regular(price: np.ndarray, hold: int = 4) -> np.ndarray:
    """A price held for `hold` weeks or more becomes the regular price; otherwise the previous regular price holds."""
    reg = np.empty_like(price)
    cur = price[0]
    run_start = 0
    for t in range(len(price)):
        if t > 0 and price[t] != price[t - 1]:
            run_start = t
        if t - run_start + 1 >= hold:
            cur = price[t]
            reg[run_start:t + 1] = cur  # the run is regular from its first week (known to the retailer at the time)
        reg[t] = cur
    return reg


def m5() -> pd.DataFrame:
    root = DATA_RAW / "m5"
    cal = pd.read_csv(root / "calendar.csv", usecols=["d", "wm_yr_wk", "date"])
    sales = pd.read_csv(root / "sales_train_evaluation.csv")
    sales = sales[(sales["dept_id"] == "FOODS_3") & sales["store_id"].isin(["CA_1", "TX_1", "WI_1"])]
    days = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=["item_id", "store_id"], value_vars=days, var_name="d", value_name="units")
    long = long.merge(cal[["d", "wm_yr_wk"]], on="d")
    wk = long.groupby(["store_id", "item_id", "wm_yr_wk"], as_index=False)["units"].sum()
    prices = pd.read_csv(root / "sell_prices.csv")
    wk = wk.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="inner")  # weeks the item was on sale
    week_end = pd.to_datetime(cal.groupby("wm_yr_wk")["date"].max())
    wk[DATE] = wk["wm_yr_wk"].map(week_end)
    wk = wk.sort_values(["store_id", "item_id", DATE])
    wk[RR] = wk.groupby(["store_id", "item_id"])["sell_price"].transform(
        lambda s: pd.Series(constructed_regular(s.to_numpy()), index=s.index))
    out = pd.DataFrame({
        SERIES: wk["store_id"] + "_" + wk["item_id"], STORE: wk["store_id"], ITEM: wk["item_id"], DATE: wk[DATE],
        UNITS: wk["units"].astype(float), PRICE: wk["sell_price"].astype(float), "promo": 0, "display": 0,
        RR: wk[RR].astype(float)})
    return out.sort_values([SERIES, DATE]).reset_index(drop=True)


def complete_journey() -> pd.DataFrame:
    """Amendment 7: The Complete Journey, purchases summed to product by week over all stores."""
    import zipfile
    z = zipfile.ZipFile(DATA_RAW / "complete_journey" / "dunnhumby_The-Complete-Journey.zip")
    base = "dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/"
    t = pd.read_csv(z.open(base + "transaction_data.csv"),
                    usecols=["PRODUCT_ID", "WEEK_NO", "QUANTITY", "SALES_VALUE", "RETAIL_DISC", "COUPON_MATCH_DISC"])
    t = t[t["QUANTITY"] > 0]
    g = t.groupby(["PRODUCT_ID", "WEEK_NO"]).agg(q=("QUANTITY", "sum"), sv=("SALES_VALUE", "sum"),
                                                 rd=("RETAIL_DISC", "sum"), cm=("COUPON_MATCH_DISC", "sum")).reset_index()
    prod = pd.read_csv(z.open(base + "product.csv"), usecols=["PRODUCT_ID", "DEPARTMENT"])
    food = {"GROCERY", "DRUG GM", "PRODUCE", "MEAT", "MEAT-PCKGD", "DELI", "PASTRY", "NUTRITION", "SEAFOOD",
            "SEAFOOD-PCKGD", "FROZEN GROCERY"}
    g = g[g["PRODUCT_ID"].isin(prod.loc[prod["DEPARTMENT"].isin(food), "PRODUCT_ID"])]
    weeks = g.groupby("PRODUCT_ID")["WEEK_NO"].nunique()
    g = g[g["PRODUCT_ID"].isin(weeks[weeks >= 95].index)]
    g["shelf"] = (g["sv"] - g["rd"] - g["cm"]) / g["q"]
    g["paid"] = (g["sv"] / g["q"]).clip(upper=g["shelf"])
    keep = set(g["PRODUCT_ID"])
    flags = []
    for c in pd.read_csv(z.open(base + "causal_data.csv"), dtype=str, chunksize=2_000_000):
        c = c[c["PRODUCT_ID"].astype(int).isin(keep)]
        flags.append(pd.DataFrame({"PRODUCT_ID": c["PRODUCT_ID"].astype(int), "WEEK_NO": c["WEEK_NO"].astype(int),
                                   "display": (c["display"] != "0").astype(int),
                                   "mailer": (c["mailer"] != "0").astype(int)}))
    f = pd.concat(flags).groupby(["PRODUCT_ID", "WEEK_NO"])[["display", "mailer"]].max().reset_index()
    g = g.merge(f, on=["PRODUCT_ID", "WEEK_NO"], how="left").fillna({"display": 0, "mailer": 0})
    g = g[(g["shelf"] > 0) & (g["paid"] >= 0.05 * g["shelf"])]  # below 5% of shelf: a recording error
    date = pd.Timestamp("2010-01-03") + pd.to_timedelta(7 * (g["WEEK_NO"] - 1), unit="D")  # weeks are relative
    out = pd.DataFrame({SERIES: "all_" + g["PRODUCT_ID"].astype(str), STORE: "all", ITEM: g["PRODUCT_ID"].astype(str),
                        DATE: date, UNITS: g["q"].astype(float), PRICE: g["paid"].astype(float),
                        "promo": g["mailer"].astype(int), "display": g["display"].astype(int),
                        RR: g["shelf"].astype(float), PURE: ((g["mailer"] == 0) & (g["display"] == 0)).astype(int)})
    return out.sort_values([SERIES, DATE]).reset_index(drop=True)


def split(panel: pd.DataFrame, data: str, origin: int, deep: float = DEEP):
    dates = np.sort(panel[DATE].unique())
    start = origin - 1 if origin > 0 else len(dates) + origin  # m5 origins count back from the end
    window = dates[start:start + HORIZON]
    train = panel[panel[DATE] < dates[start]].copy()
    test = panel[panel[DATE].isin(window)].copy()
    test = test[test[SERIES].isin(train[SERIES].unique())]
    depth = 1 - train[PRICE] / train[RR]
    if data in ("frat", "cj"):
        train = train[depth <= deep]  # no model sees a cut deeper than this (30%; 20% in amendment 6)
    last_reg = train.groupby(SERIES)[RR].last()
    move = (test[RR] / test[SERIES].map(last_reg) - 1).abs()
    tdepth = 1 - test[PRICE] / test[RR]
    test["group"] = np.where((tdepth > deep) & (data in ("frat", "cj")),
                             np.where(test.get(PURE, 0) == 1, "deep_pure", "deep"),
                             np.where(move >= MOVE[data], "regular_move", "rest"))
    train = train.drop(columns=[PURE], errors="ignore")
    return train, test


def measures(test: pd.DataFrame, pred: np.ndarray) -> dict:
    t = test.assign(pred=pred)
    out = {}
    for g, x in list(t.groupby("group")) + [("all", t)]:
        s = x[UNITS].sum()
        out[g] = {"rows": int(len(x)), "series": int(x[SERIES].nunique()), "sales": float(s),
                  "bias": float(x["pred"].sum() / s) if s > 0 else float("nan"),
                  "abs_error": float((x["pred"] - x[UNITS]).abs().sum() / s) if s > 0 else float("nan")}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", required=True, choices=["frat", "m5", "cj"])
    p.add_argument("--origin", type=int, required=True)
    p.add_argument("--versions", nargs="*")
    p.add_argument("--steps", type=int, help="override training steps (smoke test only)")
    p.add_argument("--deep", type=float, default=DEEP, help="amendment 6: cuts deeper than this leave training")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panel = frat() if a.data == "frat" else m5() if a.data == "m5" else complete_journey()
    train, test = split(panel, a.data, a.origin, a.deep)
    tag = "" if a.deep == DEEP else f"d{round(100 * a.deep)}_"
    V = versions()
    for name in a.versions or MODELS:
        f = out / f"real_{a.data}_{tag}o{a.origin}_{name}.json"
        if f.exists():
            continue
        kind, settings = V[name][0], dict(V[name][1])
        if a.steps and kind == "state_model":
            settings["steps"] = a.steps
        if kind == "state_model":
            settings["train_periods"] = 156
        keep_rr = name.endswith("_rr") or "_shape" in name
        fit_on = train if keep_rr else train.drop(columns=[RR])
        t0 = time.time()
        model = (LogLinear(**settings).fit(fit_on) if kind == "loglinear"
                 else Split(**settings).fit(fit_on) if kind == "split"
                 else build_model(kind, **settings).fit(fit_on))
        plan = test.drop(columns=[UNITS, "group"] + ([] if keep_rr else [RR]), errors="ignore") \
            .drop(columns=[PURE], errors="ignore")
        pred = np.clip(np.asarray(model.predict(plan), dtype=float), 0, None)
        rec = {"data": a.data, "origin": a.origin, "deep_threshold": a.deep, "version": name, "kind": kind, **measures(test, pred),
               "seconds": round(time.time() - t0, 1)}
        f.write_text(json.dumps(rec, indent=1))
        test.assign(pred=pred)[[SERIES, DATE, UNITS, "group", "pred"]].to_csv(f.with_suffix(".csv.gz"), index=False)
        print(name, json.dumps({g: rec[g] for g in rec if isinstance(rec[g], dict)}), flush=True)


if __name__ == "__main__":
    main()
