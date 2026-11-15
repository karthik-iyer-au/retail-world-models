"""Paper 1: what each of Dominick's 28 categories looks like, so the six used can be placed among them (Karthik,
16 November 2026: reviewers will ask why these six). Descriptive only; added after the results; no model is run.

  python scripts/dominicks_eda.py --out <folder>       # downloads any category file not yet on disk

Writes <folder>/categories.csv: one row per category, over the weeks flagged OK by the Kilts Center:
items, stores, weeks with sales, units and dollars per week, units per item-store-week, share of rows with no unit
sold, share of rows with a deal code and share of units sold on one, share of item-store rows whose price changed
from the previous recorded week, share of rows priced at least 5% below the item-store's highest price of the previous 12 weeks,
the number of items that make 80% of units, and the share of rows with a price far from the same item's same-week
median across stores (the cleaning rule's factor of 5).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data import fetch
from rwm.utils.paths import DATA_RAW

NAMES = {"ana": "Analgesics", "bat": "Bath soap", "ber": "Beer", "bjc": "Bottled juice", "cer": "Cereals",
         "che": "Cheese", "cig": "Cigarettes", "coo": "Cookies", "cra": "Crackers", "cso": "Canned soup",
         "did": "Dish detergent", "fec": "Front-end candies", "frd": "Frozen dinners", "fre": "Frozen entrees",
         "frj": "Frozen juices", "fsf": "Fabric softeners", "gro": "Grooming products", "lnd": "Laundry detergents",
         "oat": "Oatmeal", "ptw": "Paper towels", "sdr": "Soft drinks", "sha": "Shampoos", "sna": "Snack crackers",
         "soa": "Soaps", "tbr": "Toothbrushes", "tna": "Canned tuna", "tpa": "Toothpastes", "tti": "Toilet tissue"}
USED = {"che", "sna", "sdr", "bjc", "ana", "fre"}
COLS = ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "OK"]


def stats(code: str, root: Path) -> dict:
    mov = next(root / n for n in (f"w{code}.zip", f"w{code}_csv.zip") if (root / n).exists())
    d = pd.read_csv(mov, usecols=COLS, dtype={"SALE": "category"})
    d = d[d["OK"] == 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        d["p"] = (d["PRICE"] / d["QTY"]).where(d["PRICE"] > 0)
    d = d.sort_values(["UPC", "STORE", "WEEK"])
    sold = d[d["MOVE"] > 0]
    weeks = sold["WEEK"].nunique()
    units = sold["MOVE"].sum()
    dollars = (sold["MOVE"] * sold["p"]).sum()
    g = d.groupby(["UPC", "STORE"], sort=False)
    prev = g["p"].shift(1)
    changed = (d["p"].notna() & prev.notna() & ((d["p"] - prev).abs() > 0.005))
    top12 = (prev.groupby([d["UPC"], d["STORE"]], sort=False).rolling(12, min_periods=4).max()
             .reset_index(level=[0, 1], drop=True).reindex(d.index))
    cut = d["p"].notna() & top12.notna() & (d["p"] < 0.95 * top12)
    by_item = sold.groupby("UPC")["MOVE"].sum().sort_values(ascending=False)
    med = d.groupby(["UPC", "WEEK"])["p"].transform("median")
    n = d.groupby(["UPC", "WEEK"])["p"].transform("count")
    far = (n >= 3) & ((d["p"] > 5 * med) | (d["p"] < med / 5))
    deal = d["SALE"].notna()
    return {
        "code": code, "category": NAMES[code], "used": code in USED,
        "items": int(sold["UPC"].nunique()), "stores": int(sold["STORE"].nunique()), "weeks": int(weeks),
        "units_per_week": units / weeks, "dollars_per_week": dollars / weeks,
        "units_per_item_store_week": float(sold.groupby(["UPC", "STORE"])["MOVE"].mean().median()),
        "share_rows_zero": float((d["MOVE"] == 0).mean()),
        "share_rows_deal": float(deal.mean()),
        "share_units_on_deal": float(d.loc[deal, "MOVE"].sum() / max(d["MOVE"].sum(), 1)),
        "share_price_changed": float(changed.sum() / max((d["p"].notna() & prev.notna()).sum(), 1)),
        "share_price_cut": float(cut.sum() / max((d["p"].notna() & top12.notna()).sum(), 1)),
        "items_for_80pct_units": int((by_item.cumsum() / by_item.sum() < 0.8).sum() + 1),
        "share_rows_price_far": float(far.mean()),
        "rows": len(d),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--codes", nargs="*", default=sorted(NAMES))
    a = ap.parse_args(argv)
    root = DATA_RAW / "dominicks"
    missing = [c for c in a.codes if not any((root / n).exists() for n in (f"w{c}.zip", f"w{c}_csv.zip"))]
    if missing:
        fetch.main(["dominicks", *missing])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in a.codes:
        try:
            rows.append(stats(c, root))
            print(c, "done", flush=True)
        except Exception as err:  # one unreadable file is reported, not fatal
            print(c, "failed", repr(err), flush=True)
        pd.DataFrame(rows).to_csv(out / "categories.csv", index=False)


if __name__ == "__main__":
    main()
