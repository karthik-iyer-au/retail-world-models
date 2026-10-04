"""Dominick's Finer Foods loader.

Weekly sales by UPC and store for a Chicago-area chain, 1989 to 1997, from
the Kilts Center for Marketing, University of Chicago Booth School of
Business. Academic research use only; the Kilts Center must be acknowledged
in any paper using the data.

Reads, unchanged, for each category code (for example "ana" for analgesics):
  w<code>.zip or w<code>_csv.zip   weekly movement
  upc<code>.csv                    item descriptions
and, if present, demo_stata.zip for each store's price tier.

What the data records, following the Kilts data manual:
- units: MOVE, the number of single items sold.
- price: PRICE divided by QTY, the price of one item (PRICE is for a bundle
  of QTY items). Empty when the file has no price for the week.
- promo: 1 when a deal code is set. The manual warns the code is not set
  consistently, so 0 does not prove there was no promotion. The code itself
  is kept in promo_type (B bonus buy, C coupon, S simple price reduction; G
  also occurs and is not described in the manual).
- cost: price x (1 - PROFIT/100). PROFIT is gross margin based on the
  chain's average acquisition cost, which lags replacement cost.
- Weeks flagged as suspect by the Kilts Center (OK = 0) are dropped.

Not recorded: display, feature advertising, shelf space.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.utils.paths import DATA_RAW

WEEK_1_START = pd.Timestamp("1989-09-14")  # from the manual's week table
HIERARCHY = [[], ["cat_id"], ["store_id"], ["cat_id", "store_id"], ["item_id"], ["item_id", "store_id"]]
COLUMNS = ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"]


def _movement_file(root: Path, code: str) -> Path:
    for name in (f"w{code}.zip", f"w{code}_csv.zip"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"no movement file for category '{code}' in {root}. See data/README.md.")


def _price_tiers(path: Path) -> pd.Series:
    demo = pd.read_stata(path, columns=["store", "priclow", "pricmed", "prichigh"]).dropna(subset=["store"])
    tier = np.select(
        [demo["priclow"] == 1, demo["pricmed"] == 1, demo["prichigh"] == 1],
        ["low", "medium", "high"],
        default="other",
    )
    return pd.Series(tier, index=demo["store"].astype(int))


@register_dataset("dominicks")
def load_dominicks(categories: list[str], root: str | None = None) -> Dataset:
    root_path = Path(root) if root else DATA_RAW / "dominicks"
    files, parts = [], []
    for code in categories:
        mov, upc = _movement_file(root_path, code), root_path / f"upc{code}.csv"
        files += [mov, upc]
        d = pd.read_csv(mov, usecols=COLUMNS, dtype={"SALE": "category"})
        d = d[d["OK"] == 1]
        items = pd.read_csv(upc, encoding="latin-1").drop_duplicates("UPC").set_index("UPC")
        with np.errstate(invalid="ignore", divide="ignore"):
            price = (d["PRICE"] / d["QTY"]).where(d["PRICE"] > 0).astype(np.float32)
        parts.append(
            pd.DataFrame(
                {
                    "store": d["STORE"].astype(np.int32),
                    "upc": d["UPC"].astype(np.int64),
                    "week": d["WEEK"].astype(np.int16),
                    UNITS: d["MOVE"].astype(np.float32),
                    PRICE: price,
                    "promo": d["SALE"].notna().astype(np.int8),
                    "promo_type": d["SALE"].astype(str).where(d["SALE"].notna(), "none"),
                    "cost": (price * (1 - d["PROFIT"] / 100)).astype(np.float32),
                    "cat_id": code,
                    "com_code": items["COM_CODE"].reindex(d["UPC"]).to_numpy(),
                }
            )
        )
    d = pd.concat(parts, ignore_index=True)
    del parts

    # Fixed-width text keys so that text order equals numeric order.
    stores = np.sort(d["store"].unique())
    upcs = np.sort(d["upc"].unique())
    store_names = pd.Index([f"{s:03d}" for s in stores])
    upc_names = pd.Index([f"{u:012d}" for u in upcs])
    s_code = np.searchsorted(stores, d["store"].to_numpy())
    u_code = np.searchsorted(upcs, d["upc"].to_numpy())
    pair = u_code.astype(np.int64) * len(stores) + s_code
    pairs = np.unique(pair)
    series_names = pd.Index([f"{upc_names[p // len(stores)]}_{store_names[p % len(stores)]}" for p in pairs])
    order = np.lexsort((d["week"].to_numpy(), pair))

    panel = pd.DataFrame(
        {
            SERIES: pd.Categorical.from_codes(np.searchsorted(pairs, pair), categories=series_names),
            STORE: pd.Categorical.from_codes(s_code, categories=store_names),
            ITEM: pd.Categorical.from_codes(u_code, categories=upc_names),
            DATE: WEEK_1_START + pd.to_timedelta((d["week"].to_numpy().astype(np.int64) - 1) * 7, unit="D"),
            UNITS: d[UNITS].to_numpy(),
            PRICE: d[PRICE].to_numpy(),
            "promo": d["promo"].to_numpy(),
            "promo_type": pd.Categorical(d["promo_type"]),
            "cost": d["cost"].to_numpy(),
            "cat_id": pd.Categorical(d["cat_id"]),
            "com_code": d["com_code"].to_numpy(),
        }
    ).iloc[order].reset_index(drop=True)

    demo = root_path / "demo_stata.zip"
    if demo.exists():
        files.append(demo)
        tiers = _price_tiers(demo).reindex(stores).fillna("other")
        tiers.index = store_names
        panel["price_tier"] = pd.Categorical(panel[STORE].map(tiers))
    return Dataset("dominicks", panel, ["price", "promo", "cost"], files=files, hierarchy=HIERARCHY)
