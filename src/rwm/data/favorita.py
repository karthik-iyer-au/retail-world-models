"""Corporación Favorita loader (Kaggle competition favorita-grocery-sales-forecasting).

The data set kept out of development (docs/PAPER1_REVISION_PLAN.md, study D):
daily unit sales by item and store for the Favorita grocery chain in Ecuador,
2013 to 15 August 2017. Reads the competition's CSV files as extracted from
the download: train.csv, items.csv, stores.csv, holidays_events.csv.

What the data record, following the competition's data page:
- units: unit_sales; a row is absent on a day with no sales (counted as 0).
  Negative values are returns and are set to 0. Weighed items have
  fractional units.
- promo: onpromotion, the item was on promotion in the store that day;
  missing before April 2014 and counted as off.
- event: a national holiday or event that was not moved to another day.
- payday: the 15th and the last day of each month, when public-sector wages
  are paid.
Not recorded: prices, display, feature, cost. Price is held at 1, so the
revenue weights of WRMSSE become unit weights.

The series are the `n_stores` stores with the most units sold in
`select_year`, crossed with the items sold in every one of those stores in
that year (the rule of the study, fixed before the data were read). The
first load writes the panel to a cache file next to the raw files, keyed by
the settings, so later loads skip the 4.7 GB CSV.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.utils.paths import DATA_RAW

FILES = ["train.csv", "items.csv", "stores.csv", "holidays_events.csv"]
HIERARCHY = [[], ["store_id"], ["family"], ["store_id", "family"], ["item_id"], ["item_id", "store_id"]]
HOLIDAY_TYPES = ("Holiday", "Transfer", "Additional", "Bridge", "Event")


def _sales(path: Path, start: pd.Timestamp) -> pd.DataFrame:
    """train.csv from `start` on, read in chunks with compact types."""
    parts = []
    for chunk in pd.read_csv(path, usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
                             dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32",
                                    "onpromotion": "object"}, chunksize=10_000_000):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"] >= start]
        if len(chunk):
            promo = chunk["onpromotion"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
            parts.append(chunk.assign(onpromotion=promo.astype("int8")))
    return pd.concat(parts, ignore_index=True)


def _national_events(path: Path) -> set:
    h = pd.read_csv(path, parse_dates=["date"])
    h["transferred"] = h["transferred"].astype(str).str.lower().eq("true")
    keep = (h["locale"] == "National") & h["type"].isin(HOLIDAY_TYPES) & ~h["transferred"]
    return set(h.loc[keep, "date"])


@register_dataset("favorita")
def load_favorita(root: str | None = None, start: str = "2015-01-01", n_stores: int = 10,
                  select_year: int = 2016, cache: bool = True) -> Dataset:
    root_path = Path(root) if root else DATA_RAW / "favorita"
    paths = [root_path / f for f in FILES]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: python -m rwm.data.fetch favorita")
    try:  # the cache is a parquet file; without a parquet engine the panel is rebuilt each time
        import pyarrow  # noqa: F401
    except ImportError:
        cache = False
    settings = {"start": start, "n_stores": n_stores, "select_year": select_year, "version": 1}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    cached = root_path / f"panel_{key}.parquet"
    if cache and cached.exists():
        panel = pd.read_parquet(cached)
        return Dataset("favorita", panel, ["promo"], files=paths, hierarchy=HIERARCHY)

    sales = _sales(paths[0], pd.Timestamp(start))
    items = pd.read_csv(paths[1])
    stores = pd.read_csv(paths[2])
    events = _national_events(paths[3])

    year = sales[(sales["date"].dt.year == select_year) & (sales["unit_sales"] > 0)]
    top = year.groupby("store_nbr")["unit_sales"].sum().sort_values(ascending=False).index[:n_stores]
    in_top = year[year["store_nbr"].isin(top)]
    counts = in_top.groupby("item_nbr")["store_nbr"].nunique()
    keep_items = np.sort(counts[counts == n_stores].index.to_numpy())
    store_ids, item_ids = np.sort(np.asarray(top)), keep_items
    sales = sales[sales["store_nbr"].isin(store_ids) & sales["item_nbr"].isin(item_ids)]

    dates = pd.date_range(start, sales["date"].max(), freq="D")  # days with no row (25 December) count as 0
    n_s, n_i, d = len(store_ids), len(item_ids), len(dates)
    row = np.searchsorted(store_ids, sales["store_nbr"].to_numpy()) * n_i + np.searchsorted(item_ids, sales["item_nbr"].to_numpy())
    col = (sales["date"].to_numpy() - dates[0].to_datetime64()) // np.timedelta64(1, "D")
    units = np.zeros((n_s * n_i, d), dtype=np.float32)
    promo = np.zeros((n_s * n_i, d), dtype=np.int8)
    units[row, col] = np.clip(sales["unit_sales"].to_numpy(), 0, None)
    promo[row, col] = sales["onpromotion"].to_numpy()

    info = items.set_index("item_nbr").loc[item_ids]
    city = stores.set_index("store_nbr").loc[store_ids, "city"].to_numpy()
    s_idx, i_idx = np.repeat(np.arange(n_s), n_i), np.tile(np.arange(n_i), n_s)
    attrs = pd.DataFrame({
        "store_id": [f"S{s:02d}" for s in store_ids[s_idx]],
        "item_id": [f"I{i}" for i in item_ids[i_idx]],
        "family": info["family"].to_numpy()[i_idx],
        "class": info["class"].astype(str).to_numpy()[i_idx],
        "city": city[s_idx],
    })
    attrs["series"] = attrs["item_id"] + "_" + attrs["store_id"]
    order = attrs["series"].sort_values().index.to_numpy()
    attrs, units, promo = attrs.loc[order].reset_index(drop=True), units[order], promo[order]

    def repeat(c):
        cat = pd.Categorical(attrs[c])
        return pd.Categorical.from_codes(np.repeat(cat.codes, d), categories=cat.categories)

    event = dates.isin(list(events)).astype(np.int8)
    payday = ((dates.day == 15) | dates.is_month_end).astype(np.int8)
    n = len(attrs)
    panel = pd.DataFrame({
        SERIES: repeat("series"), STORE: repeat("store_id"), ITEM: repeat("item_id"),
        DATE: np.tile(dates.to_numpy(), n), UNITS: units.ravel(),
        PRICE: np.ones(n * d, dtype=np.float32),  # not recorded
        "promo": promo.ravel(), "family": repeat("family"), "class": repeat("class"), "city": repeat("city"),
        "event": np.tile(event, n), "payday": np.tile(payday, n),
    }, copy=False)
    if cache:
        panel.to_parquet(cached, index=False)
    return Dataset("favorita", panel, ["promo"], files=paths, hierarchy=HIERARCHY)
