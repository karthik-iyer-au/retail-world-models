"""M5 (Walmart) loader.

Reads the three files from the Kaggle competition, unchanged:
  calendar.csv, sell_prices.csv, sales_train_evaluation.csv

With `include_test`, the 28 days of the competition's final test period are
added from sales_test_evaluation.csv. That file is published by the
organisers (github.com/Mcompetitions/M5-methods), not in the Kaggle download.

M5 records price and calendar information only. It has no promotion,
display, feature or cost columns, so the only lever it provides is price.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.utils.paths import DATA_RAW

FILES = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]
TEST_FILE = "sales_test_evaluation.csv"

# The 12 levels of the M5 hierarchy, in the organisers' order.
HIERARCHY = [
    [],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]


EVENT_TYPES = ("National", "Religious", "Cultural", "Sporting")  # the kinds in calendar.csv


@register_dataset("m5")
def load_m5(
    root: str | None = None,
    stores: list[str] | None = None,
    max_items: int | None = None,
    include_test: bool = False,
) -> Dataset:
    """`stores` and `max_items` cut the data down for development runs.
    Reported runs leave both unset. Both are recorded in the run config."""
    root_path = Path(root) if root else DATA_RAW / "m5"
    paths = [root_path / f for f in FILES + ([TEST_FILE] if include_test else [])]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. See data/README.md.")

    calendar = pd.read_csv(paths[0])
    prices = pd.read_csv(paths[1], dtype={"store_id": "category", "item_id": "category"})
    sales = pd.read_csv(paths[2])
    if include_test:
        keys = ["item_id", "store_id"]
        test = pd.read_csv(paths[3]).drop(columns=["dept_id", "cat_id", "state_id"])
        sales = sales.merge(test, on=keys, how="left", validate="one_to_one")
        if sales.isna().any().any():
            raise ValueError("test file does not cover every series")

    if stores:
        sales = sales[sales["store_id"].isin(stores)]
    if max_items:
        keep = sorted(sales["item_id"].unique())[:max_items]
        sales = sales[sales["item_id"].isin(keep)]

    # One row per series, sorted, so row order is fixed whatever the file order.
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    order = (sales["item_id"] + "_" + sales["store_id"]).sort_values().index
    units = sales.loc[order, day_cols].to_numpy(dtype=np.float32)
    sales = sales.loc[order, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    sales = sales.reset_index(drop=True)
    sales["series"] = sales["item_id"] + "_" + sales["store_id"]
    n, d = units.shape

    cal = calendar.set_index("d").loc[day_cols]
    dates = pd.to_datetime(cal["date"]).to_numpy()
    weeks = cal["wm_yr_wk"].to_numpy()

    # The full table has about 59 million rows, so it is built from arrays
    # (series x day) and compact column types, not by joining text columns.
    week_ids = np.unique(weeks)
    price_by_week = np.full((n, len(week_ids)), np.nan, dtype=np.float32)
    keys = pd.DataFrame(
        {
            k: pd.Categorical.from_codes(
                prices[k].cat.categories.get_indexer(sales[k]), dtype=prices[k].dtype
            )
            for k in ("store_id", "item_id")
        }
    ).assign(row=np.arange(n))
    pr = prices.merge(keys, on=["store_id", "item_id"], how="inner")
    pr = pr[pr["wm_yr_wk"].isin(week_ids)]
    price_by_week[pr["row"].to_numpy(), np.searchsorted(week_ids, pr["wm_yr_wk"].to_numpy())] = (
        pr["sell_price"].to_numpy(dtype=np.float32)
    )
    price = price_by_week[:, np.searchsorted(week_ids, weeks)]
    del prices, pr, price_by_week, keys

    snap_by_state = {st: cal[f"snap_{st}"].to_numpy(dtype=np.int8) for st in ("CA", "TX", "WI")}
    snap = np.stack([snap_by_state[st] for st in sales["state_id"]])

    def repeat(col: str) -> pd.Categorical:
        c = pd.Categorical(sales[col])
        return pd.Categorical.from_codes(np.repeat(c.codes, d), categories=c.categories)

    panel = pd.DataFrame(
        {
            SERIES: repeat("series"),
            STORE: repeat("store_id"),
            ITEM: repeat("item_id"),
            DATE: np.tile(dates, n),
            UNITS: units.ravel(),
            PRICE: price.ravel(),  # empty before the item is first sold
            "dept_id": repeat("dept_id"),
            "cat_id": repeat("cat_id"),
            "state_id": repeat("state_id"),
            "event": np.tile(cal["event_name_1"].notna().to_numpy(dtype=np.int8), n),
            # the event by kind, from either of the day's two event columns
            **{
                f"event_{k.lower()}": np.tile(
                    ((cal["event_type_1"] == k) | (cal["event_type_2"] == k)).to_numpy(dtype=np.int8), n
                )
                for k in EVENT_TYPES
            },
            "snap": snap.ravel(),
        },
        copy=False,
    )
    return Dataset("m5", panel, ["price"], files=paths, hierarchy=HIERARCHY)
