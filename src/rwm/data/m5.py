"""M5 (Walmart) loader.

Reads the three files from the Kaggle competition, unchanged:
  calendar.csv, sell_prices.csv, sales_train_evaluation.csv

M5 records price and calendar information only. It has no promotion,
display, feature or cost columns, so the only lever it provides is price.
"""
from pathlib import Path

import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.utils.paths import DATA_RAW

FILES = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]


@register_dataset("m5")
def load_m5(
    root: str | None = None,
    stores: list[str] | None = None,
    max_items: int | None = None,
) -> Dataset:
    """`stores` and `max_items` cut the data down for development runs.
    Reported runs leave both unset. Both are recorded in the run config."""
    root_path = Path(root) if root else DATA_RAW / "m5"
    paths = [root_path / f for f in FILES]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. See data/README.md.")

    calendar = pd.read_csv(paths[0])
    prices = pd.read_csv(paths[1])
    sales = pd.read_csv(paths[2])

    if stores:
        sales = sales[sales["store_id"].isin(stores)]
    if max_items:
        keep = sorted(sales["item_id"].unique())[:max_items]
        sales = sales[sales["item_id"].isin(keep)]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name=UNITS,
    )
    cal = calendar[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]]
    long = long.merge(cal, on="d", how="left", validate="many_to_one")
    long = long.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )

    snap = long["snap_CA"].where(long["state_id"] == "CA", 0)
    snap = snap.where(long["state_id"] != "TX", long["snap_TX"])
    snap = snap.where(long["state_id"] != "WI", long["snap_WI"])

    panel = pd.DataFrame(
        {
            SERIES: long["item_id"] + "_" + long["store_id"],
            STORE: long["store_id"],
            ITEM: long["item_id"],
            DATE: pd.to_datetime(long["date"]),
            UNITS: long[UNITS].astype(float),
            PRICE: long["sell_price"],  # empty before the item is first sold
            "dept_id": long["dept_id"],
            "cat_id": long["cat_id"],
            "state_id": long["state_id"],
            "event": long["event_name_1"].notna().astype(int),
            "snap": snap.astype(int),
        }
    )
    return Dataset("m5", panel, ["price"], files=paths)
