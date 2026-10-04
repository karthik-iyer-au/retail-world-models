"""The one table shape every dataset is converted to.

One row per series per period. A series is one item in one store.
Models and evaluation only ever see this shape, so adding a dataset means
writing one loader and nothing else.
"""
import pandas as pd

SERIES = "series_id"
STORE = "store_id"
ITEM = "item_id"
DATE = "date"
UNITS = "units"
PRICE = "price"

REQUIRED = [SERIES, STORE, ITEM, DATE, UNITS]

# Decision columns a dataset may provide. A dataset config lists which of
# these it has. A lever that is not listed is absent, not zero.
LEVERS = ["price", "promo", "display", "feature", "cost"]


def validate(panel: pd.DataFrame, levers: list[str]) -> pd.DataFrame:
    missing = [c for c in REQUIRED + list(levers) if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    unknown = [c for c in levers if c not in LEVERS]
    if unknown:
        raise ValueError(f"unknown lever names: {unknown}")
    if panel.duplicated([SERIES, DATE]).any():
        raise ValueError("panel has more than one row for a series and date")
    if (panel[UNITS] < 0).any():
        raise ValueError("panel has negative units")
    return panel.sort_values([SERIES, DATE]).reset_index(drop=True)
