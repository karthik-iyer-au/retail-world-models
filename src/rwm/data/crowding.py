"""How crowded each lever is: the share of a store's items that have the lever
on in the same period.

Circular pages and display space are limited, so an item's promotion shares
attention with the others running at the same time. These columns come from
the plan (which items are promoted when), so they are known for future
periods.

Caveat: where a dataset only has rows for periods with sales, an item that
was promoted and sold nothing is not counted.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES, STORE


def add_crowding(panel: pd.DataFrame, levers: list[str], within: str | None = None) -> list[str]:
    """Adds `<lever>_share` to `panel`, in place: of the other items the store
    has ever carried, the share with the lever on that period. With `within`
    (for example a category column) also adds `<lever>_share_within`, the
    same among the item's own group. Returns the new column names."""
    added = []
    scopes = [("share", [STORE])] + ([("share_within", [STORE, within])] if within else [])
    for label, keys in scopes:
        carried = panel.groupby(keys, observed=True)[SERIES].transform("nunique").to_numpy()
        for lever in levers:
            on = panel[lever].fillna(0).to_numpy(dtype=np.float32)
            total = panel.assign(_on=on).groupby(keys + [DATE], observed=True)["_on"].transform("sum").to_numpy()
            panel[f"{lever}_{label}"] = ((total - on) / np.maximum(carried - 1, 1)).astype(np.float32)
            added.append(f"{lever}_{label}")
    return added
