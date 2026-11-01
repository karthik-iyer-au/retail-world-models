"""Add a panel up to a higher level of the hierarchy, such as one series per store.

Study P (docs/PAPER1_REVISION_PLAN.md, entry of 1 November 2026) forecasts the
store totals directly, then reconciles them with the item-by-store forecasts.
A store total has no single price, so its plan columns become shares:

  units  the sum over the items
  price  the mean over the items of each item's price as a share of its
         regular price (its highest price in the last `regular_window` periods,
         this one included); 1 when nothing is cut
  cost   the mean over the items of cost as a share of the same regular price
  promo  the share of the items on promotion

Every column is worked out from the same period's plan and the item's own
past, so a total is known for future periods exactly when the items' plans are.

  aggregate_panel(panel, by=["store_id"])
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS

ALL = "all"


def aggregate_panel(panel: pd.DataFrame, by: list[str] | tuple = (STORE,), regular_window: int = 12) -> pd.DataFrame:
    by = list(by)
    p = panel.sort_values([SERIES, DATE])
    g = p.groupby(SERIES, observed=True)[PRICE]
    regular = g.transform(lambda s: s.rolling(regular_window, min_periods=1).max())
    frame = pd.DataFrame({k: p[k].astype(str).to_numpy() for k in by})
    frame[DATE] = p[DATE].to_numpy()
    frame[UNITS] = p[UNITS].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        frame[PRICE] = (p[PRICE] / regular).to_numpy(dtype=np.float64)
        if "cost" in p:
            frame["cost"] = (p["cost"] / regular).to_numpy(dtype=np.float64)
    if "promo" in p:
        frame["promo"] = p["promo"].to_numpy(dtype=np.float64)
    agg = {UNITS: "sum", PRICE: "mean", **({"cost": "mean"} if "cost" in frame else {}), **({"promo": "mean"} if "promo" in frame else {})}
    out = frame.groupby(by + [DATE], sort=True).agg(agg).reset_index()
    out[SERIES] = out[by].astype(str).agg("_".join, axis=1)
    for col in (STORE, ITEM):
        if col not in by:
            out[col] = ALL
    if "cat_id" in panel and "cat_id" not in by:
        out["cat_id"] = str(panel["cat_id"].iloc[0]) if panel["cat_id"].nunique() == 1 else ALL
    for col in (SERIES, STORE, ITEM, *(["cat_id"] if "cat_id" in out else [])):
        out[col] = pd.Categorical(out[col])
    out[UNITS] = out[UNITS].astype(np.float32)
    return out.sort_values([SERIES, DATE]).reset_index(drop=True)
