"""Item roles: how each item responds to a lever, measured from its own past.

Three measures per item, pooled over the stores that carry it:

  on_share   share of its units sold in periods with the lever on
  lift       its usual sales in a lever-on period over its usual sales in an
             ordinary period (each store's sales first divided by that store's
             own average, so big and small stores count alike)
  presence   share of periods it had sales in, from its first to its last

plus its share of all units sold, and one label built from them, in the
terms a merchant uses. The first rule that fits wins:

  new_line          fewer than `min_periods` selling periods to judge from
  in_and_out        sold in under half of its periods
  kvi               a top seller (in the top `top` share of items by units)
                    that sells at least `lift` times its ordinary rate on deal
  long_tail         among the slowest sellers that together make up the last
                    `tail` share of units
  hi_lo             half or more of its units sold on deal
  promo_responsive  sells at least `lift` times its ordinary rate on deal
  core              everything else: steady everyday sellers

KVI here is a proxy from sales and deal response. A retailer's own KVI list
also uses shopper price perception and basket data, which these datasets do
not have.

Only periods up to `until` are used, so a role never sees the periods a model
is tested on. The label is fixed per item; it is a description of the item,
not of a period.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, ITEM, SERIES, UNITS

ROLE = "item_role"


def item_measures(panel: pd.DataFrame, lever: str = "promo", until=None) -> pd.DataFrame:
    """One row per item: units, selling periods, on_share, lift and presence."""
    p = panel if until is None else panel[panel[DATE] <= until]
    p = p[[SERIES, ITEM, DATE, UNITS, lever]]
    units = p[UNITS].fillna(0).to_numpy(dtype=np.float64)
    on = p[lever].fillna(0).to_numpy() > 0
    item = p[ITEM].astype(str).to_numpy()
    mean = p.groupby(SERIES, observed=True)[UNITS].transform("mean").to_numpy(dtype=np.float64)
    rel = np.divide(units, mean, out=np.zeros_like(units), where=mean > 0)
    d = pd.DataFrame({
        "item": item, "date": p[DATE].to_numpy(), "units": units, "sold": units > 0,
        "on_units": units * on, "rel_on": np.where(on, rel, np.nan), "rel_off": np.where(~on, rel, np.nan),
    })
    g = d.groupby("item")
    out = pd.DataFrame({
        "units": g["units"].sum(),
        "periods": d[d["sold"]].groupby("item")["date"].nunique().reindex(g.size().index, fill_value=0),
        "on_share": g["on_units"].sum() / g["units"].sum().replace(0, np.nan),
        "lift": g["rel_on"].mean() / g["rel_off"].mean().replace(0, np.nan),
    })
    # presence: selling periods over the periods between the item's first and last
    all_dates = np.sort(p[DATE].unique())
    pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
    sold = d[d["sold"]]
    span = sold.groupby("item")["date"].agg(["min", "max"])
    length = (pos.reindex(span["max"]).to_numpy() - pos.reindex(span["min"]).to_numpy() + 1)
    out["presence"] = (out["periods"].reindex(span.index) / length).reindex(out.index)
    return out


def add_roles(
    panel: pd.DataFrame, lever: str = "promo", until=None, lift: float = 2.0, min_periods: int = 8,
    top: float = 0.05, tail: float = 0.05,
) -> pd.DataFrame:
    """Adds the `item_role` label to `panel`, in place, from periods up to
    `until`. Returns the per-item measures with the role."""
    m = item_measures(panel, lever, until)
    by_units = m["units"].sort_values(ascending=False)
    top_seller = pd.Series(np.arange(len(by_units)) < round(top * len(by_units)), index=by_units.index)
    slow = pd.Series(by_units.cumsum().to_numpy() > (1 - tail) * by_units.sum(), index=by_units.index)
    responsive = (m["lift"] >= lift).to_numpy()
    role = np.full(len(m), "core", dtype=object)
    role[responsive] = "promo_responsive"
    role[(m["on_share"] >= 0.5).to_numpy()] = "hi_lo"
    role[slow.reindex(m.index).to_numpy()] = "long_tail"
    role[top_seller.reindex(m.index).to_numpy() & (responsive | (m["on_share"] >= 0.5).to_numpy())] = "kvi"
    role[(m["presence"] < 0.5).to_numpy()] = "in_and_out"
    role[(m["periods"] < min_periods).to_numpy()] = "new_line"
    m[ROLE] = role
    # items first seen after `until` have no row in `m`
    panel[ROLE] = pd.Categorical(panel[ITEM].astype(str).map(m[ROLE]).fillna("new_line"))
    return m
