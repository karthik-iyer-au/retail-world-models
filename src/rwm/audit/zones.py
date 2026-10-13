"""C7 of the paper 2 plan: zone price deviations on Dominick's.

Dominick's priced by zone: within a zone every store was meant to carry the
same regular price. A store whose regular prices sit at a steady ratio to the
rest of its zone for many weeks has had its prices moved on purpose (the
chain's pricing experiments did exactly this, but the public files do not say
which stores or weeks; Hoch, Dreze and Purk 1994; Abrams, Gui and Hortacsu
2018). Such a spell is used here as a quasi-experiment: its effect on the
store's category units is estimated by difference-in-differences against
stores in the same zone that kept the zone price.

Every threshold below is fixed in docs/PAPER2_PLAN.md before any model lift
is computed. Nothing in this module reads a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_WEEKS = 12        # consecutive weeks a deviation must last
MIN_CONTROLS = 3      # control stores needed for an interval
SCORED_WEEKS = 8      # weeks of an episode the models are scored on (their horizon)
MIN_SIZE = 0.05       # episode's median price ratio at least 5% away from the zone
WEEK_SIZE = 0.03      # each week of an episode at least 3% away, in the same direction
SAME_SIGN = 0.6       # share of the store's items moved in the deviation's direction
CONTROL_SIZE = 0.01   # controls: median ratio within 1% of the zone price over the episode
PRE_WEEKS = 26        # weeks before the episode used as the comparison period
CUT = 0.95            # a price under 95% of the item's highest in the last 12 weeks is a temporary cut


def zones(demo_path) -> pd.Series:
    """Each store's price zone from the Kilts demographics file (stores without one left out)."""
    demo = pd.read_stata(demo_path, columns=["store", "zone"]).dropna()
    return demo.set_index(demo["store"].astype(int))["zone"].astype(int)


def regular_prices(move: pd.DataFrame) -> pd.DataFrame:
    """Rows of one category's movement file reduced to regular prices: OK weeks,
    no deal code, and not a temporary cut against the item's recent highest price
    in that store. Returns store, upc, week, price, units."""
    d = move[(move["OK"] == 1) & (move["PRICE"] > 0) & (move["QTY"] > 0)].copy()
    d["price"] = d["PRICE"] / d["QTY"]
    d = d.sort_values(["STORE", "UPC", "WEEK"])
    top = d.groupby(["STORE", "UPC"])["price"].transform(lambda s: s.rolling(12, min_periods=1).max())
    deal = d["SALE"].notna() & (d["SALE"].astype(str).str.strip() != "")
    keep = ~deal & (d["price"] >= CUT * top)
    out = d.loc[keep, ["STORE", "UPC", "WEEK", "price", "MOVE"]]
    return out.rename(columns={"STORE": "store", "UPC": "upc", "WEEK": "week", "MOVE": "units"})


def store_ratios(reg: pd.DataFrame, zone: pd.Series) -> pd.DataFrame:
    """Per store and week: how far the store's regular prices sit from the rest of
    its zone, relative to their usual gap. For each store and item, the log of
    the week's regular price over the item's median regular price in that store
    over all weeks; then, for each zone, item and week, the median of that across
    the zone's stores is subtracted, so moves made zone-wide cancel and a store
    moved on its own stands out. Needs at least three stores per zone, item and
    week. Returns the median over the store's items, and the shares moved up and
    down by more than 1%."""
    r = reg[reg["store"].isin(zone.index)].copy()
    r["zone"] = r["store"].map(zone)
    r["lr"] = np.log(r["price"] / r.groupby(["store", "upc"])["price"].transform("median"))
    key = ["zone", "upc", "week"]
    g = r.groupby(key)["lr"]
    r["n"] = g.transform("size")
    r = r[r["n"] >= 3]
    r["adj"] = r["lr"] - r.groupby(key)["lr"].transform("median")
    r["is_up"] = r["adj"] > 0.01
    r["is_down"] = r["adj"] < -0.01
    out = r.groupby(["zone", "store", "week"]).agg(
        median=("adj", "median"), up=("is_up", "mean"), down=("is_down", "mean"), items=("adj", "size")
    ).reset_index()
    return out


@dataclass
class Episode:
    zone: int
    store: int
    first_week: int
    last_week: int
    log_ratio: float          # median log price ratio to the zone over the episode
    controls: list[int]


def find_episodes(ratios: pd.DataFrame, min_items: int = 10) -> list[Episode]:
    """Spells of at least MIN_WEEKS consecutive weeks in which a store's median
    ratio is at least MIN_SIZE from the zone price, in one direction, with at
    least SAME_SIGN of its items moved that way."""
    eps = []
    r = ratios[ratios["items"] >= min_items]
    for (z, s), d in r.groupby(["zone", "store"]):
        d = d.sort_values("week")
        up = (d["median"] >= WEEK_SIZE) & (d["up"] >= SAME_SIGN)
        down = (d["median"] <= -WEEK_SIZE) & (d["down"] >= SAME_SIGN)
        for flag in (up, down):
            weeks = d.loc[flag, "week"].to_numpy()
            if len(weeks) == 0:
                continue
            breaks = np.flatnonzero(np.diff(weeks) != 1)
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, len(weeks) - 1]
            for a, b in zip(starts, ends):
                span = d[(d["week"] >= weeks[a]) & (d["week"] <= weeks[b])]
                if weeks[b] - weeks[a] + 1 >= MIN_WEEKS and abs(span["median"].median()) >= MIN_SIZE:
                    eps.append(Episode(int(z), int(s), int(weeks[a]), int(weeks[b]),
                                       float(span["median"].median()), []))
    for e in eps:
        span = r[(r["zone"] == e.zone) & r["week"].between(e.first_week, e.last_week) & (r["store"] != e.store)]
        med = span.groupby("store")["median"].median()
        e.controls = sorted(int(x) for x in med[med.abs() <= CONTROL_SIZE].index)
    return [e for e in eps if e.controls]


def did(units: pd.DataFrame, e: Episode, last_week: int | None = None) -> dict:
    """Difference-in-differences on log category units (store-week totals):
    treated store against its controls, PRE_WEEKS before the episode against
    the episode weeks, with store and week fixed effects; standard error
    clustered by store (CR1). Returns the estimate, its standard error and a
    95% interval (t with stores minus one degrees of freedom)."""
    from scipy.stats import t as tdist

    stores = [e.store] + e.controls
    w0, w1 = e.first_week - PRE_WEEKS, (e.last_week if last_week is None else last_week)
    u = units[units["store"].isin(stores) & units["week"].between(w0, w1)]
    tot = u.groupby(["store", "week"])["units"].sum().reset_index()
    tot = tot[tot["units"] > 0]
    tot["y"] = np.log(tot["units"])
    tot["d"] = ((tot["store"] == e.store) & (tot["week"] >= e.first_week)).astype(float)
    S = pd.get_dummies(tot["store"], drop_first=True, dtype=float)
    W = pd.get_dummies(tot["week"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(tot)), tot["d"].to_numpy(), S.to_numpy(), W.to_numpy()])
    y = tot["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = tot["store"].to_numpy()
    for g in np.unique(G):
        sg = X[G == g].T @ resid[G == g]
        meat += np.outer(sg, sg)
    n, k, c = len(y), X.shape[1], len(np.unique(G))
    V = XtX_inv @ meat @ XtX_inv * (c / (c - 1)) * ((n - 1) / (n - k))
    se = float(np.sqrt(max(V[1, 1], 0)))
    q = tdist.ppf(0.975, max(c - 1, 1))
    est = float(beta[1])
    return {"estimate": est, "se": se, "lo": est - q * se, "hi": est + q * se, "stores": c, "rows": n}
