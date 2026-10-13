"""Generated weekly store data where the effect of every part of the plan is known.

Used only for the known-answer study of paper 2 (docs/PAPER2_PLAN.md), never for
a forecasting result. Sales follow

  mean = level * (price / regular) ** cut_elasticity
              * (1 + promo_effect) ** promo * (1 + display_effect) ** display
              * (1 + both_effect) ** (promo * display)
              * (1 - dip) ** (planned last week and not this week)

where `level` holds the item's size, its season, an unobserved demand shock
and its response to the regular price, (regular / list) ** regular_elasticity.
A temporary cut and a change in the regular price therefore have different
effects, as in the retail literature (temporary cuts move far more units).

Stores are grouped into price zones. In each zone one store has its regular
prices raised and one lowered by `episode_size` for `episode_weeks`, at random
weeks, like the chain-wide pricing tests the Dominick's zone deviations stand
in for. These spells are chosen at random, so they are a clean experiment.

With `confounding` above zero, merchants plan cuts, promotions and displays in
weeks when the unobserved demand shock is high, so sales in planned weeks
overstate what the plan adds.

The returned panel carries hidden columns starting with `true_` that no model
reads; `expected_units` uses them to give the expected sales under any plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS


@dataclass
class Truth:
    cut_elasticity: float = -2.5
    regular_elasticity: float = -0.5
    promo_effect: float = 0.40
    display_effect: float = 0.25
    both_effect: float = 0.15
    dip: float = 0.15


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@register_dataset("synthetic_levers")
def load_synthetic_levers(
    n_zones: int = 3, stores_per_zone: int = 4, n_items: int = 20, n_periods: int = 260, seed: int = 0,
    confounding: float = 0.0, shock_sd: float = 0.25, shock_rho: float = 0.7,
    episode_size: float = 0.10, episode_weeks: int = 16, test_origin: int | None = None, **truth,
) -> Dataset:
    t = Truth(**truth)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-06", periods=n_periods, freq="7D")
    weeks = np.arange(n_periods)
    items = [dict(list_price=rng.uniform(1.5, 8.0), size=np.exp(rng.uniform(np.log(3), np.log(60))),
                  phase=rng.uniform(0, 2 * np.pi), season=rng.uniform(0.05, 0.3)) for _ in range(n_items)]
    # one chain-wide regular price rise of 3% to 8% per item, at a random week
    rises = [(rng.integers(30, n_periods - 30), rng.uniform(0.03, 0.08)) for _ in range(n_items)]
    # per zone: one store up and one down at random weeks during training, and,
    # with `test_origin`, one up and one down starting at that week (1-based), so
    # a model fitted up to the week before is scored on them
    episodes = []
    for z in range(n_zones):
        stores = rng.permutation(stores_per_zone)
        for k, sign in enumerate((1, -1)):
            start = int(rng.integers(60, (test_origin or n_periods) - episode_weeks - 4))
            episodes.append(dict(zone=z, store=z * stores_per_zone + int(stores[k]), first_week=start + 1,
                                 last_week=start + episode_weeks, log_ratio=sign * np.log(1 + episode_size),
                                 at_origin=False))
            if test_origin:
                episodes.append(dict(zone=z, store=z * stores_per_zone + int(stores[2 + k]), first_week=test_origin,
                                     last_week=min(test_origin + episode_weeks - 1, n_periods),
                                     log_ratio=sign * np.log(1 + episode_size), at_origin=True))
    ep_factor = np.ones((n_zones * stores_per_zone, n_periods))
    for e in episodes:
        ep_factor[e["store"], e["first_week"] - 1:e["last_week"]] = np.exp(e["log_ratio"])

    rows = []
    base_p = {"cut": 0.12, "display": 0.08}
    for s in range(n_zones * stores_per_zone):
        for i, it in enumerate(items):
            # demand shock the model never sees: AR(1) on the log scale
            u = np.zeros(n_periods)
            u[0] = rng.normal(0, shock_sd)
            e = rng.normal(0, shock_sd * np.sqrt(1 - shock_rho ** 2), n_periods)
            for w in range(1, n_periods):
                u[w] = shock_rho * u[w - 1] + e[w]
            z_score = confounding * u / shock_sd
            cut = rng.random(n_periods) < _sigmoid(_logit(base_p["cut"]) + z_score)
            depth = np.where(cut, rng.uniform(0.10, 0.40, n_periods), 0.0)
            promo = np.where(cut, rng.random(n_periods) < 0.6, rng.random(n_periods) < _sigmoid(_logit(0.03) + z_score))
            display = rng.random(n_periods) < _sigmoid(_logit(base_p["display"]) + z_score)
            wk, amt = rises[i]
            regular = it["list_price"] * np.where(weeks >= wk, 1 + amt, 1.0) * ep_factor[s]
            price = regular * (1 - depth)
            season = 1 + it["season"] * np.sin(2 * np.pi * weeks / 52 + it["phase"])
            level = it["size"] * season * np.exp(u) * (regular / it["list_price"]) ** t.regular_elasticity
            frame = pd.DataFrame({
                SERIES: f"S{s:02d}_I{i:02d}", STORE: f"S{s:02d}", ITEM: f"I{i:02d}", DATE: dates,
                PRICE: price.round(4), "promo": promo.astype(int), "display": display.astype(int),
                "true_level": level, "true_regular": regular, "true_zone": s // stores_per_zone,
            })
            frame["true_prev_planned"] = 0
            mean = expected_units(frame, t)
            frame[UNITS] = rng.poisson(mean).astype(float)
            rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel.attrs["episodes"] = episodes
    panel.attrs["truth"] = asdict(t)
    return Dataset("synthetic_levers", panel, [PRICE, "promo", "display"])


def planned(frame: pd.DataFrame) -> np.ndarray:
    """A week is planned when the price is cut or any lever is on."""
    return ((frame[PRICE].to_numpy() < frame["true_regular"].to_numpy() * (1 - 1e-6))
            | (frame["promo"].to_numpy() > 0) | (frame["display"].to_numpy() > 0))


def expected_units(frame: pd.DataFrame, t: Truth | dict) -> np.ndarray:
    """Expected sales under the plan in `frame` (rows of one or more series, any
    order). The dip after a planned week uses the previous row of the same
    series; for a series' first row it uses `true_prev_planned`."""
    t = Truth(**t) if isinstance(t, dict) else t
    f = frame.sort_values([SERIES, DATE])
    on = pd.Series(planned(f), index=f.index)
    prev = on.groupby(f[SERIES]).shift(1)
    first = prev.isna()
    prev = prev.astype(object).where(~first, f["true_prev_planned"].astype(bool)).astype(bool)
    dip = prev & ~on
    m = (f["true_level"] * (f[PRICE] / f["true_regular"]) ** t.cut_elasticity
         * (1 + t.promo_effect) ** f["promo"] * (1 + t.display_effect) ** f["display"]
         * (1 + t.both_effect) ** (f["promo"] * f["display"]) * np.where(dip, 1 - t.dip, 1.0))
    return m.reindex(frame.index).to_numpy()
