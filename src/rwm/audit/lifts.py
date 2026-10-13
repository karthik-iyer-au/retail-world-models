"""Checks C1 to C6 of the paper 2 plan, for any model with `predict(future)`.

Definitions used throughout:
- The plan as made: the future frame as given (prices and levers as recorded).
- Nothing planned: each item at its regular price (its highest price over the
  last `price_window` training periods, the model's own rule) and every lever
  off.
- A lever's contribution: forecast with the plan as made minus forecast with
  that lever switched off (for price: set to the regular price) and the rest
  as made. Contributions need not add up to the total lift; the shortfall is
  the interaction between levers, reported as such.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rwm.data.schema import DATE, PRICE, SERIES, UNITS


def regular_prices(train: pd.DataFrame, price_window: int) -> pd.Series:
    """Each series' highest price over its last `price_window` periods of training."""
    dates = np.sort(train[DATE].unique())[-price_window:]
    recent = train[train[DATE].isin(dates)]
    return recent.groupby(SERIES)[PRICE].max()


def off_plan(future: pd.DataFrame, regular: pd.Series, levers: list[str], price: bool = True) -> pd.DataFrame:
    """The same future with nothing planned."""
    off = future.drop(columns=[UNITS], errors="ignore").copy()
    for lever in levers:
        off[lever] = 0
    if price:
        reg = off[SERIES].map(regular)
        off[PRICE] = np.where(reg.notna(), reg, off[PRICE])
    return off


def lever_contributions(model, future: pd.DataFrame, regular: pd.Series, levers: list[str],
                        price: bool = True) -> pd.DataFrame:
    """Forecast as planned, forecast with nothing planned, and each lever's
    on/off contribution, one row per row of `future`."""
    plan = future.drop(columns=[UNITS], errors="ignore")
    fc = model.predict(plan)
    out = pd.DataFrame({"forecast": fc}, index=future.index)
    out["nothing_planned"] = model.predict(off_plan(plan, regular, levers, price))
    out["total_lift"] = out["forecast"] - out["nothing_planned"]
    for lever in levers:
        out[f"lift_{lever}"] = fc - model.predict(plan.assign(**{lever: 0}))
    if price:
        reg = plan[SERIES].map(regular)
        out["lift_price"] = fc - model.predict(plan.assign(**{PRICE: np.where(reg.notna(), reg, plan[PRICE])}))
    parts = [c for c in out if c.startswith("lift_")]
    out["interaction"] = out["total_lift"] - out[parts].sum(axis=1)
    return out


def construction_checks(model, future: pd.DataFrame, regular: pd.Series, levers: list[str]) -> dict:
    """C1. Largest violation of each property, relative to the largest forecast."""
    plan = future.drop(columns=[UNITS], errors="ignore")
    on, off = model.breakdown(plan), model.breakdown(off_plan(plan, regular, levers))
    top = max(float(on["forecast"].max()), 1e-9)
    return {
        "zero_lift_when_nothing_planned": float(np.abs(off["lift"]).max() / top),
        "baseline_moves_with_plan": float(np.abs(on["baseline"] - off["baseline"]).max() / top),
        "parts_do_not_add_up": float(np.abs(on["baseline"] + on["lift"] - on["forecast"]).max() / top),
    }


PARTS = {  # the network parts re-initialised in turn, from the output back to the input
    "lift readout": ["lift"],
    "baseline readout": ["head"],
    "transition": ["cell", "roll_norm", "effect", "effect_in"],
    "encoder": ["inp", "pos", "encoder", "norm", "emb", "step"],
}


def randomised_copies(model, seed: int = 0):
    """C2. Copies of a fitted StateModel with its parts re-initialised one after
    another (cascading), as in the model-randomisation sanity check of Adebayo
    et al. (2018). Yields (part name, copy)."""
    import torch

    torch.manual_seed(seed)
    current = copy.deepcopy(model)
    for name, attrs in PARTS.items():
        touched = False
        for a in attrs:
            mod = getattr(current._net, a, None)
            if mod is None:
                continue
            touched = True
            if isinstance(mod, torch.nn.Parameter):
                with torch.no_grad():
                    mod.normal_(0, 0.02)
                continue
            for m in mod.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        if touched:
            yield name, copy.deepcopy(current)


def rank_agreement(a, b) -> float:
    """Spearman correlation over rows where either value is non-zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = (np.abs(a) > 0) | (np.abs(b) > 0)
    keep &= np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


def price_monotonicity(model, future: pd.DataFrame, cuts=(0.05, 0.10, 0.20, 0.30), tol: float = 1e-3) -> dict:
    """C4. Share of rows where a deeper price cut gives a lower forecast than the
    shallower one before it (by more than `tol` of the forecast)."""
    plan = future.drop(columns=[UNITS], errors="ignore")
    prev = model.predict(plan)
    worse, n = 0, 0
    for c in cuts:
        cur = model.predict(plan.assign(**{PRICE: plan[PRICE] * (1 - c)}))
        drop = cur < prev - tol * np.maximum(prev, 1e-9)
        worse += int(drop.sum())
        n += len(cur)
        prev = cur
    return {"share_lower_after_deeper_cut": worse / max(n, 1)}


def lift_vs_actual(forecast, baseline, actual, mask) -> dict:
    """C5. On the rows in `mask` (for example promoted item-weeks): predicted
    lift (forecast over the model's own baseline) against observed lift (sales
    over the same baseline). Bias is summed forecast over summed sales."""
    f, b, y, m = (np.asarray(v, float) for v in (forecast, baseline, actual, mask))
    m = m.astype(bool) & np.isfinite(y) & (b > 0)
    if m.sum() == 0:
        return {"n": 0, "bias": float("nan"), "rank": float("nan")}
    pred, obs = f[m] / b[m], y[m] / b[m]
    return {"n": int(m.sum()), "bias": float(f[m].sum() / max(y[m].sum(), 1e-9)),
            "rank": rank_agreement(pred, obs)}
