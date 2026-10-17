"""Paper 1 revision, study G (docs/PAPER1_REVISION_PLAN.md, amendment of 17 October 2026):
paired uncertainty and sensitivity of the category-plan cost, from the per-node
files of scripts/category_cost_nodes.py.

  python scripts/category_cost_uncertainty.py --nodes <folder of nodes_*.csv> --out <tables folder> [--boot 5000]

The cost first reported divided each category-by-store node's error by its sales
in the window, floored at 1e-9, so a node that sold nothing in a window (a store
closed or not reporting) entered at about 10^9 times its error; on Dominick's
those nodes decided the result. The plan defines the cost as the absolute error
of category-by-store totals weighted by dollar sales. It is computed here as the
error in dollars over sales in dollars: each node's units are valued at its
average price in the weight window before the forecast origin,
  cost = sum_n price_n * err_n / sum_n price_n * sales_n,
which is defined whenever the data set sells anything in the window.

Writes category_cost.csv: for each data set, variant and model, the cost (mean over
windows), the world model's cost minus that model's with a 95% interval from a
bootstrap over category-by-store nodes within each window (the same nodes drawn
for both models), and the number of windows in which the world model's cost is
lower; and category_cost_nodes_zero.csv, the nodes with no sales in a window.
Variants:
  dollars          error of each period's node total, in dollars (the corrected cost)
  first_half       the same over the first half of the window
  window_total     error of the node's total over the whole window, in dollars
  weekly           error of weekly node totals (daily data summed to weeks), in dollars
  units            each unit of error counted alike, in place of its dollar value
  node_mean        the mean over nodes that sold in the window of each node's error over its sales
  as_first_run     the first, flawed cost, for the record
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan", "seasonal_naive"]
VARIANTS = ["dollars", "first_half", "window_total", "weekly", "units", "node_mean", "as_first_run"]


def num_den(d: pd.DataFrame, variant: str) -> tuple[pd.Series, pd.Series]:
    """Per-node numerator and denominator; a model's cost is sum(num) / sum(den)."""
    price = (d.dollars / d.units_before.where(d.units_before > 0)).fillna(0.0)
    sold = d.actual_window > 0
    if variant == "dollars":
        return price * d.abs_period, price * d.actual_window
    if variant == "first_half":
        return price * d.abs_half, price * d.actual_half
    if variant == "window_total":
        return price * (d.actual_window - d.fc_window).abs(), price * d.actual_window
    if variant == "weekly":
        return price * d.abs_week, price * d.actual_week_sum
    if variant == "units":
        return d.abs_period, d.actual_window
    if variant == "node_mean":
        return (d.abs_period / d.actual_window.where(sold)).fillna(0.0), sold.astype(float)
    if variant == "as_first_run":
        return d.dollars * d.abs_period / np.maximum(d.actual_window, 1e-9), d.dollars
    raise ValueError(variant)


def summarise(d: pd.DataFrame, boot: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for key, g in d.groupby("key"):
        models = [m for m in MODELS if m in set(g.model)]
        origins = sorted(g.origin.unique())
        for variant in VARIANTS:
            cost, draws = {m: [] for m in models}, {m: np.zeros(boot) for m in models}
            for o in origins:
                w_ = g[g.origin == o]
                n_, d_ = num_den(w_, variant)
                num = n_.groupby([w_.model, w_.node]).sum().unstack(0)[models]
                den = d_.groupby([w_.model, w_.node]).sum().unstack(0)[models]
                n = len(num)
                idx = rng.integers(0, n, size=(boot, n))
                counts = np.apply_along_axis(np.bincount, 1, idx, minlength=n)  # boot x nodes
                for m in models:
                    cost[m].append(num[m].sum() / den[m].sum())
                    draws[m] += (counts @ num[m].to_numpy()) / np.maximum(counts @ den[m].to_numpy(), 1e-12)
            for m in models:
                c = np.array(cost[m])
                row = {"key": key, "variant": variant, "model": m, "cost": c.mean(), "windows": len(origins)}
                if "seasonal_naive" in models:
                    row["ratio_to_naive"] = c.mean() / np.mean(cost["seasonal_naive"])
                if m != "world_model" and "world_model" in models:
                    cw = np.array(cost["world_model"])
                    diff = (draws["world_model"] - draws[m]) / len(origins)
                    row.update(world_minus=(cw - c).mean(), lo=np.quantile(diff, 0.025), hi=np.quantile(diff, 0.975),
                               world_lower_windows=int((cw < c).sum()))
                rows.append(row)
    return pd.DataFrame(rows)


def zero_nodes(d: pd.DataFrame) -> pd.DataFrame:
    """Category-by-store nodes with no sales in a window but dollar sales before it."""
    one = d.drop_duplicates(["key", "origin", "node"])
    z = one[(one.actual_window <= 0) & (one.dollars > 0)]
    tot = one.groupby("key").size().rename("node_windows")
    return (z.groupby("key").size().rename("zero_node_windows").to_frame().join(tot, how="right").fillna(0)
            .astype(int).reset_index())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=5000)
    a = p.parse_args(argv)
    d = pd.concat([pd.read_csv(f) for f in sorted(Path(a.nodes).glob("nodes_*.csv"))])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zero_nodes(d).to_csv(out / "category_cost_nodes_zero.csv", index=False)
    summarise(d[d.model.isin(MODELS)], a.boot).to_csv(out / "category_cost.csv", index=False, float_format="%.5g")
    print("done", flush=True)


if __name__ == "__main__":
    main()
