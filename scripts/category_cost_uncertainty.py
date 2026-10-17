"""Paper 1 revision, study G (docs/PAPER1_REVISION_PLAN.md, amendment of 17 October 2026):
paired uncertainty and sensitivity of the category-plan cost, from the per-node
files of scripts/category_cost_nodes.py.

  python scripts/category_cost_uncertainty.py --nodes <folder of nodes_*.csv> --out <tables folder> [--boot 5000]

Writes category_cost.csv: for each data set, variant and model, the cost (mean over
windows), the world model's cost minus that model's with a 95% interval from a
bootstrap over category-by-store nodes within each window (the same nodes drawn
for both models), and the number of windows in which the world model's cost is
lower. Variants:
  reported      sum of period errors over the window, nodes weighted by dollar sales (the paper's table)
  first_half    the same over the first half of the window
  window_total  error of the window total
  weekly        sum of weekly errors (daily data summed to weeks; equals reported on weekly data)
  units_weight  reported error, nodes weighted by units sold before the window
  equal_weight  reported error, every node weighted equally
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan"]


def errors_and_weights(d: pd.DataFrame, variant: str) -> tuple[pd.Series, pd.Series]:
    den = lambda s: np.maximum(s, 1e-9)  # noqa: E731
    err = {
        "reported": d.abs_period / den(d.actual_window),
        "first_half": d.abs_half / den(d.actual_half),
        "window_total": (d.actual_window - d.fc_window).abs() / den(d.actual_window),
        "weekly": d.abs_week / den(d.actual_week_sum),
        "units_weight": d.abs_period / den(d.actual_window),
        "equal_weight": d.abs_period / den(d.actual_window),
    }[variant]
    w = {"units_weight": d.units_before, "equal_weight": pd.Series(1.0, index=d.index)}.get(variant, d.dollars)
    return err, w


VARIANTS = ["reported", "first_half", "window_total", "weekly", "units_weight", "equal_weight"]


def summarise(d: pd.DataFrame, boot: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for key, g in d.groupby("key"):
        models = [m for m in MODELS if m in set(g.model)]
        origins = sorted(g.origin.unique())
        for variant in VARIANTS:
            # per window: node x model matrices of weight * error and of weight
            cost, draws = {m: [] for m in models}, {m: np.zeros(boot) for m in models}
            for o in origins:
                w_ = g[g.origin == o]
                e, w = errors_and_weights(w_, variant)
                we = (w * e).groupby([w_.model, w_.node]).sum().unstack(0)[models]
                ww = w.groupby([w_.model, w_.node]).sum().unstack(0)[models]
                n = len(we)
                idx = rng.integers(0, n, size=(boot, n))
                counts = np.apply_along_axis(np.bincount, 1, idx, minlength=n)  # boot x nodes
                for m in models:
                    cost[m].append(we[m].sum() / ww[m].sum())
                    draws[m] += (counts @ we[m].to_numpy()) / (counts @ ww[m].to_numpy())
            for m in models:
                c = np.array(cost[m])
                row = {"key": key, "variant": variant, "model": m, "cost": c.mean(), "windows": len(origins)}
                if m != "world_model" and "world_model" in models:
                    cw = np.array(cost["world_model"])
                    diff = (draws["world_model"] - draws[m]) / len(origins)
                    row.update(world_minus=(cw - c).mean(), lo=np.quantile(diff, 0.025), hi=np.quantile(diff, 0.975),
                               world_lower_windows=int((cw < c).sum()))
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=5000)
    a = p.parse_args(argv)
    d = pd.concat([pd.read_csv(f) for f in sorted(Path(a.nodes).glob("nodes_*.csv"))])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summarise(d[d.model.isin(MODELS)], a.boot).to_csv(out / "category_cost.csv", index=False, float_format="%.5g")
    print("done", flush=True)


if __name__ == "__main__":
    main()
