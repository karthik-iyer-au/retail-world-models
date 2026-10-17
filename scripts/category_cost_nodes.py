"""Paper 1 revision, study G (docs/PAPER1_REVISION_PLAN.md, amendment of 17 October 2026):
category-by-store errors per node, so the category-plan cost can be compared
between models with paired uncertainty and under other horizons and weights.

  python scripts/category_cost_nodes.py --results <runs folder> --out <folder> [--keys ...]

Writes nodes_<key>.csv, one row per data set, window, model and category-by-store
node, with the actual and forecast totals needed to rebuild the cost:
  abs_period, actual_window     sum over periods of the absolute error, and the actual total (the reported cost)
  abs_half, actual_half         the same over the first half of the window
  fc_window                     the forecast total over the window (error of the window total)
  abs_week, actual_week_sum     sum over weeks of the absolute weekly error, and of actuals
                                (daily data are summed to weeks of seven days first; weekly data equal abs_period)
  dollars, units_before         the node's dollar sales and units in the weight window before
The reported cost of a model is sum(dollars * abs_period / actual_window) / sum(dollars)
(rwm.evaluation.compare.category_planning_loss).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_analysis import CATLEVEL, DAILY, frames_and_forecasts, index, matrices  # noqa: E402

from rwm.evaluation.hierarchy import _sum_by_group  # noqa: E402


def node_rows(fr, origin: int, fc: np.ndarray, cols, daily: bool) -> pd.DataFrame:
    t0, t1, rev = fr.cut(origin)
    codes = fr.codes(cols)
    g = int(codes.max()) + 1
    a = _sum_by_group(fr.units[:, t0:t1], codes, g)
    f = _sum_by_group(fc, codes, g)
    H = a.shape[1]
    half = max(1, H // 2)
    if daily:
        k = H // 7
        aw = a[:, : 7 * k].reshape(g, k, 7).sum(axis=2)
        fw = f[:, : 7 * k].reshape(g, k, 7).sum(axis=2)
    else:
        aw, fw = a, f
    units_before = _sum_by_group(fr.units[:, t0 - fr.window:t0].sum(axis=1)[:, None], codes, g)[:, 0]
    return pd.DataFrame({
        "node": np.arange(g),
        "abs_period": np.abs(a - f).sum(axis=1), "actual_window": a.sum(axis=1), "fc_window": f.sum(axis=1),
        "abs_half": np.abs(a - f)[:, :half].sum(axis=1), "actual_half": a[:, :half].sum(axis=1),
        "abs_week": np.abs(aw - fw).sum(axis=1), "actual_week_sum": aw.sum(axis=1),
        "dollars": _sum_by_group(rev[:, None], codes, g)[:, 0], "units_before": units_before,
    })


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keys", nargs="*")
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = index(results)
    for key in a.keys or sorted(idx["key"].unique()):
        path = out / f"nodes_{key}.csv"
        if path.exists():
            print(f"{key}: exists, skipped", flush=True)
            continue
        rows = idx[(idx["key"] == key) & (idx["set"] == "rolling")]
        if rows.empty:
            continue
        fr, fcs = frames_and_forecasts(results, rows)
        catlevel = CATLEVEL.get(key, CATLEVEL["default"])
        cols = next(c for c in fr.levels if "x".join(c) == catlevel)
        parts = []
        for o in sorted({o for f in fcs.values() for o in f["origin"].unique()}):
            for m, f in matrices(fr, fcs, o).items():
                parts.append(node_rows(fr, o, f, cols, key in DAILY).assign(key=key, origin=o, model=m))
        pd.concat(parts).to_csv(path, index=False, float_format="%.6g")
        print(f"{key}: {len(parts)} model-windows written", flush=True)


if __name__ == "__main__":
    main()
