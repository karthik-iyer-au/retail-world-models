"""Paper 1 revision, study A of the amendment of 15 October 2026
(docs/PAPER1_REVISION_PLAN.md): what the world model adds to the equal-weight
ensemble, on saved forecasts of the rolling windows only.

  python scripts/ensemble_ablation.py --results <runs folder> --out <folder> [--keys ...] [--boot 1000]
  python scripts/ensemble_ablation.py --tables <folder> --out <tables folder>

Writes, per data set and window (saved after every window, so a lost runtime
costs one window, and windows already in the CSVs are skipped):
  scores.csv   WRMSSE by level of every variant
  grid.csv     WRMSSE by level on the weight grid (steps of 0.05 over the three weights, and equal weights)
  weights.csv  the weights chosen from the earlier windows, by target
  paired.csv   variant minus the ensemble, with the 95% item-bootstrap interval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_analysis import frames_and_forecasts, index  # noqa: E402
from revision_tables import paired as window_paired  # noqa: E402

from rwm.evaluation import compare as C  # noqa: E402

MEMBERS = ["world_model", "lightgbm", "timesfm_plan"]
VARIANTS = {
    "ensemble": ["world_model", "lightgbm", "timesfm_plan"],
    "without_world_model": ["lightgbm", "timesfm_plan"],
    "without_lightgbm": ["world_model", "timesfm_plan"],
    "without_timesfm": ["world_model", "lightgbm"],
    "lift_for_world_model": ["lift", "lightgbm", "timesfm_plan"],
}
TARGETS = ["bottom", "total", "all_levels"]
STEP = 0.05


def grid(step: float = STEP) -> np.ndarray:
    """Every (w_world_model, w_lightgbm, w_timesfm) on the simplex in steps of
    `step`, and the equal weights (not on a grid of twentieths) as the last point."""
    k = int(round(1 / step))
    pts = [(i / k, j / k, (k - i - j) / k) for i in range(k + 1) for j in range(k + 1 - i)]
    return np.array(pts + [(1 / 3, 1 / 3, 1 / 3)])


def variants(mats: dict) -> dict:
    out = {}
    for name, members in VARIANTS.items():
        if all(m in mats for m in members):
            out[name] = np.mean([mats[m] for m in members], axis=0)
    return out


def target_value(g: pd.DataFrame, target: str, bottom: str) -> pd.Series:
    """Mean WRMSSE per grid point over the rows in `g`, for one target."""
    if target == "bottom":
        g = g[g["level"] == bottom]
    elif target == "total":
        g = g[g["level"] == "total"]
    return g.groupby("point")["wrmsse"].mean()


def choose(prior: pd.DataFrame, target: str, bottom: str, pts: np.ndarray) -> np.ndarray:
    """The grid weights with the lowest mean WRMSSE over earlier windows; ties go to the lowest point index."""
    v = target_value(prior, target, bottom)
    return pts[int(v.idxmin())]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results")
    p.add_argument("--out", required=True)
    p.add_argument("--keys", nargs="*")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--tables")
    a = p.parse_args(argv)
    if a.tables:
        return tables(Path(a.tables), Path(a.out))
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = index(results)
    idx = idx[idx["set"] == "rolling"]
    parts = ("scores", "grid", "weights", "paired")
    acc = {k: (pd.read_csv(out / f"{k}.csv").to_dict("records") if (out / f"{k}.csv").exists() else []) for k in parts}
    done = {(r["key"], r["origin"]) for r in acc["paired"]}
    pts = grid()
    for key in a.keys or sorted(idx["key"].unique()):
        rows = idx[idx["key"] == key]
        if not set(MEMBERS) <= set(rows["model"]):
            print(f"{key}: members missing, skipped", flush=True)
            continue
        t_start = time.time()
        fr, fcs = frames_and_forecasts(results, rows)
        bottom = "x".join(fr.levels[-1])
        origins = sorted(set.intersection(*[set(fcs[m]["origin"]) for m in MEMBERS]))
        for o in origins:
            if (key, o) in done:
                continue
            mats = {m: C.forecast_matrix(f, fr, o) for m, f in fcs.items() if o in set(f["origin"])}
            # the weight grid; the curve of the world model's weight is the part with equal weights on the other two
            for i, w in enumerate(pts):
                fc = w[0] * mats["world_model"] + w[1] * mats["lightgbm"] + w[2] * mats["timesfm_plan"]
                for lv, v in C.level_scores(fr, o, fc).items():
                    acc["grid"].append({"key": key, "origin": o, "point": i, "w_world_model": w[0],
                                        "w_lightgbm": w[1], "w_timesfm_plan": w[2], "level": lv, "wrmsse": v})
            ens = variants(mats)
            gr = pd.DataFrame(acc["grid"])
            prior = gr[(gr["key"] == key) & (gr["origin"] < o)]
            if len(prior):
                for t in TARGETS:
                    w = choose(prior, t, bottom, pts)
                    acc["weights"].append({"key": key, "origin": o, "target": t, "w_world_model": w[0],
                                           "w_lightgbm": w[1], "w_timesfm_plan": w[2],
                                           "earlier_windows": int(prior["origin"].nunique())})
                    ens[f"fitted_{t}"] = w[0] * mats["world_model"] + w[1] * mats["lightgbm"] + w[2] * mats["timesfm_plan"]
            point = {m: C.level_scores(fr, o, f) for m, f in ens.items()}
            for m, s in point.items():
                for lv, v in s.items():
                    acc["scores"].append({"key": key, "origin": o, "model": m, "level": lv, "wrmsse": v})
            bs = C.Bootstrap(fr, o, n=a.boot, seed=o)
            boot = bs.scores(ens)
            pairs = [(m, "ensemble") for m in ens if m != "ensemble"]
            for r in C.paired(boot, point, bs.level_names, pairs).to_dict("records"):
                acc["paired"].append({"key": key, "origin": o, **r})
            for k in parts:
                if acc[k]:
                    pd.DataFrame(acc[k]).to_csv(out / f"{k}.csv", index=False)
            print(f"{key} window {o} done, {time.time() - t_start:.0f} s", flush=True)
    print("done", flush=True)


def tables(src: Path, out: Path) -> None:
    """The small tables the paper reads, from the per-window CSVs."""
    out.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(src / "scores.csv")
    acc = sc.groupby(["key", "model", "level"]).agg(wrmsse=("wrmsse", "mean"), windows=("wrmsse", "size")).reset_index()
    acc.to_csv(out / "ablation_accuracy.csv", index=False, float_format="%.5g")
    pr = pd.read_csv(src / "paired.csv").assign(set="rolling")
    window_paired(pr).drop(columns="set").to_csv(out / "ablation_paired.csv", index=False, float_format="%.5g")
    # fitted weights are scored from the second window on; compare them with the ensemble on the same windows
    wt = pd.read_csv(src / "weights.csv")
    wt.to_csv(out / "ablation_weights.csv", index=False, float_format="%.5g")
    gr = pd.read_csv(src / "grid.csv")
    sweep = gr[np.isclose(gr["w_lightgbm"], gr["w_timesfm_plan"])]
    sweep.groupby(["key", "level", "w_world_model"])["wrmsse"].mean().reset_index().to_csv(
        out / "ablation_sweep.csv", index=False, float_format="%.5g")
    mcs = []
    for (key, lv), g in sc.groupby(["key", "level"]):
        wide = g[g["model"].isin(VARIANTS)].pivot_table(index="origin", columns="model", values="wrmsse").dropna(axis=1)
        if len(wide) >= 3 and wide.shape[1] >= 2:
            kept = C.model_confidence_set(wide)
            for m in wide.columns:
                mcs.append({"key": key, "level": lv, "model": m, "in_set": m in kept, "windows": len(wide)})
    pd.DataFrame(mcs).to_csv(out / "ablation_mcs.csv", index=False)
    print({f.name: len(pd.read_csv(f)) for f in sorted(out.glob("ablation_*.csv"))})


if __name__ == "__main__":
    main()
