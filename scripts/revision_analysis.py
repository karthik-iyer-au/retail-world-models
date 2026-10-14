"""Paper 1 revision analyses on saved runs (docs/PAPER1_REVISION_PLAN.md).

  python scripts/revision_analysis.py --results <runs folder> --out results/paper1_revision [--parts ...] [--boot 1000]

Parts (all by default): scores, paired, mcs, intervals, decisions, reconcile.
Writes one small CSV per part; a part whose CSV exists is skipped, so the
script can be started again after a dropped runtime.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.evaluation import compare as C

MODELS = ["world_model", "lift", "lightgbm", "seasonal_naive", "timesfm_mean", "timesfm_plan", "chronos", "tft",
          "graph", "ensemble"]
ENSEMBLE = ["world_model", "lightgbm", "timesfm_plan"]
PAIRS_WITH = ["world_model", "lightgbm"]
CATLEVEL = {"m5": "store_idxcat_id", "default": "cat_idxstore_id"}


def label(config_name: str) -> tuple[str, str, str] | None:
    """(data key, window set, model) for a run worth analysing, from its config name."""
    n = re.sub(r"_seed\d+$", "", config_name)
    m = re.match(r"rev_(dominicks_\w{3}|frat|m5)_(rolling|finalwindow)_(\w+)$", n)
    if m:
        return m.group(1), "rolling" if m.group(2) == "rolling" else "final", m.group(3)
    finals = {
        r"dominicks_(\w{3})_final_world_model_switching_avg3_roll_norm": "world_model",
        r"dominicks_(\w{3})_final_lightgbm": "lightgbm", r"dominicks_(\w{3})_final_timesfm_mean": "timesfm_mean",
        r"dominicks_(\w{3})_final_timesfm_plan": "timesfm_plan",
        r"frat_final_world_model": "world_model", r"frat_final_lift": "lift", r"frat_final_lightgbm": "lightgbm",
        r"frat_final_timesfm_mean": "timesfm_mean", r"frat_final_timesfm_plan": "timesfm_plan",
        r"m5_test_world_model_roll_norm": "world_model", r"m5_test_lift": "lift", r"m5_test_lightgbm": "lightgbm",
        r"m5_test_timesfm_mean": "timesfm_mean", r"m5_test_timesfm_plan": "timesfm_plan", r"m5_test_graph": "graph",
        r"m5_test_seasonal_naive": "seasonal_naive",
    }
    for pat, model in finals.items():
        mm = re.fullmatch(pat, n)
        if mm:
            key = f"dominicks_{mm.group(1)}" if n.startswith("dominicks") else ("frat" if n.startswith("frat") else "m5")
            return key, "final", model
    return None


def index(results: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(results.iterdir()):
        if not (d / "manifest.json").exists() or not (d / "forecasts.csv").exists():
            continue
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        lab = label(cfg["name"])
        if lab is None:
            continue
        seed = re.search(r"_seed(\d+)$", cfg["name"])
        rows.append({"key": lab[0], "set": lab[1], "model": lab[2], "seed": int(seed.group(1)) if seed else 0,
                     "run": d.name, "config": cfg["name"]})
    idx = pd.DataFrame(rows)
    # one run per model: the lowest seed (seed 1 where seeds were run), the latest run if repeated
    idx = idx.sort_values(["key", "set", "model", "seed", "run"]).groupby(["key", "set", "model"]).head(1)
    return idx.reset_index(drop=True)


def frames_and_forecasts(results: Path, rows: pd.DataFrame):
    """The data frame of this key and set (from any of its runs) and every model's forecasts."""
    cfg, _ = C.load_run(results / rows["run"].iloc[0])
    fr = C.frame(cfg)
    fcs = {r["model"]: pd.read_csv(results / r["run"] / "forecasts.csv") for _, r in rows.iterrows()}
    return fr, fcs


def matrices(fr, fcs, origin):
    mats = {m: C.forecast_matrix(f, fr, origin) for m, f in fcs.items() if origin in set(f["origin"])}
    if all(m in mats for m in ENSEMBLE):
        mats["ensemble"] = np.mean([mats[m] for m in ENSEMBLE], axis=0)
    return mats


def save(acc: dict, out: Path) -> None:
    for part, rowsl in acc.items():
        if rowsl and part != "mcs":
            pd.DataFrame(rowsl).to_csv(out / f"{part}.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parts", nargs="*", default=["scores", "paired", "mcs", "intervals", "decisions", "reconcile"])
    p.add_argument("--keys", nargs="*")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = index(results)
    idx.to_csv(out / "runs.csv", index=False)
    keys = a.keys or sorted(idx["key"].unique())
    acc = {part: [] for part in a.parts}
    for key in keys:
        for wset in ("final", "rolling"):
            rows = idx[(idx["key"] == key) & (idx["set"] == wset)]
            if rows.empty:
                continue
            t_start = time.time()
            fr, fcs = frames_and_forecasts(results, rows)
            origins = sorted({o for f in fcs.values() for o in f["origin"].unique()})
            catlevel = CATLEVEL["m5" if key == "m5" else "default"]
            catcols = next(c for c in fr.levels if "x".join(c) == catlevel)
            bottom = fr.levels[-1]
            season = 7 if key == "m5" else 52
            errors = {}  # model -> level -> list of scaled errors from earlier windows
            for o in origins:
                mats = matrices(fr, fcs, o)
                point = {m: C.level_scores(fr, o, f) for m, f in mats.items()}
                base = dict(key=key, set=wset, origin=o)
                if "scores" in acc:
                    for m, s in point.items():
                        for lv, v in s.items():
                            acc["scores"].append({**base, "model": m, "level": lv, "wrmsse": v})
                if "paired" in acc:
                    bs = C.Bootstrap(fr, o, n=a.boot, seed=o)
                    boot = bs.scores(mats)
                    pairs = {tuple(sorted((x, y))) for x in PAIRS_WITH for y in mats if x != y and x in mats}
                    df = C.paired(boot, point, bs.level_names, sorted(pairs))
                    for r in df.to_dict("records"):
                        acc["paired"].append({**base, **r})
                if "intervals" in acc or "decisions" in acc:
                    for m, f in mats.items():
                        for cols in fr.levels:
                            lv = "x".join(cols) if cols else "total"
                            prior = errors.get(m, {}).get(lv, [])
                            if prior and "intervals" in acc:
                                q = C.conformal_quantiles(prior)
                                acc["intervals"].append({**base, "model": m, "level": lv, "windows_used": len(prior),
                                                         **C.interval_scores(fr, o, f, q, cols)})
                            if prior and "decisions" in acc and cols == bottom:
                                q = C.conformal_quantiles(prior)
                                for ratio in (0.5, 0.8, 0.95):
                                    acc["decisions"].append({**base, "model": m, "decision": f"replenish_{ratio}",
                                                             "cost": C.replenishment_cost(fr, o, f, q, ratio)})
                            errors.setdefault(m, {}).setdefault(lv, []).append(C.scaled_errors(fr, o, f, cols))
                        if "decisions" in acc:
                            acc["decisions"].append({**base, "model": m, "decision": "category_planning",
                                                     "cost": C.category_planning_loss(fr, o, f, catcols)})
                        if "intervals" in acc and m in fcs:
                            nat = C.native_scores(fr, o, fcs[m])
                            if nat:
                                acc["intervals"].append({**base, "model": m, "level": "native_bottom", **nat})
                if "reconcile" in acc:
                    from rwm.evaluation.reconcile import reconcile_all
                    recs = reconcile_all(fr, o, season, {m: mats[m] for m in ("world_model", "lightgbm") if m in mats})
                    for name, b in recs.items():
                        for lv, v in C.level_scores(fr, o, b).items():
                            acc["reconcile"].append({**base, "method": name, "level": lv, "wrmsse": v})
                save(acc, out)  # after every window, so a lost runtime costs one window
                print(f"{key} {wset} window {o} done, {time.time() - t_start:.0f} s", flush=True)
            print(f"{key} {wset}: {len(origins)} windows, {len(fcs)} models, {time.time() - t_start:.0f} s", flush=True)
    if "mcs" in a.parts and (out / "scores.csv").exists():
        sc = pd.read_csv(out / "scores.csv")
        mcs = []
        for (key, lv), g in sc[sc["set"] == "rolling"].groupby(["key", "level"]):
            wide = g.pivot_table(index="origin", columns="model", values="wrmsse").dropna(axis=1)
            if len(wide) >= 3 and wide.shape[1] >= 2:
                kept = C.model_confidence_set(wide)
                for m in wide.columns:
                    mcs.append({"key": key, "level": lv, "model": m, "in_set": m in kept,
                                "mean_wrmsse": wide[m].mean(), "windows": len(wide)})
        pd.DataFrame(mcs).to_csv(out / "mcs.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
