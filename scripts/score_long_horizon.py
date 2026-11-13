"""Score the 52-week test (docs/PAPER1_REVISION_PLAN.md, 13 November 2026).

  python scripts/score_long_horizon.py --results <folder of h52 runs> --out <folder> [--boot 1000]

Reads every run named h52_<key>_<world_model|lightgbm>_<blocks|direct>_w<window>. For each data set, window and
setup, scores the world model, LightGBM and their average over the 52 weeks and over each block of 13 weeks
(91 days on M5), at every level: WRMSSE with the history and revenue weights of the window's start, and forecast over
actual at the bottom level. Paired item bootstrap (as in RQ1) of the differences within the blocks setup, and of the
world model in blocks against LightGBM forecasting directly. Writes scores.csv and paired.csv.
"""
import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.evaluation import compare as C
from rwm.evaluation.splits import Split

NAME = re.compile(r"h52_(dominicks_\w{3}|m5)_(world_model|lightgbm)_(blocks|direct)_w(\d)")


def index(results: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(results.iterdir()):
        if not (d / "manifest.json").exists() or not (d / "forecasts.csv").exists():
            continue
        name = json.loads((d / "manifest.json").read_text())["config"]["name"]
        m = NAME.fullmatch(name)
        if m:
            rows.append(dict(key=m.group(1), model=m.group(2), setup=m.group(3), window=int(m.group(4)), run=d.name))
    idx = pd.DataFrame(rows).sort_values("run")
    return idx.groupby(["key", "model", "setup", "window"]).tail(1).reset_index(drop=True)


def block_view(fr: C.Frame, a: int, b: int) -> C.Frame:
    """The frame with periods a to b of the window as its test window and the history of the window's start:
    the periods of the window before a are left out, so scales and revenue weights are those of the start."""
    t0, t1, _ = fr.cut(0)
    keep = np.r_[np.arange(t0), np.arange(t0 + a, t0 + b)]
    dates = fr.dates[keep]
    sp = Split(origin=0, train_end=pd.Timestamp(fr.dates[t0 - 1]), test_dates=tuple(pd.Timestamp(x) for x in dates[t0:]))
    promo = None if fr.promo is None else fr.promo[:, keep]
    return replace(fr, units=fr.units[:, keep], price=fr.price[:, keep], dates=dates, splits=[sp], promo=promo)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = index(results)
    idx.to_csv(out / "runs.csv", index=False)
    scores, paired = [], []
    for (key, window), rows in idx.groupby(["key", "window"]):
        cfg, _ = C.load_run(results / rows["run"].iloc[0])
        fr = C.frame(cfg)
        mats = {}
        for _, r in rows.iterrows():
            f = pd.read_csv(results / r["run"] / "forecasts.csv")
            mats[f"{r['model']}_{r['setup']}"] = C.forecast_matrix(f, fr, 0)
        for setup in ("blocks", "direct"):
            if f"world_model_{setup}" in mats and f"lightgbm_{setup}" in mats:
                mats[f"average_{setup}"] = (mats[f"world_model_{setup}"] + mats[f"lightgbm_{setup}"]) / 2
        t0, t1, _ = fr.cut(0)
        h = t1 - t0
        q = h // 4
        spans = {"all": (0, h), **{str(i + 1): (i * q, (i + 1) * q) for i in range(4)}}
        for span, (s0, s1) in spans.items():
            view = fr if span == "all" else block_view(fr, s0, s1)
            sub = {m: f[:, s0:s1] for m, f in mats.items()}
            point = {m: C.level_scores(view, 0, f) for m, f in sub.items()}
            act = view.units[:, t0:t0 + (s1 - s0)]
            for m, s in point.items():
                fa = float(np.nansum(np.where(np.isnan(act), np.nan, sub[m])) / max(np.nansum(act), 1e-9))
                for lv, v in s.items():
                    scores.append(dict(key=key, window=window, span=span, model=m, level=lv, wrmsse=v,
                                       forecast_to_actual=fa))
            pairs = [(x, y) for x, y in [("world_model_blocks", "lightgbm_blocks"),
                                         ("average_blocks", "world_model_blocks"),
                                         ("average_blocks", "lightgbm_blocks"),
                                         ("world_model_blocks", "lightgbm_direct"),
                                         ("world_model_direct", "lightgbm_direct")] if x in sub and y in sub]
            bs = C.Bootstrap(view, 0, n=a.boot, seed=window)
            boot = bs.scores({m: sub[m] for m in {m for pr in pairs for m in pr}})
            for r in C.paired(boot, point, bs.level_names, pairs).to_dict("records"):
                paired.append(dict(key=key, window=window, span=span, **r))
            print(key, window, span, flush=True)
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
        pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
