"""The history-length test scored (plan entry of 15 November 2026).

  python scripts/score_history.py --hist <hist runs folder> --final <final runs folder> --out <folder>

For each data set and training length: the world model, LightGBM and their average, WRMSSE at every level on every
rolling window, and the paired item bootstrap of world model minus LightGBM and average minus LightGBM (1,000
resamples, as the paper). The final runs give the length they were trained on (Dominick's 3 years, M5 2 years);
Favorita's three lengths all come from the history runs, on its 2013 load. Writes scores.csv, paired.csv, runs.csv.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.evaluation import compare as C



def runs(folder: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for d in sorted(folder.iterdir()):
        if not (d / "manifest.json").exists() or not (d / "forecasts.csv").exists():
            continue
        name = json.loads((d / "manifest.json").read_text())["config"]["name"]
        m = re.fullmatch(pattern, name)
        if m:
            rows.append({**m.groupdict(), "run": str(d)})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hist", required=True)
    p.add_argument("--final", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = runs(Path(a.hist), r"hist_(?P<key>dominicks_\w{3}|m5|favorita)_(?P<model>world_model|lightgbm)_(?P<length>\d+y)")
    f = runs(Path(a.final), r"final_(?P<key>dominicks_\w{3}|m5)_(?P<model>world_model|lightgbm)")
    f["length"] = f["key"].map(lambda k: "2y" if k == "m5" else "3y")
    idx = pd.concat([h, f]).sort_values("run").groupby(["key", "model", "length"]).tail(1)
    idx.to_csv(out / "runs.csv", index=False)
    scores, paired = [], []
    for key, rows in idx.groupby("key"):
        # one frame per data set: Favorita's from a history run (2013 load), the others' from the final run
        src = rows[rows["run"].str.contains("hist_")] if key == "favorita" else rows[rows["run"].str.contains("final_")]
        if src.empty:
            continue
        cfg, _ = C.load_run(Path(src["run"].iloc[0]))
        fr = C.frame(cfg)
        for length, g in rows.groupby("length"):
            if set(g["model"]) != {"world_model", "lightgbm"}:
                continue
            fcs = {r["model"]: pd.read_csv(Path(r["run"]) / "forecasts.csv") for _, r in g.iterrows()}
            origins = sorted(set(fcs["world_model"]["origin"]) & set(fcs["lightgbm"]["origin"]))
            for o in origins:
                mats = {m: C.forecast_matrix(x, fr, o) for m, x in fcs.items()}
                mats["average"] = (mats["world_model"] + mats["lightgbm"]) / 2
                point = {m: C.level_scores(fr, o, x) for m, x in mats.items()}
                for m, s in point.items():
                    for lv, v in s.items():
                        scores.append(dict(key=key, length=length, origin=o, model=m, level=lv, wrmsse=v))
                bs = C.Bootstrap(fr, o, n=a.boot, seed=o)
                df = C.paired(bs.scores(mats), point, bs.level_names,
                              [("world_model", "lightgbm"), ("average", "lightgbm"), ("average", "world_model")])
                for r in df.to_dict("records"):
                    paired.append(dict(key=key, length=length, origin=o, **r))
            print(key, length, "done", flush=True)
            pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
            pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
