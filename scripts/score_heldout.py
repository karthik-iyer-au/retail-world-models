"""The held-out periods scored (plan entry of 15 November 2026).

  python scripts/score_heldout.py --results <heldout runs folder> --out <folder>

For each data set: the final world model, LightGBM and their average on the held-out windows, WRMSSE at every level
per window, and the paired item bootstrap of each pair (1,000 resamples), as the paper. On M5 the last window is the
competition's final 28 days. Writes scores.csv, paired.csv and runs.csv.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from rwm.evaluation import compare as C

NAME = re.compile(r"heldout_(?P<key>dominicks_\w{3}|frat|m5)_(?P<model>world_model|lightgbm)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(res.iterdir()):
        if (d / "manifest.json").exists() and (d / "forecasts.csv").exists():
            m = NAME.fullmatch(json.loads((d / "manifest.json").read_text())["config"]["name"])
            if m:
                rows.append({**m.groupdict(), "run": d.name})
    idx = pd.DataFrame(rows).sort_values("run").groupby(["key", "model"]).tail(1)
    idx.to_csv(out / "runs.csv", index=False)
    scores, paired = [], []
    for key, g in idx.groupby("key"):
        if set(g["model"]) != {"world_model", "lightgbm"}:
            continue
        cfg, _ = C.load_run(res / g["run"].iloc[0])
        fr = C.frame(cfg)
        fcs = {r["model"]: pd.read_csv(res / r["run"] / "forecasts.csv") for _, r in g.iterrows()}
        last = max(s.origin for s in fr.splits)
        for o in sorted(set(fcs["world_model"]["origin"]) & set(fcs["lightgbm"]["origin"])):
            mats = {m: C.forecast_matrix(f, fr, o) for m, f in fcs.items()}
            mats["average"] = (mats["world_model"] + mats["lightgbm"]) / 2
            point = {m: C.level_scores(fr, o, f) for m, f in mats.items()}
            tag = "competition final 28 days" if key == "m5" and o == last else "held out"
            for m, s in point.items():
                for lv, v in s.items():
                    scores.append(dict(key=key, origin=o, window=tag, model=m, level=lv, wrmsse=v))
            bs = C.Bootstrap(fr, o, n=a.boot, seed=o)
            df = C.paired(bs.scores(mats), point, bs.level_names,
                          [("average", "lightgbm"), ("average", "world_model"), ("world_model", "lightgbm")])
            for r in df.to_dict("records"):
                paired.append(dict(key=key, origin=o, window=tag, **r))
        print(key, "done", flush=True)
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
        pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
