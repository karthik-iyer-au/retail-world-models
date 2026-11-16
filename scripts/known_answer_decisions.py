"""Paper 2: what a wrong lift costs a planner, in the known-answer worlds (plan amendment of 16 November 2026).

  python scripts/known_answer_decisions.py --known <folder of *_lifts.csv.gz> --out <csv>

From each world's and model's saved lifts (no model is refitted), on the planned item-weeks:
  choice: keep the half of planned item-weeks with the largest stated total lift; the true lift of that half over the
          true lift of the half with the largest true lift (1 is as good as knowing the truth). A random half is
          reported once per world (mean of 1,000 draws).
  stock:  the absolute difference between stated and true total lift, summed over planned item-weeks and divided by
          the summed true lift.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def score(stated: np.ndarray, true: np.ndarray) -> dict:
    k = len(true) // 2
    best = np.sort(true)[::-1][:k].sum()
    chosen = true[np.argsort(-stated, kind="stable")[:k]].sum()
    return {"choice": chosen / best if best else np.nan,
            "stock": np.abs(stated - true).sum() / true.sum() if true.sum() else np.nan,
            "stated_over_true": stated.sum() / true.sum() if true.sum() else np.nan, "planned": len(true)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--known", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows, worlds = [], {}
    for folder in a.known:
        for f in sorted(Path(folder).glob("*_lifts.csv.gz")):
            m = re.match(r"(.+?)_(world_\w+|lightgbm\w*)_lifts\.csv\.gz", f.name)
            if not m:
                continue
            world, model = m.groups()
            d = pd.read_csv(f)
            d = d[d["planned"]]
            true = d["true_total_lift"].to_numpy()
            rows.append({"world": world, "model": model, **score(d["total_lift"].to_numpy(), true)})
            if world not in worlds:
                rng = np.random.default_rng(0)
                k, best = len(true) // 2, np.sort(true)[::-1][: len(true) // 2].sum()
                worlds[world] = np.mean([true[rng.permutation(len(true))[:k]].sum() / best for _ in range(1000)])
    out = pd.DataFrame(rows)
    out["choice_random"] = out["world"].map(worlds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
