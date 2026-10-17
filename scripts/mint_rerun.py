"""Study G (docs/PAPER1_REVISION_PLAN.md, 17 October 2026): ETS reconciliation of
one data set's rolling windows, rerun after the MinT-shrink fix. Needs only the
data (any saved run of that data set gives its windows), not the forecasts.

  python scripts/mint_rerun.py --results <runs folder> --key frat --out <csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_analysis import DAILY, frames_and_forecasts, index  # noqa: E402

from rwm.evaluation import compare as C  # noqa: E402
from rwm.evaluation.reconcile import reconcile_all  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    idx = index(Path(a.results))
    rows = idx[(idx.key == a.key) & (idx.set == "rolling")].head(1)
    fr, fcs = frames_and_forecasts(Path(a.results), rows)
    origins = sorted(next(iter(fcs.values()))["origin"].unique())
    out = []
    for o in origins:
        for name, b in reconcile_all(fr, o, 7 if a.key in DAILY else 52).items():
            for lv, v in C.level_scores(fr, o, b).items():
                out.append({"key": a.key, "set": "rolling", "origin": o, "method": name, "level": lv, "wrmsse": v})
        pd.DataFrame(out).to_csv(a.out, index=False, float_format="%.6g")
        print(f"window {o} done", flush=True)


if __name__ == "__main__":
    main()
