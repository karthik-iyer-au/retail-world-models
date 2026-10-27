"""Study K result (docs/PAPER1_REVISION_PLAN.md, entry of 27 October 2026,
exploratory): failures and scores of each world-model variant by window.

  python scripts/reliability_analysis.py <folder holding the study K run folders>

Writes results/paper1_revision/tables/reliability.csv (one row per data set,
variant, window and seed) and prints the summary the plan names.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper1_revision" / "tables"
NAME = re.compile(r"reliability_dominicks_(?P<cat>[a-z]+)_(?P<variant>.+)_seed(?P<seed>\d+)$")


def main(folder: str) -> None:
    rows = []
    for d in sorted(Path(folder).iterdir()):
        if not (d / "metrics.json").exists():
            continue
        name = json.loads((d / "manifest.json").read_text())["config"]["name"]
        m = NAME.match(name)
        if not m:
            continue
        for s in json.loads((d / "metrics.json").read_text())["splits"]:
            rows.append({**m.groupdict(), "window": s["test_start"], "wrmsse": s["wrmsse"],
                         "forecast_to_actual": s["forecast_to_actual"]})
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "reliability.csv", index=False)
    bad = r.groupby(["cat", "window"])["wrmsse"].max() > 2
    clean = ~r.set_index(["cat", "window"]).index.isin(bad[bad].index)
    summary = r.groupby(["cat", "variant"]).agg(
        runs=("wrmsse", "size"), failures=("wrmsse", lambda x: int((x > 2).sum())),
        worst=("wrmsse", "max"), mean=("wrmsse", "mean"))
    summary["mean_where_none_failed"] = r[clean].groupby(["cat", "variant"])["wrmsse"].mean()
    print(summary.round(3).to_string())
    print()
    print(r.pivot_table(index=["cat", "variant"], columns="window", values="wrmsse", aggfunc="max").round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
