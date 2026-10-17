"""Study G (docs/PAPER1_REVISION_PLAN.md, 17 October 2026): put the Frat MinT-shrink
rerun after the singular-covariance fix into reconcile.csv. The rows from before
the fix are kept as method `ets_mint_shrink_before_fix`.

  python scripts/apply_mint_fix.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "paper1_revision"


def main():
    rec = pd.read_csv(ROOT / "tables" / "reconcile.csv")
    new = pd.read_csv(ROOT / "reconcile" / "reconcile_frat_mint_fixed.csv")
    new = new[new.method == "ets_mint_shrink"]
    mean = new.groupby(["key", "set", "method", "level"]).agg(wrmsse=("wrmsse", "mean"), windows=("wrmsse", "size")).reset_index()
    old = (rec.key == "frat") & (rec.set == "rolling") & (rec.method == "ets_mint_shrink")
    if old.any():
        rec.loc[old, "method"] = "ets_mint_shrink_before_fix"
    rec = rec[~((rec.key == "frat") & (rec.set == "rolling") & (rec.method == "ets_mint_shrink"))]
    rec = pd.concat([rec, mean], ignore_index=True).sort_values(["key", "set", "method", "level"])
    rec.to_csv(ROOT / "tables" / "reconcile.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
