"""Paper 1 revision, study B of the amendment of 15 October 2026: the table the
paper reads, from the CSV of scripts/input_perturbation.py.

  python scripts/perturbation_tables.py --csv <perturbation.csv> --out results/paper1_revision/tables/perturbation.csv

For each data set, model, input and operation, the mean over windows of the
change in WRMSSE at item by store and at the total (as a share of the score
with the true plan) and of the change in forecast units in the periods where
the input was on.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(bottom_change=df["bottom"] / df["bottom_true"] - 1, total_change=df["total"] / df["total_true"] - 1)
    out = df.groupby(["key", "model", "input", "operation"]).agg(
        bottom_change=("bottom_change", "mean"), total_change=("total_change", "mean"),
        windows=("origin", "nunique"),
        **({"units_change_on": ("units_change_on", "mean")} if "units_change_on" in df else {})).reset_index()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = table(pd.read_csv(a.csv))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.5g")
    print(f"{len(out)} rows")


if __name__ == "__main__":
    main()
