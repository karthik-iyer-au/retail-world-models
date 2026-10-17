"""The 90% model confidence set from a saved per-window scores table.

  python scripts/mcs_from_scores.py SCORES.csv OUT.csv --keys frat [--check dominicks_ana]

`revision_analysis.py --parts mcs` computes the set when it scores the windows;
for Frat that part did not run in the job that scored its rolling windows, so
the set is computed here from the job's scores.csv, with the same function and
the same models (the Timestral models of study F, scored later, are left out).
`--check` recomputes a data set already in results/paper1_revision/tables/mcs.csv
and stops if the two disagree.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwm.evaluation import compare as C  # noqa: E402

MODELS = ["chronos", "ensemble", "lift", "lightgbm", "seasonal_naive", "tft", "timesfm_mean", "timesfm_plan",
          "world_model"]
REF = Path(__file__).resolve().parents[1] / "results" / "paper1_revision" / "tables" / "mcs.csv"


def mcs(sc: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    g0 = sc[(sc["set"] == "rolling") & (sc["key"] == key) & sc["model"].isin(MODELS)]
    for lv, g in g0.groupby("level"):
        wide = g.pivot_table(index="origin", columns="model", values="wrmsse").dropna(axis=1)
        kept = C.model_confidence_set(wide)
        for m in wide.columns:
            rows.append({"key": key, "level": lv, "model": m, "in_set": m in kept,
                         "mean_wrmsse": wide[m].mean(), "windows": len(wide)})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("out")
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--check", nargs="*", default=[])
    a = p.parse_args(argv)
    sc = pd.read_csv(a.scores)
    ref = pd.read_csv(REF)
    for key in a.check:
        mine = mcs(sc, key).set_index(["level", "model"]).in_set.sort_index()
        theirs = ref[ref.key == key].set_index(["level", "model"]).in_set.astype(bool).sort_index()
        if not mine.reindex(theirs.index).equals(theirs):
            raise SystemExit(f"{key}: recomputed set differs from {REF.name}")
        print(f"{key}: matches {REF.name}")
    out = pd.concat([mcs(sc, k) for k in a.keys], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
