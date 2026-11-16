"""C4 within the prices each item had in training (plan amendment of 16 November 2026), from
results/paper2/tuned/c4_observed.csv (scripts/c4_observed.py). Writes generated/c4_observed_macros.tex: for the six
Dominick's categories, the range over categories of the mean share of steps failing C4 on all steps and on steps
inside the observed price range, and the share of steps kept; the same for M5 and Frat; Favorita's share kept.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    d = pd.read_csv(ROOT / "results/paper2/tuned/c4_observed.csv")
    d["key"] = d["config"].str.replace("final_", "").str.replace("_world_model", "")
    m = d.groupby("key")[["c4_all", "c4_observed", "share_kept"]].mean()
    dom = m[m.index.str.startswith("dominicks")]
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    M = {"cfoAllDomLo": pct(dom.c4_all.min()), "cfoAllDomHi": pct(dom.c4_all.max()),
         "cfoObsDomLo": pct(dom.c4_observed.min()), "cfoObsDomHi": pct(dom.c4_observed.max()),
         "cfoKeptDomLo": pct(dom.share_kept.min()), "cfoKeptDomHi": pct(dom.share_kept.max()),
         "cfoObsCloser": str(int((dom.c4_observed < dom.c4_all).sum()))}
    for k, tag in (("m5", "Mfive"), ("frat", "Frat"), ("favorita", "Fav")):
        if k in m.index:
            M[f"cfoAll{tag}"], M[f"cfoKept{tag}"] = pct(m.loc[k, "c4_all"]), pct(m.loc[k, "share_kept"])
            M[f"cfoObs{tag}"] = "n/a" if pd.isna(m.loc[k, "c4_observed"]) else pct(m.loc[k, "c4_observed"])
    (ROOT / "paper2/generated/c4_observed_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    print(m.round(3)); print(M)


if __name__ == "__main__":
    main()
