"""Paper 1: the history-length test (plan entry of 15 November 2026), from results/history/ (scripts/score_history.py).

Writes paper/generated/hist_tab.tex and hist_values.tex. For each data set and training length: WRMSSE (mean over
levels and windows, Dominick's duplicate levels counted once) of the world model, LightGBM and their average, the world
model's gap to LightGBM, and the paired comparisons (levels x windows) where the world model is clearly more or less
accurate. Reading, fixed before the runs: years of history help the world model if its gap to LightGBM shrinks as the
length grows on Dominick's; observations per series help if it shrinks only on the daily data (M5, Favorita).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "results" / "history"
NAMES = {"dominicks_ana": "Analgesics", "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese",
         "dominicks_fre": "Frozen entrees", "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snack crackers",
         "m5": "M5", "favorita": "Favorita"}
WORDS = "none one two three four five six seven eight nine".split()
DUP = ["cat_id", "cat_idxstore_id"]


def years(tag: str) -> int:
    return int(tag.rstrip("y"))


def main():
    s = pd.read_csv(H / "scores.csv")
    s = s[~(s.key.str.startswith("dominicks") & s.level.isin(DUP))]
    p = pd.read_csv(H / "paired.csv")
    p = p[~(p.key.str.startswith("dominicks") & p.level.isin(DUP))]
    over = s.groupby(["key", "length", "model", "origin"]).wrmsse.mean().groupby(["key", "length", "model"]).mean().unstack()
    wl = p[(p.a == "world_model") & (p.b == "lightgbm")]
    rows, v, trend = [], {}, {}
    for key in NAMES:
        if key not in over.index.get_level_values(0):
            continue
        lens = sorted(over.loc[key].index, key=years)
        gaps = []
        for i, ln in enumerate(lens):
            r = over.loc[(key, ln)]
            q = wl[(wl.key == key) & (wl.length == ln)]
            gap = r.world_model - r.lightgbm
            gaps.append((years(ln), gap))
            best = min(("world_model", "lightgbm", "average"), key=lambda m: r[m])
            cell = lambda m: (f"\\textbf{{{r[m]:.3f}}}" if m == best else f"{r[m]:.3f}")  # noqa: E731
            rows.append(f"{NAMES[key] if i == 0 else ''} & {years(ln)} & {cell('world_model')} & {cell('lightgbm')} & "
                        f"{cell('average')} & {gap:+.3f} & {int((q.hi < 0).sum())} / {int((q.lo > 0).sum())} \\\\"
                        .replace("-0.", "$-$0.").replace("+0.", "+0."))
        rows.append(r"\addlinespace[2pt]")
        y, g = zip(*gaps)
        trend[key] = np.polyfit(y, g, 1)[0] if len(y) > 1 else np.nan  # gap change per year of history
    dom = [k for k in trend if k.startswith("dominicks")]
    v["histDomShrinks"] = WORDS[int(sum(trend[k] < 0 for k in dom))]
    v["histDomSets"] = WORDS[len(dom)]
    for k, t in (("m5", "Mfive"), ("favorita", "Fav")):
        if k in trend:
            v[f"hist{t}Trend"] = f"{trend[k]:+.3f}".replace("-", "$-$")
            g = over.loc[k]
            lens = sorted(g.index, key=years)
            v[f"hist{t}GapShort"] = f"{g.loc[lens[0]].world_model - g.loc[lens[0]].lightgbm:+.3f}".replace("-", "$-$")
            v[f"hist{t}GapLong"] = f"{g.loc[lens[-1]].world_model - g.loc[lens[-1]].lightgbm:+.3f}".replace("-", "$-$")
            v[f"hist{t}YearsShort"], v[f"hist{t}YearsLong"] = str(years(lens[0])), str(years(lens[-1]))
    v["histDomTrendLo"] = f"{min(trend[k] for k in dom):+.3f}".replace("-", "$-$")
    v["histDomTrendHi"] = f"{max(trend[k] for k in dom):+.3f}".replace("-", "$-$")
    # data sets where the world model is ahead of LightGBM with the least and with the most history
    short = [over.loc[k].loc[min(over.loc[k].index, key=years)] for k in NAMES if k in trend]
    long_ = [over.loc[k].loc[max(over.loc[k].index, key=years)] for k in NAMES if k in trend]
    v["histAheadShort"] = WORDS[int(sum(r.world_model < r.lightgbm for r in short))]
    v["histAheadLong"] = WORDS[int(sum(r.world_model < r.lightgbm for r in long_))]
    v["histSets"] = WORDS[len(short)]
    # where the average stands: lowest of the three at each data set and length
    av = (over.idxmin(axis=1) == "average")
    v["histAvgBest"], v["histCells"] = str(int(av.sum())), str(len(av))
    tab = (r"""\begin{table}[t]
\centering
\caption{\textbf{Training on more history.} WRMSSE, mean over levels and windows, of each model trained on the given
years of history before each window, settings otherwise those of the final runs (Dominick's at 3 years and M5 at 2 are
the final runs; Favorita, loaded from 2013 for this test, at every length). Gap: world model minus LightGBM (negative,
the world model is more accurate). Clear: comparisons (levels and windows) where the world model is clearly more / less
accurate than LightGBM. Bold: lowest. Planned and added in \thisrev{}.}
\label{tab:history}
\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrcccrc}
\toprule
Data & Years & WM & LGB & Avg & Gap & Clear \\
\midrule
""" + "\n".join(rows[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")
    gen = ROOT / "paper" / "generated"
    (gen / "hist_tab.tex").write_text(tab)
    (gen / "hist_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(v.items())))
    print(v)
    print({k: round(t, 4) for k, t in trend.items()})


if __name__ == "__main__":
    main()
