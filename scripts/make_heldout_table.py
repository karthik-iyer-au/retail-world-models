"""Paper 1: the held-out periods scored (plan entry of 15 November 2026), from results/heldout/ (score_heldout.py).

Writes paper/generated/heldout_tab.tex and heldout_values.tex: for each data set, WRMSSE (mean over levels and the
held-out windows, Dominick's duplicate levels counted once) of the world model, LightGBM and their average, beside the
same models' score on the paper's rolling windows (results/analysis_tuned/scores.csv), and the paired comparisons of
the average with each model; on M5 the competition's final 28 days on their own.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"dominicks_ana": "Analgesics", "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese",
         "dominicks_fre": "Frozen entrees", "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snack crackers",
         "frat": "Frat", "m5": "M5"}
WORDS = "none one two three four five six seven eight nine".split()
DUP = ["cat_id", "cat_idxstore_id"]
MODELS = [("world_model", "WM"), ("lightgbm", "LGB"), ("average", "Avg")]


def drop_dup(d):
    return d[~(d.key.str.startswith("dominicks") & d.level.isin(DUP))]


def main():
    s = drop_dup(pd.read_csv(ROOT / "results/heldout/scores.csv"))
    p = drop_dup(pd.read_csv(ROOT / "results/heldout/paired.csv"))
    held = s[s.window == "held out"].groupby(["key", "model", "origin"]).wrmsse.mean().groupby(["key", "model"]).mean().unstack()
    comp = s[s.window != "held out"].groupby(["key", "model"]).wrmsse.mean().unstack()
    r = pd.read_csv(ROOT / "results/analysis_tuned/scores.csv")
    if "set" in r:
        r = r[r["set"] == "rolling"]
    r = drop_dup(r.replace({"model": {"ensemble_2": "average"}}))
    roll = r.groupby(["key", "model", "origin"]).wrmsse.mean().groupby(["key", "model"]).mean().unstack()
    ph = p[p.window == "held out"]
    rows, v = [], {}
    avg_best = avg_beats_lgb = avg_beats_wm = 0
    order_same = 0
    for k, name in NAMES.items():
        if k not in held.index:
            continue
        h = held.loc[k]
        best = min((m for m, _ in MODELS), key=lambda m: h[m])
        avg_best += best == "average"
        q = ph[ph.key == k]
        al = q[(q.a == "average") & (q.b == "lightgbm")]
        aw = q[(q.a == "average") & (q.b == "world_model")]
        avg_beats_lgb += (al.hi < 0).sum() > (al.lo > 0).sum()
        avg_beats_wm += (aw.hi < 0).sum() > (aw.lo > 0).sum()
        if k in roll.index:
            order_same += min((m for m, _ in MODELS), key=lambda m: roll.loc[k][m]) == best
        cells = " & ".join((f"\\textbf{{{h[m]:.3f}}}" if m == best else f"{h[m]:.3f}") for m, _ in MODELS)
        rc = " & ".join(f"{roll.loc[k][m]:.3f}" if k in roll.index else "n/a" for m, _ in MODELS)
        rows.append(f"{name} & {cells} & {rc} & {int((al.hi < 0).sum())} / {int((al.lo > 0).sum())} & "
                    f"{int((aw.hi < 0).sum())} / {int((aw.lo > 0).sum())} \\\\")
    n = len(rows)
    v.update(hoSets=WORDS[n], hoAvgBest=WORDS[avg_best], hoAvgBeatsLgb=WORDS[avg_beats_lgb],
             hoAvgBeatsWm=WORDS[avg_beats_wm], hoOrderSame=WORDS[order_same])
    if "m5" in comp.index:
        c = comp.loc["m5"]
        for m, t in MODELS:
            v[f"hoMfiveFinal{t}"] = f"{c[m]:.3f}"
    tab = (r"""\begin{table*}[t]
\centering
\caption{\textbf{The periods every final run held out.} WRMSSE, mean over
levels and the held-out windows (three of 8 weeks on Dominick's and Frat, three of 28 days on M5), each window trained on
the history before it, beside the same models on the paper's rolling windows. Avg against LGB, Avg against WM:
comparisons (levels and windows) where the average is clearly more / less accurate (paired item bootstrap). Bold: lowest
on the held-out windows. Favorita held nothing out. Planned 15 November 2026, after the other results.}
\label{tab:heldout}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccrr}
\toprule
 & \multicolumn{3}{c}{Held-out periods} & \multicolumn{3}{c}{Rolling windows} & & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Data set & WM & LGB & Avg & WM & LGB & Avg & Avg vs LGB & Avg vs WM \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    gen = ROOT / "paper" / "generated"
    (gen / "heldout_tab.tex").write_text(tab)
    (gen / "heldout_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(v.items())))
    print(v)


if __name__ == "__main__":
    main()
