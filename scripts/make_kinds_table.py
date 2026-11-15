"""Paper 1, RQ2 extended (plan entry of 15 November 2026; Karthik, 16:09, "Yes agreed"): the world model and the
average against LightGBM by kind of line and kind of week, from results/kinds/ (scripts/score_kinds.py).
Writes paper/generated/rq_tab_kinds.tex and kinds_values.tex."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORDS = {i: w for i, w in enumerate("none one two three four five six seven eight nine".split())}
KINDS = [("short_history", "Short history"), ("long_history", "Long history"), ("slow", "Slow lines"),
         ("fast", "Fast lines"), ("intermittent", "Sells in under half the periods"),
         ("steady", "Sells in nine periods of ten or more"), ("no_promotion", "Steady price (no cut)"),
         ("light_promotion", "Occasional cuts (under a quarter of periods)"),
         ("heavy_promotion", "Frequent cuts (a quarter of periods or more)"), ("other_periods", "Periods at regular price"),
         ("price_cut_periods", "Price-cut periods")]
KEYS = ["dominicks_ana", "dominicks_bjc", "dominicks_che", "dominicks_fre", "dominicks_sdr", "dominicks_sna", "frat",
        "m5", "favorita"]
SHORT = {"dominicks_ana": "Ana", "dominicks_bjc": "Bjc", "dominicks_che": "Che", "dominicks_fre": "Fre",
         "dominicks_sdr": "Sdr", "dominicks_sna": "Sna", "frat": "Frat", "m5": "M5", "favorita": "Fav"}


TAG = {**SHORT, "m5": "Mfive"}  # macro names take no digits


def mark(r) -> str:
    return "$+$" if r.hi < 0 else ("$-$" if r.lo > 0 else "$\\cdot$")


def main():
    d = pd.read_csv(ROOT / "results" / "kinds" / "kinds.csv")
    sh = pd.read_csv(ROOT / "results" / "kinds" / "kind_shares.csv").groupby(["kind", "key"]).share.mean()
    wm, avg = d[d.model == "world_model"].set_index(["kind", "key"]), d[d.model == "average"].set_index(["kind", "key"])
    rows, v = [], {}
    for kind, label in KINDS:
        cells = [mark(wm.loc[(kind, k)]) if (kind, k) in wm.index else "" for k in KEYS]
        b = sum(c == "$+$" for c in cells)
        w = sum(c == "$-$" for c in cells)
        ab = sum((kind, k) in avg.index and avg.loc[(kind, k)].hi < 0 for k in KEYS)
        aw = sum((kind, k) in avg.index and avg.loc[(kind, k)].lo > 0 for k in KEYS)
        rows.append(f"{label} & " + " & ".join(cells) + f" & {b} & {w} & {ab} & {aw} \\\\")
        tag = "".join(x.capitalize() for x in kind.split("_"))
        v[f"kWmBetter{tag}"], v[f"kWmWorse{tag}"] = WORDS[b], WORDS[w]
        v[f"kAvgBetter{tag}"], v[f"kAvgWorse{tag}"] = WORDS[ab], WORDS[aw]
    # shares of series by promotion kind, by data set
    for kind in ("no_promotion", "heavy_promotion"):
        tag = "".join(x.capitalize() for x in kind.split("_"))
        dom = [sh[(kind, k)] for k in KEYS if k.startswith("dominicks")]
        v[f"kShare{tag}DomLo"], v[f"kShare{tag}DomHi"] = f"{100 * min(dom):.0f}", f"{100 * max(dom):.0f}"
        for k in ("frat", "m5", "favorita"):
            v[f"kShare{tag}{TAG[k]}"] = f"{100 * sh[(kind, k)]:.0f}"
    # data sets where the world model is ahead on most kinds
    ahead = {k: sum(wm.loc[(kind, k)].hi < 0 for kind, _ in KINDS if (kind, k) in wm.index) for k in KEYS}
    behind = {k: sum(wm.loc[(kind, k)].lo > 0 for kind, _ in KINDS if (kind, k) in wm.index) for k in KEYS}
    v["kKinds"] = WORDS[len(KINDS)] if len(KINDS) < 10 else str(len(KINDS))
    for k in KEYS:
        v[f"kAhead{TAG[k]}"], v[f"kBehind{TAG[k]}"] = str(ahead[k]), str(behind[k])
    head = " & ".join(SHORT[k] for k in KEYS)
    tab = (r"""\begin{table*}[t]
\centering
\caption{\textbf{The world model is ahead of LightGBM on new lines and lines sold at a steady price, and behind on lines
driven by frequent price cuts.} Each mark compares the world model with LightGBM on the item-by-store series (or periods)
of one kind, over the rolling windows of the final runs: $+$ clearly more accurate, $-$ clearly less, $\cdot$ not
clearly different (95\% interval of the difference in revenue-weighted RMSSE, 1,000 resamples of series). Every kind is
fixed from the year before each window; a price cut is a price at most 95\% of the line's most common price that year
(Favorita, which records no prices: its promotion flag). The last columns count data sets where the world model, and
the average of the two models, are clearly more or less accurate than LightGBM. Added in \thisrev{} after the 52-week
test.}
\label{tab:rq-kinds}
\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l""" + "c" * len(KEYS) + r"""cccc}
\toprule
 & \multicolumn{""" + str(len(KEYS)) + r"""}{c}{World model against LightGBM} & \multicolumn{2}{c}{WM} & \multicolumn{2}{c}{Avg} \\
\cmidrule(lr){2-""" + str(len(KEYS) + 1) + r"""}\cmidrule(lr){""" + str(len(KEYS) + 2) + "-" + str(len(KEYS) + 3) + r"""}\cmidrule(lr){""" + str(len(KEYS) + 4) + "-" + str(len(KEYS) + 5) + r"""}
Kind of line or period & """ + head + r""" & ahead & behind & ahead & behind \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    gen = ROOT / "paper" / "generated"
    (gen / "rq_tab_kinds.tex").write_text(tab)
    (gen / "kinds_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(v.items())))
    print(tab)
    print(v)


if __name__ == "__main__":
    main()
