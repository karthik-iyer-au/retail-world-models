"""Table, figure and values of the 52-week test (docs/PAPER1_REVISION_PLAN.md, 13 November 2026).

  python scripts/make_long_horizon_tables.py [--scores results/long_horizon]

Reads scores.csv and paired.csv written by scripts/score_long_horizon.py and writes
paper/drafts/generated/{lh_tab.tex, lh_values.tex} and paper/drafts/fig_long_horizon.pdf.
Overall scores are the mean over levels (Dominick's duplicate levels counted once) and over the two windows.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "figs"))
OUT = ROOT / "paper" / "drafts"
KEYS = {"dominicks_ana": "Analgesics", "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese",
        "dominicks_fre": "Frozen entrees", "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snacks", "m5": "M5"}
DUP = {"cat_id", "cat_idxstore_id"}
TRIO = ["world_model_blocks", "lightgbm_blocks", "average_blocks"]
CELL = {"world_model_blocks": "wmcell", "lightgbm_blocks": "lgbcell", "average_blocks": "avgcell"}
WORDS = "none one two three four five six seven eight nine ten eleven twelve".split()


def _w(n: int) -> str:
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def distinct(d: pd.DataFrame) -> pd.DataFrame:
    return d[~(d.key.str.startswith("dominicks") & d.level.isin(DUP))]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(ROOT / "results" / "long_horizon"))
    a = p.parse_args(argv)
    s = distinct(pd.read_csv(Path(a.scores) / "scores.csv"))
    pr = distinct(pd.read_csv(Path(a.scores) / "paired.csv"))
    (OUT / "generated").mkdir(parents=True, exist_ok=True)
    keys = [k for k in KEYS if k in set(s.key)]
    over = s.groupby(["key", "span", "model", "window"]).wrmsse.mean().groupby(["key", "span", "model"]).mean()
    item = s[s.level == "item_idxstore_id"].groupby(["key", "span", "model"]).wrmsse.mean()
    fa = s.groupby(["key", "span", "model"]).forecast_to_actual.mean()

    def clear(key, span, x, y):
        q = pr[(pr.key == key) & (pr.span == span) & (pr.a == x) & (pr.b == y)]
        return int((q.hi < 0).sum()), int((q.lo > 0).sum()), len(q)

    rows, v = [], {}
    wins = {"world_model_blocks": 0, "lightgbm_blocks": 0, "average_blocks": 0}
    wm_beats_best, best_beats_wm = 0, 0
    for k in keys:
        o = over.loc[k, "all"]
        vals = [o.get(m, np.nan) for m in TRIO]
        best = np.nanmin(vals)
        wins[TRIO[int(np.nanargmin(vals))]] += 1
        cells = ["n/a" if np.isnan(x) else
                 (f"\\cellcolor{{{CELL[m]}}}\\textbf{{{x:.3f}}}" if abs(x - best) < 5e-4 else f"{x:.3f}")
                 for m, x in zip(TRIO, vals)]
        lgb_best = "lightgbm_direct" if o.get("lightgbm_direct", np.inf) < o.get("lightgbm_blocks", np.inf) \
            else "lightgbm_blocks"
        wl = clear(k, "all", "world_model_blocks", "lightgbm_blocks")
        wd = clear(k, "all", "world_model_blocks", "lightgbm_direct")
        if o.get("world_model_blocks", np.inf) < o.get(lgb_best, np.inf):
            wm_beats_best += 1
        else:
            best_beats_wm += 1
        f = lambda m: f"{o[m]:.3f}" if m in o and np.isfinite(o[m]) else "n/a"  # noqa: E731
        growth = [item.get((k, str(b), m), np.nan) for m in ("world_model_blocks", "lightgbm_blocks") for b in (1, 4)]
        rows.append(f"{KEYS[k]} & " + " & ".join(cells) + f" & {f('lightgbm_direct')} & {f('world_model_direct')} & "
                    f"{wl[0]} / {wl[1]} of {wl[2]} & {wd[0]} / {wd[1]} of {wd[2]} & "
                    + " & ".join("n/a" if np.isnan(x) else f"{x:.3f}" for x in growth) + r" \\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Forecasts 52 weeks ahead (364 days on M5). WRMSSE over the 52 weeks, mean over levels and two windows, for
the world model (WM), LightGBM (LGB) and their average (Avg), each forecasting four blocks of 13 weeks with each block's
forecast taken in as sales before the next; and each model forecasting all 52 weeks at once (direct; not run for M5's
world model). Paired item bootstrap over levels and windows: the number of comparisons in which the world model in
blocks is clearly lower / clearly higher. Last four columns: WRMSSE item by store in the first and last block of 13
weeks. Shading marks the lowest of the three in blocks: \colorbox{wmcell}{world model}, \colorbox{lgbcell}{LightGBM},
\colorbox{avgcell}{average}.}
\label{tab:long-horizon}
\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l ccc cc cc cc cc}
\toprule
 & \multicolumn{3}{c}{In blocks} & \multicolumn{2}{c}{Direct} & \multicolumn{2}{c}{WM in blocks against} & \multicolumn{2}{c}{WM, item by store} & \multicolumn{2}{c}{LGB, item by store}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
Data set & WM & LGB & Avg & LGB & WM & LGB in blocks & LGB direct & Weeks 1--13 & 40--52 & Weeks 1--13 & 40--52\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table*}
"""
    (OUT / "generated" / "lh_tab.tex").write_text(tab.replace("1--13", "1 to 13").replace("40--52", "40 to 52"))
    n = len(keys)
    v["lhSets"] = _w(n)
    v["lhWmBest"], v["lhLgbBest"], v["lhAvgBest"] = (_w(wins[m]) for m in TRIO)
    v["lhWmBeatsLgbBest"], v["lhLgbBestBeatsWm"] = _w(wm_beats_best), _w(best_beats_wm)
    tot = [clear(k, "all", "world_model_blocks", "lightgbm_blocks") for k in keys]
    v["lhPWmBetter"], v["lhPWmWorse"], v["lhPN"] = (str(sum(t[i] for t in tot)) for i in range(3))
    tot = [clear(k, "all", "average_blocks", "world_model_blocks") for k in keys]
    v["lhPAvgWmBetter"], v["lhPAvgWmWorse"] = (str(sum(t[i] for t in tot)) for i in range(2))
    tot = [clear(k, "all", "average_blocks", "lightgbm_blocks") for k in keys]
    v["lhPAvgLgbBetter"], v["lhPAvgLgbWorse"] = (str(sum(t[i] for t in tot)) for i in range(2))
    # growth from the first to the last block, item by store, mean over data sets
    for m, tag in (("world_model_blocks", "Wm"), ("lightgbm_blocks", "Lgb"), ("average_blocks", "Avg")):
        g = [item.get((k, "4", m), np.nan) - item.get((k, "1", m), np.nan) for k in keys]
        v[f"lhGrow{tag}Lo"], v[f"lhGrow{tag}Hi"] = f"{np.nanmin(g):+.3f}", f"{np.nanmax(g):+.3f}"
        r = [fa.get((k, "all", m), np.nan) for k in keys]
        v[f"lhFa{tag}Lo"], v[f"lhFa{tag}Hi"] = f"{100 * np.nanmin(r):.0f}", f"{100 * np.nanmax(r):.0f}"
    (OUT / "generated" / "lh_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in v.items()))

    import matplotlib.pyplot as plt
    import style

    style.use_style()
    fig, axes = plt.subplots(2, 4, figsize=(style.FULL, 3.6), sharex=True)
    show = [("world_model_blocks", "world", "World model"), ("lightgbm_blocks", "lgb", "LightGBM"),
            ("average_blocks", "ensemble", "Average")]
    for ax, k in zip(axes.flat, keys):
        for m, c, lab in show:
            y = [over.get((k, str(b), m), np.nan) for b in (1, 2, 3, 4)]
            ax.plot([1, 2, 3, 4], y, color=style.COLOR[c], marker=style.MARKER.get(c, "D"), ms=3.5, lw=1.2, label=lab)
        ax.set_title(KEYS[k], fontsize=8)
        ax.set_xticks([1, 2, 3, 4], ["1-13", "14-26", "27-39", "40-52"], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        style.despine(ax)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    last = list(axes.flat)[len(keys)] if len(keys) < 8 else axes.flat[-1]
    last.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=8)
    fig.supxlabel("Weeks ahead (blocks of 13 weeks; 91 days on M5)", fontsize=8)
    fig.supylabel("WRMSSE, mean over levels", fontsize=8)
    fig.tight_layout()
    style.save(fig, OUT / "fig_long_horizon.pdf")
    print(v)


if __name__ == "__main__":
    main()
