"""Table, figure and values of the 52-week test (docs/PAPER1_REVISION_PLAN.md, 13 November 2026).

  python scripts/make_long_horizon_tables.py [--scores results/long_horizon]

Reads scores.csv and paired.csv written by scripts/score_long_horizon.py and writes
paper/generated/{lh_tab.tex, lh_tab_all.tex, lh_values.tex} and paper/figs/fig_long_horizon.pdf.
Overall scores are the mean over levels (Dominick's duplicate levels counted once) and over the two windows.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "figs"))
OUT = ROOT / "paper"  # generated/ and figs/ of paper 1 (the draft in paper/drafts reads the same files)
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
    SIX = [f"{m}_{st}" for st in ("blocks", "direct") for m in ("world_model", "lightgbm", "average")]
    colour = {m: CELL[m.replace("_direct", "_blocks")] for m in SIX}
    lowest = {m: 0 for m in SIX}
    wm_beats_best, best_beats_wm = 0, 0
    for k in keys:
        o = over.loc[k, "all"]
        vals = [o.get(m, np.nan) for m in SIX]
        best = np.nanmin(vals)
        lowest[SIX[int(np.nanargmin(vals))]] += 1
        cells = ["n/a" if np.isnan(x) else
                 (f"\\cellcolor{{{colour[m]}}}\\textbf{{{x:.3f}}}" if abs(x - best) < 5e-4 else f"{x:.3f}")
                 for m, x in zip(SIX, vals)]
        lgb_best = "lightgbm_direct" if o.get("lightgbm_direct", np.inf) < o.get("lightgbm_blocks", np.inf) \
            else "lightgbm_blocks"
        if o.get("world_model_blocks", np.inf) < o.get(lgb_best, np.inf):
            wm_beats_best += 1
        else:
            best_beats_wm += 1
        wl = clear(k, "all", "world_model_blocks", "lightgbm_blocks")
        ad = clear(k, "all", "average_direct", "lightgbm_direct")
        pair = lambda t: f"{t[0]} / {t[1]} of {t[2]}" if t[2] else "n/a"  # noqa: E731
        growth = [item.get((k, str(b), m), np.nan) for m in ("world_model_blocks", "lightgbm_blocks") for b in (1, 4)]
        rows.append(f"{KEYS[k]} & " + " & ".join(cells) + f" & {pair(wl)} & {pair(ad)} & "
                    + " & ".join("n/a" if np.isnan(x) else f"{x:.3f}" for x in growth) + r" \\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Forecasts 52 weeks ahead (364 days on M5). WRMSSE over the 52 weeks, mean over levels and two windows, for
the world model (WM), LightGBM (LGB) and their average (Avg). In blocks: each model forecasts four blocks of 13 weeks,
each block's forecast taken in as sales before the next. Direct: each model forecasts all 52 weeks at once (not run for
M5's world model). Paired item bootstrap over levels and windows: comparisons clearly lower / clearly higher, of all.
Last four columns: WRMSSE item by store in the first and last block of 13 weeks, in blocks. Shading marks the lowest
of the six: \colorbox{wmcell}{world model}, \colorbox{lgbcell}{LightGBM}, \colorbox{avgcell}{average}.}
\label{tab:long-horizon}
\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l ccc ccc cc cc cc}
\toprule
 & \multicolumn{3}{c}{In blocks} & \multicolumn{3}{c}{Direct} & \multicolumn{2}{c}{Paired} & \multicolumn{2}{c}{WM, item by store} & \multicolumn{2}{c}{LGB, item by store}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}
Data set & WM & LGB & Avg & WM & LGB & Avg & WM vs LGB, blocks & Avg vs LGB, direct & Weeks 1--13 & 40--52 & Weeks 1--13 & 40--52\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table*}
"""
    (OUT / "generated" / "lh_tab.tex").write_text(tab.replace("1--13", "1 to 13").replace("40--52", "40 to 52"))
    n = len(keys)
    v["lhSets"] = _w(n)
    for m in SIX:
        v["lhLowest" + "".join(w.capitalize() for w in m.replace("world_model", "wm").replace("lightgbm", "lgb")
                                   .replace("average", "avg").split("_"))] = _w(lowest[m])
    over_all = over.xs("all", level="span")
    for m, tag in (("world_model", "Wm"), ("lightgbm", "Lgb")):
        both = [k for k in keys if (k, f"{m}_direct") in over_all.index and (k, f"{m}_blocks") in over_all.index]
        v[f"lhDirectBeatsBlocks{tag}"] = _w(sum(over_all[(k, f"{m}_direct")] < over_all[(k, f"{m}_blocks")] for k in both))
        v[f"lhDirectSets{tag}"] = _w(len(both))
    tot = [clear(k, "all", "average_direct", "lightgbm_direct") for k in keys]
    v["lhPAvgDirBetter"], v["lhPAvgDirWorse"], v["lhPAvgDirN"] = (str(sum(t[i] for t in tot)) for i in range(3))
    tot = [clear(k, "all", "average_direct", "world_model_direct") for k in keys]
    v["lhPAvgDirWmBetter"], v["lhPAvgDirWmWorse"] = (str(sum(t[i] for t in tot)) for i in range(2))
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
        v[f"lhGrow{tag}Lo"], v[f"lhGrow{tag}Hi"] = f"{np.nanmin(g):+.3f}".replace("-", "\u2212"), f"{np.nanmax(g):+.3f}".replace("-", "\u2212")
        r = [fa.get((k, "all", m), np.nan) for k in keys]
        v[f"lhFa{tag}Lo"], v[f"lhFa{tag}Hi"] = f"{100 * np.nanmin(r):.0f}", f"{100 * np.nanmax(r):.0f}"
    v.update(every(s, pr, keys))
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
    style.save(fig, OUT / "figs" / "fig_long_horizon.pdf")
    print(v)


# added 14 November 2026: every forecast over 52 weeks, the designs added after the pre-registered test, the other
# models, and (when run) each model's settings tuned for 52 weeks
ALL = [("world_model_direct", "WM"), ("world_model_rollon", "WM rolled on"), ("lightgbm_direct", "LGB"),
       ("lightgbm_byblock", "LGB by block"), ("average_direct", "Avg"), ("average_rollon", "Avg, WM rolled on"),
       ("average_byblock", "Avg by block"),
       ("world_model_tdirect", "WM tuned"), ("lightgbm_tdirect", "LGB tuned"), ("lightgbm_tbyblock", "LGB tuned by block"),
       ("average_tdirect", "Avg tuned"), ("average_tbyblock", "Avg tuned by block"),
       ("seasonal_naive_direct", "Seasonal naive"), ("chronos_direct", "Chronos"), ("timesfm_mean_direct", "TimesFM"),
       ("timesfm_plan_direct", "TimesFM with plan"), ("timestral_plan_direct", "Timestral with plan"),
       ("tft_direct", "TFT")]
UNABLE = {"timestral_plan_direct"}  # forecasts 1.5 to 9.5 times actual from the first 13 weeks: left out of 'lowest'


def every(s: pd.DataFrame, pr: pd.DataFrame, keys: list[str]) -> dict:
    over = s[s.span == "all"].groupby(["key", "model", "window"]).wrmsse.mean().groupby(["key", "model"]).mean()
    fa = s[s.span == "all"].groupby(["key", "model"]).forecast_to_actual.mean()
    fa1 = s[s.span == "1"].groupby(["key", "model"]).forecast_to_actual.mean()
    cols = [(m, lab) for m, lab in ALL if m in set(over.index.get_level_values(1))]
    rows, lowest, v = [], {}, {}
    for k in keys:
        vals = {m: over.get((k, m), np.nan) for m, _ in cols}
        able = {m: x for m, x in vals.items() if m not in UNABLE and not np.isnan(x)}
        best = min(able, key=able.get)
        lowest[best] = lowest.get(best, 0) + 1
        cells = ["n/a" if np.isnan(x) else (f"\\textbf{{{x:.3f}}}" if m == best else f"{x:.3f}") for m, x in vals.items()]
        rows.append(f"{KEYS[k]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"\\rotatebox{{90}}{{{lab}}}" for _, lab in cols)
    tab = (r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Every forecast 52 weeks ahead (364 days on M5): WRMSSE over the 52 weeks, mean over levels and two windows.
Each forecasts all 52 weeks at once except the world model rolled on (its state rolled 13 weeks at a time on the plan,
without taking in its own forecasts) and LightGBM by block (one model per block of 13 weeks). Averages: the world
model with LightGBM directly (Avg; with the world model rolled on, the M5 average) or by block.""" + (" Tuned: settings searched again for 52 weeks on Dominick's." if any("_t" in m for m, _ in cols) else "") + r""" Bold: lowest
of each row, leaving out Timestral with the plan, which could not forecast 52 weeks. Of these, only WM, LGB and Avg
were in the planned test; the rest were added after it ran.}
\label{tab:long-horizon-all}
\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l""" + "c" * len(cols) + r"""}
\toprule
Data set & """ + head + r"""\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table*}
""")
    (OUT / "generated" / "lh_tab_all.tex").write_text(tab)
    tag = {m: "".join(w.capitalize() for w in m.replace("world_model", "wm").replace("lightgbm", "lgb")
                      .replace("average", "avg").replace("seasonal_naive", "snaive").replace("timesfm_mean", "timesfm")
                      .replace("timesfm_plan", "timesfmplan").replace("timestral_plan", "timestral").split("_"))
           for m, _ in cols}
    for m, _ in cols:
        v[f"lhAllLowest{tag[m]}"] = _w(lowest.get(m, 0))
    if "timestral_plan_direct" in dict(cols):
        r = [fa1.get((k, "timestral_plan_direct"), np.nan) for k in keys]
        v["lhTimestralFaOneLo"], v["lhTimestralFaOneHi"] = f"{np.nanmin(r):.1f}", f"{np.nanmax(r):.1f}"
        r = [fa.get((k, "timestral_plan_direct"), np.nan) for k in keys]
        v["lhTimestralFaLo"], v["lhTimestralFaHi"] = f"{np.nanmin(r):.1f}", f"{np.nanmax(r):.1f}"
    if "world_model_rollon" in dict(cols):
        r = [fa.get((k, "world_model_rollon"), np.nan) for k in keys]
        v["lhRollonFaLo"], v["lhRollonFaHi"] = f"{np.nanmin(r):.2f}", f"{np.nanmax(r):.2f}"

    def clear(x, y):
        q = pr[(pr.span == "all") & (pr.a == x) & (pr.b == y)]
        return int((q.hi < 0).sum()), int((q.lo > 0).sum()), len(q)

    for x, y, nm in [("average_byblock", "lightgbm_byblock", "AvgBbLgbBb"), ("world_model_rollon", "lightgbm_direct", "RollonLgb"),
                     ("average_rollon", "lightgbm_direct", "AvgRollonLgb"), ("world_model_rollon", "world_model_blocks", "RollonBlocks"),
                     ("world_model_tdirect", "world_model_direct", "TunedWm"), ("lightgbm_tdirect", "lightgbm_direct", "TunedLgb"),
                     ("average_tdirect", "lightgbm_tdirect", "TunedAvgLgb"), ("average_tdirect", "world_model_tdirect", "TunedAvgWm"),
                     ("world_model_tdirect", "lightgbm_tdirect", "TunedWmLgb")]:
        b, w, n = clear(x, y)
        if n:
            v[f"lhP{nm}Better"], v[f"lhP{nm}Worse"], v[f"lhP{nm}N"] = str(b), str(w), str(n)
    for x, y, nm in [("average_rollon", "lightgbm_direct", "AvgRollonLgb"), ("world_model_rollon", "world_model_blocks", "RollonBlocks")]:
        q = pr[(pr.span == "all") & (pr.key == "m5") & (pr.a == x) & (pr.b == y)]
        if len(q):
            v[f"lhMfive{nm}Better"], v[f"lhMfive{nm}Worse"], v[f"lhMfive{nm}N"] = \
                str(int((q.hi < 0).sum())), str(int((q.lo > 0).sum())), str(len(q))
    for m, nm in [("average_rollon", "AvgRollon"), ("lightgbm_direct", "LgbDirect"), ("world_model_rollon", "WmRollon"),
                  ("world_model_blocks", "WmBlocks")]:
        if ("m5", m) in over.index:
            v[f"lhMfive{nm}"] = f"{over[('m5', m)]:.3f}"
    v["lhRollonCheese"] = f"{over.get(('dominicks_che', 'world_model_rollon'), np.nan):.2f}"
    v["lhRollonCheeseFa"] = f"{fa.get(('dominicks_che', 'world_model_rollon'), np.nan):.2f}"
    return v


if __name__ == "__main__":
    main()
