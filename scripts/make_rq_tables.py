"""Tables of the results by research question (paper 1, Framework section; plan entries of 11 November 2026).

Reads the tuned-model analysis (results/analysis_tuned: scores.csv, blocks.csv, segments.csv, mcs.csv, paired.csv),
the transfer runs (results/ftransfer), the pooled runs (results/pooled) and the final runs (results/final), and writes
paper/generated/rq_tab_grain.tex, rq_tab_lines.tex, rq_tab_transfer.tex, rq_tab_retailer.tex and rq_values.tex.

  python scripts/make_rq_tables.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "analysis_tuned"
OUT = ROOT / "paper" / "generated"
KEYS = {"dominicks_ana": "Analgesics", "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese",
        "dominicks_fre": "Frozen entrees", "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snacks",
        "frat": "Frat", "m5": "M5", "favorita": "Favorita"}
TRIO = ["world_model", "lightgbm", "ensemble_2"]


CELL = ["wmcell", "lgbcell", "avgcell"]  # world model, LightGBM, average (paper/main.tex)
KEY = (r"Shading marks the lowest of the three: \colorbox{wmcell}{world model}, \colorbox{lgbcell}{LightGBM}, "
       r"\colorbox{avgcell}{average}.")


def _fmt(vals: list[float]) -> list[str]:
    """Values in the order world model, LightGBM, average; the lowest is bold and shaded in its model's tint."""
    best = np.nanmin(vals)
    return [(f"\\cellcolor{{{CELL[j]}}}\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}")
            for j, v in enumerate(vals)]


def _shade(clear: str) -> str:
    cell = {"World model": "wmcell", "LightGBM": "lgbcell"}.get(clear)
    return f"\\cellcolor{{{cell}}}{clear}" if cell else clear


def _groups(p: pd.DataFrame) -> tuple[list, list]:
    item = [lv for lv in p.index if "item" in lv]
    return [lv for lv in p.index if lv not in item], item


def grain() -> tuple[str, dict]:
    s = pd.read_csv(A / "scores.csv").query("set == 'rolling'")
    b = pd.read_csv(A / "blocks.csv").query("set == 'rolling'")
    rows, wins = [], {"coarse": 0, "item": 0, "block": 0}
    for k, name in KEYS.items():
        p = s[s.key == k].groupby(["level", "model"]).wrmsse.mean().unstack()[TRIO]
        q = b[b.key == k].groupby(["level", "model"]).wrmsse.mean().unstack()[TRIO]
        coarse, item = _groups(p)
        c, i, bl = p.loc[coarse].mean(), p.loc[item].mean(), q.mean()
        for g, v in (("coarse", c), ("item", i), ("block", bl)):
            wins[g] += v.idxmin() == "ensemble_2"
        rows.append(f"{name} & " + " & ".join(_fmt(list(c)) + _fmt(list(i)) + _fmt(list(bl))) + r" \\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{RQ1, grain. WRMSSE of the tuned world model (WM), the tuned LightGBM (LGB) and their equal-weight average
(Avg), averaged over the levels at store grain and above, over the item levels, and over every level for the whole
forecast window summed as one block; mean over the paper's windows. """ + KEY + r"""}
\label{tab:rq-grain}
\resizebox{\textwidth}{!}{\begin{tabular}{l ccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{Store and above} & \multicolumn{3}{c}{Item levels} & \multicolumn{3}{c}{Whole block}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Data set & WM & LGB & Avg & WM & LGB & Avg & WM & LGB & Avg\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table*}
"""
    return tab, wins


def lines() -> tuple[str, dict]:
    d = pd.read_csv(A / "segments.csv").query("set == 'rolling' and view == 'period'")
    rows, facts = [], {}
    for k, name in KEYS.items():
        g = d[d.key == k]
        share = g.fast_item_share.mean()
        cells = [f"{100 * share:.0f}\\%"]
        for seg in ("fast", "slow", "short_history"):
            p = g[g.segment == seg].groupby(["level", "model"]).wrmsse.mean().unstack()
            v = p[TRIO].mean() if set(TRIO) <= set(p.columns) else pd.Series([np.nan] * 3, index=TRIO)
            cells += _fmt(list(v))
            facts[(k, seg)] = v
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{RQ2, nature of the line. Share of items that make 80\% of unit sales over the year before the forecast, and
WRMSSE averaged over every level, by period, on the fast lines (those items), the slow lines (the other items) and the
series with short history (first sale less than 26 weeks or 182 days before the forecast). WM, LGB and Avg as in
Table~\ref{tab:rq-grain}. """ + KEY + r"""}
\label{tab:rq-lines}
\resizebox{\textwidth}{!}{\begin{tabular}{l c ccc ccc ccc}
\toprule
 & Items making & \multicolumn{3}{c}{Fast lines} & \multicolumn{3}{c}{Slow lines} & \multicolumn{3}{c}{Short history}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Data set & 80\% of units & WM & LGB & Avg & WM & LGB & Avg & WM & LGB & Avg\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table*}
"""
    return tab, facts


def transfer() -> tuple[str, list]:
    """Kept-out series scored for each model and the average (part keptout of scripts/revision_analysis.py, which
    reproduces the transfer runs' own kept-out scores); the cost is against the trained-on-all runs."""
    d = pd.read_csv(A / "keptout.csv")
    M = ["world_model", "lightgbm", "ensemble_2"]
    rng = np.random.default_rng(0)
    out, lines_ = [], []

    def clear(x: np.ndarray, a: str, b: str) -> str:
        lo, hi = np.quantile(rng.choice(x, (10000, len(x))).mean(1), [0.05, 0.95])
        return a if hi < 0 else (b if lo > 0 else "Neither")

    for key in KEYS:
        for kind in ("item", "store"):
            ko = d[(d.key == key) & (d.set == f"kept_out_{kind}")].pivot_table(index="origin", columns="model",
                                                                               values="kept_out")
            if ko.empty or not set(M) <= set(ko.columns):
                continue
            al = d[(d.key == key) & (d.set == f"trained_on_all_{kind}")].pivot_table(index="origin", columns="model",
                                                                                     values="kept_out")
            cost = (ko[M] - al[M]).mean()
            fa = d[(d.key == key) & (d.set == f"kept_out_{kind}")].groupby("model").forecast_to_actual.mean()
            wl = clear((ko.world_model - ko.lightgbm).to_numpy(), "World model", "LightGBM")
            wa = clear((ko.world_model - ko.ensemble_2).to_numpy(), "World model", "Average")
            out.append(dict(data=key, kind=kind, score=ko[M].mean(), cost=cost, fa=fa[M], wm_lgb=wl, wm_avg=wa))
            shade = lambda t: _shade(t) if t != "Average" else f"\\cellcolor{{avgcell}}{t}"  # noqa: E731
            lines_.append(f"{KEYS[key]} & {'Items' if kind == 'item' else 'Stores'} & "
                          + " & ".join(_fmt(list(ko[M].mean()))) + " & "
                          + " & ".join(f"{v:.3f}" for v in cost) + " & "
                          + " & ".join(f"{v:.2f}" for v in fa[M]) + f" & {shade(wl)} & {shade(wa)} \\\\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{RQ3, seen and unseen data. A fifth of the items, or of the stores, is kept out of training. WRMSSE on that
fifth for the world model (WM), LightGBM (LGB) and their average (Avg); the cost of not training on them, which is that
WRMSSE minus the WRMSSE on the same fifth of the same forecast trained on everything; and forecast over actual on the
fifth. Means over windows. The last two columns name the forecast that is clearly lower on the kept-out fifth when a
90\% bootstrap interval over windows of the difference excludes zero. """ + KEY + r"""}
\label{tab:rq-transfer}
\resizebox{\textwidth}{!}{\begin{tabular}{ll ccc ccc ccc ll}
\toprule
 & & \multicolumn{3}{c}{WRMSSE, kept out} & \multicolumn{3}{c}{Cost of not training on them} & \multicolumn{3}{c}{Forecast over actual} & \multicolumn{2}{c}{Clearly lower}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-13}
Data set & Kept out & WM & LGB & Avg & WM & LGB & Avg & WM & LGB & Avg & WM or LGB & WM or Avg\\
\midrule
""" + "\n".join(lines_) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table*}
"""
    return tab, out


def _score(pattern: str) -> tuple[float, float] | None:
    fs = sorted(glob.glob(str(ROOT / "results" / pattern / "metrics.json")))
    if not fs:
        return None
    m = json.load(open(fs[-1]))
    ib = [lv["wrmsse"] for s in m["splits"] for lv in s["by_level"] if lv["level"] == "item_idxstore_id"]
    return m["wrmsse"], float(np.mean(ib))


def retailer() -> tuple[str, dict]:
    rows, vals = [], {}
    f = lambda t: "--" if t is None else f"{t[0]:.3f} / {t[1]:.3f}"  # noqa: E731
    for k, name in KEYS.items():
        cells = []
        for mdl in ("world_model", "lightgbm"):
            own = _score(f"final/*_final_{k}_{mdl}_*")
            suf = "_v2" if (mdl == "lightgbm" and k not in ("m5", "favorita")) else ""
            pooled = _score(f"pooled/*_pooled_{k}_{mdl}{suf}_*")
            cells += [f(own), f(pooled)]
            vals[(k, mdl)] = (own, pooled)
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    carried = {mdl: _score(f"ftransfer/*_fnewretailer_favorita_{mdl}_*") for mdl in ("world_model", "lightgbm")}
    vals["carried"] = carried
    rows.append(r"\midrule")
    rows.append("Favorita, M5's settings & " + f(carried["world_model"]) + " & & " + f(carried["lightgbm"]) + r" & \\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{RQ4, transferability. WRMSSE overall / item by store of each tuned model trained on its own data set and
trained on it together with other data sets (each Dominick's category with the other five and Frat; Frat with the
six Dominick's categories; M5 with Favorita and Favorita with M5). The last row carries M5's tuned settings
unchanged to Favorita, to be read against Favorita's own-data row.}
\label{tab:rq-retailer}
\resizebox{\textwidth}{!}{\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{World model} & \multicolumn{2}{c}{LightGBM}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Data set & Own data & Pooled & Own data & Pooled\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table*}
"""
    return tab, vals


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t1, wins = grain()
    t2, _ = lines()
    t3, tr = transfer()
    t4, _ = retailer()
    for name, t in (("rq_tab_grain", t1), ("rq_tab_lines", t2), ("rq_tab_transfer", t3), ("rq_tab_retailer", t4)):
        (OUT / f"{name}.tex").write_text(t)
    mc = pd.read_csv(A / "mcs.csv")
    names = {"world_model": "WorldModel", "lightgbm": "Lightgbm", "ensemble_2": "Average",
             "ensemble": "Ensemble", "timesfm_plan": "Timesfm"}
    vals = {f"rqMcs{n}": int(mc[mc.model == m].in_set.sum()) for m, n in names.items()}
    vals["rqMcsCells"] = int(mc[mc.model == "ensemble_2"].shape[0])
    vals.update({f"rqAvgBest{g.capitalize()}": n for g, n in wins.items()})
    (OUT / "rq_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()))
    print(vals)


if __name__ == "__main__":
    main()
