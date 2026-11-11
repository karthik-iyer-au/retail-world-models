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


def _fmt(vals: list[float]) -> list[str]:
    best = np.nanmin(vals)
    return [("\\textbf{%.3f}" if abs(v - best) < 5e-4 else "%.3f") % v for v in vals]


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
forecast window summed as one block; mean over the paper's windows. Lowest of the three in bold.}
\label{tab:rq-grain}
\resizebox{\textwidth}{!}{\begin{tabular}{l ccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{Store and above, by period} & \multicolumn{3}{c}{Item levels, by period} & \multicolumn{3}{c}{Whole block, all levels}\\
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
Table~\ref{tab:rq-grain}; lowest in bold.}
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
    rows = []
    for f in glob.glob(str(ROOT / "results" / "ftransfer" / "*" / "metrics.json")):
        m = re.search(r"ftransfer_(?P<data>m5|favorita|frat|dominicks_[a-z]+)_(?P<kind>store|item)_"
                      r"(?P<model>world_model|lightgbm)(?:_pooled)?_(?P<form>kept_out|trained_on_all)_[0-9a-f]{8}/", f)
        if not m:
            continue
        for s in json.load(open(f))["splits"]:
            rows.append({**m.groupdict(), "window": s["test_start"], "kept_out": s["kept_out"],
                         "fa": s.get("kept_out_forecast_to_actual")})
    r = pd.DataFrame(rows)
    k = ["data", "kind", "model", "window"]
    o = r[r.form == "kept_out"].set_index(k)
    ref = r[r.form == "trained_on_all"].set_index(k)["kept_out"]
    o = o.assign(comp=ref.reindex(o.index)).dropna(subset=["comp"])
    o["cost"] = o.kept_out - o.comp
    c = o.reset_index()
    rng = np.random.default_rng(0)
    out, lines_ = [], []
    for (d, kd), g in sorted(c.groupby(["data", "kind"]), key=lambda t: (list(KEYS).index(t[0][0]), t[0][1])):
        w = g.pivot(index="window", columns="model", values="cost")
        fa = g.groupby("model").fa.mean()
        x = (w.world_model - w.lightgbm).dropna().to_numpy()
        lo, hi = np.quantile(rng.choice(x, (10000, len(x))).mean(1), [0.05, 0.95])
        clear = "World model" if hi < 0 else ("LightGBM" if lo > 0 else "Neither")
        out.append(dict(data=d, kind=kd, wm=w.world_model.mean(), lgb=w.lightgbm.mean(), lo=lo, hi=hi,
                        fa_wm=fa["world_model"], fa_lgb=fa["lightgbm"], clear=clear))
        lines_.append(f"{KEYS[d]} & {'Items' if kd == 'item' else 'Stores'} & {w.world_model.mean():.3f} & "
                      f"{w.lightgbm.mean():.3f} & [{lo:+.3f}, {hi:+.3f}] & {clear} & {fa['world_model']:.2f} & "
                      f"{fa['lightgbm']:.2f} \\\\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{RQ3, seen and unseen data. A fifth of the items, or of the stores, is kept out of training; the cost is the
WRMSSE on that fifth minus the WRMSSE on the same fifth of the same model trained on everything, mean over windows.
The interval is a 90\% bootstrap interval over windows for the world model's cost minus LightGBM's; ``clearly lower''
names the model whose cost is lower when the interval excludes zero. Forecast over actual is on the kept-out series.}
\label{tab:rq-transfer}
\resizebox{\textwidth}{!}{\begin{tabular}{ll cc c l cc}
\toprule
 & & \multicolumn{2}{c}{Cost of not training on them} & & & \multicolumn{2}{c}{Forecast over actual}\\
\cmidrule(lr){3-4}\cmidrule(lr){7-8}
Data set & Kept out & WM & LGB & WM $-$ LGB, 90\% & Clearly lower & WM & LGB\\
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
