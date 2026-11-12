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
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
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


WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
CANDIDATES = ["world_model", "lightgbm", "ensemble_2", "ensemble", "timesfm_plan"]  # the models RQ1 compares


def _w(n: int) -> str:
    return WORDS[n] if n < len(WORDS) else str(n)


def _paired(x: str, y: str) -> dict:
    d = pd.read_csv(A / "paired.csv").query("set == 'rolling'").drop_duplicates()
    g = d[((d.a == x) & (d.b == y)) | ((d.a == y) & (d.b == x))].copy()
    s = g.a.eq(x).map({True: 1, False: -1})
    lo, hi = g.lo.where(s > 0, -g.hi), g.hi.where(s > 0, -g.lo)
    g["better"], g["worse"] = hi < 0, lo > 0
    g["grain"] = g.level.map(lambda lv: "item" if "item" in lv else "coarse")
    out = {"better": int(g.better.sum()), "worse": int(g.worse.sum()), "n": len(g)}
    for gr, h in g.groupby("grain"):
        out.update({f"{gr}better": int(h.better.sum()), f"{gr}worse": int(h.worse.sum()), f"{gr}n": len(h)})
    return out


def text_values(tr: list) -> dict:
    """Every number quoted in the text of the results by research question, the abstract, the discussion and the
    conclusion (plan entry of 12 November 2026: the paper is rebuilt from the result files by make)."""
    from rwm.evaluation import compare as C

    v = {}
    s = pd.read_csv(A / "scores.csv").query("set == 'rolling'")
    b = pd.read_csv(A / "blocks.csv").query("set == 'rolling'")
    # RQ1: best and worst of the three in each of the 27 cells of the grain table
    best = {m: 0 for m in TRIO}
    worst = dict(best)
    blk = {}
    for k in KEYS:
        p = s[s.key == k].groupby(["level", "model"]).wrmsse.mean().unstack()[TRIO]
        q = b[b.key == k].groupby(["level", "model"]).wrmsse.mean().unstack()[TRIO]
        coarse, item = _groups(p)
        blk[k] = q.mean()
        for col in (p.loc[coarse].mean(), p.loc[item].mean(), q.mean()):
            best[col.idxmin()] += 1
            worst[col.idxmax()] += 1
    v["rqCells"] = 3 * len(KEYS)
    v["rqAvgBestCells"], v["rqAvgWorstCells"] = best["ensemble_2"], worst["ensemble_2"]
    v["rqWmWorstCells"], v["rqLgbWorstCells"] = worst["world_model"], worst["lightgbm"]
    v["rqAvgSecondBlock"] = sum(1 for k in KEYS if blk[k].rank()["ensemble_2"] == 2)
    for k, tag in (("m5", "Mfive"), ("favorita", "Fav"), ("dominicks_sdr", "Sdr")):
        for m, mt in zip(TRIO, ("Wm", "Lgb", "Avg")):
            v[f"rqBlock{tag}{mt}"] = f"{blk[k][m]:.3f}"
    # paired tests
    for (x, y), tag in {("ensemble_2", "world_model"): "AvgWm", ("ensemble_2", "lightgbm"): "AvgLgb",
                        ("world_model", "lightgbm"): "WmLgb"}.items():
        for k2, n in _paired(x, y).items():
            v[f"rqP{tag}{k2.capitalize()}"] = n
    v["rqAvgItemWorseMax"] = max(v["rqPAvgWmItemworse"], v["rqPAvgLgbItemworse"])
    # model confidence set over the models RQ1 compares
    kept = []
    for (k, lv), g in s[s.model.isin(CANDIDATES)].groupby(["key", "level"]):
        wide = g.pivot_table(index="origin", columns="model", values="wrmsse").dropna(axis=1)
        if len(wide) >= 3 and wide.shape[1] >= 2:
            ks = C.model_confidence_set(wide)
            kept += [{"model": m, "in": m in ks} for m in wide.columns]
    kp = pd.DataFrame(kept)
    for m, n in {"world_model": "WorldModel", "lightgbm": "Lightgbm", "ensemble_2": "Average",
                 "ensemble": "Ensemble", "timesfm_plan": "Timesfm"}.items():
        v[f"rqMcs{n}"] = int(kp[kp.model == m]["in"].sum())
    v["rqMcsCells"] = int((kp.model == "ensemble_2").sum())
    # RQ2
    g = pd.read_csv(A / "segments.csv").query("set == 'rolling' and view == 'period'")
    dom = [k for k in KEYS if k.startswith("dominicks")]
    rest = [k for k in KEYS if k not in dom]
    share = g.groupby("key").fast_item_share.mean() * 100
    shortshare = (g[g.segment == "short_history"].groupby("key").short_history_series.mean()
                  / g.groupby("key").series.max() * 100)
    v["rqFastShareDomLo"], v["rqFastShareDomHi"] = f"{share[dom].min():.0f}", f"{share[dom].max():.0f}"
    v["rqFastShareRestLo"], v["rqFastShareRestHi"] = f"{share[rest].min():.0f}", f"{share[rest].max():.0f}"
    v["rqShortShareDomLo"], v["rqShortShareDomHi"] = f"{shortshare[dom].min():.0f}", f"{shortshare[dom].max():.0f}"
    v["rqShortShareRestMax"] = f"{np.ceil(shortshare[rest].max()):.0f}"
    seg = {(k, sg): g[(g.key == k) & (g.segment == sg)].groupby(["level", "model"]).wrmsse.mean().unstack()[TRIO].mean()
           for k in KEYS for sg in ("fast", "slow", "short_history")}
    v["rqAvgBestFast"] = _w(sum(seg[(k, "fast")].idxmin() == "ensemble_2" for k in KEYS))
    v["rqAvgBestSlow"] = _w(sum(seg[(k, "slow")].idxmin() == "ensemble_2" for k in KEYS))
    v["rqShortWmBeatsLgb"] = _w(sum(seg[(k, "short_history")]["world_model"] < seg[(k, "short_history")]["lightgbm"]
                                    for k in KEYS))
    for k, tag in (("favorita", "Fav"), ("m5", "Mfive")):
        v[f"rqShort{tag}Wm"] = f"{seg[(k, 'short_history')]['world_model']:.3f}"
        v[f"rqShort{tag}Lgb"] = f"{seg[(k, 'short_history')]['lightgbm']:.3f}"
    v["rqNineWord"] = _w(len(KEYS))
    # snacks overall, and the forecast by history against the average (overall, all levels)
    allv = {k: s[s.key == k].groupby(["level", "model"]).wrmsse.mean().unstack().mean() for k in KEYS}
    for m, mt in zip(TRIO, ("Wm", "Lgb", "Avg")):
        v[f"rqSna{mt}"] = f"{allv['dominicks_sna'][m]:.3f}"
    if all("by_history" in allv[k] for k in KEYS):
        v["rqRuleLevel"] = _w(sum(abs(allv[k]["by_history"] - allv[k]["ensemble_2"]) <= 0.005 for k in KEYS))
        v["rqRuleSna"] = f"{allv['dominicks_sna']['by_history']:.3f}"
    # RQ3
    items = [r for r in tr if r["kind"] == "item"]
    stores = [r for r in tr if r["kind"] == "store"]
    rng_ = lambda xs, f: (f.format(min(xs)), f.format(max(xs)))  # noqa: E731
    v["rqCostWmItemLo"], v["rqCostWmItemHi"] = rng_([r["cost"]["world_model"] for r in items], "{:.3f}")
    v["rqCostLgbItemLo"], v["rqCostLgbItemHi"] = rng_([r["cost"]["lightgbm"] for r in items], "{:.3f}")
    for m, tag in zip(TRIO, ("Wm", "Lgb", "Avg")):
        lo, hi = rng_([100 * r["fa"][m] for r in items], "{:.0f}")
        v[f"rqFa{tag}ItemLo"], v[f"rqFa{tag}ItemHi"] = lo, hi
    v["rqWmFaItemOff"] = f"{100 * max(abs(r['fa']['world_model'] - 1) for r in items):.0f}"
    v["rqKeptSets"] = _w(len(items))
    v["rqItemWmBeatsAvg"] = _w(sum(r["wm_avg"] == "World model" for r in items))
    v["rqStoreWmBeatsLgb"] = _w(sum(r["wm_lgb"] == "World model" for r in stores))
    v["rqStoreAvgBeatsWm"] = _w(sum(r["wm_avg"] == "Average" for r in stores))
    v["rqStoreWmBeatsAvg"] = _w(sum(r["wm_avg"] == "World model" for r in stores))
    v["rqStoreLevel"] = _w(sum(r["wm_avg"] == "Neither" for r in stores))
    v["rqLgbFaItemLoFrac"] = "two thirds" if min(r["fa"]["lightgbm"] for r in items) < 0.67 else \
        f"{100 * min(r['fa']['lightgbm'] for r in items):.0f}\\%"
    return v


def retailer_values(vals: dict) -> dict:
    v = {}
    own_wm, _ = vals[("favorita", "world_model")]
    own_lgb, _ = vals[("favorita", "lightgbm")]
    c = vals["carried"]
    v["rqCarriedWm"], v["rqOwnWm"] = f"{c['world_model'][0]:.3f}", f"{own_wm[0]:.3f}"
    v["rqCarriedLgb"], v["rqOwnLgb"] = f"{c['lightgbm'][0]:.3f}", f"{own_lgb[0]:.3f}"
    v["rqCarriedWmIxs"] = f"{c['world_model'][1] - own_wm[1]:.3f}"
    v["rqCarriedLgbIxs"] = f"{c['lightgbm'][1] - own_lgb[1]:.3f}"
    gains, losses = [], []
    for k in KEYS:
        for m in ("world_model", "lightgbm"):
            own, pooled = vals[(k, m)]
            if own and pooled:
                gains.append((own[0] - pooled[0], k, m))
                losses.append((pooled[0] - own[0], k, m))
    v["rqPoolGainMax"] = f"{max(gains)[0]:.3f}"
    best_wm = max(x for x in gains if x[2] == "world_model")
    best_lgb = max(x for x in gains if x[2] == "lightgbm")
    worst_wm = max(x for x in losses if x[2] == "world_model")
    worst_lgb = max(x for x in losses if x[2] == "lightgbm")
    v["rqPoolGainWm"], v["rqPoolGainWmSet"] = f"{best_wm[0]:.3f}", KEYS[best_wm[1]].lower()
    v["rqPoolGainLgb"], v["rqPoolGainLgbSet"] = f"{best_lgb[0]:.3f}", KEYS[best_lgb[1]].lower()
    v["rqPoolLossWm"], v["rqPoolLossWmSet"] = f"{worst_wm[0]:.3f}", KEYS[worst_wm[1]].lower()
    v["rqPoolLossLgb"], v["rqPoolLossLgbSet"] = f"{worst_lgb[0]:.3f}", KEYS[worst_lgb[1]].lower()
    v["rqPoolGainMaxTwo"] = f"{np.ceil(max(gains)[0] * 100) / 100:.2f}"
    return v


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t1, wins = grain()
    t2, _ = lines()
    t3, tr = transfer()
    t4, rv = retailer()
    for name, t in (("rq_tab_grain", t1), ("rq_tab_lines", t2), ("rq_tab_transfer", t3), ("rq_tab_retailer", t4)):
        (OUT / f"{name}.tex").write_text(t)
    vals = {f"rqAvgBest{g.capitalize()}": n for g, n in wins.items()}
    vals.update({f"rqAvgBest{g.capitalize()}Word": _w(n) for g, n in wins.items()})
    vals.update(text_values(tr))
    vals.update(retailer_values(rv))
    (OUT / "rq_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()))
    print(len(vals), "values")


if __name__ == "__main__":
    main()
