"""Tables and numbers of paper 1's supporting analyses, rerun on the tuned models (plan entry of 12 November 2026).

  python scripts/make_support_tables.py

Reads results/analysis_tuned (scores of the tuned pair and the comparison models, the ensemble ablation, input
perturbation, mechanism tests, intervals with the agreed fix, decision costs, reconciliation and ranking) and, for
the comparison models on M5, Frat and Favorita, which were not rerun because nothing in them was tuned, the scores of
study F (results/paper1_revision/scores). Writes paper/generated/sup_*.tex and sup_values.tex.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_rq_tables import CELL, DUPLICATE_LEVELS, KEYS, _w  # noqa: E402

A = ROOT / "results" / "analysis_tuned"
GEN = ROOT / "paper" / "generated"
DOM = [k for k in KEYS if k.startswith("dominicks")]
ITEM_BY_STORE = "item_idxstore_id"


def shade_low(vals: list[float], colours: list[str] | None = None) -> list[str]:
    """Bold the lowest; shade it in its model's tint where a colour is given for that column."""
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    best = np.nanmin(arr) if np.isfinite(arr).any() else np.nan
    out = []
    for j, v in enumerate(arr):
        if not np.isfinite(v):
            out.append("--")
        elif abs(v - best) < 5e-4:
            c = colours[j] if colours and colours[j] else None
            out.append((f"\\cellcolor{{{c}}}" if c else "") + f"\\textbf{{{v:.3f}}}")
        else:
            out.append(f"{v:.3f}")
    return out


def _dedup(d: pd.DataFrame) -> pd.DataFrame:
    """Each Dominick's set is one category: its category levels repeat other levels and are counted once."""
    if "level" in d and "key" in d:
        d = d[~(d.key.astype(str).str.startswith("dominicks") & d.level.isin(DUPLICATE_LEVELS))]
    return d


def _read(path: Path) -> pd.DataFrame:
    return _dedup(pd.read_csv(path)) if path.exists() else pd.DataFrame()


def _cat(*paths: Path, keys=("key", "set", "origin", "model", "level")) -> pd.DataFrame:
    frames = [_dedup(pd.read_csv(p)) for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=[k for k in keys if k in d], keep="last")


# ------------------------------------------------------------------ comparison models
COMP = [("world_model", "WM"), ("lightgbm", "LGB"), ("ensemble_2", "Avg"), ("ensemble", "Ens"),
        ("timesfm_plan", "TFM+"), ("timesfm_mean", "TFM"), ("chronos", "Chr"), ("timestral_plan", "TST+"),
        ("timestral_mean", "TST"), ("tft", "TFT"), ("seasonal_naive", "SN")]


def comparison(v: dict) -> str:
    tuned = _read(A / "scores.csv").query("set == 'rolling'")
    comp = _cat(A / "comparison" / "scores.csv", A / "comparison_rest" / "scores.csv").query("set == 'rolling'")
    sf = _cat(ROOT / "results/paper1_revision/scores/studyF_scores_daily.csv",
              ROOT / "results/paper1_revision/scores/studyF_scores_weekly.csv")
    sf = sf[sf.set == "rolling"] if "set" in sf else sf
    rows, beat, table = [], {m: 0 for m, _ in COMP}, {}
    for k, name in KEYS.items():
        score = table[k] = {}
        for m, _ in COMP:
            if m in ("world_model", "lightgbm", "ensemble_2", "ensemble", "timesfm_plan"):
                src = tuned[(tuned.key == k) & (tuned.model == m)]
            elif k in DOM:
                src = comp[(comp.key == k) & (comp.model == m)]
            else:
                src = sf[(sf.key == k) & (sf.model == m)]
            score[m] = src.groupby("level").wrmsse.mean().mean() if len(src) else None
        vals = [score[m] for m, _ in COMP]
        colours = ["wmcell", "lgbcell", "avgcell"] + [None] * (len(COMP) - 3)
        rows.append(f"{name} & " + " & ".join(shade_low(vals, colours)) + r" \\")
        best_other = min((x for m, x in score.items() if x is not None and m not in ("world_model", "lightgbm",
                                                                                    "ensemble_2", "ensemble")),
                         default=None)
        for m, x in score.items():
            if x is not None and m not in ("world_model", "lightgbm", "ensemble_2", "ensemble"):
                beat[m] += x < min(score["world_model"], score["lightgbm"])
    v["supCompBeatsTuned"] = _w(sum(beat.values()))
    untrained = ["timesfm_plan", "timesfm_mean", "chronos", "timestral_plan", "timestral_mean", "seasonal_naive"]
    v["supTfmPlanBestUntrained"] = _w(sum(1 for k in KEYS if all(
        table[k].get("timesfm_plan") is not None and (table[k].get(m) is None or table[k]["timesfm_plan"] <= table[k][m])
        for m in untrained)))
    for k, tag in (("frat", "Frat"), ("dominicks_ana", "Ana")):
        for m, mt in (("timesfm_plan", "TfmPlan"), ("world_model", "Wm"), ("lightgbm", "Lgb"), ("ensemble_2", "Avg"),
                      ("timestral_plan", "TstPlan")):
            x = table[k].get(m)
            v[f"sup{tag}{mt}"] = f"{x:.3f}" if x is not None else "--"
    head = " & ".join(lab for _, lab in COMP)
    return r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Every model on the rolling windows: WRMSSE averaged over all levels, mean over windows. WM and LGB: the tuned
world model and LightGBM; Avg: their average; Ens: the three-model ensemble with TimesFM given the plan; TFM+ and
TFM: TimesFM with and without the plan; Chr: Chronos-Bolt; TST+ and TST: Timestral with and without the plan; TFT:
the Temporal Fusion Transformer, with the settings of the first version; SN: seasonal naive. The comparison models
are scored on cleaned prices on Dominick's; on M5, Frat and Favorita they are the runs of the first revision on the
same windows, since nothing in them was tuned. -- marks a model not run on that data set. """ + \
        r"""Shading marks the lowest: \colorbox{wmcell}{world model}, \colorbox{lgbcell}{LightGBM}, \colorbox{avgcell}{average}.}
\label{tab:sup-comparison}
\resizebox{\textwidth}{!}{\begin{tabular}{l""" + "c" * len(COMP) + r"""}
\toprule
Data set & """ + head + r"""\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table*}
"""


# ------------------------------------------------------------------ ensemble ablation
def ablation(v: dict) -> str:
    sc = _read(A / "ablation" / "scores.csv")
    names = [("ensemble", "Three-model"), ("without_world_model", "No world model"), ("without_lightgbm", "No LightGBM"),
             ("without_timesfm", "No TimesFM"), ("fitted_all_levels", "Fitted weights")]
    rows = []
    hurt = 0
    fitted_worse = 0
    for k, name in KEYS.items():
        g = sc[sc.key == k]
        if g.empty:
            continue
        ixs = [g[(g.model == m) & (g.level == ITEM_BY_STORE)].wrmsse.mean() for m, _ in names]
        rows.append(f"{name} & " + " & ".join(shade_low(ixs)) + r" \\")
        hurt += ixs[1] > ixs[0]
        fitted_worse += ixs[4] > ixs[0]
    pr = pd.read_csv(A / "ablation" / "paired.csv").query("level == @ITEM_BY_STORE")
    for a, tag in (("without_world_model", "NoWm"), ("without_lightgbm", "NoLgb"), ("without_timesfm", "NoTfm"),
                   ("fitted_bottom", "FitBottom"), ("fitted_all_levels", "FitAll")):
        g = pr[(pr.a == a) & (pr.b == "ensemble")]
        v[f"supP{tag}Better"], v[f"supP{tag}Worse"], v[f"supP{tag}N"] = int((g.hi < 0).sum()), int((g.lo > 0).sum()), len(g)
    v["supNoWmWorse"] = _w(hurt)
    v["supFittedWorse"] = _w(fitted_worse)
    v["supAblationSets"] = _w(len(rows))
    return r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{What each member adds to the three-model ensemble: WRMSSE item by store, mean over the rolling windows, of
the ensemble, of the ensemble without each member (without TimesFM it is the average of the two tuned models), and of
the ensemble with weights fitted on earlier windows. Lowest in bold.}
\label{tab:sup-ablation}
\resizebox{\columnwidth}{!}{\begin{tabular}{lccccc}
\toprule
Data set & Three & No WM & No LGB & No TFM & Fitted\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table}
"""


# ------------------------------------------------------------------ input perturbation
def perturbation(v: dict) -> str:
    # perturbation.csv is the first job; perturbation_<key>.csv are the same script run one data set per container
    # (13 November, to finish in time); a data set in its own file is read from there
    parts = {f.stem.split("perturbation_")[1]: _read(f) for f in sorted(A.glob("perturbation_*.csv"))}
    d = _read(A / "perturbation.csv")
    d = pd.concat([d[~d.key.isin([k for k, x in parts.items() if not x.empty])] if not d.empty else d]
                  + [x for x in parts.values() if not x.empty], ignore_index=True)
    if d.empty:
        return "% perturbation pending\n"
    d = d.assign(dbottom=d.bottom - d.bottom_true)
    occ = d[d.operation == "occlude"].groupby(["key", "model", "input"]).dbottom.mean()
    for m, tag in (("world_model", "Wm"), ("lightgbm", "Lgb")):
        for inp, itag in (("price", "Price"), ("whole_plan", "Plan")):
            x = occ.xs((m, inp), level=("model", "input")) if (m, inp) in occ.droplevel(0).index else pd.Series(dtype=float)
            v[f"supPert{itag}Helps{tag}"] = _w(int((x > 0).sum()))
    v["supPertSets"] = _w(int(d.key.nunique()))
    un = d[(d.input == "whole_plan") & (d.operation == "occlude")].groupby(["key", "model"]).units_change_on.mean()
    for m, tag in (("world_model", "Wm"), ("lightgbm", "Lgb")):
        x = 100 * un.xs(m, level="model")
        v[f"supPertUnits{tag}Lo"], v[f"supPertUnits{tag}Hi"] = f"{-x.max():.0f}", f"{-x.min():.0f}"
    h = d[(d.model == "world_model") & d.input.str.startswith("history_last")].groupby("key").dbottom.max()
    v["supPertHistHelps"] = _w(int((h > 0).sum()))
    rows = []
    for k, name in KEYS.items():
        cells = []
        for m in ("world_model", "lightgbm"):
            g = d[(d.key == k) & (d.model == m)]
            price = g[(g.input == "price") & (g.operation == "occlude")].dbottom.mean() if len(g) else np.nan
            plan = g[(g.input == "whole_plan") & (g.operation == "occlude")].dbottom.mean() if len(g) else np.nan
            units = g[(g.input == "whole_plan") & (g.operation == "occlude")].units_change_on.mean() if len(g) else np.nan
            cells += [price, plan, 100 * units if np.isfinite(units) else np.nan]
        hist = d[(d.key == k) & (d.model == "world_model") & d.input.str.startswith("history_last")]
        hmax = hist.dbottom.max() if len(hist) else np.nan
        cells.append(hmax)
        fmt = lambda x, pct=False: "--" if not np.isfinite(x) else (f"{x:+.0f}\\%" if pct else f"{x:+.3f}")  # noqa: E731
        rows.append(f"{name} & {fmt(cells[0])} & {fmt(cells[1])} & {fmt(cells[2], True)} & {fmt(cells[3])} & "
                    f"{fmt(cells[4])} & {fmt(cells[5], True)} & {fmt(cells[6])} \\\\")
    return r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Which plan inputs each tuned model uses. Change in WRMSSE item by store, mean over windows, when the price
is set to the regular price (price off) and when every plan input is set to nothing planned (plan off); the change
in forecast units on the rows where the plan had something on; and, for the world model, the largest change when its
history before the last few periods is removed. A positive change means the input was helping.}
\label{tab:sup-perturbation}
\begin{tabular}{l ccc ccc c}
\toprule
 & \multicolumn{3}{c}{World model} & \multicolumn{3}{c}{LightGBM} & World model\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}
Data set & Price off & Plan off & Units, plan rows & Price off & Plan off & Units, plan rows & History cut\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table*}
"""


# ------------------------------------------------------------------ mechanism tests
def mechanism(v: dict) -> str:
    d = _read(A / "mechanism.csv")
    if d.empty:
        return "% mechanism pending\n"
    d["data"] = d.config.str.replace("final_", "").str.replace("_world_model", "")
    d["timing"] = 100 * (d.bottom_permuted / d.bottom_true - 1)
    rows, worse, probe_gain = [], 0, []
    for k, name in KEYS.items():
        g = d[d.data == k]
        if g.empty:
            continue
        r = g.mean(numeric_only=True)
        worse += r.timing > 0
        probe_gain.append(r.probe_after_r2 - r.probe_before_r2)
        pc = lambda x: "--" if not np.isfinite(x) else f"{100 * x:+.0f}\\%"  # noqa: E731
        r2 = lambda x: "--" if not np.isfinite(x) else f"{x:.2f}"  # noqa: E731
        rows.append(f"{name} & {r.timing:+.0f}\\% & {pc(r.persist_h1)} & {pc(r.persist_h2)} & {pc(r.persist_h4)} & "
                    f"{r2(r.probe_before_r2)} & {r2(r.probe_after_r2)} \\\\")
    d2 = d[d.data.isin(KEYS)]
    v["supTimingWorse"], v["supMechSets"] = _w(int(worse)), _w(len(rows))
    v["supTimingLo"], v["supTimingHi"] = f"{d2.groupby('data').timing.mean().min():.0f}", f"{d2.groupby('data').timing.mean().max():.0f}"
    v["supPersistTwoLo"] = f"{100 * d2.groupby('data').persist_h2.mean().min():.0f}"
    v["supPersistTwoHi"] = f"{100 * d2.groupby('data').persist_h2.mean().max():.0f}"
    v["supPersistRiseSets"] = _w(int((d2.groupby('data').persist_h2.mean() > 0).sum()))
    v["supProbeGainMax"] = f"{max(probe_gain):.2f}" if probe_gain else "--"
    v["supProbeGainMin"] = f"{min(probe_gain):.2f}" if probe_gain else "--"
    return r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{The tuned world model's carried state, mean over windows. Timing: change in WRMSSE item by store when the
plan's timing is shuffled within each series. Carry-over: forecast change in periods 1, 2 and 4 after a promotion and a
20\% price cut added in period 1 to series with nothing planned, as a share of the forecast without them. Probe: $R^2$
of log sales after the window from the state before the window and from the state rolled through it (first of the
five copies).}
\label{tab:sup-mechanism}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}
\toprule
Data set & Timing & Period 1 & Period 2 & Period 4 & Probe before & Probe after\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table}
"""


# ------------------------------------------------------------------ intervals, decisions, reconciliation, ranking
TRIO = [("world_model", "WM"), ("lightgbm", "LGB"), ("ensemble_2", "Avg")]


def decisions(v: dict) -> str:
    # the rerun with the corrected category-plan cost (plan entry of 13 November 2026)
    d = _cat(*sorted(A.glob("decisions_fixed_*/decisions.csv")), keys=("key", "origin", "model", "decision"))
    if d.empty:
        return "% decisions pending\n"
    rows = []
    low = {m: 0 for m, _ in TRIO}
    for k, name in KEYS.items():
        g = d[d.key == k]
        if g.empty:
            continue
        cells = []
        for dec in ("replenish_0.8", "category_planning"):
            vals = [g[(g.model == m) & (g.decision == dec)].cost.mean() for m, _ in TRIO]
            if dec == "category_planning":
                low[TRIO[int(np.nanargmin(vals))][0]] += 1
                vals = [100 * x for x in vals]
                cells += [("\\cellcolor{" + CELL[j] + "}\\textbf{" + f"{x:.1f}" + "}") if x == min(vals) else f"{x:.1f}"
                          for j, x in enumerate(vals)]
            else:
                tot = [g[(g.model == m) & (g.decision == dec)].cost.sum() for m, _ in TRIO]
                ref = tot[1]
                cells += [f"{x / ref:.3f}" if ref else "--" for x in tot]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    v["supCatPlanAvgLowest"] = _w(low["ensemble_2"])
    v["supCatPlanLgbLowest"] = _w(low["lightgbm"])
    v["supCatPlanWmLowest"] = _w(low["world_model"])
    order = {}
    for k in KEYS:
        g = d[(d.key == k) & (d.decision == "replenish_0.8")]
        if not g.empty:
            order[k] = {m: g[g.model == m].cost.sum() for m, _ in TRIO}
    # below means below at the table's three decimals (snacks against LightGBM is 0.99994 and shows as 1.000)
    v["supOrderAvgBelowLgb"] = _w(sum(round(o["ensemble_2"] / o["lightgbm"], 3) < 1 for o in order.values()))
    v["supOrderAvgBelowWm"] = _w(sum(round(o["ensemble_2"] / o["world_model"], 3) < 1 for o in order.values()))
    v["supOrderSets"] = _w(len(order))
    v["supCatPlanSets"] = _w(len(rows))
    return r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Decision costs of the tuned models. Store orders: cost of ordering each forecast's 0.8 quantile, summed over
windows, relative to LightGBM's. Category plan: absolute error in dollars over sales in dollars at category by store
(\%), mean over windows; lowest shaded.}
\label{tab:sup-decisions}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Store orders, ratio 0.8} & \multicolumn{3}{c}{Category plan (\%)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Data set & WM & LGB & Avg & WM & LGB & Avg\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table}
"""


def intervals(v: dict) -> str:
    d = _cat(A / "intervals_fixed" / "intervals.csv")
    if d.empty:
        return "% intervals pending\n"
    for lv, lt in ((ITEM_BY_STORE, "Ixs"), ("total", "Total")):
        for m, mt in (("world_model", "Wm"), ("lightgbm", "Lgb"), ("ensemble_2", "Avg")):
            h = d[(d.model == m) & (d.level == lv)]
            v[f"supCov{lt}{mt}"] = f"{100 * h.coverage_80.mean():.0f}"
            v[f"supCovNinetyFive{lt}{mt}"] = f"{100 * h.coverage_95.mean():.0f}"
            v[f"supSpl{lt}{mt}"] = f"{h.spl.mean():.3f}"
    rows = []
    for k, name in KEYS.items():
        g = d[d.key == k]
        if g.empty:
            continue
        cells = []
        for lv in (ITEM_BY_STORE, "total"):
            for m, _ in TRIO:
                h = g[(g.model == m) & (g.level == lv)]
                cells.append(f"{100 * h.coverage_80.mean():.0f}" if len(h) else "--")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Coverage (\%) of each model's 80\% interval, from its own errors in earlier windows, item by store and at
the total, mean over the scored windows.}
\label{tab:sup-intervals}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Item by store} & \multicolumn{3}{c}{Total}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Data set & WM & LGB & Avg & WM & LGB & Avg\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table}
"""


def reconcile(v: dict) -> str:
    r = _cat(A / "reconcile.csv", *sorted(A.glob("idr*/reconcile.csv")), keys=("key", "origin", "method", "level"))
    s = _read(A / "scores.csv").query("set == 'rolling'")
    if r.empty:
        return "% reconciliation pending\n"
    rows = []
    beats = 0
    for k, name in KEYS.items():
        g = r[r.key == k]
        if g.empty:
            continue
        ets = g[g.method.str.startswith("ets")].groupby(["method", "level"]).wrmsse.mean().groupby("method").mean()
        best_ets = ets.min()
        mixed = g[~g.method.str.startswith("ets")].groupby(["method", "level"]).wrmsse.mean().groupby("method").mean()
        avg = s[(s.key == k) & (s.model == "ensemble_2")].groupby("level").wrmsse.mean().mean()
        vals = [best_ets, mixed.get("world_model_with_ets_wls_var", np.nan), mixed.get("lightgbm_with_ets_wls_var", np.nan), avg]
        beats += avg < best_ets
        rows.append(f"{name} & " + " & ".join(shade_low(vals, [None, None, None, "avgcell"])) + r" \\")
    v["supAvgBeatsEts"] = _w(beats)
    v["supReconSets"] = _w(len(rows))
    return r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Reconciliation against summing one forecast: WRMSSE averaged over all levels, mean over windows, of the best
reconciled ETS forecast, of each tuned model combined with ETS above it by weighted least squares, and of the average
summed upward.}
\label{tab:sup-reconcile}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccc}
\toprule
Data set & Best ETS & WM + ETS & LGB + ETS & Avg\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}}
\end{table}
"""


def main() -> None:
    v = {}
    parts = {"sup_tab_comparison": comparison, "sup_tab_ablation": ablation, "sup_tab_perturbation": perturbation,
             "sup_tab_mechanism": mechanism, "sup_tab_decisions": decisions, "sup_tab_intervals": intervals,
             "sup_tab_reconcile": reconcile}
    for name, f in parts.items():
        (GEN / f"{name}.tex").write_text(f(v))
    (GEN / "sup_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in v.items()))
    print(v)


if __name__ == "__main__":
    main()
