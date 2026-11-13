"""Figures and tables of the online supplement's ablation, occlusion, calibration and exploration sections.

  cd paper/figs && python make_supplement_figures.py

Reads results/analysis_tuned (ensemble ablation, input perturbation, intervals with the agreed fix), the logs of the
loss and JEPA rounds of the tuned searches (results/research/lossT_*, hybridU_*) and the first version's loss ledger
(results/paper1_revision/tables/loss_ledger.csv). Writes paper/figs/fig_sup_*.pdf, paper/generated/sup_tab_explore.tex,
sup_tab_devlosses.tex and supfig_values.tex. Every number comes from those files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from style import COLOR, EDGE, FULL, GRID, INK, MUTED, despine, panel_label, save, use_style  # noqa: E402
from make_rq_tables import KEYS, _w  # noqa: E402
from make_support_tables import _cat, _read  # noqa: E402

A = ROOT / "results" / "analysis_tuned"
LOGS = ROOT / "results" / "research"
GEN = ROOT / "paper" / "generated"
IXS = "item_idxstore_id"
NAMES = list(KEYS.values())
WM, LGB, AVG, TFM = COLOR["world"], COLOR["lgb"], "#009E73", COLOR["tfm_plan"]

LEVELS = [("total", "Total"), ("cat_id", "Category"), ("store_id", "Store"), ("cat_idxstore_id", "Category\nby store"),
          ("item_id", "Item"), (IXS, "Item\nby store")]


def _search(prefix: str, key: str) -> str:
    if key.startswith("dominicks"):
        return f"{prefix}_clean_avg5_world_model_{key}"
    return {"frat": f"{prefix}_avg5_world_model_frat", "m5": f"{prefix}_pure_m5",
            "favorita": f"{prefix}_r1_avg5_world_model_favorita"}[key]


def _log(name: str) -> list[dict]:
    """A search log with restarts read once (a restart reads finished runs back)."""
    p = LOGS / name / "log.jsonl"
    if not p.exists():
        return []
    out, seen = [], set()
    for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()):
        k = (r["kind"], r.get("round"), r.get("change"))
        if r["kind"] in ("rung", "try", "kept") and (k in seen or r.get("score") is None):
            continue
        seen.add(k)
        out.append(r)
    return out


# ------------------------------------------------------------------ ablation
ABL = [("without_world_model", "Without the world model", "o", WM),
       ("without_lightgbm", "Without LightGBM", "s", LGB),
       ("without_timesfm", "Without TimesFM (the average)", "P", TFM),
       ("fitted_all_levels", "Weights fitted on earlier windows", "D", MUTED)]


def fig_ablation(v: dict):
    sc = _read(A / "ablation" / "scores.csv")
    sc = sc[sc.level == IXS].groupby(["key", "model"]).wrmsse.mean()
    w = pd.read_csv(A / "ablation" / "weights.csv").groupby(["key", "target"])[
        ["w_world_model", "w_lightgbm", "w_timesfm_plan"]].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.45})
    y = np.arange(len(KEYS))[::-1]
    worse = {m: 0 for m, *_ in ABL}
    for j, (m, lab, mk, c) in enumerate(ABL):
        d = [sc.get((k, m), np.nan) - sc.get((k, "ensemble"), np.nan) for k in KEYS]
        worse[m] = int(np.nansum(np.array(d) > 0))
        a.scatter(d, y + (j - 1.5) * 0.15, marker=mk, s=18, color=c, edgecolor="white", linewidth=0.5, label=lab,
                  zorder=3)
    a.axvline(0, color=INK, lw=0.6)
    a.set_yticks(y, NAMES)
    a.set_xlabel("Change in WRMSSE item by store against the three-model ensemble")
    a.grid(axis="x")
    a.text(0.99, 1.01, "right of the line: the member was helping", transform=a.transAxes, ha="right", va="bottom",
           fontsize=6.6, color=MUTED)
    despine(a)
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=6.6, handletextpad=0.2, columnspacing=0.8)
    panel_label(a, "a")
    hgt = 0.36
    for t, off, alpha in (("bottom", hgt / 2, 1.0), ("total", -hgt / 2, 0.55)):
        left = np.zeros(len(KEYS))
        for col, c in (("w_world_model", WM), ("w_lightgbm", LGB), ("w_timesfm_plan", TFM)):
            vals = np.array([w.loc[(k, t), col] if (k, t) in w.index else np.nan for k in KEYS])
            b.barh(y + off, vals, hgt, left=left, color=c, alpha=alpha, edgecolor="white", linewidth=0.4)
            left += np.nan_to_num(vals)
    b.set_yticks(y, NAMES)
    b.set_xlim(0, 1)
    b.set_xlabel("Weight fitted on earlier windows (mean over windows)")
    despine(b)
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (WM, LGB, TFM)]
             + [plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=a_) for a_ in (1.0, 0.55)],
             labels=["World model", "LightGBM", "TimesFM with the plan", "Fitted to item by store (upper bar)",
                     "Fitted to the total (lower bar)"],
             loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=6.6, handletextpad=0.3, columnspacing=0.8)
    panel_label(b, "b")
    save(fig, HERE / "fig_sup_ablation.pdf")
    v["sfAblNoWmWorse"] = _w(worse["without_world_model"])
    v["sfAblNoLgbWorse"] = _w(worse["without_lightgbm"])
    v["sfAblNoTfmWorse"] = _w(worse["without_timesfm"])
    wb = w.xs("bottom", level="target")
    wt = w.xs("total", level="target")
    v["sfWtTfmBottomMean"] = f"{wb.w_timesfm_plan.mean():.2f}"
    v["sfWtTfmTotalMean"] = f"{wt.w_timesfm_plan.mean():.2f}"
    v["sfWtWmBottomMean"] = f"{wb.w_world_model.mean():.2f}"
    v["sfWtLgbBottomMean"] = f"{wb.w_lightgbm.mean():.2f}"


# ------------------------------------------------------------------ occlusion
INPUTS = [("price", "Price"), ("promo", "Promotion"), ("display", "Display"), ("feature", "Feature ad"),
          ("tpr_only", "Price cut only"), ("event", "Events"), ("snap", "SNAP days"), ("payday", "Paydays"),
          ("whole_plan", "Whole plan")]


def _perturbation() -> pd.DataFrame:
    parts = {f.stem.split("perturbation_")[1]: _read(f) for f in sorted(A.glob("perturbation_*.csv"))}
    d = _read(A / "perturbation.csv")
    d = pd.concat([d[~d.key.isin([k for k, x in parts.items() if not x.empty])]] + [x for x in parts.values()
                                                                                     if not x.empty], ignore_index=True)
    return d.assign(dbottom=d.bottom - d.bottom_true)


def fig_occlusion(v: dict):
    d = _perturbation()
    occ = d[d.operation == "occlude"].groupby(["key", "model", "input"]).dbottom.mean()
    per = d[d.operation == "permute"].groupby(["key", "model", "input"]).dbottom.mean()
    fig = plt.figure(figsize=(FULL, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.12)
    gs.update(right=0.98)
    mats = {}
    for m in ("world_model", "lightgbm"):
        mats[m] = np.array([[occ.get((k, m, i), np.nan) for i, _ in INPUTS] for k in KEYS])
    lim = np.nanmax(np.abs(np.concatenate([mats["world_model"], mats["lightgbm"]])))
    norm = TwoSlopeNorm(0, -lim * 0.25, lim)
    for j, (m, title) in enumerate((("world_model", "World model"), ("lightgbm", "LightGBM"))):
        ax = fig.add_subplot(gs[0, j])
        mm = np.ma.masked_invalid(mats[m])
        im = ax.pcolormesh(np.arange(mm.shape[1] + 1) - 0.5, np.arange(mm.shape[0] + 1) - 0.5, mm, cmap="RdBu_r",
                           norm=norm, edgecolors="white", linewidth=0.6)
        ax.set_xlim(-0.5, mm.shape[1] - 0.5)
        ax.set_ylim(mm.shape[0] - 0.5, -0.5)
        ax.set_facecolor("#f4f5f7")
        for r in range(mm.shape[0]):
            for c in range(mm.shape[1]):
                if np.isfinite(mats[m][r, c]):
                    x = mats[m][r, c]
                    ax.text(c, r, f"{x:.2f}".replace("-0.00", "0.00").replace("0.", "."), ha="center", va="center",
                            fontsize=6.0, color="white" if abs(x) > lim * 0.55 else INK)
        ax.set_xticks(range(len(INPUTS)), [lab for _, lab in INPUTS], rotation=55, ha="right")
        ax.set_yticks(range(len(KEYS)), NAMES if j == 0 else [])
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        panel_label(ax, "ab"[j], dx=-0.02 if j else -0.3)
    cax = fig.add_axes([0.13, -0.17, 0.4, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Change in WRMSSE item by store with the input set to nothing planned (red: the input helps)",
                 fontsize=6.4)
    cb.ax.tick_params(labelsize=6.2)
    ax = fig.add_subplot(gs[0, 2])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    h = d[(d.model == "world_model") & d.input.str.startswith("history_last")].copy()
    h["kept"] = h.input.str.replace("history_last_", "").astype(int)
    hm = h.groupby(["key", "kept"]).dbottom.mean().reset_index()
    cols = plt.cm.tab10.colors
    for jx, (k, name) in enumerate(KEYS.items()):
        g = hm[hm.key == k].sort_values("kept")
        if g.empty:
            continue
        ax.plot(g.kept, g.dbottom, marker="o", ms=3, lw=0.9, color=cols[jx], label=name)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.6)
    ax.set_ylim(-0.02, 12)
    ax.set_yticks([0, 0.05, 0.1, 1, 10], ["0", "0.05", "0.1", "1", "10"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([4, 13, 28, 56], ["4", "13", "26, 28", "56"])
    ax.set_xlim(3, 70)
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(-0.05, -0.2), ncol=3, handlelength=1.0,
              columnspacing=0.6, labelspacing=0.25)
    ax.set_xlabel("Periods of history kept")
    ax.set_ylabel("Change in WRMSSE item by store")
    ax.set_title("World model, history cut")
    ax.grid(axis="y")
    despine(ax)
    panel_label(ax, "c")
    save(fig, HERE / "fig_sup_occlusion.pdf")
    # occlusion against permutation: shuffling the plan across rows keeps its distribution but breaks its timing
    both = pd.concat([occ.rename("occ"), per.rename("per")], axis=1).dropna()
    both = both[both.index.get_level_values("input") != "whole_plan"]
    v["sfPermAgree"] = _w(int(((both.occ > 0) == (both.per > 0)).sum()))
    v["sfPermPairs"] = _w(len(both))
    pr = occ.xs("promo", level="input") if "promo" in occ.index.get_level_values("input") else pd.Series(dtype=float)
    v["sfPromoSets"] = _w(pr.index.get_level_values("key").nunique())
    pw = pr.xs("world_model", level="model")
    pl = pr.xs("lightgbm", level="model")
    v["sfPromoWmHelps"] = _w(int((pw > 0).sum()))
    v["sfPromoLgbHelps"] = _w(int((pl > 0).sum()))
    cal = occ[occ.index.get_level_values("input").isin(["event", "snap", "payday"])]
    v["sfCalendarMax"] = f"{cal.max():.3f}"
    v["sfFavHistFour"] = f"{hm[(hm.key == 'favorita') & (hm.kept == 4)].dbottom.iloc[0]:.1f}"
    shortest = hm.loc[hm.groupby("key").kept.idxmin()]
    v["sfHistShortWorse"] = _w(int((shortest.dbottom > 0).sum()))
    v["sfHistSets"] = _w(len(shortest))


# ------------------------------------------------------------------ calibration and reliability
TRIO = [("world_model", "World model", WM, "o"), ("lightgbm", "LightGBM", LGB, "s"),
        ("ensemble_2", "Average", AVG, "D")]


def fig_calibration(v: dict):
    d = _cat(A / "intervals_fixed" / "intervals.csv")
    per_key = d.groupby(["key", "model", "level"])[["coverage_50", "coverage_80", "coverage_95", "spl"]].mean()
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.55), gridspec_kw={"wspace": 0.42})
    nominal = np.array([50, 80, 95])
    for ax, lv, title, letter in ((axs[0], IXS, "Item by store", "a"), (axs[1], "total", "Total", "b")):
        ax.plot([40, 100], [40, 100], color=EDGE, lw=0.7, ls="--")
        for j, (m, lab, c, mk) in enumerate(TRIO):
            g = per_key.xs((m, lv), level=("model", "level"))
            emp = 100 * g[["coverage_50", "coverage_80", "coverage_95"]].to_numpy()
            for row in emp:
                ax.scatter(nominal + (j - 1) * 1.6, row, s=5, color=c, alpha=0.3, linewidth=0)
            ax.plot(nominal + (j - 1) * 1.6, emp.mean(0), marker=mk, ms=4, color=c, lw=1.0, label=lab,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xticks(nominal, ["50%", "80%", "95%"])
        ax.set_xlim(42, 100)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Interval's stated coverage")
        ax.set_ylabel("Share of actual sales inside (%)")
        ax.set_title(title)
        ax.grid(axis="y")
        despine(ax)
        panel_label(ax, letter)
    axs[0].legend(loc="lower right", fontsize=6.6)
    ax = axs[2]
    lv_keep = [(lv, lab) for lv, lab in LEVELS if lv not in ("cat_id", "cat_idxstore_id")]
    for j, (m, lab, c, mk) in enumerate(TRIO):
        ys = []
        for lv, _ in lv_keep:
            g = per_key.xs((m, lv), level=("model", "level")) if (m, lv) in per_key.droplevel(0).index else None
            ys.append(100 * g.coverage_80.mean() if g is not None else np.nan)
        ax.plot(range(len(lv_keep)), ys, marker=mk, ms=4, color=c, lw=1.0, label=lab, markeredgecolor="white",
                markeredgewidth=0.5)
    ax.axhline(80, color=EDGE, lw=0.7, ls="--")
    ax.set_xticks(range(len(lv_keep)), [lab for _, lab in lv_keep], fontsize=6.6)
    ax.set_ylabel("Coverage of the 80% interval (%)")
    ax.set_title("By level, mean over data sets")
    ax.grid(axis="y")
    despine(ax)
    panel_label(ax, "c")
    save(fig, HERE / "fig_sup_calibration.pdf")
    for lv, lt in ((IXS, "Ixs"), ("total", "Total")):
        for m, mt in (("world_model", "Wm"), ("lightgbm", "Lgb"), ("ensemble_2", "Avg")):
            g = per_key.xs((m, lv), level=("model", "level"))
            v[f"sfCovFifty{lt}{mt}"] = f"{100 * g.coverage_50.mean():.0f}"
            v[f"sfCovFiftyMin{lt}{mt}"] = f"{100 * g.coverage_50.min():.0f}"
            v[f"sfCovFiftyMax{lt}{mt}"] = f"{100 * g.coverage_50.max():.0f}"
            v[f"sfCovEightyMin{lt}{mt}"] = f"{100 * g.coverage_80.min():.0f}"
            v[f"sfCovEightyMax{lt}{mt}"] = f"{100 * g.coverage_80.max():.0f}"
    for lv, lt in (("total", "Total"), ("store_id", "Store"), ("item_id", "Item"), (IXS, "Ixs")):
        for m, mt in (("world_model", "Wm"), ("lightgbm", "Lgb"), ("ensemble_2", "Avg")):
            g = per_key.xs((m, lv), level=("model", "level"))
            v[f"sfCovLevel{lt}{mt}"] = f"{100 * g.coverage_80.mean():.0f}"
    # window-to-window spread of the 80% coverage at the total (how far one window's coverage strays)
    tw = d[(d.level == "total") & d.model.isin(["world_model", "lightgbm", "ensemble_2"])]
    v["sfCovTotalWindowMin"] = f"{100 * tw.coverage_80.min():.0f}"
    v["sfCovTotalWindowMax"] = f"{100 * tw.coverage_80.max():.0f}"
    v["sfCalWindows"] = f"{d.groupby('key').origin.nunique().min()}"
    v["sfCalWindowsMax"] = f"{d.groupby('key').origin.nunique().max()}"


# ------------------------------------------------------------------ loss and JEPA rounds of the tuned searches
LOSSES = [("negative_binomial", "NB"), ("negative_binomial_item_spread", "NB,\nspread by item"),
          ("zinb_item_spread", "Zero-inflated NB,\nspread by item"), ("hurdle_nb_item_spread", "Hurdle NB,\nspread by item")]
JEPA = [("jepa_hybrid", "Short\npretraining"), ("jepa_hybrid_long", "Long\npretraining")]


def _prob() -> pd.DataFrame:
    fs = sorted((A / "probabilistic").glob("*.csv"))
    return pd.concat([_read(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def fig_probabilistic(v: dict):
    """Quantile reliability diagrams, the zero-sales reliability diagram, CRPS and the Brier score
    (Karthik, 13 November 2026), from results/analysis_tuned/probabilistic."""
    d = _prob()
    if d.empty:
        return
    from rwm.evaluation.compare import QFINE, ZERO_BINS
    d = d[d.model.isin([m for m, *_ in TRIO])]
    mid = {qv: (d[f"le_{qv}"] + d[f"lt_{qv}"]) / 2 for qv in QFINE}
    for qv in QFINE:
        d[f"mid_{qv}"] = mid[qv]
    fig = plt.figure(figsize=(FULL, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42)
    qs = np.array(QFINE)
    for j, (lv, title) in enumerate(((IXS, "Item by store"), ("store_id", "Store"), ("total", "Total"))):
        ax = fig.add_subplot(gs[0, j])
        ax.plot([0, 1], [0, 1], color=EDGE, lw=0.7, ls="--")
        for m, lab, c, mk in TRIO:
            g = d[(d.model == m) & (d.level == lv)].groupby("key")[[f"mid_{x}" for x in QFINE]].mean()
            for _, row in g.iterrows():
                ax.plot(qs, row.to_numpy(), color=c, alpha=0.18, lw=0.6)
            ax.plot(qs, g.mean().to_numpy(), color=c, lw=1.3, label=lab)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Quantile of the forecast")
        ax.set_ylabel("Share of actuals below it")
        ax.set_title(title)
        ax.grid()
        despine(ax)
        panel_label(ax, "abc"[j])
        if j == 0:
            ax.legend(loc="upper left", fontsize=6.4)
    # zero-sales reliability, pooled over data sets and windows
    ax = fig.add_subplot(gs[1, 0])
    bot = d[d.level == IXS]
    ax.plot([0, 1], [0, 1], color=EDGE, lw=0.7, ls="--")
    for m, lab, c, mk in TRIO:
        g = bot[bot.model == m]
        n = np.array([g[f"zb{i}_n"].sum() for i in range(ZERO_BINS)])
        p = np.array([g[f"zb{i}_p"].sum() for i in range(ZERO_BINS)])
        y = np.array([g[f"zb{i}_y"].sum() for i in range(ZERO_BINS)])
        ok = n > 100
        ax.plot(p[ok] / n[ok], y[ok] / n[ok], marker=mk, ms=3.5, color=c, lw=1.0, label=lab,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Forecast chance of selling nothing")
    ax.set_ylabel("Share that sold nothing")
    ax.set_title("Zero sales, item by store")
    ax.grid()
    despine(ax)
    panel_label(ax, "d")
    # CRPS by data set, item by store, relative to LightGBM
    ax = fig.add_subplot(gs[1, 1:])
    c_ixs = d[d.level == IXS].groupby(["key", "model"]).crps.mean().unstack()
    c_tot = d[d.level == "total"].groupby(["key", "model"]).crps.mean().unstack()
    x = np.arange(len(KEYS))
    for j, (m, lab, c, mk) in enumerate(TRIO):
        vals = [c_ixs.loc[k, m] if k in c_ixs.index else np.nan for k in KEYS]
        ax.scatter(x + (j - 1) * 0.18, vals, marker=mk, s=20, color=c, edgecolor="white", linewidth=0.5, label=lab,
                   zorder=3)
    ax.set_xticks(x, NAMES, rotation=30, ha="right")
    ax.set_ylabel("Scaled CRPS, item by store")
    ax.set_title("CRPS by data set (lower is better)")
    ax.grid(axis="y")
    despine(ax)
    panel_label(ax, "e")
    save(fig, HERE / "fig_sup_probabilistic.pdf")
    # numbers and table
    br = bot.groupby(["key", "model"])[["brier", "brier_clim", "zero_share"]].mean()
    v["sfProbSets"] = _w(c_ixs.shape[0])
    best_ixs = c_ixs[["world_model", "lightgbm", "ensemble_2"]].idxmin(axis=1)
    best_tot = c_tot[["world_model", "lightgbm", "ensemble_2"]].idxmin(axis=1)
    v["sfCrpsAvgBestIxs"] = _w(int((best_ixs == "ensemble_2").sum()))
    v["sfCrpsLgbBestIxs"] = _w(int((best_ixs == "lightgbm").sum()))
    v["sfCrpsWmBestIxs"] = _w(int((best_ixs == "world_model").sum()))
    v["sfCrpsAvgBestTot"] = _w(int((best_tot == "ensemble_2").sum()))
    v["sfCrpsLgbBestTot"] = _w(int((best_tot == "lightgbm").sum()))
    v["sfCrpsWmBestTot"] = _w(int((best_tot == "world_model").sum()))
    for m, mt in (("world_model", "Wm"), ("lightgbm", "Lgb"), ("ensemble_2", "Avg")):
        b = br.xs(m, level="model")
        v[f"sfBrierBeatsClim{mt}"] = _w(int((b.brier < b.brier_clim).sum()))
        v[f"sfBrier{mt}"] = f"{b.brier.mean():.3f}"
        g = d[(d.model == m) & (d.level == IXS)].groupby("key")[[f"mid_{x}" for x in QFINE]].mean().mean()
        v[f"sfRelNinety{mt}"] = f"{100 * g['mid_0.9']:.0f}"
        v[f"sfRelTen{mt}"] = f"{100 * g['mid_0.1']:.0f}"
        v[f"sfRelHalf{mt}"] = f"{100 * g['mid_0.5']:.0f}"
        gt = d[(d.model == m) & (d.level == "total")].groupby("key")[[f"mid_{x}" for x in QFINE]].mean().mean()
        v[f"sfRelNinetyTot{mt}"] = f"{100 * gt['mid_0.9']:.0f}"
        v[f"sfRelTenTot{mt}"] = f"{100 * gt['mid_0.1']:.0f}"
    v["sfBrierClim"] = f"{br.xs('ensemble_2', level='model').brier_clim.mean():.3f}"
    ba = br.xs("ensemble_2", level="model")
    v["sfBrierBeatsClimNames"] = ", ".join(KEYS[k] for k in ba.index if ba.loc[k, "brier"] < ba.loc[k, "brier_clim"]) or "none"
    rows = []
    for k, name in KEYS.items():
        if k not in c_ixs.index:
            continue
        cells = []
        for tab in (c_ixs, c_tot):
            vals = [tab.loc[k, m] for m in ("world_model", "lightgbm", "ensemble_2")]
            lo = min(vals)
            cells += [f"\\textbf{{{x:.3f}}}" if x == lo else f"{x:.3f}" for x in vals]
        bv = [br.loc[(k, m), "brier"] for m in ("world_model", "lightgbm", "ensemble_2")]
        clim = br.loc[(k, "ensemble_2"), "brier_clim"]
        lo = min(bv + [clim])
        cells += [f"\\textbf{{{x:.3f}}}" if x == lo else f"{x:.3f}" for x in bv + [clim]]
        cells.append(f"{100 * br.loc[(k, 'ensemble_2'), 'zero_share']:.0f}\\%")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Probabilistic scores of the forecast ranges, mean over windows. Scaled CRPS item by store and at the
total, from each model's point forecast and the quantiles of its scaled errors in earlier windows on a grid of 23
quantiles, weighted by revenue as WRMSSE. Brier score of the forecast chance of selling nothing, item by store,
against each series' share of zero-sale periods over the year before (Past). Lowest in bold.}
\label{tab:sup-probabilistic}
\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l ccc ccc cccc c}
\toprule
 & \multicolumn{3}{c}{CRPS, item by store} & \multicolumn{3}{c}{CRPS, total} & \multicolumn{4}{c}{Brier score, zero sales} & Periods\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-11}
Data set & WM & LGB & Avg & WM & LGB & Avg & WM & LGB & Avg & Past & with none\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table*}
"""
    (GEN / "sup_tab_probabilistic.tex").write_text(tex)


def _round(prefix: str, changes: list) -> dict:
    """Per data set: start score and keep bar, one-seed screen and three-seed try per change, and what was kept."""
    out = {}
    for k in KEYS:
        rs = _log(_search(prefix, k))
        st = next((r for r in rs if r["kind"] == "start"), None)
        if st is None:
            continue
        e = {"start": st["score"], "start_held": st.get("held_out"), "bar": st.get("keep_bar"), "kept": set()}
        for r in rs:
            if r["kind"] == "rung":
                e.setdefault("screen", {})[r["change"]] = r["score"]
            elif r["kind"] == "try":
                e.setdefault("try", {})[r["change"]] = (r["score"], r.get("held_out"))
            elif r["kind"] == "kept":
                e["kept"].add(r["change"])
        out[k] = e
    return out


def fig_explore(v: dict):
    loss, jepa = _round("lossT", LOSSES), _round("hybridU", JEPA)
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw={"width_ratios": [len(LOSSES), len(JEPA) + 0.6],
                                                                   "wspace": 0.18}, sharey=True)
    cols = plt.cm.tab10.colors
    for ax, res, changes, title, letter in ((axs[0], loss, LOSSES, "Likelihood round", "a"),
                                             (axs[1], jepa, JEPA, "JEPA pretraining round", "b")):
        for j, (k, name) in enumerate(KEYS.items()):
            e = res.get(k)
            if not e:
                continue
            for i, (ch, _) in enumerate(changes):
                x = i + (j - 4) * 0.07
                s = e.get("screen", {}).get(ch)
                if s is not None:
                    ax.scatter(x, 100 * (s - e["start"]) / e["start"], s=9, color=cols[j], alpha=0.45, linewidth=0)
                t = e.get("try", {}).get(ch)
                if t is not None:
                    kept = ch in e["kept"]
                    ax.scatter(x, 100 * (t[0] - e["start"]) / e["start"], s=26 if kept else 16, color=cols[j],
                               edgecolor=INK if kept else "white", linewidth=1.0 if kept else 0.4, zorder=3,
                               marker="*" if kept else "o")
        bar = np.nanmean([100 * e["bar"] / e["start"] for e in res.values() if e.get("bar")])
        ax.axhspan(-bar, 0, color=GRID, alpha=0.6, lw=0)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(len(changes)), [lab for _, lab in changes], fontsize=6.4)
        ax.set_xlim(-0.6, len(changes) - 0.4)
        ax.set_title(title)
        ax.grid(axis="y")
        despine(ax)
        panel_label(ax, letter)
    axs[0].set_ylabel("Change on the search windows against\nthe starting settings (%, lower is better)")
    hs = [Line2D([], [], marker="o", ls="none", color=cols[j], ms=4, label=n) for j, n in enumerate(NAMES)]
    hs += [Line2D([], [], marker="o", ls="none", color=MUTED, alpha=0.45, ms=3, label="One-seed screen"),
           Line2D([], [], marker="o", ls="none", color=MUTED, ms=4, label="Three-seed try"),
           Line2D([], [], marker="*", ls="none", color=MUTED, markeredgecolor=INK, ms=7, label="Kept"),
           plt.Rectangle((0, 0), 1, 1, color=GRID, label="Inside the mean keep bar")]
    axs[1].legend(handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.3, handletextpad=0.2)
    save(fig, HERE / "fig_sup_explore.pdf")
    # numbers for the text and the table
    v["sfLossSets"] = _w(len(loss))
    v["sfLossTried"] = _w(len(LOSSES))
    v["sfLossKeptSets"] = _w(sum(1 for e in loss.values() if e["kept"]))
    v["sfLossKeptNames"] = ", ".join(KEYS[k] for k, e in loss.items() if e["kept"]) or "none"
    v["sfJepaSets"] = _w(len(jepa))
    v["sfJepaKeptSets"] = _w(sum(1 for e in jepa.values() if e["kept"]))
    v["sfJepaKeptNames"] = ", ".join(KEYS[k] for k, e in jepa.items() if e["kept"]) or "none"
    screens = [(s - e["start"]) / e["start"] for e in loss.values() for s in e.get("screen", {}).values()]
    v["sfLossScreens"] = f"{len(screens)}"
    v["sfLossScreenBetter"] = f"{sum(x < 0 for x in screens)}"
    js = [(s - e["start"]) / e["start"] for e in jepa.values() for s in e.get("screen", {}).values()]
    v["sfJepaScreens"] = f"{len(js)}"
    v["sfJepaScreenBetter"] = f"{sum(x < 0 for x in js)}"
    _explore_table(loss, jepa)


def _cell(e: dict, ch: str) -> str:
    t = e.get("try", {}).get(ch)
    s = e.get("screen", {}).get(ch)
    if t is not None:
        txt = f"{t[0]:.3f}"
        return f"\\textbf{{{txt}}}" if ch in e["kept"] else txt
    return f"({s:.3f})" if s is not None else "--"


def _explore_table(loss: dict, jepa: dict):
    rows = []
    for k, name in KEYS.items():
        e, f = loss.get(k, {}), jepa.get(k, {})
        if not e:
            continue
        cells = [f"{e['start']:.3f}"] + [_cell(e, c) for c, _ in LOSSES]
        cells += [f"{f['start']:.3f}" if f else "--"] + ([_cell(f, c) for c, _ in JEPA] if f else ["--", "--"])
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Every likelihood and JEPA pretraining recipe tried on the tuned world model, WRMSSE on the search windows.
Start: the settings each round began from.
A number in brackets is a one-seed screen that did not earn a three-seed try; a plain number is the three-seed mean;
bold marks the change kept, which had to beat the start by more than the keep bar on the search windows.}
\label{tab:sup-explore}
\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l ccccc ccc}
\toprule
 & \multicolumn{5}{c}{Likelihood round} & \multicolumn{3}{c}{JEPA pretraining round}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-9}
Data set & Start & NB & NB, spread by item & ZINB, spread by item & Hurdle NB, spread by item & Start & Short & Long\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}\end{adjustbox}
\end{table*}
"""
    (GEN / "sup_tab_explore.tex").write_text(tex)


# ------------------------------------------------------------------ first version's loss experiments
def fig_devlosses(v: dict):
    d = pd.read_csv(ROOT / "results" / "paper1_revision" / "tables" / "loss_ledger.csv")
    s = d[d.status == "scored"].dropna(subset=["diff"]).copy()
    order = s.groupby("change")["diff"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(FULL, 0.24 * len(order) + 0.8))
    cats = {"ana": "Analgesics", "bjc": "Bottled juice", "che": "Cheese", "fre": "Frozen entrees",
            "sdr": "Soft drinks", "sna": "Snacks"}
    cols = plt.cm.tab10.colors
    for i, ch in enumerate(order):
        g = s[s.change == ch]
        y = len(order) - 1 - i
        ax.plot([g["diff"].min(), g["diff"].max()], [y, y], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        for _, r in g.iterrows():
            cat = str(r.config).split("_")[1] if str(r.config).startswith("dominicks_") else None
            c = list(cats).index(cat) if cat in cats else 6
            ax.scatter(r["diff"], y, s=14, color=cols[c], edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(0.37, y, f"{len(g)}", va="center", ha="right", fontsize=6, color=MUTED)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(order))[::-1], order, fontsize=6.6)
    ax.set_xlabel("Change in WRMSSE against the settings it changed (first version, development windows; left is better)")
    ax.set_xlim(-0.3, 0.4)
    ax.grid(axis="x")
    despine(ax)
    hs = [Line2D([], [], marker="o", ls="none", color=cols[j], ms=4, label=n) for j, n in enumerate(cats.values())]
    hs.append(Line2D([], [], marker="o", ls="none", color=cols[6], ms=4, label="Frat"))
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.3, ncol=7, handletextpad=0.1)
    ax.text(0.37, len(order) - 0.4, "runs", ha="right", fontsize=6, color=MUTED)
    save(fig, HERE / "fig_sup_devlosses.pdf")
    v["sfDevChanges"] = _w(d.change.nunique())
    v["sfDevRuns"] = f"{len(d)}"
    v["sfDevScored"] = f"{len(s)}"
    v["sfDevBetter"] = f"{int((s['diff'] < 0).sum())}"
    v["sfDevLatentMax"] = f"{s[s.change.str.startswith('Latent')]['diff'].abs().max():.3f}"
    v["sfDevNotRun"] = f"{int((d.status == 'not run on full data').sum())}"


def main():
    use_style()
    v: dict = {}
    fig_ablation(v)
    fig_occlusion(v)
    fig_calibration(v)
    fig_probabilistic(v)
    fig_explore(v)
    fig_devlosses(v)
    (GEN / "supfig_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(v.items())))
    for k, x in sorted(v.items()):
        print(f"  {k} = {x}")


if __name__ == "__main__":
    main()
