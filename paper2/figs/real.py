"""Audit of the saved model versions: pass rules, the main test, figures and
tables, from the summary tables committed under results/paper2/summary/
(written by scripts/audit_summary.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from style import ACCENT, FILL, FULL, GRID, INK, MUTED, despine, panel_label, save

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / "results" / "paper2" / "summary"
DATASETS = {"m5": "M5", "dominicks": "Dominick's", "frat": "Frat"}
DCOL = {"m5": "#3E6E9A", "dominicks": "#B5523B", "frat": "#4F7F6A"}
PARTS = ["lift_readout", "baseline_readout", "transition", "encoder"]


def available() -> bool:
    return (SUM / "versions.csv").exists()


def dataset_of(config: str) -> str:
    for k in DATASETS:
        if f"{k}_" in config or config.startswith(k):
            return k
    return "other"


def base(config: str) -> str:
    import re
    return re.sub(r"_seed\d+$", "", config)


def load():
    v = pd.read_csv(SUM / "versions.csv")
    st = pd.read_csv(SUM / "stability.csv") if (SUM / "stability.csv").exists() else pd.DataFrame()
    ag = pd.read_csv(SUM / "agreement.csv") if (SUM / "agreement.csv").exists() else pd.DataFrame()
    z = pd.read_csv(SUM / "zones.csv") if (SUM / "zones.csv").exists() else pd.DataFrame()
    if len(z):  # only episodes scored under the corrected rule (docs/PAPER2_PLAN.md)
        ep = pd.read_csv(ROOT / "results" / "paper2" / "zone_episodes.csv")
        keep = ep.loc[ep["scored"], ["category", "store", "first_week"]]
        z = z.merge(keep, on=["category", "store", "first_week"])
    v["dataset"] = v["config"].map(dataset_of)
    v["base"] = v["config"].map(base)
    return v, st, ag, z


def score(v, st, ag) -> pd.DataFrame:
    """Each version's check scores and passes under the plan's rules."""
    d = v.copy()
    c1 = [c for c in d.columns if c.startswith("c1_")]
    d["C1"] = d[c1].max(axis=1) if c1 else np.nan
    d["C1_pass"] = np.where(d["C1"].notna(), d["C1"] < 1e-3, np.nan)
    d["C2"] = d.get("c2_lift_readout", pd.Series(np.nan, index=d.index))
    d["C2_pass"] = np.where(d["C2"].notna(), d["C2"] < 0.5, np.nan)
    if len(st):
        s = st.set_index(["config", "origin"])["lift_rank_agreement"]
        d["C3"] = [s.get((b, o), np.nan) for b, o in zip(d["base"], d["origin"])]
    else:
        d["C3"] = np.nan
    d["C3_pass"] = np.where(d["C3"].notna(), d["C3"] >= 0.7, np.nan)
    d["C4"] = d["c4_share_lower_after_deeper_cut"]
    d["C4_pass"] = d["C4"] < 0.01
    d["C5_bias"], d["C5_rank"] = d["c5_any_plan_bias"], d["c5_any_plan_rank"]
    d["C5_pass"] = d["C5_bias"].between(0.8, 1.25) & (d["C5_rank"] >= 0.3)
    if len(ag):
        a = ag.assign(base=ag["model_config"].map(base)).set_index(["base", "seed", "origin"])["total_lift_onoff"]
        d["C6"] = [a.get((b, s, o), np.nan) for b, s, o in zip(d["base"], d["seed"], d["origin"])]
    else:
        d["C6"] = np.nan
    d["C6_pass"] = np.where(d["C6"].notna(), d["C6"] >= 0.5, np.nan)
    passes = d[[f"C{k}_pass" for k in range(1, 7)]].astype(float)
    d["passed"] = passes.sum(axis=1, skipna=True)
    d["applicable"] = passes.notna().sum(axis=1)
    # exploratory, added after the results were read (docs/PAPER2_PLAN.md, 17 October 2026):
    # C5 compares the forecast with sales over the same baseline, so it shares
    # information with accuracy; C1 failures come from the audit's regular price
    d["passed_no_c5"] = passes.drop(columns="C5_pass").sum(axis=1, skipna=True)
    d["passed_no_c1_c5"] = passes.drop(columns=["C1_pass", "C5_pass"]).sum(axis=1, skipna=True)
    return d


SCORES = ["passed", "passed_no_c5", "passed_no_c1_c5", "applicable", "C1", "C2", "C3", "C4", "C5_rank", "C5_bias", "C6", "item_by_store"]


def per_version(d: pd.DataFrame) -> pd.DataFrame:
    """One row per version (a saved configuration and seed): the mean of each score
    and of checks passed over the version's windows (docs/PAPER2_PLAN.md: the main
    test is across versions)."""
    g = d.groupby(["dataset", "run", "config", "base", "seed"], dropna=False)
    v = g[SCORES].mean().reset_index()
    v["windows"] = g.size().to_numpy()
    v["model"] = np.where(v["config"].str.contains("lift"), "lift", "world")
    return v


def main_test(d: pd.DataFrame, n_perm: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Spearman of checks passed (and each check's score) with item-by-store WRMSSE,
    within data set, across versions; pooled by permuting scores within data set."""
    rng = np.random.default_rng(seed)
    cols = ["passed", "C1", "C2", "C3", "C4", "C5_rank", "C5_bias", "C6", "passed_no_c5", "passed_no_c1_c5"]
    rows = []
    for col in cols:
        per = {}
        for ds, g in d.groupby("dataset"):
            g = g[[col, "item_by_store"]].dropna()
            if len(g) >= 4 and g[col].nunique() > 1:
                per[ds] = (g[col].to_numpy(), g["item_by_store"].to_numpy())
        if not per:
            continue
        obs = {ds: spearmanr(x, y).statistic for ds, (x, y) in per.items()}
        stat = np.mean(list(obs.values()))
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = np.mean([spearmanr(rng.permutation(x), y).statistic for x, y in per.values()])
        p = float((np.abs(null) >= abs(stat)).mean())
        rows.append({"score": col, "pooled_rho": stat, "p": p,
                     **{f"rho_{ds}": r for ds, r in obs.items()}, **{f"n_{ds}": len(x) for ds, (x, _) in per.items()}})
    return pd.DataFrame(rows)


def fig_real_checks(d, st, ag, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.6), gridspec_kw={"wspace": 0.45, "hspace": 0.6})
    a, b, c, e, f, g = axes.ravel()
    order = [k for k in DATASETS if k in set(d["dataset"])]
    rng = np.random.default_rng(0)
    # C1
    for k, ds in enumerate(order):
        x = d.loc[d["dataset"] == ds, "C1"].dropna()
        a.scatter(np.maximum(x, 1e-9), k + rng.uniform(-0.2, 0.2, len(x)), s=8, color=DCOL[ds], alpha=0.8, lw=0)
    a.axvline(1e-3, color=INK, lw=0.7, ls=(0, (3, 2)))
    a.set_xscale("log")
    a.set_yticks(range(len(order)))
    a.set_yticklabels([DATASETS[k] for k in order])
    a.set_xlabel("Largest violation (share of forecast)")
    a.set_title("C1 By construction")
    despine(a)
    panel_label(a, "a")
    # C2 cascade
    cols = [f"c2_{p}" for p in PARTS if f"c2_{p}" in d.columns]
    for _, r in d.dropna(subset=cols[:1]).iterrows():
        b.plot(range(len(cols) + 1), [1.0] + [r[c] for c in cols], color=DCOL[r["dataset"]], lw=0.5, alpha=0.6)
    b.axhline(0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    b.set_xticks(range(len(cols) + 1))
    b.set_xticklabels(["none", "+lift", "+base", "+trans.", "+enc."][:len(cols) + 1], fontsize=6.2)
    b.set_ylabel("Rank agreement")
    b.set_ylim(-0.4, 1.05)
    b.set_title("C2 Randomisation")
    despine(b)
    panel_label(b, "b")
    # C3
    if len(st):
        s = st.assign(dataset=st["config"].map(dataset_of))
        for ds, gg in s.groupby("dataset"):
            c.scatter(gg["forecast_rank_agreement"], gg["lift_rank_agreement"], s=12, color=DCOL.get(ds, MUTED),
                      edgecolor="white", lw=0.3, label=DATASETS.get(ds, ds))
    c.plot([0, 1], [0, 1], color=GRID, lw=0.8)
    c.axhline(0.7, color=INK, lw=0.7, ls=(0, (3, 2)))
    c.set_xlabel("Forecast agreement between seeds")
    c.set_ylabel("Lift agreement between seeds")
    c.set_title("C3 Seed stability")
    c.legend(fontsize=5.8, loc="lower right")
    despine(c)
    panel_label(c, "c")
    # C4
    for k, ds in enumerate(order):
        x = d.loc[d["dataset"] == ds, "C4"].dropna() * 100
        e.scatter(x, k + rng.uniform(-0.2, 0.2, len(x)), s=8, color=DCOL[ds], alpha=0.8, lw=0)
    e.axvline(1, color=INK, lw=0.7, ls=(0, (3, 2)))
    e.set_yticks(range(len(order)))
    e.set_yticklabels([DATASETS[k] for k in order])
    e.set_xlabel("Item-weeks lower after a deeper cut, %")
    e.set_title("C4 Price direction")
    despine(e)
    panel_label(e, "d")
    # C5
    for ds, gg in d.groupby("dataset"):
        f.scatter(gg["C5_bias"], gg["C5_rank"], s=10, color=DCOL.get(ds, MUTED), edgecolor="white", lw=0.3)
    f.add_patch(plt.Rectangle((0.8, 0.3), 0.45, 0.7, facecolor=FILL, edgecolor="none", zorder=0))
    f.set_xlabel("Forecast over sales, planned weeks")
    f.set_ylabel("Rank of predicted\nwith observed lift")
    f.set_title("C5 Against what sold")
    despine(f)
    panel_label(f, "e")
    # C6
    if len(ag):
        levers = sorted({c[:-6] for c in ag.columns if c.endswith("_onoff") and c != "total_lift_onoff"})
        for k, lv in enumerate(levers):
            for kind, off, mk in (("onoff", -0.15, "o"), ("shap", 0.15, "s")):
                col = f"{lv}_{kind}"
                if col in ag:
                    x = ag[col].dropna()
                    g.scatter(x, k + off + rng.uniform(-0.06, 0.06, len(x)), s=8, marker=mk,
                              color=ACCENT if kind == "onoff" else MUTED, alpha=0.8, lw=0)
        g.set_yticks(range(len(levers)))
        g.set_yticklabels(levers, fontsize=6.4)
        g.scatter([], [], marker="o", color=ACCENT, s=8, label="on/off")
        g.scatter([], [], marker="s", color=MUTED, s=8, label="TreeSHAP")
        g.legend(fontsize=5.8, loc="lower left")
    g.axvline(0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    g.set_xlabel("Rank agreement with LightGBM")
    g.set_title("C6 Against LightGBM")
    despine(g)
    panel_label(g, "f")
    save(fig, out / "fig_real_checks.pdf")


def fig_real_main(d, out: Path):
    order = [k for k in DATASETS if k in set(d["dataset"])]
    fig, axes = plt.subplots(1, len(order), figsize=(FULL, 2.4), gridspec_kw={"wspace": 0.35})
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(1)
    for ax, ds in zip(axes, order):
        g = d[d["dataset"] == ds].dropna(subset=["item_by_store"])
        ax.scatter(g["passed"] + rng.uniform(-0.15, 0.15, len(g)), g["item_by_store"], s=12, color=DCOL[ds],
                   edgecolor="white", lw=0.3)
        if len(g) > 3 and g["passed"].nunique() > 1:
            r = spearmanr(g["passed"], g["item_by_store"]).statistic
            ax.text(0.03, 0.95, f"Spearman {r:.2f}\n{len(g)} versions", transform=ax.transAxes, va="top", fontsize=6.2)
        ax.set_xlabel("Checks passed (C1 to C6)")
        ax.set_title(DATASETS[ds])
        despine(ax)
    axes[0].set_ylabel("Item-by-store WRMSSE")
    save(fig, out / "fig_real_main.pdf")


def fig_real_c7(z, measured_slope, measured_lo, measured_hi, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1.3, 1]})
    for k, (col, lab, colr) in enumerate((("world_implied", "World model", ACCENT), ("lightgbm_implied", "LightGBM", MUTED))):
        if col not in z:
            continue
        a.scatter(z["price_log_ratio"] * 100, z[col] * 100, s=12, color=colr, edgecolor="white", lw=0.3, label=lab)
    a.scatter(z["price_log_ratio"] * 100, z["did_estimate"] * 100, s=10, marker="x", color=INK, lw=0.7, label="measured")
    xs = np.linspace(-12, 12, 5)
    a.plot(xs, -0.3 * xs, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    a.axhline(0, color=GRID, lw=0.8)
    a.axvline(0, color=GRID, lw=0.8)
    a.set_xlabel("Price against zone peers, %")
    a.set_ylabel("Change in category units, %")
    a.legend(fontsize=6, loc="upper right")
    a.set_title("Stated and measured response, each episode")
    despine(a)
    panel_label(a, "a")
    from make_figures import ols_slope
    rows = [("Measured", measured_slope, (measured_hi - measured_lo) / 3.92)]
    for col, lab in (("world_implied", "World model"), ("lightgbm_implied", "LightGBM")):
        if col in z:
            sl, se = ols_slope(z[col], z["price_log_ratio"])
            rows.append((lab, sl, se))
    for k, (lab, sl, se) in enumerate(rows):
        b.plot([sl - 1.96 * se, sl + 1.96 * se], [k, k], color=INK if k == 0 else ACCENT, lw=1)
        b.scatter([sl], [k], s=16, color=INK if k == 0 else ACCENT, zorder=3)
    b.axvline(-0.3, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    b.axvline(0, color=INK, lw=0.5)
    lo = min(sl - 1.96 * se for _, sl, se in rows)
    hi = max(sl + 1.96 * se for _, sl, se in rows)
    b.set_xlim(lo - 0.08, max(0.1, hi + 0.08))
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([r[0] for r in rows])
    b.set_ylim(len(rows) - 0.4, -0.6)
    b.set_xlabel("Pooled slope, 95% interval")
    b.set_title("Pooled response")
    b.tick_params(axis="y", length=0)
    despine(b, keep=("bottom",))
    panel_label(b, "b")
    save(fig, out / "fig_real_c7.pdf")
    return rows


def category_of(config: str) -> str:
    """The data and weeks a version was scored on: its configuration's data set
    (with the Dominick's category) and evaluation windows. Versions with the same
    value were scored on the same series and weeks."""
    import json

    import yaml

    f = ROOT / "configs" / "experiments" / f"{base(config)}.yaml"
    if not f.exists():
        return f"unknown:{base(config)}"
    c = yaml.safe_load(open(f))
    return json.dumps([c["dataset"], c["evaluation"]], sort_keys=True, default=str)


def within_window_test(d: pd.DataFrame, col: str = "passed", n_perm: int = 5000, n_boot: int = 2000,
                       seed: int = 0) -> dict:
    """Checks passed against accuracy, comparing only configurations fitted on the
    same data and scored on the same window (added after the results,
    docs/PAPER2_PLAN.md, 18 October 2026).

    Seeds of a configuration are averaged first, so a configuration counts once
    per window. Within each data (Dominick's category, Frat, M5) and window,
    the Spearman correlation across configurations of checks passed with
    item-by-store WRMSSE; pooled as the mean weighted by the number of
    configurations. Inference keeps the dependence: the permutation reassigns
    whole configurations (their scores in every window together) within each
    data, and the bootstrap resamples configurations within each data."""
    rng = np.random.default_rng(seed)
    x = d.assign(cat=d["config"].map(category_of))
    x = x.groupby(["dataset", "cat", "base", "origin"], as_index=False)[[col, "item_by_store"]].mean()
    blocks = []  # one per data: configurations by windows, for the score and the error
    for (ds, cat), g in x.groupby(["dataset", "cat"]):
        s = g.pivot(index="base", columns="origin", values=col)
        e = g.pivot(index="base", columns="origin", values="item_by_store")
        if len(s) >= 4:
            blocks.append((ds, cat, s.to_numpy(), e.to_numpy()))

    def stat(bl, perms=None, picks=None):
        num, den, per_ds = 0.0, 0.0, {}
        for i, (ds, _, s, e) in enumerate(bl):
            if perms is not None:
                s = s[perms[i]]
            if picks is not None:
                s, e = s[picks[i]], e[picks[i]]
            for w in range(s.shape[1]):
                ok = ~np.isnan(s[:, w]) & ~np.isnan(e[:, w])
                if ok.sum() >= 4 and np.unique(s[ok, w]).size > 1 and np.unique(e[ok, w]).size > 1:
                    r = spearmanr(s[ok, w], e[ok, w]).statistic
                    num, den = num + r * ok.sum(), den + ok.sum()
                    a = per_ds.setdefault(ds, [0.0, 0.0])
                    a[0], a[1] = a[0] + r * ok.sum(), a[1] + ok.sum()
        return num / den if den else np.nan, {k: v[0] / v[1] for k, v in per_ds.items()}

    obs, per_ds = stat(blocks)
    null = np.array([stat(blocks, perms=[rng.permutation(len(b[2])) for b in blocks])[0] for _ in range(n_perm)])
    boot = np.array([stat(blocks, picks=[rng.integers(0, len(b[2]), len(b[2])) for b in blocks])[0]
                     for _ in range(n_boot)])
    return {"score": col, "rho": obs, "p": float((np.abs(null) >= abs(obs)).mean()),
            "lo": float(np.nanquantile(boot, 0.025)), "hi": float(np.nanquantile(boot, 0.975)),
            "configs": int(sum(len(b[2]) for b in blocks)), "data": len(blocks),
            **{f"rho_{k}": v for k, v in per_ds.items()}}


def write_tables(d, dv, mt, gen: Path, mtv=None):
    lines = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             "Data & Versions & Windows & C1 & C2 & C3 & C4 & C5 & C6 & All \\\\", "\\midrule"]
    for ds in [k for k in DATASETS if k in set(d["dataset"])]:
        g = d[d["dataset"] == ds]
        cells = [DATASETS[ds], str(int((dv["dataset"] == ds).sum())), str(len(g))]
        for k in range(1, 7):
            p = g[f"C{k}_pass"].astype(float).dropna()
            cells.append(f"{int(p.sum())}/{len(p)}" if len(p) else "n/a")
        cells.append(f"{int((g['passed'] == g['applicable']).sum())}/{len(g)}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "tab_real_pass.tex").write_text("\n".join(lines) + "\n")
    names = {"passed": "Checks passed", "passed_no_c5": "Checks passed, C5 left out$^\\dagger$",
             "passed_no_c1_c5": "Checks passed, C1 and C5 left out$^\\dagger$", "C1": "C1 violation", "C2": "C2 agreement after randomising",
             "C3": "C3 seed agreement", "C4": "C4 share lower", "C5_rank": "C5 rank", "C5_bias": "C5 bias",
             "C6": "C6 agreement with LightGBM"}
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{5}{c}{Version and window} & \\multicolumn{2}{c}{Version} \\\\",
             "\\cmidrule(lr){2-6}\\cmidrule(l){7-8}",
             "Score & M5 & Dom. & Frat & Pooled & $p$ & Pooled & $p$ \\\\", "\\midrule"]
    for _, r in mt.iterrows():
        cells = [names.get(r["score"], r["score"])]
        for ds in DATASETS:
            v = r.get(f"rho_{ds}")
            cells.append(f"${v:.2f}$" if pd.notna(v) else "n/a")
        cells += [f"${r['pooled_rho']:.2f}$", f"{r['p']:.3f}"]
        if mtv is not None:
            q = mtv[mtv["score"] == r["score"]].iloc[0]
            cells += [f"${q['pooled_rho']:.2f}$", f"{q['p']:.3f}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "tab_real_main.tex").write_text("\n".join(lines) + "\n")


def make(out: Path, measured):
    v, st, ag, z = load()
    d = score(v, st, ag)
    gen = out.parent / "generated"
    d.to_csv(gen / "real_scores.csv", index=False)
    dv = per_version(d)
    dv.to_csv(gen / "real_versions.csv", index=False)
    # the unit fixed in the manuscript before any result: a version is a
    # configuration, seed and window; the mean over a version's windows is reported beside it
    mt = main_test(d)
    mt.to_csv(gen / "real_main_test.csv", index=False)
    mtv = main_test(dv)
    mtv.to_csv(gen / "real_main_test_by_version.csv", index=False)
    ww = pd.DataFrame([within_window_test(d, c) for c in ("passed", "passed_no_c5", "passed_no_c1_c5")])
    ww.to_csv(gen / "real_within_window.csv", index=False)
    fig_real_checks(d, st, ag, out)
    fig_real_main(dv, out)
    rows = fig_real_c7(z, *measured, out) if len(z) else []
    write_tables(d, dv, mt, gen, mtv)
    write_macros(d, dv, mt, z, rows, gen, mtv, ww)
    write_diag_macros(gen)
    return d, mt, rows


def write_macros(d, dv, mt, z, rows, gen: Path, mtv=None, ww=None):
    """Every number the audit section quotes, as LaTeX macros (generated/real_macros.tex)."""
    out = []

    def put(name, value):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    tag = {"m5": "M", "dominicks": "D", "frat": "F"}
    f2 = lambda x: f"{x:.2f}".replace("-", "$-$") if pd.notna(x) else "n/a"  # noqa: E731
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    put("rVersions", int(len(dv)))
    put("rWindows", int(len(d)))
    for ds, t in tag.items():
        g, gv = d[d["dataset"] == ds], dv[dv["dataset"] == ds]
        put(f"rVer{t}", int(len(gv)))
        put(f"rWin{t}", int(len(g)))
        for k in range(1, 7):
            p = g[f"C{k}_pass"].astype(float).dropna()
            put(f"rPass{t}C{'abcdef'[k - 1]}", f"{int(p.sum())} of {len(p)}" if len(p) else "none")
        put(f"rAll{t}", int((g["passed"] == g["applicable"]).sum()))
        for col, nm in (("C1", "Ca"), ("C2", "Cb"), ("C3", "Cc"), ("C5_rank", "CeRank"), ("C5_bias", "CeBias"), ("C6", "Cf")):
            put(f"rMed{t}{nm}", f2(g[col].median()))
        put(f"rMed{t}CdPct", pct(g["C4"].median()))
        put(f"rMed{t}CaPct", f"{100 * g['C1'].median():.1f}")
        put(f"rMax{t}CaPct", f"{100 * g['C1'].max():.0f}")
    put("rCaPartsMax", f"{d['c1_parts_do_not_add_up'].max():.0g}")
    put("rCaBaseMax", f"{100 * d['c1_baseline_moves_with_plan'].max():.1f}")
    st = pd.read_csv(SUM / "stability.csv")
    st["dataset"] = st["config"].map(dataset_of)
    ag = pd.read_csv(SUM / "agreement.csv")
    ag["dataset"] = ag["model_config"].map(dataset_of)
    for ds, t in tag.items():
        g = st[st["dataset"] == ds]
        put(f"rStabLift{t}", f2(g["lift_rank_agreement"].median()))
        put(f"rStabFc{t}", f2(g["forecast_rank_agreement"].median()))
        a = ag[ag["dataset"] == ds]
        for col in ("promo_onoff", "promo_shap", "price_onoff", "price_shap", "event_onoff", "snap_onoff"):
            if col in a and a[col].notna().any():
                put(f"rAg{t}{col.replace('_', '').replace('onoff', 'On').replace('shap', 'Shap')}", f2(a[col].median()))
        g = d[d["dataset"] == ds]
        for col in ("c5_event_rank", "c5_snap_rank", "c5_promo_rank", "c5_price_cut_only_rank"):
            if g[col].notna().any():
                put(f"rCe{t}{col.replace('c5_', '').replace('_rank', '').replace('_', '')}", f2(g[col].median()))
    for _, r in mt.iterrows():
        key = {"passed": "Passed", "passed_no_c5": "PassedNoCe", "passed_no_c1_c5": "PassedNoCaCe", "C1": "Ca",
               "C2": "Cb", "C3": "Cc", "C4": "Cd", "C5_rank": "CeRank", "C5_bias": "CeBias", "C6": "Cf"}[r["score"]]
        for ds, t in tag.items():
            put(f"rRho{key}{t}", f2(r.get(f"rho_{ds}")))
        put(f"rRho{key}Pool", f2(r["pooled_rho"]))
        put(f"rP{key}", f"{r['p']:.3f}")
        if mtv is not None:
            q = mtv[mtv["score"] == r["score"]].iloc[0]
            put(f"rRho{key}PoolV", f2(q["pooled_rho"]))
            put(f"rP{key}V", f"{q['p']:.3f}")
            for ds, t in tag.items():
                put(f"rRho{key}{t}V", f2(q.get(f"rho_{ds}")))
    if ww is not None:
        for _, r in ww.iterrows():
            key = {"passed": "", "passed_no_c5": "NoCe", "passed_no_c1_c5": "NoCaCe"}[r["score"]]
            put(f"rWw{key}Rho", f2(r["rho"]))
            put(f"rWw{key}Lo", f2(r["lo"]))
            put(f"rWw{key}Hi", f2(r["hi"]))
            put(f"rWw{key}P", f"{r['p']:.2f}")
            for ds, t in tag.items():
                put(f"rWw{key}Rho{t}", f2(r.get(f"rho_{ds}")))
        put("rWwConfigs", int(ww["configs"].iloc[0]))
        put("rWwData", int(ww["data"].iloc[0]))
    if len(z):
        put("rZEpisodes", int(len(z)))
        inside = ((z["world_implied"] >= z["did_lo"]) & (z["world_implied"] <= z["did_hi"])).sum()
        put("rZWorldInside", int(inside))
        put("rZLgbInside", int(((z["lightgbm_implied"] >= z["did_lo"]) & (z["lightgbm_implied"] <= z["did_hi"])).sum()))
        same = lambda c: int((np.sign(z[c]) == np.sign(-z["price_log_ratio"])).sum())  # noqa: E731
        put("rZWorldSign", same("world_implied"))
        put("rZLgbSign", same("lightgbm_implied"))
        put("rZMeasuredSign", same("did_estimate"))
        for lab, sl, se in rows:
            t = {"Measured": "Meas", "World model": "World", "LightGBM": "Lgb"}[lab]
            put(f"rZSlope{t}", f2(sl))
            put(f"rZSlope{t}Lo", f2(sl - 1.96 * se))
            put(f"rZSlope{t}Hi", f2(sl + 1.96 * se))
    (gen / "real_macros.tex").write_text("\n".join(out) + "\n")


def write_diag_macros(gen: Path):
    """Numbers from the diagnosis of C1 and C4 on fourteen saved runs
    (results/paper2/diagnose/audit_diagnose.csv, scripts/audit_diagnose.py)."""
    f = ROOT / "results" / "paper2" / "diagnose" / "audit_diagnose.csv"
    if not f.exists():
        return
    x = pd.read_csv(f)
    x["dataset"] = x["config"].map(dataset_of)
    new = x["config"].str.contains("new_items")
    regcol = x["config"].str.startswith("frat_dev_state_model_lift")
    out = []

    def put(name, value):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    pc = lambda v: f"{100 * v:.0f}"  # noqa: E731
    p1 = lambda v: f"{100 * v:.1f}"  # noqa: E731
    put("dRuns", x["run"].nunique())
    put("dWindows", len(x))
    other_dom = x[(x["dataset"] == "dominicks") & ~new]
    put("dNewLo", pc(x.loc[new, "c1_audit_zero_lift_when_nothing_planned"].min()))
    put("dNewHi", pc(x.loc[new, "c1_audit_zero_lift_when_nothing_planned"].max()))
    put("dDomLo", p1(other_dom["c1_keep_missing_zero_lift_when_nothing_planned"].min()))
    put("dDomHi", p1(other_dom["c1_keep_missing_zero_lift_when_nothing_planned"].max()))
    put("dDomRows", p1(other_dom["c1_keep_missing_share_rows_over_0.001"].max()))
    put("dFratLo", pc(x.loc[regcol, "c1_audit_zero_lift_when_nothing_planned"].min()))
    put("dFratHi", pc(x.loc[regcol, "c1_audit_zero_lift_when_nothing_planned"].max()))
    put("dFratRowsLo", pc(x.loc[regcol, "c1_keep_missing_share_rows_over_0.001"].min()))
    put("dFratRowsHi", pc(x.loc[regcol, "c1_keep_missing_share_rows_over_0.001"].max()))
    put("dBaseMax", p1(x["c1_audit_baseline_moves_with_plan"].max()))
    for ds, t in (("dominicks", "D"), ("m5", "M"), ("frat", "F")):
        g = x[x["dataset"] == ds]
        put(f"dCdAll{t}", pc(g["c4_all_lower"].median()))
        put(f"dCdTenth{t}", pc(g["c4_tenth_lower"].median()))
        put(f"dCdBig{t}", pc(g["c4_all_lower_5pct"].median()))
        put(f"dCdFive{t}", pc(g["c4_all_lower_at_5"].median()))
    (gen / "diag_macros.tex").write_text("\n".join(out) + "\n")
