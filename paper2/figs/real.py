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
    return d


def main_test(d: pd.DataFrame, n_perm: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Spearman of checks passed (and each check's score) with item-by-store WRMSSE,
    within data set; pooled by permuting scores within data set."""
    rng = np.random.default_rng(seed)
    cols = ["passed", "C1", "C2", "C3", "C4", "C5_rank", "C5_bias", "C6"]
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
    a.plot(xs, -0.4 * xs, color=MUTED, lw=0.8, ls=(0, (4, 3)))
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
    b.axvline(-0.4, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    b.axvline(0, color=INK, lw=0.5)
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


def write_tables(d, mt, gen: Path):
    lines = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
             "Data & Versions & C1 & C2 & C3 & C4 & C5 & C6 & All \\\\", "\\midrule"]
    for ds in [k for k in DATASETS if k in set(d["dataset"])]:
        g = d[d["dataset"] == ds]
        cells = [DATASETS[ds], str(len(g))]
        for k in range(1, 7):
            p = g[f"C{k}_pass"].astype(float).dropna()
            cells.append(f"{int(p.sum())}/{len(p)}" if len(p) else "n/a")
        cells.append(f"{int((g['passed'] == g['applicable']).sum())}/{len(g)}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "tab_real_pass.tex").write_text("\n".join(lines) + "\n")
    names = {"passed": "Checks passed", "C1": "C1 violation", "C2": "C2 agreement after randomising",
             "C3": "C3 seed agreement", "C4": "C4 share lower", "C5_rank": "C5 rank", "C5_bias": "C5 bias",
             "C6": "C6 agreement with LightGBM"}
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "Score & M5 & Dominick's & Frat & Pooled & $p$ \\\\", "\\midrule"]
    for _, r in mt.iterrows():
        cells = [names.get(r["score"], r["score"])]
        for ds in DATASETS:
            v = r.get(f"rho_{ds}")
            cells.append(f"${v:.2f}$" if pd.notna(v) else "n/a")
        cells += [f"${r['pooled_rho']:.2f}$", f"{r['p']:.3f}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "tab_real_main.tex").write_text("\n".join(lines) + "\n")


def make(out: Path, measured):
    v, st, ag, z = load()
    d = score(v, st, ag)
    gen = out.parent / "generated"
    d.to_csv(gen / "real_scores.csv", index=False)
    mt = main_test(d)
    mt.to_csv(gen / "real_main_test.csv", index=False)
    fig_real_checks(d, st, ag, out)
    fig_real_main(d, out)
    rows = fig_real_c7(z, *measured, out) if len(z) else []
    write_tables(d, mt, gen)
    return d, mt, rows
