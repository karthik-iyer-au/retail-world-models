"""Known-answer study: load the records, score every check with the plan's pass
rules, and draw its figures and tables (results/paper2/known_answer/)."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from style import ACCENT, FILL, FULL, GRID, INK, MUTED, despine, panel_label, save

ROOT = Path(__file__).resolve().parents[2]
KA = Path(os.environ.get("KNOWN_ANSWER", ROOT / "results" / "paper2" / "known_answer"))
WORLDS = ["clean", "confounded"]
MODELS = ["world_seed1", "world_seed2", "world_seed3", "world_price_blind", "world_lever_blind",
          "lightgbm", "lightgbm_price_blind"]
LABEL = {"world_seed1": "World model, seed 1", "world_seed2": "World model, seed 2",
         "world_seed3": "World model, seed 3", "world_price_blind": "World model, blind to price",
         "world_lever_blind": "World model, blind to levers", "lightgbm": "LightGBM",
         "lightgbm_price_blind": "LightGBM, blind to price"}
SHORT = {"world_seed1": "WM s1", "world_seed2": "WM s2", "world_seed3": "WM s3", "world_price_blind": "WM no price",
         "world_lever_blind": "WM no levers", "lightgbm": "LGB", "lightgbm_price_blind": "LGB no price"}
PASS, FAIL, NA = "#4F7F6A", "#B5523B", "#EDEBE6"
CHECKS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def available() -> bool:
    return all((KA / f"{w}_{m}.json").exists() for w in WORLDS for m in MODELS)


def rank(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ((np.abs(a) > 0) | (np.abs(b) > 0)) & np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[keep], b[keep]).statistic) if keep.sum() > 2 else float("nan")


def load():
    rec = {(w, m): json.loads((KA / f"{w}_{m}.json").read_text()) for w in WORLDS for m in MODELS}
    lifts = {(w, m): pd.read_csv(KA / f"{w}_{m}_lifts.csv.gz") for w in WORLDS for m in MODELS}
    worlds = {w: json.loads((KA / f"{w}_world.json").read_text()) for w in WORLDS}
    return rec, lifts, worlds


def scores(rec, lifts) -> pd.DataFrame:
    """One row per world and model: each check's score and pass (True/False/None), and the
    agreement of its explanations with the truth."""
    rows = []
    for w in WORLDS:
        seeds = [m for m in MODELS if m.startswith("world_seed")]
        seed_r = np.mean([rank(lifts[(w, a)]["total_lift"], lifts[(w, b)]["total_lift"])
                          for a, b in itertools.combinations(seeds, 2)])
        seed_fc = np.mean([rank(lifts[(w, a)]["forecast"], lifts[(w, b)]["forecast"])
                           for a, b in itertools.combinations(seeds, 2)])
        lgb = lifts[(w, "lightgbm")]
        for m in MODELS:
            r, L = rec[(w, m)], lifts[(w, m)]
            row = {"world": w, "model": m}
            c1 = max(r["construction"].values()) if "construction" in r else None
            row["C1"], row["C1_pass"] = c1, (c1 < 1e-3) if c1 is not None else None
            c2 = r.get("randomisation", {}).get("lift readout")
            row["C2"], row["C2_pass"] = c2, (c2 < 0.5) if c2 is not None else None
            row["C3"], row["C3_pass"] = (seed_r, seed_r >= 0.7) if m in seeds else (None, None)
            c4 = r["monotonicity"]["share_lower_after_deeper_cut"]
            row["C4"], row["C4_pass"] = c4, c4 < 0.01
            s = r["against_sales"]["any_plan"]
            row["C5_bias"], row["C5_rank"] = s["bias"], s["rank"]
            row["C5"], row["C5_pass"] = s["rank"], (0.8 <= s["bias"] <= 1.25) and s["rank"] >= 0.3
            if r["kind"] == "lightgbm_direct":
                row["C6"], row["C6_pass"] = None, None
            else:
                c6 = rank(L["total_lift"], lgb["total_lift"])
                row["C6"], row["C6_pass"] = c6, c6 >= 0.5
                for lv in ("price", "promo", "display"):
                    row[f"C6_{lv}"] = rank(L[f"lift_{lv}"], lgb[f"lift_{lv}"])
                    row[f"C6_{lv}_shap"] = rank(L[f"lift_{lv}"], lgb[f"shap_{lv}"])
                row["C6_lever_min"] = min(row[f"C6_{lv}"] for lv in ("price", "promo", "display"))
            eps = r["origin_episodes"]
            from make_figures import ols_slope
            x = [e["log_ratio"] for e in eps]
            ms, mse = ols_slope([e["did"]["estimate"] for e in eps], x)
            ss, _ = ols_slope([e["stated"] for e in eps], x)
            row["C7_episode_inside"] = float(np.mean([e["did"]["lo"] <= e["stated"] <= e["did"]["hi"] for e in eps]))
            row["C7_measured_slope"], row["C7_measured_lo"], row["C7_measured_hi"] = ms, ms - 1.96 * mse, ms + 1.96 * mse
            row["C7"] = ss
            row["C7_pass"] = ms - 1.96 * mse <= ss <= ms + 1.96 * mse
            row["C7_err_vs_truth"] = float(np.mean([e["stated"] - e["true"] for e in eps]))
            row["C7_stated_elasticity"] = float(np.median([e["stated"] / e["log_ratio"] for e in eps]))
            t = r["truth"]
            row["truth_rank"], row["truth_bias"] = t["total_lift"]["rank"], t["total_lift"]["bias"]
            for lv in ("price", "promo", "display"):
                row[f"truth_{lv}_rank"], row[f"truth_{lv}_bias"] = t[f"lift_{lv}"]["rank"], t[f"lift_{lv}"]["bias"]
            row["truth_log_err"] = t["lift_ratio_median_abs_log_error"]
            row["rmsse"] = r["accuracy"]["item_by_store_rmsse"]
            row["fit_seconds"] = r["fit_seconds"]
            applicable = [row[f"{c}_pass"] for c in CHECKS[:6] if row[f"{c}_pass"] is not None]
            row["passed"], row["applicable"] = int(sum(applicable)), len(applicable)
            row["seed_forecast_rank"] = seed_fc if m in seeds else None
            rows.append(row)
    return pd.DataFrame(rows)


def fig_known_checks(sc: pd.DataFrame, out: Path):
    """Pass/fail grid: models by checks, in each world, with the truth alongside."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.75), gridspec_kw={"wspace": 0.05})
    cols = CHECKS[:6] + ["C6L", "C7", "truth"]
    for k, (ax, w) in enumerate(zip(axes, WORLDS)):
        d = sc[sc["world"] == w].set_index("model").loc[MODELS]
        for i, m in enumerate(MODELS):
            for j, c in enumerate(cols):
                if c == "truth":
                    v = d.loc[m, "truth_rank"]
                    ok = v >= 0.7 and 0.8 <= d.loc[m, "truth_bias"] <= 1.25
                    face, txt = (PASS if ok else FAIL), f"{v:.2f}\n×{d.loc[m, 'truth_bias']:.2f}"
                elif c == "C6L":
                    v = d.loc[m, "C6_lever_min"] if "C6_lever_min" in d else np.nan
                    if pd.isna(v):
                        face, txt = NA, "n/a"
                    else:
                        face, txt = (PASS if v >= 0.5 else FAIL), f"{v:.2f}"
                else:
                    p, v = d.loc[m, f"{c}_pass"], d.loc[m, c]
                    if p is None or (isinstance(p, float) and np.isnan(p)):
                        face, txt = NA, "n/a"
                    else:
                        face = PASS if p else FAIL
                        txt = f"{v:.0e}" if c == "C1" else (f"{v * 100:.1f}%" if c == "C4" else f"{v:.2f}")
                        txt = txt.replace("-", "\u2212")
                        if c == "C5":
                            txt = f"{d.loc[m, 'C5_rank']:.2f}\n×{d.loc[m, 'C5_bias']:.2f}"
                x = j + (0.35 if c == "truth" else 0)
                ax.add_patch(Rectangle((x + 0.04, i + 0.06), 0.92, 0.88, facecolor=face, edgecolor="none",
                                       alpha=0.9 if face != NA else 1))
                ax.text(x + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5.2,
                        color="white" if face != NA else MUTED, linespacing=0.95)
        ax.set_xlim(0, len(cols) + 0.35)
        ax.set_ylim(len(MODELS), 0)
        ax.set_xticks([j + 0.5 + (0.35 if c == "truth" else 0) for j, c in enumerate(cols)])
        ax.set_xticklabels(["C1", "C2", "C3", "C4", "C5", "C6", "C6\nlevers", "C7", "Truth"], fontsize=6.4)
        ax.xaxis.tick_top()
        ax.set_yticks([i + 0.5 for i in range(len(MODELS))])
        ax.set_yticklabels([LABEL[m] for m in MODELS] if k == 0 else [], fontsize=6.6)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{'Plans at random' if w == 'clean' else 'Plans in high-demand weeks'}", pad=16)
    save(fig, out / "fig_known_checks.pdf")


def fig_known_scatter(lifts, out: Path):
    """Each model's lift against the true lift, item-week by item-week."""
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.9), gridspec_kw={"wspace": 0.38, "hspace": 0.55})
    show = ["world_seed1", "world_price_blind", "lightgbm", "lightgbm_price_blind"]
    for i, w in enumerate(WORLDS):
        for j, m in enumerate(show):
            ax = axes[i, j]
            L = lifts[(w, m)]
            L = L[L["planned"]]
            x, y = L["true_total_lift"], L["total_lift"]
            lim = np.nanpercentile(np.abs(np.r_[x, y]), 99.5)
            ax.scatter(x, y, s=1.5, color=ACCENT, alpha=0.35, lw=0, rasterized=True)
            ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.7, ls=(0, (3, 2)))
            ax.set_xlim(-0.35 * lim, lim)
            ax.set_ylim(-0.35 * lim, lim)
            ax.axhline(0, color=GRID, lw=0.6)
            ax.axvline(0, color=GRID, lw=0.6)
            r = rank(y, x)
            ax.text(0.04, 0.95, f"rank {r:.2f}\nsum ×{y.sum() / x.sum():.2f}", transform=ax.transAxes, va="top",
                    fontsize=6, color=INK)
            if i == 0:
                ax.set_title(LABEL[m].replace(", ", ",\n"), fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{'At random' if w == 'clean' else 'High-demand weeks'}\nmodel's lift, units")
            if i == 1:
                ax.set_xlabel("True lift, units")
            despine(ax)
    save(fig, out / "fig_known_scatter.pdf")


def fig_known_c7(rec, out: Path):
    """Stated response to the price spells at the origin, against the truth and DiD."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7), gridspec_kw={"wspace": 0.08})
    rows = ["did"] + MODELS
    for k, (ax, w) in enumerate(zip(axes, WORLDS)):
        for i, m in enumerate(rows):
            if m == "did":
                eps = rec[(w, MODELS[0])]["origin_episodes"]
                st = [e["did"]["estimate"] / e["log_ratio"] for e in eps]
                col = INK
            else:
                eps = rec[(w, m)]["origin_episodes"]
                st = [e["stated"] / e["log_ratio"] for e in eps]
                col = ACCENT
            ax.scatter(st, [i] * len(st), s=12, color=col, edgecolor="white", lw=0.4, zorder=3)
            ax.plot([np.median(st)] * 2, [i - 0.3, i + 0.3], color=col, lw=1.3)
        ax.axhline(0.5, color=GRID, lw=0.8)
        ax.axvline(-0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.axvline(-2.5, color=MUTED, lw=0.8, ls=(0, (1, 2)))
        ax.text(-0.45, len(rows) - 0.35, "true regular\nresponse −0.5", fontsize=5.8, ha="left", va="top", color=INK)
        ax.text(-2.45, len(rows) - 0.35, "temporary\ncut −2.5", fontsize=5.8, ha="left", va="top", color=MUTED)
        ax.set_ylim(len(rows) + 0.6, -0.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(["Measured (DiD)"] + [LABEL[m] for m in MODELS] if k == 0 else [], fontsize=6.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Response per unit of log regular price")
        ax.set_title(f"{'Plans at random' if w == 'clean' else 'Plans in high-demand weeks'}")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        despine(ax, keep=("bottom",))
    save(fig, out / "fig_known_c7.pdf")


def fig_known_cascade(rec, out: Path):
    """C2: rank agreement with the original lifts as parts are re-initialised."""
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    parts = ["original", "lift readout", "baseline readout", "transition", "encoder"]
    for w, ls in zip(WORLDS, ["-", (0, (3, 2))]):
        for m in [m for m in MODELS if m.startswith("world")]:
            r = rec[(w, m)].get("randomisation")
            if not r:
                continue
            ys = [1.0] + [r[p] for p in parts[1:]]
            col = ACCENT if "seed" in m else MUTED
            ax.plot(range(len(parts)), ys, color=col, lw=0.9, ls=ls, marker="o", ms=2.2, alpha=0.9)
    ax.axhline(0.5, color=GRID, lw=0.8)
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(["none", "+ lift\nreadout", "+ baseline\nreadout", "+ transition", "+ encoder"], fontsize=6.2)
    ax.set_ylabel("Rank agreement with\nthe original lifts")
    ax.set_ylim(-0.3, 1.05)
    ax.plot([], [], color=ACCENT, label="world model, seeds")
    ax.plot([], [], color=MUTED, label="planted failures")
    ax.plot([], [], color=INK, ls=(0, (3, 2)), label="high-demand weeks")
    ax.legend(fontsize=6, loc="lower left")
    despine(ax)
    save(fig, out / "fig_known_cascade.pdf")


def fig_known_series(out: Path, world: str = "confounded"):
    """What the data look like: one item in one store, its plan, sales and the true baseline."""
    d = pd.read_csv(KA / f"{world}_example.csv")
    base, full = d["expected_nothing_planned"].to_numpy(), d["expected"].to_numpy()
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.3, 2.7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    x = np.arange(len(d))
    a.bar(x, d["units"], color="#D9D4CB", width=0.85, label="sales")
    a.plot(x, full, color=ACCENT, lw=1.0, label="expected under the plan")
    a.plot(x, base, color=INK, lw=0.9, ls=(0, (3, 2)), label="expected with nothing planned")
    a.set_ylabel("Units")
    a.legend(fontsize=5.8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    despine(a)
    disc = 1 - d["price"] / d["true_regular"]
    b.bar(x, disc * 100, color=MUTED, width=0.85)
    for c, y, col in (("promo", -8, ACCENT), ("display", -16, INK)):
        on = d[c].to_numpy() > 0
        b.scatter(x[on], [y] * on.sum(), s=5, marker="s", color=col)
        b.text(len(d) + 0.5, y, c, fontsize=5.6, ha="left", va="center", color=col)
    b.set_ylabel("Cut, %")
    b.set_ylim(-20, 45)
    b.set_xlabel("Week")
    despine(b)
    save(fig, out / "fig_known_series.pdf")


def write_tables(sc: pd.DataFrame, rec, worlds, gen: Path):
    """Macros and the per-model table."""
    def g(w, m, c):
        return sc[(sc["world"] == w) & (sc["model"] == m)][c].iloc[0]
    M = {}
    for w, tag in (("clean", "Cl"), ("confounded", "Co")):
        M[f"ka{tag}SeedRank"] = f"{g(w, 'world_seed1', 'C3'):.2f}"
        M[f"ka{tag}SeedFc"] = f"{g(w, 'world_seed1', 'seed_forecast_rank'):.2f}"
        for m, mt in (("world_seed1", "Wm"), ("world_price_blind", "WmNp"), ("world_lever_blind", "WmNl"),
                      ("lightgbm", "Lgb"), ("lightgbm_price_blind", "LgbNp")):
            M[f"ka{tag}{mt}Truth"] = f"{g(w, m, 'truth_rank'):.2f}"
            M[f"ka{tag}{mt}Bias"] = f"{g(w, m, 'truth_bias'):.2f}"
            M[f"ka{tag}{mt}Passed"] = f"{int(g(w, m, 'passed'))} of {int(g(w, m, 'applicable'))}"
            M[f"ka{tag}{mt}Cseven"] = f"{g(w, m, 'C7_stated_elasticity'):.2f}"
            M[f"ka{tag}{mt}Rmsse"] = f"{g(w, m, 'rmsse'):.3f}"
            M[f"ka{tag}{mt}Cfive"] = f"{g(w, m, 'C5_bias'):.2f}"
            M[f"ka{tag}{mt}Cfour"] = f"{g(w, m, 'C4') * 100:.1f}"
            for lv in ("price", "promo", "display"):
                M[f"ka{tag}{mt}{lv.capitalize()}Bias"] = f"{g(w, m, f'truth_{lv}_bias'):.2f}"
        M[f"ka{tag}Observed"] = f"{worlds[w]['observed_over_true_nothing_planned']:.2f}"
        M[f"ka{tag}TrueLift"] = f"{worlds[w]['true_forecast_over_true_nothing_planned']:.2f}"
        did = [e["did"]["estimate"] / e["log_ratio"] for e in rec[(w, "world_seed1")]["origin_episodes"]]
        M[f"ka{tag}DidMed"] = f"{np.median(did):.2f}"
    sub = sc[sc["model"].isin(MODELS)]
    rho = spearmanr(sub["passed"] / sub["applicable"], sub["truth_rank"]).statistic
    M["kaPassTruthRho"] = f"{rho:.2f}"
    M["kaPassBiasRho"] = f"{spearmanr(sub['passed'] / sub['applicable'], -np.abs(np.log(sub['truth_bias'].clip(lower=1e-3)))).statistic:.2f}"
    M["kaRmsseTruthRho"] = f"{spearmanr(sub['rmsse'], sub['truth_rank']).statistic:.2f}"
    seeds = [m for m in MODELS if m.startswith("world_seed")]
    enc = [rec[(w, m)]["randomisation"]["encoder"] for w in WORLDS for m in MODELS if "randomisation" in rec[(w, m)]]
    M["kaEncLo"], M["kaEncHi"] = f"{min(enc):.2f}", f"{max(enc):.2f}"
    lr = [rec[(w, m)]["randomisation"]["lift readout"] for w in WORLDS for m in MODELS if "randomisation" in rec[(w, m)]]
    M["kaLiftReadLo"], M["kaLiftReadHi"] = f"{min(lr):.2f}", f"{max(lr):.2f}"
    for w, tag in (("clean", "Cl"), ("confounded", "Co")):
        eps = rec[(w, "world_seed1")]["origin_episodes"]
        M[f"ka{tag}DidCover"] = f"{sum(e['did']['lo'] <= e['true'] <= e['did']['hi'] for e in eps)}"
        row = sc[(sc["world"] == w) & (sc["model"] == "world_seed1")].iloc[0]
        M[f"ka{tag}MeasSlope"] = f"{row['C7_measured_slope']:.2f}"
        M[f"ka{tag}MeasLo"] = f"{row['C7_measured_lo']:.2f}"
        M[f"ka{tag}MeasHi"] = f"{row['C7_measured_hi']:.2f}"
        ws = sc[(sc["world"] == w) & sc["model"].isin(seeds)]
        M[f"ka{tag}WmBiasLo"], M[f"ka{tag}WmBiasHi"] = f"{ws['truth_bias'].min():.2f}", f"{ws['truth_bias'].max():.2f}"
        M[f"ka{tag}WmDispLo"], M[f"ka{tag}WmDispHi"] = f"{ws['truth_display_bias'].min():.2f}", f"{ws['truth_display_bias'].max():.2f}"
        M[f"ka{tag}WmCsixPromo"] = f"{row['C6_promo']:.2f}"
        M[f"ka{tag}WmCsix"] = f"{row['C6']:.2f}"
        M[f"ka{tag}WmCsevenSlope"] = f"{ws['C7'].median():.2f}"
        M[f"ka{tag}WmCsevenLo"], M[f"ka{tag}WmCsevenHi"] = f"{ws['C7'].min():.2f}", f"{ws['C7'].max():.2f}"
        M[f"ka{tag}WmTimesLo"] = f"{(ws['C7'] / -0.5).min():.1f}"
        M[f"ka{tag}WmTimesHi"] = f"{(ws['C7'] / -0.5).max():.1f}"
        nl = sc[(sc["world"] == w) & (sc["model"] == "world_lever_blind")].iloc[0]
        M[f"ka{tag}NlCsixPromo"], M[f"ka{tag}NlCsixDisplay"] = f"{nl['C6_promo']:.2f}", f"{nl['C6_display']:.2f}"
        M[f"ka{tag}NlCsix"] = f"{nl['C6']:.2f}"
        lg = sc[(sc["world"] == w) & (sc["model"] == "lightgbm")].iloc[0]
        M[f"ka{tag}LgbCsevenSlope"] = f"{lg['C7']:.2f}"
    M["kaFitMinutes"] = f"{sc['fit_seconds'].sum() / 60:.0f}"
    M["kaPlanned"] = f"{worlds['clean']['planned_item_weeks']:,}"
    M["kaRows"] = f"{worlds['clean']['rows']:,}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}".replace("{-", "{−") for k, v in M.items()]
    (gen / "known_macros.tex").write_text("\n".join(lines) + "\n")
    rows = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
            " & \\multicolumn{3}{c}{Lift against truth} & & & & \\\\",
            "\\cmidrule(lr){2-4}",
            "Model & rank & sum & price & C4 & C5 & Passed & RMSSE \\\\", "\\midrule"]
    for w in WORLDS:
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{'Plans at random' if w == 'clean' else 'Plans in high-demand weeks'}}}}} \\\\")
        for m in MODELS:
            r = sc[(sc["world"] == w) & (sc["model"] == m)].iloc[0]
            rows.append(f"{LABEL[m]} & {r['truth_rank']:.2f} & {r['truth_bias']:.2f} & {r['truth_price_bias']:.2f} & "
                        f"{r['C4'] * 100:.1f}\\% & {r['C5_bias']:.2f} & {int(r['passed'])}/{int(r['applicable'])} & "
                        f"{r['rmsse']:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (gen / "tab_known.tex").write_text("\n".join(rows).replace("& -", "& −") + "\n")


def make(out: Path):
    rec, lifts, worlds = load()
    sc = scores(rec, lifts)
    gen = out.parent / "generated"
    sc.to_csv(gen / "known_scores.csv", index=False)
    fig_known_series(out)
    fig_known_checks(sc, out)
    fig_known_scatter(lifts, out)
    fig_known_c7(rec, out)
    fig_known_cascade(rec, out)
    write_tables(sc, rec, worlds, gen)
    return sc
