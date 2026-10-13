"""Paper 2 figures, from results/paper2/ (vector PDF into paper2/figs/).

  python paper2/figs/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from style import ACCENT, EDGE, FILL, FULL, GRID, INK, MUTED, despine, panel_label, save, use_style

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
EP = ROOT / "results" / "paper2" / "zone_episodes.csv"
CAT = {"che": "Cheese", "sna": "Snack crackers", "sdr": "Soft drinks", "bjc": "Bottled juice",
       "ana": "Analgesics", "fre": "Frozen entrees"}
UP, DOWN = "#B5523B", "#3E6E9A"  # price moved up, down


def fig_audit():
    """The eight checks, grouped by what they compare the explanation with."""
    fig = plt.figure(figsize=(FULL, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # the explanation in the middle
    ax.add_patch(FancyBboxPatch((7.2, 3.6), 5.6, 2.8, boxstyle="round,pad=0.02,rounding_size=0.25",
                                facecolor=ACCENT, edgecolor="none"))
    ax.text(10, 5.75, "One forecast, explained", ha="center", fontsize=8, weight="bold", color="white")
    ax.text(10, 4.9, "baseline (nothing planned)", ha="center", fontsize=6.6, color="white")
    ax.text(10, 4.35, "× lift: promotion, price, display, ...", ha="center", fontsize=6.6, color="white")
    groups = [
        (0.2, 6.3, "Against the model itself", ["C1  holds by construction", "C2  changes when weights are randomised",
                                                 "C3  same across seeds", "C8  plan readable from the state"]),
        (0.2, 0.4, "Against common sense", ["C4  a deeper cut never lowers the forecast"]),
        (13.6, 6.3, "Against what sold", ["C5  predicted lift against sales in", "      promoted and price-cut weeks"]),
        (13.6, 0.4, "Against another model and a cause", ["C6  agrees with LightGBM (on/off, TreeSHAP)",
                                                          "C7  matches the measured response to", "      zone price deviations"]),
    ]
    for x, y, title, lines in groups:
        h = 3.3
        ax.add_patch(FancyBboxPatch((x, y), 6.2, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    facecolor=FILL, edgecolor="none"))
        ax.text(x + 0.25, y + h - 0.35, title, fontsize=7.2, weight="bold", color=INK, va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 0.25, y + h - 1.05 - i * 0.55, ln, fontsize=6.3, va="top", color=INK)
        cx = x + 6.2 if x < 5 else x
        cy = y + h / 2
        tx = 7.2 if x < 5 else 12.8
        ax.add_patch(FancyArrowPatch((cx, cy), (tx, 5.0 + (0.6 if y > 5 else -0.6)), arrowstyle="-",
                                     color=EDGE, lw=0.9, connectionstyle="arc3,rad=0.0"))
    ax.text(10, 2.6, "Question: do versions whose explanations\npass more checks also forecast better?",
            ha="center", va="top", fontsize=6.8, color=ACCENT, style="italic")
    save(fig, OUT / "fig_audit.pdf")


def fig_zones():
    d = pd.read_csv(EP)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw={"wspace": 0.32, "width_ratios": [1.35, 1]})
    cats = list(CAT)
    for i, c in enumerate(cats):
        e = d[d["category"] == c].sort_values("first_week").reset_index(drop=True)
        n = max(len(e), 1)
        for j, r in e.iterrows():
            y = i + (j - (n - 1) / 2) * min(0.7 / n, 0.12)
            col = UP if r["price_log_ratio"] > 0 else DOWN
            a.plot([r["first_week"], r["last_week"]], [y, y], color=col, lw=1.6 if r["scored"] else 0.8,
                   alpha=1 if r["scored"] else 0.45, solid_capstyle="butt")
    a.axvspan(190, 240, color=FILL, zorder=0)
    a.text(215, -0.75, "most episodes", ha="center", fontsize=6.2, color=MUTED)
    a.set_yticks(range(len(cats)))
    a.set_yticklabels([f"{CAT[c]} ({(d['category'] == c).sum()})" for c in cats], fontsize=6.8)
    a.set_ylim(len(cats) - 0.4, -1.6)
    weeks = [1, 69, 121, 173, 225, 277, 329]
    a.set_xticks(weeks)
    a.set_xticklabels([str(1989 + (w + 15) // 52) if True else "" for w in weeks], fontsize=6.5)
    a.set_xlabel("Year")
    a.grid(axis="x")
    a.set_axisbelow(True)
    a.tick_params(axis="y", length=0)
    despine(a, keep=("bottom",))
    a.set_title("Store price deviations of 12 weeks or more (count)")
    a.plot([], [], color=UP, lw=1.6, label="prices up against the zone")
    a.plot([], [], color=DOWN, lw=1.6, label="prices down")
    a.legend(loc="upper right", fontsize=6.2)
    panel_label(a, "a")
    # (b) measured response against the price change
    s = d[d["scored"]]
    for _, r in s.iterrows():
        col = UP if r["price_log_ratio"] > 0 else DOWN
        b.plot([r["price_log_ratio"] * 100] * 2, [r["lo_8wk"] * 100, r["hi_8wk"] * 100], color=col, lw=0.6, alpha=0.5)
        b.scatter([r["price_log_ratio"] * 100], [r["estimate_8wk"] * 100], s=14, color=col, edgecolor="white",
                  lw=0.4, zorder=3)
    slope, icpt = np.polyfit(s["price_log_ratio"], s["estimate_8wk"], 1)
    xs = np.linspace(-11, 11, 10)
    b.plot(xs, (icpt + slope * xs / 100) * 100, color=INK, lw=0.9)
    b.plot(xs, -0.4 * xs, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    b.text(10.8, -0.4 * 10.8 - 2.0, "Hoch et al.\n−0.4", fontsize=6.0, color=MUTED, ha="right", va="top")
    b.text(-10.8, (icpt - slope * 0.108) * 100 + 2.2, f"fitted slope {slope:.2f}", fontsize=6.2, color=INK)
    b.axhline(0, color=GRID, lw=0.8)
    b.axvline(0, color=GRID, lw=0.8)
    b.set_xlabel("Price against zone peers, %")
    b.set_ylabel("Change in category units, %\n(difference in differences, 8 weeks)")
    b.set_title("Measured response: the ground truth")
    b.grid(True)
    b.set_axisbelow(True)
    despine(b)
    panel_label(b, "b")
    (OUT.parent / "generated").mkdir(exist_ok=True)
    sig = ((s["lo_8wk"] > 0) | (s["hi_8wk"] < 0)).mean()
    sign = (np.sign(s["price_log_ratio"]) != np.sign(s["estimate_8wk"])).mean()
    (OUT.parent / "generated" / "zone_macros.tex").write_text("\n".join([
        f"\\newcommand{{\\zEpisodes}}{{{len(d)}}}",
        f"\\newcommand{{\\zScored}}{{{len(s)}}}",
        f"\\newcommand{{\\zSlope}}{{{slope:.2f}}}",
        f"\\newcommand{{\\zMedianElasticity}}{{{(s['estimate_8wk'] / s['price_log_ratio']).median():.2f}}}",
        f"\\newcommand{{\\zSignShare}}{{{sign * 100:.0f}}}",
        f"\\newcommand{{\\zSigShare}}{{{sig * 100:.0f}}}",
        f"\\newcommand{{\\zSizeLo}}{{{s['price_log_ratio'].abs().min() * 100:.0f}}}",
        f"\\newcommand{{\\zSizeHi}}{{{s['price_log_ratio'].abs().max() * 100:.0f}}}",
    ]) + "\n")
    save(fig, OUT / "fig_zones.pdf")


def ols_slope(y, x):
    """Slope of y on x with an intercept, and its heteroskedasticity-robust (HC1) standard error."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    X = np.c_[np.ones(n), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * r ** 2) @ X @ XtXi * n / (n - 2)
    return float(b[1]), float(np.sqrt(V[1, 1]))


def fig_zone_checks():
    """Price path, event study, placebo and pooled slopes (results/paper2/, data only)."""
    R = ROOT / "results" / "paper2"
    ev, pl, pp = pd.read_csv(R / "event_study.csv"), pd.read_csv(R / "placebo.csv"), pd.read_csv(R / "price_path.csv")
    ep = pd.read_csv(EP)
    ep = ep[ep["scored"]]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.6), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
    a, b, c, dd = axes.ravel()
    # (a) price path
    pp["signed"] = pp["ratio"] * np.sign(pp["price_log_ratio"])
    g = pp.groupby("rel_week")["signed"]
    q = pd.DataFrame({"med": g.median(), "lo": g.quantile(0.25), "hi": g.quantile(0.75)}).reindex(range(-52, 8))
    a.fill_between(q.index, q["lo"] * 100, q["hi"] * 100, color=ACCENT, alpha=0.15, lw=0)
    a.plot(q.index, q["med"] * 100, color=ACCENT, lw=1.3)
    a.axvspan(-26, -1, color=FILL, zorder=0)
    a.axvline(-0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    a.axhline(0, color=GRID, lw=0.8)
    a.text(-13.5, 10.8, "comparison\nweeks", ha="center", fontsize=5.8, color=MUTED)
    a.set_ylim(-4, 13)
    a.set_xlabel("Weeks from the detected start")
    a.set_ylabel("Price against zone peers, %\n(signed by the episode)")
    a.set_title("Prices move early")
    despine(a)
    panel_label(a, "a")
    # (b) event study
    ev["signed"] = ev["coef"] * np.sign(ev["price_log_ratio"])
    g = ev.groupby("rel_week")["signed"]
    q2 = pd.DataFrame({"med": g.median(), "lo": g.quantile(0.25), "hi": g.quantile(0.75)}).reindex(range(-26, 8))
    b.fill_between(q2.index, q2["lo"] * 100, q2["hi"] * 100, color=ACCENT, alpha=0.15, lw=0)
    b.plot(q2.index, q2["med"] * 100, color=ACCENT, lw=1.3, marker="o", ms=1.8)
    b.axvline(-0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    b.axhline(0, color=GRID, lw=0.8)
    b.set_ylim(-25, 25)
    b.set_xlabel("Weeks from the detected start")
    b.set_ylabel("Units against controls, %\n(signed, week $-1$ = 0)")
    b.set_title("Event study")
    despine(b)
    panel_label(b, "b")
    # (c) placebo against real
    m = pl.merge(ep[["category", "store", "first_week", "estimate_8wk"]], on=["category", "store", "first_week"])
    for _, r in m.iterrows():
        col = UP if r["price_log_ratio"] > 0 else DOWN
        c.scatter(r["estimate_8wk"] * 100, r["placebo_estimate"] * 100, s=10, color=col, edgecolor="white", lw=0.4, zorder=3)
    lim = 1.08 * max(m["estimate_8wk"].abs().max(), m["placebo_estimate"].abs().max()) * 100
    c.plot([-lim, lim], [-lim, lim], color=GRID, lw=0.8)
    c.axhline(0, color=INK, lw=0.5)
    c.axvline(0, color=INK, lw=0.5)
    c.set_xlim(-lim, lim)
    c.set_ylim(-lim, lim)
    c.set_xlabel("Real start, units %")
    c.set_ylabel("Start 26 weeks earlier, units %")
    c.set_title("Placebo")
    despine(c)
    panel_label(c, "c")
    # (d) pooled slopes with intervals
    rows = [("8 weeks (plan)", *ols_slope(ep["estimate_8wk"], ep["price_log_ratio"]), len(ep)),
            ("Full episode", *ols_slope(ep["estimate"], ep["price_log_ratio"]), len(ep)),
            ("Placebo", *ols_slope(pl["placebo_estimate"], pl["price_log_ratio"]), len(pl))]
    for cat, d in ep.groupby("category"):
        if len(d) >= 5:
            rows.append((CAT[cat], *ols_slope(d["estimate_8wk"], d["price_log_ratio"]), len(d)))
    XL = (-2.0, 1.5)
    for k, (name, sl, se, n) in enumerate(rows):
        col = INK if k < 2 else (MUTED if k == 2 else ACCENT)
        lo, hi = sl - 1.96 * se, sl + 1.96 * se
        dd.plot([max(lo, XL[0]), min(hi, XL[1])], [k, k], color=col, lw=1.0)
        if hi > XL[1]:
            dd.annotate("", xy=(XL[1], k), xytext=(XL[1] - 0.15, k), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
        if lo < XL[0]:
            dd.annotate("", xy=(XL[0], k), xytext=(XL[0] + 0.15, k), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
        if XL[0] <= sl <= XL[1]:
            dd.scatter([sl], [k], s=14, color=col, zorder=3)
    dd.set_xlim(*XL)
    dd.text(-0.36, -0.75, "Hoch et al. $-0.4$", fontsize=5.8, color=MUTED, ha="left", va="bottom")
    dd.axvline(-0.4, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    dd.axvline(0, color=INK, lw=0.5)
    dd.axhline(2.5, color=GRID, lw=0.6)
    dd.set_yticks(range(len(rows)))
    dd.set_yticklabels([f"{r[0]} ({r[3]})" for r in rows], fontsize=6.2)
    dd.set_ylim(len(rows) - 0.4, -1.0)
    dd.set_xlabel("Pooled slope, 95% interval")
    dd.set_title("Response to price")
    dd.tick_params(axis="y", length=0)
    despine(dd, keep=("bottom",))
    panel_label(dd, "d")
    s8, se8 = rows[0][1], rows[0][2]
    sf, sef = rows[1][1], rows[1][2]
    sp, sep = rows[2][1], rows[2][2]
    pre8 = pp[pp["rel_week"].between(-8, -1)]
    early = (pre8["signed"] >= 0.03).mean()
    sig = ((pl["placebo_lo"] > 0) | (pl["placebo_hi"] < 0)).mean()
    neg = lambda v: f"{v:.2f}"
    macros = {
        "zSlopeLo": neg(s8 - 1.96 * se8), "zSlopeHi": neg(s8 + 1.96 * se8),
        "zFullSlope": neg(sf), "zFullLo": neg(sf - 1.96 * sef), "zFullHi": neg(sf + 1.96 * sef),
        "zPlaceboSlope": neg(sp), "zPlaceboLo": neg(sp - 1.96 * sep), "zPlaceboHi": neg(sp + 1.96 * sep),
        "zPlaceboSig": f"{sig * 100:.0f}", "zPlaceboN": f"{len(pl)}",
        "zEarlyShare": f"{early * 100:.0f}",
        "zEpisodeSE": f"{ep['se_8wk'].median():.2f}", "zPlaceboSD": f"{pl['placebo_estimate'].std():.2f}",
        "zCatNeg": f"{sum(r[1] < 0 for r in rows[3:])}", "zCatN": f"{len(rows) - 3}",
    }
    sens = pd.read_csv(R / "sensitivity.csv")
    macros.update({"zSensLo": f"{int(sens['scored'].min())}", "zSensHi": f"{int(sens['scored'].max())}",
                   "zSensSlopeLo": neg(sens["slope_8wk"].min()), "zSensSlopeHi": neg(sens["slope_8wk"].max())})
    gen = OUT.parent / "generated"
    (gen / "zone_check_macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    lines = ["\\begin{tabular}{@{}rrrrr@{}}", "\\toprule",
             "Each week & Episode median & Episodes & Scored & Slope \\\\", "\\midrule"]
    for _, r in sens.iterrows():
        cells = [f"{r['week_size'] * 100:.0f}\\%", f"{r['min_size'] * 100:.0f}\\%", f"{int(r['episodes'])}",
                 f"{int(r['scored'])}", f"${r['slope_8wk']:.2f}$"]
        if abs(r["week_size"] - 0.03) < 1e-9 and abs(r["min_size"] - 0.05) < 1e-9:
            cells = [f"\\textbf{{{x}}}" for x in cells]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")
    save(fig, OUT / "fig_zone_checks.pdf")


def tab_episodes():
    """Every episode found, for the appendix."""
    d = pd.read_csv(EP).sort_values(["category", "first_week", "store"])
    m = lambda v: f"{v * 100:.1f}".replace("-", "\u2212")
    lines = ["\\begin{tabular}{@{}llrrrrrrl@{}}", "\\toprule",
             "Category & Zone & Store & Weeks & Price & Controls & 8 weeks & 95\\% interval & Full \\\\", "\\midrule"]
    for _, r in d.iterrows():
        lines.append(" & ".join([CAT[r["category"]], str(r["zone"]), str(r["store"]),
                                 f"{r['first_week']}--{r['last_week']}", m(r["price_log_ratio"]), str(r["n_controls"]),
                                 m(r["estimate_8wk"]), f"{m(r['lo_8wk'])} to {m(r['hi_8wk'])}", m(r["estimate"])])
                     + ("" if r["scored"] else " \\textsuperscript{*}") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT.parent / "generated" / "tab_episodes.tex").write_text("\n".join(lines) + "\n")


def main():
    use_style()
    tab_episodes()
    fig_audit()
    fig_zones()
    if (ROOT / "results" / "paper2" / "sensitivity.csv").exists():
        fig_zone_checks()


if __name__ == "__main__":
    main()
