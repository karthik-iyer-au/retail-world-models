"""Manuscript figures, drawn from the committed run records and the recorded
analyses (figs/rwdata.py). Vector PDF into paper/figs/.

  python paper/figs/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import rwdata as D
from style import (ACCENT, COLOR, EDGE, FILL, FULL, GRID, INK, LABEL, MARKER, MUTED, SINGLE, despine,
                   model_handles, panel_label, save, use_style)

OUT = Path(__file__).resolve().parent
WIN_SHORT = {"mar": "29 Feb to 27 Mar", "apr": "28 Mar to 24 Apr", "practice": "25 Apr to 22 May\n(practice)",
             "test": "23 May to 19 Jun\n(official test)"}


# ------------------------------------------------------------- helpers ---
def dot(ax, y, vals, model, lo, hi, big=34, label_clip=True):
    """Mean as a large marker, each seed as a small open marker, seed range as a thin bar.
    Values past `hi` are drawn at the edge with an arrow and their value."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return
    c, m = COLOR[model], MARKER[model]
    clipped = [min(v, hi) for v in vals]
    mu = np.mean(vals)
    if len(vals) > 1:
        ax.plot([min(clipped), max(clipped)], [y, y], color=c, lw=0.7, solid_capstyle="butt", zorder=2)
        ax.scatter(clipped, [y] * len(vals), s=9, marker=m, facecolor="white", edgecolor=c, lw=0.6, zorder=3)
    ax.scatter([min(mu, hi)], [y], s=big, marker=m, color=c, edgecolor="white", lw=0.7, zorder=4)
    for v in vals:
        if v > hi and label_clip:
            ax.text(hi - (hi - lo) * 0.025, y - 0.13, f"{v:.2f}", ha="right", va="center", fontsize=6.0,
                    color=c, zorder=5)
            ax.annotate("", xy=(hi + (hi - lo) * 0.035, y), xytext=(hi, y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.6, mutation_scale=6), annotation_clip=False)


def row_axes(ax, labels, lo, hi, xlabel, ref=None):
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    for i in range(len(labels) - 1):
        ax.axhline(i + 0.5, color=GRID, lw=0.5, zorder=0)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel(xlabel)
    despine(ax, keep=("bottom",))


def offsets(n, width=0.72):
    return [(i - (n - 1) / 2) * width / max(n - 1, 1) for i in range(n)]


def m5_vals(model, window, key):
    if model == "tfm_plan":
        r = D.TFM_PLAN_M5[window]
        return [r.get(key)] if r.get(key) is not None else []
    return D.m5(model, window)[key]


# -------------------------------------------------------- figure 1 ------
def fig_overview():
    fig = plt.figure(figsize=(FULL, 2.7))
    ax = fig.add_axes([0.0, 0.0, 0.61, 0.90])
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    def box(x, y, w, h, text, face="white", edge=EDGE, color=INK, weight="normal", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)

    def arrow(x0, y0, x1, y1, color=MUTED, lw=0.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0))

    ax.add_patch(FancyBboxPatch((0.05, 0.05), 10.1, 4.15, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=FILL, edgecolor="none"))
    box(0.2, 2.05, 1.8, 1.05, "History of one\nitem in one store:\nsales, price,\ncalendar", fs=6.2)
    arrow(2.0, 2.575, 2.3, 2.575)
    box(2.3, 2.2, 1.05, 0.75, "Encoder", edge=ACCENT, color=ACCENT, weight="bold")
    arrow(3.35, 2.575, 3.65, 2.575)
    xs = [3.65, 5.2, 6.75, 8.75]
    for i, x in enumerate(xs):
        sub = ["t", "t+1", "t+2", "t+h"][i]
        box(x, 2.25, 0.95, 0.65, r"$s_{%s}$" % sub, face=ACCENT if i else "white", edge=ACCENT,
            color="white" if i else ACCENT, fs=8.5)
        if i:
            px = x + 0.475
            box(x - 0.1, 3.4, 1.15, 0.55, r"plan, $%s$" % sub, fs=6.6)
            arrow(px, 3.4, px, 2.92, color=ACCENT)
            box(x + 0.05, 0.95, 0.85, 0.55, r"$\hat y_{%s}$" % sub, fs=7.5)
            arrow(px, 2.25, px, 1.52, color=ACCENT)
    arrow(xs[0] + 0.95, 2.575, xs[1], 2.575, color=ACCENT, lw=1.1)
    arrow(xs[1] + 0.95, 2.575, xs[2], 2.575, color=ACCENT, lw=1.1)
    ax.text((xs[2] + 0.95 + xs[3]) / 2, 2.575, "· · ·", ha="center", va="center", fontsize=9, color=ACCENT)
    ax.text(4.12, 2.05, "each step: state and\nthat period's plan\ngive the next state", fontsize=6.0,
            ha="center", va="top", color=MUTED)
    ax.text(0.3, 1.75, "Plan: price against the\nitem's regular price,\npromotion, display,\nfeature, events",
            fontsize=6.0, ha="left", va="top", color=MUTED)
    ax.text(5.2, 0.55, "Lift model: the same encoder and inputs with no roll-forward;\n"
            "every period is read from $s_t$ and that period's plan.", fontsize=6.0, ha="left", va="center",
            color=INK)
    fig.text(0.0, 0.985, "(a)", fontsize=9.5, weight="bold", va="top")
    fig.text(0.04, 0.985, "The world model carries an item state forward under the plan", fontsize=8,
             weight="bold", va="top")

    # (b) protocol
    bx = fig.add_axes([0.745, 0.13, 0.25, 0.66])
    fig.text(0.635, 0.985, "(b)", fontsize=9.5, weight="bold", va="top")
    fig.text(0.675, 0.985, "Settings fixed before each final window", fontsize=8, weight="bold", va="top")
    bx.set_ylim(-0.75, 2.75)
    bx.set_xlim(0, 10)
    segs = [(1.0, "Mar", "#c9d6e4", INK), (3.25, "Apr", "#c9d6e4", INK), (5.5, "Prac.", "#e5e7eb", INK),
            (7.75, "Test", ACCENT, "white")]
    bx.add_patch(plt.Rectangle((0, 1.78), 0.9, 0.44, color="#f3f4f6", lw=0))
    for x, nm, c, tc in segs:
        bx.add_patch(plt.Rectangle((x, 1.78), 2.15, 0.44, color=c, lw=0))
        bx.text(x + 1.07, 2, nm, ha="center", va="center", fontsize=6.2, color=tc)
    bx.text(3.2, 2.33, "never used to choose", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    bx.text(6.6, 2.33, "chosen on", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    for y in (1, 0):
        bx.add_patch(plt.Rectangle((0, y - 0.22), 2.9, 0.44, color="#f3f4f6", lw=0))
        bx.text(1.45, y, "training", ha="center", va="center", fontsize=6.0, color=MUTED)
        for k, x in enumerate((3.0, 5.0)):
            bx.add_patch(plt.Rectangle((x, y - 0.22), 1.9, 0.44, color="#e5e7eb", lw=0))
            bx.text(x + 0.95, y, f"dev {k + 1}", ha="center", va="center", fontsize=6.0)
        bx.add_patch(plt.Rectangle((7.2, y - 0.22), 2.8, 0.44, color=ACCENT, lw=0))
        bx.text(8.6, y, "final 8 wk", ha="center", va="center", fontsize=6.0, color="white")
    bx.plot([7.1, 7.1], [-0.4, 1.4], color=INK, lw=0.9)
    bx.text(7.1, -0.5, "settings fixed,\nrun once", ha="center", va="top", fontsize=5.8, color=INK)
    bx.set_yticks([2, 1, 0])
    bx.set_yticklabels(["M5,\ndaily", "Dominick's,\nweekly", "Frat,\nweekly"], fontsize=6.6)
    bx.set_xticks([])
    bx.tick_params(axis="y", length=0)
    despine(bx, keep=())
    save(fig, OUT / "fig_overview.pdf")


# -------------------------------------------------------- figure 2 ------
def fig_m5_windows():
    models = ["world", "lift", "lgb", "graph", "tfm", "tfm_plan", "naive"]
    panels = [("ixs", "Item by store", 0.80, 0.95), ("total", "Total", 0.20, 1.00),
              ("overall", "All 12 levels", 0.50, 1.00)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 3.2), sharey=True, gridspec_kw={"wspace": 0.14})
    off = offsets(len(models))
    for ax, (key, title, lo, hi), letter in zip(axes, panels, "abc"):
        for w_i, w in enumerate(D.M5_WINDOWS):
            for m, o in zip(models, off):
                if key == "total" and m == "tfm_plan":
                    continue
                dot(ax, w_i + o, m5_vals(m, w, key), m, lo, hi)
        row_axes(ax, [WIN_SHORT[w] for w in D.M5_WINDOWS], lo, hi, "WRMSSE, lower is better")
        ax.set_title(title)
        panel_label(ax, letter)
    fig.legend(handles=model_handles(models), loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.07),
               handletextpad=0.2, columnspacing=1.0)
    save(fig, OUT / "fig_m5_windows.pdf")


# -------------------------------------------------------- figure 3 ------
def fig_levels():
    models = ["world", "lgb", "graph", "tfm"]
    short = ["Total", "State", "Store", "Category", "Department", "State × category", "State × department",
             "Store × category", "Store × department", "Item", "Item × state", "Item × store"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.0), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.22, "wspace": 0.06})
    x = np.arange(len(D.M5_LEVELS))
    for ax, w, letter in zip(axes.flat, D.M5_WINDOWS, "abcd"):
        for m in models:
            runs = D.m5(m, w)["levels"]
            ys = np.array([[r[lv] for lv in D.M5_LEVELS] for r in runs])
            mu = ys.mean(axis=0)
            if len(runs) > 1:
                ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(x, mu, color=COLOR[m], lw=1.6 if m == "world" else 1.0, marker=MARKER[m], ms=3.2,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_title(D.M5_WINDOW_LABEL[w])
        panel_label(ax, letter, dx=-0.01)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        despine(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=6.4, rotation=45, ha="right", rotation_mode="anchor")
        ax.axvspan(-0.4, 0.4, color=FILL, zorder=0)
        ax.axvspan(10.6, 11.4, color=FILL, zorder=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("WRMSSE, lower is better")
    fig.legend(handles=model_handles(models), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, OUT / "fig_levels.pdf")


# -------------------------------------------------------- figure 4 ------
def fig_level_bias():
    models = ["world", "lift", "lgb", "graph", "tfm", "naive"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.7), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1]})
    off = offsets(len(models), 0.66)
    for w_i, w in enumerate(D.M5_WINDOWS):
        for m, o in zip(models, off):
            dot(a, w_i + o, D.m5(m, w)["fa"], m, 0.85, 1.10)
    row_axes(a, [WIN_SHORT[w] for w in D.M5_WINDOWS], 0.85, 1.10, "Forecast / sales over the window", ref=1.0)
    a.set_title("Total forecast against total sales")
    panel_label(a, "a")
    fig.legend(handles=model_handles(models), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08),
               handletextpad=0.2, columnspacing=1.0)
    # (b) April by March sales band
    y = np.arange(len(D.BANDS))
    for m, o in (("world", -0.14), ("lgb", 0.14)):
        b.barh(y + o, D.BAND_BIAS[m], height=0.26, color=COLOR[m], label=LABEL[m], zorder=3)
    b.axvline(0, color=INK, lw=0.6, zorder=4)
    b.set_xlim(-10, 20)
    b.annotate(f"+{D.BAND_BIAS['world'][0]:.0f}%", xy=(20, -0.14), xytext=(-2, 0), textcoords="offset points",
               ha="right", va="center", fontsize=6.3, color="white", zorder=5)
    b.set_yticks(y)
    b.set_yticklabels([f"{n}  ({u / 1000:.0f}k)" for n, u in zip(D.BANDS, D.BAND_UNITS)])
    b.set_ylim(len(y) - 0.5, -0.5)
    b.tick_params(axis="y", length=0)
    b.grid(axis="x")
    b.set_axisbelow(True)
    b.set_xlabel("Forecast against sales in April, %")
    b.set_title("April, items grouped by their March sales a day")
    despine(b, keep=("bottom",))
    b.legend(loc="lower right", fontsize=6.5, bbox_to_anchor=(1.0, 0.02))
    panel_label(b, "b")
    save(fig, OUT / "fig_level_bias.pdf")


# -------------------------------------------------------- figure 5 ------
FINAL_ROWS = ["frat"] + D.CATS
FINAL_LABEL = {"frat": "Frat", **D.CAT_LABEL}


def final_vals(model, data, key):
    if model == "tfm":
        return [D.TFM_FINAL[data].get(key)]
    if model == "tfm_plan":
        return [D.TFM_PLAN_FINAL[data].get(key)]
    if data not in D.FINAL_CONFIG.get(model, {}):
        return []
    return D.final(model, data)[key]


def fig_finals():
    models = ["world", "lift", "lgb", "tfm", "tfm_plan"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.1), sharey=True, gridspec_kw={"wspace": 0.12})
    off = offsets(len(models), 0.62)
    for ax, (key, title, lo, hi), letter in zip(axes, [("overall", "All levels", 0.35, 1.10),
                                                       ("ixs", "Item by store", 0.50, 0.95)], "ab"):
        for r_i, d in enumerate(FINAL_ROWS):
            for m, o in zip(models, off):
                dot(ax, r_i + o, final_vals(m, d, key), m, lo, hi)
        row_axes(ax, [FINAL_LABEL[d] for d in FINAL_ROWS], lo, hi, "WRMSSE, lower is better")
        ax.set_title(title)
        panel_label(ax, letter)
    fig.legend(handles=model_handles(models), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07))
    save(fig, OUT / "fig_finals.pdf")


# -------------------------------------------------------- figure 6 ------
def fig_plan():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"wspace": 0.62, "width_ratios": [1.05, 1]})
    models = ["world", "lgb", "tfm", "tfm_plan"]
    off = offsets(len(models), 0.6)
    labels = []
    for r_i, c in enumerate(D.CATS):
        share = D.final("lgb", c)["peaks"][0]["promoted"]["share_of_units"]
        labels.append(f"{D.CAT_LABEL[c]} ({share * 100:.0f}%)")
        for m, o in zip(models, off):
            if m in ("world", "lgb"):
                vals = [p["promoted"]["forecast_to_actual"] for p in D.final(m, c)["peaks"]]
            else:
                vals = [(D.TFM_FINAL if m == "tfm" else D.TFM_PLAN_FINAL)[c]["promo_fa"]]
            dot(a, r_i + o, vals, m, 0.2, 1.2)
    row_axes(a, labels, 0.2, 1.2, "Forecast / sales in promoted weeks", ref=1.0)
    a.set_title("Dominick's, promoted weeks")
    panel_label(a, "a")
    fig.legend(handles=model_handles(["world", "lift", "lgb", "graph", "tfm", "tfm_plan"]), loc="upper center",
               ncol=6, bbox_to_anchor=(0.5, 1.08), handletextpad=0.2, columnspacing=1.0)
    # (b) first price cut in a year
    rows = ["world", "lift", "graph", "tfm", "lgb"]
    for r_i, m in enumerate(rows):
        vals = D.PRICE_CUT[m]
        b.scatter(vals, [r_i] * len(vals), s=30 if len(vals) == 1 else 22, marker=MARKER[m], color=COLOR[m],
                  edgecolor="white", lw=0.6, zorder=4)
        if len(vals) > 1:
            b.plot([min(vals), max(vals)], [r_i, r_i], color=COLOR[m], lw=0.7, zorder=3)
    big = max(D.PRICE_CUT["lift"])
    b.annotate(f"{big:,.0f}", xy=(big, 1), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.0,
               color=COLOR["lift"])
    b.axvline(D.PRICE_CUT["sold"], color=INK, lw=0.8, ls=(0, (4, 3)))
    b.text(D.PRICE_CUT["sold"] * 1.12, -0.75, f"sold {D.PRICE_CUT['sold']}", fontsize=6.5, va="bottom")
    b.set_xscale("log")
    b.set_xlim(1, 1e6)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([LABEL[m] for m in rows])
    b.set_ylim(len(rows) - 0.5, -0.9)
    b.tick_params(axis="y", length=0)
    b.grid(axis="x")
    b.set_axisbelow(True)
    b.set_xlabel("Units forecast over the 28 days (log scale), one point per seed")
    b.set_title("First price cut in a year, one item, four stores")
    despine(b, keep=("bottom",))
    panel_label(b, "b")
    save(fig, OUT / "fig_plan.pdf")


# -------------------------------------------------------- figure 7 ------
def fig_frat():
    fig, (a, b) = plt.subplots(2, 1, figsize=(SINGLE, 3.4), sharex=True, gridspec_kw={"hspace": 0.18,
                                                                                         "height_ratios": [1, 1.25]})
    x = np.arange(len(D.FRAT_WEEKS))
    a.bar(x, np.array(D.FRAT_ACTUAL) / 1000, color="#c9d1db", width=0.7, zorder=3)
    a.set_ylabel("Units sold, thousands")
    a.grid(axis="y")
    a.set_axisbelow(True)
    despine(a)
    panel_label(a, "a", dx=-0.14)
    for m in ("world", "lgb"):
        b.plot(x, D.FRAT_BIAS[m], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4 if m == "world" else 1.0,
               markeredgecolor="white", markeredgewidth=0.5, label=LABEL[m], zorder=3)
    b.axhline(0, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    b.set_ylabel("Forecast against sales, %")
    b.grid(axis="y")
    b.set_axisbelow(True)
    b.set_xticks(x)
    b.set_xticklabels(D.FRAT_WEEKS, rotation=0, fontsize=6.4)
    b.set_xlabel("Week starting (2011 to 2012)")
    b.legend(loc="upper left", fontsize=6.5)
    despine(b)
    panel_label(b, "b", dx=-0.14)
    for ax in (a, b):
        ax.axvspan(2.6, 6.4, color=FILL, zorder=0)
    save(fig, OUT / "fig_frat.pdf")


# -------------------------------------------------------- figure 8 ------
def fig_readback():
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    days = [0, 1, 7]
    styles = {"mar": ("o", "-"), "apr": ("s", "--"), "test": ("D", ":")}
    names = {"mar": "29 Feb to 27 Mar", "apr": "28 Mar to 24 Apr", "test": "Official test"}
    x = np.arange(len(days))
    for w, (mk, ls) in styles.items():
        rolled = [D.READBACK[w][d][1] for d in days]
        ctrl = [D.READBACK[w][d][0] for d in days]
        ax.plot(x, rolled, color=ACCENT, marker=mk, ls=ls, lw=1.1, ms=4, markeredgecolor="white",
                markeredgewidth=0.5, label=names[w])
        ax.plot(x, ctrl, color=MUTED, marker=mk, ls=ls, lw=0.8, ms=3, markeredgecolor="white",
                markeredgewidth=0.5)
    ax.text(2.05, 0.05, "state before\nthe window", fontsize=6.3, color=MUTED, va="bottom", ha="right")
    ax.text(2.05, 0.98, "state after rolling\nthrough the plan", fontsize=6.3, color=ACCENT, va="top", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["same day", "1 day back", "7 days back"])
    ax.set_ylim(-0.03, 1.0)
    ax.set_ylabel("$R^2$ reading the discount\nback from the state")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="lower left", fontsize=6.3, bbox_to_anchor=(0.0, 0.12), title="Window", title_fontsize=6.3)
    save(fig, OUT / "fig_readback.pdf")


def main():
    use_style()
    fig_overview()
    fig_m5_windows()
    fig_levels()
    fig_level_bias()
    fig_finals()
    fig_plan()
    fig_frat()
    fig_readback()


if __name__ == "__main__":
    main()
