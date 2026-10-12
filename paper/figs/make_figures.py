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



# ======================================================== new figures ===
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.patches import Polygon, Rectangle

SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f9fc", "#c6d6e8", "#7fa3c8", ACCENT])
DIV = LinearSegmentedColormap.from_list("div", ["#2166ac", "#9ec5e0", "#f7f7f7", "#f4b183", "#b35806"])


def tint(c, a=0.25):
    r, g, b = to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def ink_on(rgb):
    r, g, b = rgb[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


# ---------------------------------------------------------- data cube ---
def fig_cube():
    fig = plt.figure(figsize=(FULL, 2.9))
    ax = fig.add_axes([0.0, 0.0, 0.47, 0.93])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.text(0.0, 0.985, "(a)", fontsize=9.5, weight="bold", va="top")
    fig.text(0.035, 0.985, "One row per item, store and period", fontsize=8, weight="bold", va="top")
    # isometric cube of small cells
    ox, oy, cw, ch, dx, dy = 0.6, 0.8, 0.42, 0.42, 0.21, 0.15
    nx, ny, nz = 8, 7, 5
    def face(pts, fc, ec=EDGE, lw=0.35):
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw))
    for j in range(ny):  # front face: week (x) by item (y)
        for i in range(nx):
            x, y = ox + i * cw, oy + j * ch
            hi = (i == 5 and j == 3)
            face([(x, y), (x + cw, y), (x + cw, y + ch), (x, y + ch)], ACCENT if hi else "white")
    for k in range(nz):  # top face: store depth
        for i in range(nx):
            x, y = ox + i * cw + k * dx, oy + ny * ch + k * dy
            face([(x, y), (x + cw, y), (x + cw + dx, y + dy), (x + dx, y + dy)], FILL)
    for k in range(nz):  # right face
        for j in range(ny):
            x, y = ox + nx * cw + k * dx, oy + j * ch + k * dy
            face([(x, y), (x + dx, y + dy), (x + dx, y + ch + dy), (x, y + ch)], "#e5e7eb")
    ax.text(ox + nx * cw / 2, oy - 0.35, "week (day on M5)", ha="center", va="top", fontsize=6.6, color=MUTED)
    ax.text(ox - 0.15, oy + ny * ch / 2, "item", ha="right", va="center", fontsize=6.6, color=MUTED, rotation=90)
    ax.text(ox + nx * cw + nz * dx + 0.1, oy + ny * ch + nz * dy - 0.05, "store", fontsize=6.6, color=MUTED)
    # call-out of the highlighted cell
    cx, cy = ox + 5.5 * cw, oy + 3.5 * ch
    ax.annotate("", xy=(cx, cy), xytext=(6.3, 4.6), arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    rows = [("units sold", "all"), ("price", "all"), ("promotion", "Dominick's, Frat"),
            ("display, feature", "Frat"), ("price reduction only", "Frat"), ("event, SNAP day", "M5"),
            ("item, store labels", "all")]
    ax.add_patch(FancyBboxPatch((6.3, 1.05), 3.6, 4.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="white", edgecolor=ACCENT, lw=0.8))
    ax.text(6.45, 5.45, "one cell", fontsize=6.8, weight="bold", color=ACCENT, va="top")
    for r, (name, where) in enumerate(rows):
        y = 4.95 - r * 0.53
        ax.text(6.45, y, name, fontsize=6.3, va="top")
        ax.text(9.8, y, where, fontsize=5.5, va="top", ha="right", color=MUTED)
    # (b) the levels and decisions
    bx = fig.add_axes([0.52, 0.02, 0.48, 0.88])
    bx.axis("off")
    bx.set_xlim(0, 10)
    bx.set_ylim(0, 10)
    fig.text(0.51, 0.985, "(b)", fontsize=9.5, weight="bold", va="top")
    fig.text(0.545, 0.985, "The levels a forecast is scored at, and who plans there", fontsize=8,
             weight="bold", va="top")
    levels = [("Chain total", "1", "financial plan"), ("State, store", "3, 10", ""),
              ("Category, department", "3, 7", "sales, units and margin plans"),
              ("Crossings of the above", "9 to 70", ""), ("Item", "3,049", "range and supplier plans"),
              ("Item by state", "9,147", ""), ("Item by store", "30,490", "price, promotion, orders")]
    n = len(levels)
    for i, (name, count, who) in enumerate(levels):
        y = 9.0 - i * 1.25
        w = 1.4 + i * 0.62
        shade = SEQ(0.15 + 0.8 * i / (n - 1))
        bx.add_patch(Rectangle((3.4 - w / 2, y - 0.45), w, 0.9, facecolor=shade, edgecolor="white", lw=0.8))
        bx.text(3.4, y, name, ha="center", va="center", fontsize=6.4, color=ink_on(shade))
        bx.text(3.4 + w / 2 + 0.12, y, count, ha="left", va="center", fontsize=5.9, color=MUTED)
        if who:
            bx.text(9.95, y, who, ha="right", va="center", fontsize=6.5, color=ACCENT)
    bx.text(3.4, 0.05, "series on M5 at each level; 12 levels in all", ha="center", fontsize=5.9, color=MUTED)
    save(fig, OUT / "fig_cube.pdf")


# ----------------------------------------------------- Frat dev forest ---
def fig_frat_dev():
    groups = {"ref": "Reference", "base": "Encoder and readout", "neighbours": "Item-to-item",
              "lever": "Lever step", "inputs": "Extra inputs", "jepa": "Latent (JEPA-style) loss",
              "training": "Training", "world": "World model"}
    order = ["ref", "base", "lever", "neighbours", "inputs", "training", "jepa", "world"]
    rows = []
    for g in order:
        rows += [r for r in D.FRAT_DEV if r[1] == g]
    fig, ax = plt.subplots(figsize=(FULL, 5.4))
    lift, lgb = 0.406, 0.421
    ax.axvspan(0.401, 0.414, color=tint(COLOR["lift"], 0.25), zorder=0, lw=0)
    ax.axvline(lift, color=COLOR["lift"], lw=0.8, zorder=1)
    ax.axvline(lgb, color=COLOR["lgb"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    y = 0
    yt, yl = [], []
    prev = None
    for label, g, mu, lo, hi in rows:
        if g != prev:
            if prev is not None:
                y += 0.6
            ax.text(0.388, y - 0.05, groups[g], fontsize=6.8, weight="bold", color=ACCENT, ha="left", va="center")
            y += 0.75
            prev = g
        c = COLOR["lgb"] if "LightGBM" in label else COLOR["world"] if g == "world" else (
            COLOR["lift"] if label.startswith("Lift model") else INK)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=c, lw=0.9, zorder=2)
            for v in (lo, hi):
                ax.plot([v, v], [y - 0.18, y + 0.18], color=c, lw=0.9, zorder=2)
        better = mu < lift - 1e-9
        ax.scatter([mu], [y], s=26, marker="D" if g == "world" else "o", color=c,
                   edgecolor="white", lw=0.6, zorder=3)
        ax.text(0.519, y, f"{mu:.3f}", fontsize=6.4, va="center", ha="right",
                color=INK, weight="bold" if better else "normal")
        yt.append(y)
        yl.append(label)
        y += 1
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.6)
    ax.set_ylim(y, -0.6)
    ax.set_xlim(0.388, 0.52)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("WRMSSE on the two Frat development windows, all six levels (mean of five seeds, bar: seed range)")
    ax.text(lift, -0.45, "lift model", color=COLOR["lift"], fontsize=6.4, ha="right", va="bottom")
    ax.text(lgb + 0.001, -0.45, "LightGBM", color=COLOR["lgb"], fontsize=6.4, ha="left", va="bottom")
    despine(ax, keep=("bottom",))
    save(fig, OUT / "fig_frat_dev.pdf")


# -------------------------------------------- 28 Dominick's categories ---
def fig_dom28():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.6), gridspec_kw={"wspace": 0.28, "width_ratios": [1.15, 1]})
    lim = (0.28, 1.6)
    a.plot(lim, lim, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    a.fill_between(lim, lim, [lim[1]] * 2, color=tint(COLOR["lgb"], 0.08), lw=0)
    a.text(0.33, 1.52, "LightGBM more accurate", fontsize=6.5, color=COLOR["lgb"])
    a.text(1.2, 0.36, "world model more\naccurate", fontsize=6.5, color=COLOR["world"])
    for code, (lgb, lift, w26, w8) in D.DOM28.items():
        moved = abs(w8 - w26) > 0.005
        col = COLOR["world"] if w8 < lgb else MUTED
        if moved:
            a.annotate("", xy=(lgb, w8), xytext=(lgb, w26),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=1.5))
        a.scatter([lgb], [w26], s=9, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
        a.scatter([lgb], [w8], s=16, color=col, zorder=4, edgecolor="white", lw=0.4)
        if code in ("sna", "bat", "che", "sdr", "ana", "fre", "bjc", "tna", "cso", "cig"):
            a.text(lgb + 0.012, w8, D.DOM28_NAME[code], fontsize=5.6, va="center", color=INK)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_aspect("equal")
    a.set_xlabel("LightGBM, WRMSSE")
    a.set_ylabel("World model, WRMSSE")
    a.set_title("Each category: 26-week level (open) to 8-week level (filled)")
    a.grid(True)
    a.set_axisbelow(True)
    despine(a)
    panel_label(a, "a")
    # (b) wins per model, two settings
    counts = []
    for idx, nm in ((2, "26-week level"), (3, "8-week level")):
        w = sum(1 for v in D.DOM28.values() if v[idx] < v[0])
        counts.append((nm, w, 28 - w))
    mean_lgb = np.mean([v[0] for v in D.DOM28.values()])
    means = [np.mean([v[i] for v in D.DOM28.values()]) for i in (2, 3)]
    for i, (nm, w, l) in enumerate(counts):
        b.barh(i, w, color=COLOR["world"], height=0.5)
        b.barh(i, l, left=w, color=tint(COLOR["lgb"], 0.75), height=0.5)
        b.text(w / 2, i, f"{w}", color="white", ha="center", va="center", fontsize=7, weight="bold")
        b.text(w + l / 2, i, f"{l}", color=INK, ha="center", va="center", fontsize=7)
        b.text(28.5, i, f"mean {means[i]:.3f}", va="center", fontsize=6.5, color=MUTED)
    b.text(0, 1.75, f"Mean WRMSSE over the 28 categories: LightGBM {mean_lgb:.3f}", va="center", fontsize=6.5,
           color=COLOR["lgb"])
    b.set_yticks([0, 1])
    b.set_yticklabels([c[0] for c in counts])
    b.set_ylim(2.0, -0.6)
    b.set_xlim(0, 28)
    b.set_xticks([0, 7, 14, 21, 28])
    b.set_xlabel("Categories of 28 where the model is more accurate")
    b.set_title("World model (blue) against LightGBM (orange)")
    b.tick_params(axis="y", length=0)
    despine(b, keep=("bottom",))
    panel_label(b, "b")
    save(fig, OUT / "fig_dom28.pdf")


# --------------------------------------------------- peaks and dips map ---
def fig_peaks():
    cats = D.CATS
    variants = ["lgb", "world", "feedback", "weight"]
    kinds = ["Promoted", "After", "Ordinary"]
    M = np.array([[D.PEAKS[c][v][k] for c in cats for k in range(3)] for v in variants])
    fig = plt.figure(figsize=(FULL, 2.55))
    ax = fig.add_axes([0.17, 0.12, 0.74, 0.62])
    norm = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.45)
    ax.imshow(M, cmap=DIV, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color=ink_on(DIV(norm(v))))
    for k in range(1, len(cats)):
        ax.axvline(3 * k - 0.5, color="white", lw=2.5)
    ax.axhline(0.5, color="white", lw=1.5)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([D.PEAK_VARIANT[v] for v in variants], fontsize=6.8)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(kinds * len(cats), fontsize=5.6, rotation=90)
    for k, c in enumerate(cats):
        ax.text(3 * k + 1, -0.75, D.CAT_LABEL[c], ha="center", va="bottom", fontsize=7, weight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.62])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax)
    cb.set_ticks([0.6, 0.8, 1.0, 1.2, 1.4])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    cax.set_title("forecast\n/ sales", fontsize=6, loc="left")
    save(fig, OUT / "fig_peaks.pdf")


# -------------------------------------------- development ladder ---
def fig_ladder():
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 4.8), gridspec_kw={"hspace": 0.95, "wspace": 0.38})
    steps = ["8-week level", "+ switching", "+ normalised", "+ average of 3", "+ 2× steps"]
    for ax, c, letter in zip(axes.flat[:6], D.CATS, "abcdef"):
        vals, lgb = D.LADDER[c]
        x = np.arange(len(vals))
        ax.axhline(lgb, color=COLOR["lgb"], lw=0.9, ls=(0, (4, 3)))
        ax.plot(x, vals, color=COLOR["world"], lw=1.2, marker="o", ms=3.5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)
        for xi, (lo, hi) in zip((0, 3), D.LADDER_RANGE[c]):
            ax.plot([xi, xi], [lo, min(hi, ax.get_ylim()[1] if False else hi)], color=COLOR["world"],
                    lw=2.4, alpha=0.25, solid_capstyle="round", zorder=2)
        ax.scatter([3], [vals[3]], s=70, facecolor="none", edgecolor=INK, lw=0.9, zorder=4)
        top = max(max(vals), lgb, D.LADDER_RANGE[c][0][1])
        top = min(top, 1.25)
        bot = min(min(vals), lgb, D.LADDER_RANGE[c][1][0])
        pad = (top - bot) * 0.12
        ax.set_ylim(bot - pad, top + pad)
        if D.LADDER_RANGE[c][1][1] > top + pad:
            ax.annotate(f"{D.LADDER_RANGE[c][1][1]:.2f}", xy=(3, top + pad), xytext=(6, -2),
                        textcoords="offset points", fontsize=5.6, color=COLOR["world"], va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(steps, fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_title(D.CAT_LABEL[c], fontsize=7.8)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        despine(ax)
        panel_label(ax, letter, dx=-0.05, dy=1.08)
    axes.flat[0].set_ylabel("WRMSSE, development")
    axes.flat[4].set_ylabel("WRMSSE, development")
    # (g) learned share of lift that is new sales
    g = axes.flat[6]
    for i, c in enumerate(D.CATS):
        lo, hi = D.SWITCH_SHARE[c]
        g.plot([lo, hi], [i, i], color=ACCENT, lw=3.2, solid_capstyle="round", alpha=0.8)
    g.set_yticks(range(len(D.CATS)))
    g.set_yticklabels([D.CAT_LABEL[c] for c in D.CATS], fontsize=6)
    g.set_ylim(len(D.CATS) - 0.4, -0.6)
    g.set_xlim(-0.05, 1.05)
    g.set_xticks([0, 0.5, 1])
    g.set_xticklabels(["0", "½", "all"], fontsize=6)
    g.set_title("Share of lift that is\nnew to the category", fontsize=7.8)
    g.tick_params(axis="y", length=0)
    g.grid(axis="x")
    g.set_axisbelow(True)
    despine(g, keep=("bottom",))
    panel_label(g, "g", dx=-0.05, dy=1.08)
    # (h) key
    h = axes.flat[7]
    h.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR["world"], marker="o", ms=4, lw=1.2, label="World model, version"),
               Line2D([], [], color=COLOR["lgb"], lw=0.9, ls=(0, (4, 3)), label="LightGBM"),
               Line2D([], [], color=COLOR["world"], lw=3, alpha=0.3, label="seed range"),
               Line2D([], [], marker="o", ms=7, markerfacecolor="none", markeredgecolor=INK, ls="none",
                      label="version taken to\nthe final test")]
    h.legend(handles=handles, loc="center left", fontsize=6.3, frameon=False)
    save(fig, OUT / "fig_ladder.pdf")


# ---------------------------------------------------- M5 heatmap ---------
def fig_m5_heat():
    models = ["world", "lift", "lgb", "graph", "tfm", "tfm_plan", "naive"]
    grains = [("ixs", "Item by store"), ("total", "Total"), ("overall", "All 12")]
    cols = [(w, k) for w in D.M5_WINDOWS for k, _ in grains]
    V = np.full((len(models), len(cols)), np.nan)
    for i, m in enumerate(models):
        for j, (w, k) in enumerate(cols):
            xs = m5_vals(m, w, k)
            if xs:
                V[i, j] = np.mean(xs)
    best = np.nanmin(V, axis=0)
    R = (V / best - 1) * 100
    fig = plt.figure(figsize=(FULL, 2.95))
    ax = fig.add_axes([0.16, 0.06, 0.76, 0.68])
    norm = plt.Normalize(0, 60)
    shown = np.where(np.isnan(R), np.nan, np.clip(R, 0, 60))
    ax.imshow(shown, cmap=SEQ, norm=norm, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if np.isnan(V[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.6, color=MUTED)
                continue
            rgb = SEQ(norm(min(R[i, j], 60)))
            is_best = abs(V[i, j] - best[j]) < 1e-12
            ax.text(j, i, f"{V[i, j]:.3f}", ha="center", va="center", fontsize=5.9,
                    color=ink_on(rgb), weight="bold" if is_best else "normal")
            if is_best:
                ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, edgecolor=INK, lw=1.0))
    for k in range(1, 4):
        ax.axvline(3 * k - 0.5, color="white", lw=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABEL[m] for m in models], fontsize=6.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([g for _, g in grains] * 4, fontsize=5.8)
    ax.xaxis.tick_top()
    for k, w in enumerate(D.M5_WINDOWS):
        ax.text(3 * k + 1, -1.35, D.M5_WINDOW_LABEL[w], ha="center", va="bottom", fontsize=6.9, weight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.935, 0.06, 0.012, 0.68])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([0, 20, 40, 60])
    cb.set_ticklabels(["best", "+20%", "+40%", "≥60%"])
    save(fig, OUT / "fig_m5_heat.pdf")


# ----------------------------------------------- winner by level ---------
def fig_winners():
    models = ["world", "lift", "lgb", "graph", "tfm"]
    short = ["Total", "State", "Store", "Category", "Department", "State × category", "State × department",
             "Store × category", "Store × department", "Item", "Item × state", "Item × store"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.5), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.25]})
    L = len(D.M5_LEVELS)
    for j, w in enumerate(D.M5_WINDOWS):
        means = {}
        for m in models:
            runs = D.m5(m, w)["levels"]
            if m == "lift":  # median over seeds: one April seed failed
                means[m] = {lv: float(np.median([r[lv] for r in runs])) for lv in D.M5_LEVELS}
            else:
                means[m] = {lv: float(np.mean([r[lv] for r in runs])) for lv in D.M5_LEVELS}
        for i, lv in enumerate(D.M5_LEVELS):
            order = sorted(models, key=lambda m: means[m][lv])
            win, second = order[0], order[1]
            gap = (means[second][lv] / means[win][lv] - 1) * 100
            a.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, facecolor=tint(COLOR[win], 0.55), lw=0))
            a.scatter([j - 0.3], [i], marker=MARKER[win], s=14, color=COLOR[win], edgecolor="white", lw=0.4)
            a.text(j + 0.08, i, f"{gap:.0f}%", ha="center", va="center", fontsize=5.8, color=INK)
    a.set_xlim(-0.5, 3.5)
    a.set_ylim(L - 0.5, -0.5)
    a.set_xticks(range(4))
    a.set_xticklabels(["Mar", "Apr", "Practice", "Test"], fontsize=6.8)
    a.xaxis.tick_top()
    a.set_yticks(range(L))
    a.set_yticklabels(short, fontsize=6.4)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("Most accurate model at each level\n(lead over the next model)", pad=22)
    panel_label(a, "a", dx=-0.05, dy=1.13)
    a.legend(handles=model_handles(models, ms=4), loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
             fontsize=6.0, handletextpad=0.1, columnspacing=0.6)
    # (b) plan-aware models relative to TimesFM by level: median over the four windows, band min to max
    x = np.arange(L)
    for m in ("world", "lgb", "graph"):
        R = []
        for w in D.M5_WINDOWS:
            mm = np.array([np.mean([r[lv] for r in D.m5(m, w)["levels"]]) for lv in D.M5_LEVELS])
            tt = np.array([D.m5("tfm", w)["levels"][0][lv] for lv in D.M5_LEVELS])
            R.append((mm / tt - 1) * 100)
        R = np.array(R)
        b.fill_between(x, R.min(axis=0), R.max(axis=0), color=COLOR[m], alpha=0.14, lw=0)
        b.plot(x, np.median(R, axis=0), color=COLOR[m], lw=1.5 if m == "world" else 1.0, marker=MARKER[m], ms=3.5,
               markeredgecolor="white", markeredgewidth=0.5, label=LABEL[m])
    b.axhline(0, color=INK, lw=0.7)
    b.axhspan(0, 12, color=tint(COLOR["tfm"], 0.12), zorder=0, lw=0)
    b.text(0.1, 6, "TimesFM more accurate", fontsize=6.3, color=COLOR["tfm"], va="center")
    b.text(0.1, -66, "plan-aware model more accurate", fontsize=6.3, color=MUTED, va="center")
    b.set_ylim(-70, 12)
    b.set_xticks(x)
    b.set_xticklabels(short, fontsize=6.1, rotation=45, ha="right", rotation_mode="anchor")
    b.set_ylabel("Error against TimesFM, %")
    b.grid(axis="y")
    b.set_axisbelow(True)
    despine(b)
    b.legend(loc="lower right", fontsize=6.2, bbox_to_anchor=(1.0, 0.12))
    b.set_title("Error against TimesFM, level by level\n(median of four months; band: range)")
    panel_label(b, "b", dx=-0.05, dy=1.05)
    save(fig, OUT / "fig_winners.pdf")


# ---------------------------------------- level against total error ----
def fig_level_scatter():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw={"wspace": 1.05, "width_ratios": [1, 1.05]})
    xs_all, ys_all = [], []
    for m in ["world", "lift", "lgb", "graph", "tfm", "naive"]:
        for w in D.M5_WINDOWS:
            s = D.m5(m, w)
            for fa, tot in zip(s["fa"], s["total"]):
                if fa is None:
                    continue
                x = (fa - 1) * 100
                a.scatter([x], [tot], s=16, marker=MARKER[m], color=COLOR[m], edgecolor="white", lw=0.4, zorder=3)
                if tot < 2:  # leaves out the one failed lift seed (April, total 2.35)
                    xs_all.append(abs(x))
                    ys_all.append(tot)
    r = np.corrcoef(xs_all, ys_all)[0, 1]
    a.axvline(0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    a.set_xlim(-12, 12)
    a.set_ylim(0.15, 1.0)
    a.set_xlabel("Total forecast against total sales, % (0 is right)")
    a.set_ylabel("Error at the chain total, WRMSSE")
    a.grid(True)
    a.set_axisbelow(True)
    despine(a)
    (OUT.parent / "generated").mkdir(exist_ok=True)
    (OUT.parent / "generated" / "fig_values.tex").write_text(
        f"\\newcommand{{\\biasErrCorr}}{{{r:.2f}}}\n\\newcommand{{\\biasErrN}}{{{len(xs_all)}}}\n")
    a.set_title("The total error follows the level")
    panel_label(a, "a")
    a.legend(handles=model_handles(["world", "lift", "lgb", "graph", "tfm", "naive"], ms=4), fontsize=5.9,
             loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1, labelspacing=0.3)
    # (b) April by March band: dumbbell
    y = np.arange(len(D.BANDS))
    for i in y:
        wv, lv = D.BAND_BIAS["world"][i], D.BAND_BIAS["lgb"][i]
        b.plot([min(wv, 15), lv], [i, i], color=GRID, lw=2.2, zorder=1)
        b.scatter([lv], [i], s=26, marker=MARKER["lgb"], color=COLOR["lgb"], zorder=3, edgecolor="white", lw=0.5)
        b.scatter([min(wv, 15)], [i], s=26, marker=MARKER["world"], color=COLOR["world"], zorder=3,
                  edgecolor="white", lw=0.5)
        if wv > 15:
            b.annotate(f"+{wv:.0f}%", xy=(15, i), xytext=(4, 0), textcoords="offset points", fontsize=6,
                       color=COLOR["world"], va="center")
        b.text(24, i, f"{D.BAND_CLOSER[i]:.1f}%", fontsize=6.3, va="center", ha="right")
    b.text(24, -0.55, "world model\ncloser", fontsize=5.8, ha="right", va="bottom", color=MUTED)
    b.axvline(0, color=INK, lw=0.6)
    b.set_xlim(-8, 24)
    b.set_xticks([-5, 0, 5, 10, 15])
    b.set_yticks(y)
    b.set_yticklabels([f"{n} ({u / 1000:.0f}k)" for n, u in zip(D.BANDS, D.BAND_UNITS)], fontsize=6.5)
    b.set_ylim(len(y) - 0.5, -0.6)
    b.tick_params(axis="y", length=0)
    b.grid(axis="x")
    b.set_axisbelow(True)
    b.set_xlabel("April forecast against April sales, %")
    b.set_title("April, items grouped by March sales a day", pad=18)
    despine(b, keep=("bottom",))
    panel_label(b, "b", dy=1.1)
    save(fig, OUT / "fig_level_scatter.pdf")


# --------------------------------------------- finals small multiples ---
def fig_finals_grid():
    models = ["world", "lift", "lgb", "tfm", "tfm_plan"]
    data = ["frat"] + D.CATS
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.9), gridspec_kw={"hspace": 0.55, "wspace": 0.42})
    for ax, d, letter in zip(axes.flat, data, "abcdefg"):
        pts = []
        for m in models:
            xs, ys = final_vals(m, d, "ixs"), final_vals(m, d, "overall")
            xs = [v for v in xs if v is not None]
            ys = [v for v in ys if v is not None]
            if not xs:
                continue
            mx, my = np.mean(xs), np.mean(ys)
            pts.append((mx, my))
            if len(xs) > 1:
                ax.plot([min(xs), max(xs)], [my, my], color=COLOR[m], lw=0.7)
                ax.plot([mx, mx], [min(ys), max(ys)], color=COLOR[m], lw=0.7)
            ax.scatter([mx], [my], s=34, marker=MARKER[m], color=COLOR[m], edgecolor="white", lw=0.6, zorder=3)
        xs_, ys_ = zip(*pts)
        px, py = (max(xs_) - min(xs_)) * 0.18 + 0.01, (max(ys_) - min(ys_)) * 0.18 + 0.02
        ax.set_xlim(min(xs_) - px, max(xs_) + px)
        ax.set_ylim(min(ys_) - py, max(ys_) + py)
        ax.set_title(FINAL_LABEL[d], fontsize=7.8)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6)
        despine(ax)
        panel_label(ax, letter, dx=-0.05, dy=1.06)
        ax.text(0.97, 0.03, "↙ better", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("All levels", fontsize=7)
    for ax in axes[1, :3]:
        ax.set_xlabel("Item by store", fontsize=7)
    axes[0, 3].set_xlabel("Item by store", fontsize=7)
    last = axes.flat[7]
    last.axis("off")
    last.legend(handles=model_handles(models), loc="center", fontsize=6.6, frameon=False)
    save(fig, OUT / "fig_finals_grid.pdf")


# ------------------------------------------------- plan: arrows + ratio ---
def fig_plan_arrows():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"wspace": 0.75, "width_ratios": [1.1, 1]})
    labels = []
    for i, c in enumerate(D.CATS):
        share = D.final("lgb", c)["peaks"][0]["promoted"]["share_of_units"]
        labels.append(f"{D.CAT_LABEL[c]} ({share * 100:.0f}%)")
        t0, t1 = D.TFM_FINAL[c]["promo_fa"], D.TFM_PLAN_FINAL[c]["promo_fa"]
        a.add_patch(Rectangle((min(t0, t1), i - 0.28), abs(t1 - t0), 0.56, color=tint(COLOR["tfm_plan"], 0.25), lw=0))
        a.annotate("", xy=(t1, i), xytext=(t0, i),
                   arrowprops=dict(arrowstyle="-|>", color=COLOR["tfm_plan"], lw=1.3, mutation_scale=8,
                                   shrinkA=2, shrinkB=2))
        a.scatter([t0], [i], s=22, marker=MARKER["tfm"], color=COLOR["tfm"], zorder=4, edgecolor="white", lw=0.5)
        for m in ("world", "lgb"):
            v = np.mean([p["promoted"]["forecast_to_actual"] for p in D.final(m, c)["peaks"]])
            a.plot([v, v], [i - 0.3, i + 0.3], color=COLOR[m], lw=1.6, zorder=3)
    a.axvline(1.0, color=INK, lw=0.7, ls=(0, (4, 3)))
    a.set_yticks(range(len(D.CATS)))
    a.set_yticklabels(labels, fontsize=6.6)
    a.set_ylim(len(D.CATS) - 0.5, -0.6)
    a.set_xlim(0.2, 1.2)
    a.tick_params(axis="y", length=0)
    a.grid(axis="x")
    a.set_axisbelow(True)
    despine(a, keep=("bottom",))
    a.set_xlabel("Forecast / sales in promoted weeks (1.0 is right)")
    a.set_title("TimesFM: history only to history plus plan")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=MARKER["tfm"], color=COLOR["tfm"], ls="none", ms=4, label="TimesFM"),
          Line2D([], [], color=COLOR["tfm_plan"], lw=1.3, label="with the plan (arrow)"),
          Line2D([], [], color=COLOR["world"], lw=1.6, label="World model"),
          Line2D([], [], color=COLOR["lgb"], lw=1.6, label="LightGBM")]
    a.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=4, fontsize=6.0,
             handletextpad=0.3, columnspacing=0.8)
    panel_label(a, "a")
    # (b) how many times over or under, per seed
    rows = ["world", "lift", "graph", "tfm", "lgb"]
    sold = D.PRICE_CUT["sold"]
    yt, yl = [], []
    y = 0
    for m in rows:
        vals = D.PRICE_CUT[m]
        for k, v in enumerate(vals):
            r = np.log10(v / sold)
            b.plot([0, r], [y, y], color=COLOR[m], lw=1.4, solid_capstyle="butt")
            b.scatter([r], [y], s=18, marker=MARKER[m], color=COLOR[m], zorder=3, edgecolor="white", lw=0.4)
            lbl = (f"{v / sold:,.0f}×" if v / sold >= 10 else f"{v / sold:.1f}×") if v >= sold else f"1/{sold / v:.0f}"
            b.text(r + (0.08 if r >= 0 else -0.08), y, lbl, fontsize=5.6, va="center",
                   ha="left" if r >= 0 else "right", color=INK)
            y += 1
        yt.append(y - (len(vals) + 1) / 2)
        yl.append(LABEL[m])
        y += 0.6
    b.axvline(0, color=INK, lw=0.8)
    b.set_yticks(yt)
    b.set_yticklabels(yl, fontsize=6.6)
    b.set_ylim(y - 0.4, -0.8)
    b.set_xlim(-2.4, 3.9)
    b.set_xticks([-2, -1, 0, 1, 2, 3])
    b.set_xticklabels(["1/100", "1/10", "sold", "10×", "100×", "1000×"], fontsize=6.3)
    b.tick_params(axis="y", length=0)
    b.grid(axis="x")
    b.set_axisbelow(True)
    despine(b, keep=("bottom",))
    b.set_xlabel(f"Forecast against the {sold} units sold, one bar per seed")
    b.set_title("First price cut in a year")
    panel_label(b, "b")
    save(fig, OUT / "fig_plan_arrows.pdf")


# --------------------------------------------- Frat overlay and slope ---
def fig_frat_story():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw={"wspace": 0.32, "width_ratios": [1.7, 1]})
    x = np.arange(len(D.FRAT_WEEKS))
    act = np.array(D.FRAT_ACTUAL) / 1000
    a.bar(x, act, color="#d5dbe3", width=0.72, zorder=2, label="Units sold")
    for m in ("world", "lgb"):
        f = act * (1 + np.array(D.FRAT_BIAS[m]) / 100)
        a.plot(x, f, color=COLOR[m], marker=MARKER[m], ms=4.5, lw=1.4 if m == "world" else 1.0,
               markeredgecolor="white", markeredgewidth=0.5, zorder=3, label=f"{LABEL[m]} forecast")
    for i, v in enumerate(D.FRAT_BIAS["world"]):
        if abs(v) >= 15:
            a.text(i, act[i] * (1 + v / 100) + (3 if v > 0 else -3), f"{v:+.0f}%", ha="center",
                   va="bottom" if v > 0 else "top", fontsize=6.0, color=COLOR["world"])
    a.axvspan(2.6, 6.4, color=FILL, zorder=0)
    a.text(4.5, 99, "holiday weeks", ha="center", fontsize=6.3, color=MUTED)
    a.set_xticks(x)
    a.set_xticklabels(D.FRAT_WEEKS, fontsize=6.3)
    a.set_ylim(0, 104)
    a.set_ylabel("Units, thousands")
    a.set_xlabel("Week starting (2011 to 2012)")
    a.legend(loc="upper left", fontsize=6.2, ncol=1)
    a.grid(axis="y")
    a.set_axisbelow(True)
    despine(a)
    a.set_title("Final 8 weeks: total right, shape missed")
    panel_label(a, "a")
    # (b) development to final
    dev = {"world": 0.408, "lift": 0.406, "lgb": 0.421}
    fin = {m: np.mean(D.final(m, "frat")["overall"]) for m in dev}
    for m in dev:
        b.plot([0, 1], [dev[m], fin[m]], color=COLOR[m], lw=1.3, marker=MARKER[m], ms=5,
               markeredgecolor="white", markeredgewidth=0.5)
        b.text(1.06, fin[m], f"{LABEL[m]} {fin[m]:.3f}", va="center", fontsize=6.3, color=COLOR[m])
    b.text(-0.14, 0.414, "0.406 to\n0.421", ha="right", va="center", fontsize=6.3, color=MUTED)
    b.set_xticks([0, 1])
    b.set_xticklabels(["Development\n(Jul to Nov)", "Final 8 weeks\n(Nov to Jan)"], fontsize=6.6)
    b.set_xlim(-0.5, 1.9)
    b.set_ylim(0.38, 0.74)
    b.set_ylabel("WRMSSE, all levels")
    b.grid(axis="y")
    b.set_axisbelow(True)
    despine(b)
    b.set_title("Level in development, apart at the holidays")
    panel_label(b, "b")
    save(fig, OUT / "fig_frat_story.pdf")


# ------------------------------------------------- the study at a glance ---
def fig_solution():
    fig = plt.figure(figsize=(FULL, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face="white", edge=EDGE, tcol=ACCENT, fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=face, edgecolor=edge, lw=0.8))
        ax.text(x + 0.15, y + h - 0.2, title, fontsize=7.0, weight="bold", color=tcol, va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.75 - i * 0.44, ln, fontsize=fs, va="top", color=INK)

    def arrow(x0, y0, x1, y1, color=ACCENT):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=color,
                                     lw=1.0, shrinkA=0, shrinkB=0))

    top, h = 5.0, 4.6
    cols = [
        (0.15, 2.95, "1  Public data", ["M5: daily, 30,490", "item-stores", "Dominick's: weekly,", "6 categories (28 in", "development)", "Frat: weekly, 3,909", "item-stores"]),
        (3.3, 3.0, "2  One table", ["item × store × period:", "units, price, and the", "plan as known future", "inputs (promotion,", "display, feature,", "events, SNAP days)"]),
        (6.5, 3.15, "3  Seven models", ["see the plan: world", "model, lift model,", "LightGBM, graph", "model, TimesFM", "with the plan", "history only: TimesFM,", "seasonal naive"]),
        (9.85, 3.05, "4  Development", ["windows before the", "final weeks only;", "each change screened", "against fixed rules;", "settings written", "down, then locked"]),
        (13.3, 3.05, "5  Final tests", ["run once, 3 seeds;", "M5: 4 months, two", "never used to choose;", "Dominick's and Frat:", "last 8 weeks; every", "losing model reported"]),
        (16.55, 3.3, "6  Scored by grain", ["every level, from item", "by store to chain total", "(12 on M5, 6 on the", "others); promoted,", "after-promotion and", "ordinary weeks"]),
    ]
    for i, (x, w, t, lines) in enumerate(cols):
        face = FILL if i in (3, 4) else "white"
        box(x, top, w, h, t, lines, face=face)
        if i < len(cols) - 1:
            nx = cols[i + 1][0]
            arrow(x + w + 0.02, top + h / 2, nx - 0.02, top + h / 2)
    ax.plot([13.12, 13.12], [top - 0.1, top + h + 0.1], color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(13.12, top + h + 0.15, "settings locked", fontsize=5.8, ha="center", color=INK)
    # answers by decision
    ax.text(0.15, 4.45, "What the grain decides", fontsize=7.6, weight="bold", color=INK, va="top")
    ans = [
        ("Store orders", "item by store", "history read by a pretrained\nmodel is enough when prices\nrarely move (M5)", COLOR["tfm"]),
        ("Promotion and price", "item by week, changed plan", "the plan is needed; history\nalone sees 29% to 63% of\npromoted-week sales", ACCENT),
        ("Category plans", "category or chain total", "get the month's level right;\nplan-aware models 21% to 47%\nlower error than TimesFM", ACCENT),
        ("Choosing a model", "any", "score several months: one\nM5 month reversed the\nranking", MUTED),
    ]
    w = 4.75
    for i, (dec, grain, res, c) in enumerate(ans):
        x = 0.15 + i * (w + 0.2)
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 3.65, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=tint(c, 0.12), edgecolor=c, lw=0.9))
        ax.text(x + 0.18, 3.62, dec, fontsize=7.2, weight="bold", color=INK, va="top")
        ax.text(x + 0.18, 3.0, grain, fontsize=6.2, color=MUTED, va="top", style="italic")
        ax.text(x + 0.18, 2.35, res, fontsize=6.4, color=INK, va="top", linespacing=1.35)
    save(fig, OUT / "fig_solution.pdf")


def main():
    use_style()
    fig_solution()
    fig_overview()
    fig_cube()
    fig_readback()
    fig_frat_dev()
    fig_dom28()
    fig_peaks()
    fig_ladder()
    fig_m5_heat()
    fig_winners()
    fig_level_scatter()
    fig_finals_grid()
    fig_plan_arrows()
    fig_frat_story()


if __name__ == "__main__":
    main()
