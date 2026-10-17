"""Revision figures, drawn from the committed result tables (results/paper1_revision/tables)
and the experiment configs (window layout). Vector PDF into paper/figs/.

  python paper/figs/make_revision_figures.py

Writes fig_study.pdf, fig_rolling.pdf, fig_ablation.pdf, fig_mechanism.pdf and fig_decisions.pdf.
Every value drawn is read from a table or a config at run time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_revision as R  # noqa: E402  (rolling_accuracy, NAME, KEYS)
from style import (ACCENT, COLOR, EDGE, FILL, FULL, GRID, INK, LABEL, MARKER, MUTED, despine,  # noqa: E402
                   panel_label, save, use_style)

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results" / "paper1_revision" / "tables"
CFG = ROOT / "configs" / "experiments"
BOTTOM = R.BOTTOM
DOM = ["dominicks_che", "dominicks_sna", "dominicks_sdr", "dominicks_bjc", "dominicks_ana", "dominicks_fre"]
KEYS = DOM + ["frat", "m5", "favorita"]
NAME = R.NAME
# result-table model name -> style key
SK = {"world_model": "world", "lift": "lift", "lightgbm": "lgb", "timesfm_plan": "tfm_plan", "timesfm_mean": "tfm",
      "chronos": "chronos", "tft": "tft", "timestral_plan": "timestral", "ensemble": "ensemble",
      "seasonal_naive": "naive"}
SMALL = 6.3  # smallest annotation text at final size

# window roles (not models): rolling dark, development light grey with hatch, final mid blue-grey
ROLE_FACE = {"rolling": ACCENT, "development": "#e3e6ea", "final": "#a9bed3"}
ROLE_TEXT = {"rolling": "white", "development": INK, "final": INK}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


def mk(ax, x, y, model, s=16, filled=True, z=4, **kw):
    c = COLOR[model]
    if model == "ensemble":  # the star reads small at the same area; enlarge it and keep its edge solid
        ax.scatter(x, y, s=s * 1.7 if s < 20 else s * 1.25, marker=MARKER[model], facecolor=c if filled else "white",
                   edgecolor=c, linewidths=0.4 if filled else 0.7, zorder=z, **kw)
        return
    ax.scatter(x, y, s=s, marker=MARKER[model], facecolor=c if filled else "white", edgecolor="white" if filled else c,
               linewidths=0.5 if filled else 0.7, zorder=z, **kw)


def handle(model, ms=4.6, filled=True, label=None):
    c = COLOR[model]
    return Line2D([], [], color=c, marker=MARKER[model], ls="none", ms=ms * (1.5 if model == "ensemble" else 1),
                  markerfacecolor=c if filled else "white",
                  markeredgecolor=c if (model == "ensemble" or not filled) else "white",
                  markeredgewidth=0.5 if filled else 0.8, label=label or LABEL[model])


def rows_axis(ax, ypos, labels, show=True):
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels if show else [])
    ax.tick_params(axis="y", length=0)
    lo, hi = min(ypos), max(ypos)
    ax.set_ylim(hi + 0.6, lo - 0.6)
    for a, b in zip(ypos[:-1], ypos[1:]):
        ax.axhline((a + b) / 2, color=GRID, lw=0.45, zorder=0)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))


def row_positions(n_dom=6, gap=0.45):
    """Dominick's categories 0..5, then the other data sets after a small gap."""
    ys = list(range(n_dom))
    for i in range(3):
        ys.append(n_dom + gap + i)
    return ys


def dom_bracket(ax, ys, x=-0.6):
    """A thin bracket with 'Dominick's' to the left of the tick labels of the first six rows."""
    t = ax.get_yaxis_transform()
    ax.plot([x, x], [ys[0] - 0.35, ys[5] + 0.35], transform=t, color=MUTED, lw=0.6, clip_on=False)
    ax.text(x - 0.03, (ys[0] + ys[5]) / 2, "Dominick's", transform=t, rotation=90, ha="right", va="center",
            fontsize=7.0, color=MUTED, clip_on=False)


def repel(vals, gap):
    """Spread sorted label positions at least `gap` apart, each run of crowded labels centred on its data."""
    groups = [[v] for v in vals]
    merged = True
    while merged:
        merged = False
        for j in range(1, len(groups)):
            top_prev = np.mean(groups[j - 1]) + (len(groups[j - 1]) - 1) * gap / 2
            low_this = np.mean(groups[j]) - (len(groups[j]) - 1) * gap / 2
            if low_this - top_prev < gap:
                groups[j - 1] += groups.pop(j)
                merged = True
                break
    pos = []
    for g in groups:
        c = np.mean(g)
        pos += [c + (j - (len(g) - 1) / 2) * gap for j in range(len(g))]
    return pos


def dodge(n, width):
    return [(i - (n - 1) / 2) * width / max(n - 1, 1) for i in range(n)]


# ===================================================== 1. study layout ===
def cfg_eval(name):
    c = yaml.safe_load((CFG / f"{name}.yaml").read_text())
    return c["evaluation"], bool((c["dataset"].get("params") or {}).get("include_test"))


def spans(name, extra=0):
    """Windows of a config as (start, end) in periods before the end of the data (end > start >= 0)."""
    ev, _ = cfg_eval(name)
    h, n, step = ev["horizon"], ev["n_origins"], ev.get("step") or ev["horizon"]
    off = (ev.get("holdout_periods") or 0) + extra
    return [(off + k * step, off + k * step + h) for k in range(n)]


def layout():
    """Window layout per data source, read from the configs; checks that the six categories agree."""
    out = {}
    lays = []
    for c in ["che", "sna", "sdr", "bjc", "ana", "fre"]:
        lays.append((spans(f"rev_dominicks_{c}_rolling_world_model"), spans(f"dominicks_{c}_dev_lightgbm"),
                     spans(f"rev_dominicks_{c}_finalwindow_lift")))
    assert all(l == lays[0] for l in lays), "Dominick's categories differ in window layout"
    h = cfg_eval("rev_dominicks_che_rolling_world_model")[0]["horizon"]
    out["dominicks"] = dict(h=h, unit="weeks", rolling=lays[0][0], development=lays[0][1], final=lays[0][2])
    h = cfg_eval("rev_frat_rolling_world_model")[0]["horizon"]
    out["frat"] = dict(h=h, unit="weeks", rolling=spans("rev_frat_rolling_world_model"),
                       development=spans("frat_dev_lightgbm"), final=spans("frat_final_lightgbm"))
    # M5: configs without include_test end before the official test month, whose length is the
    # horizon of the test config; the windows of those configs sit that much further from the end.
    ev_t, inc = cfg_eval("m5_test_lightgbm")
    assert inc
    test_len = ev_t["horizon"]

    def m5(name):
        return spans(name, extra=0 if cfg_eval(name)[1] else test_len)
    out["m5"] = dict(h=test_len, unit="days", rolling=m5("rev_m5_rolling_world_model"),
                     development=m5("m5_validation_lightgbm"),
                     final=m5("m5_confirm_mar_lightgbm") + m5("m5_confirm_apr_lightgbm") + m5("m5_test_lightgbm"),
                     test=m5("m5_test_lightgbm"))
    ev_f, _ = cfg_eval("rev_favorita_rolling_world_model")
    out["favorita"] = dict(h=ev_f["horizon"], unit="days", rolling=spans("rev_favorita_rolling_world_model"),
                           development=[], final=[])
    return out


def glyph_world(ax, x, y, c, s=0.11):
    """Three small state circles carried forward by arrows."""
    for i in range(3):
        ax.add_patch(plt.Circle((x + i * 0.42, y), s, facecolor=c if i else "white", edgecolor=c, lw=0.8))
        if i:
            ax.add_patch(FancyArrowPatch((x + (i - 1) * 0.42 + s, y), (x + i * 0.42 - s, y), arrowstyle="-|>",
                                         mutation_scale=4.5, color=c, lw=0.7, shrinkA=0, shrinkB=0))


def glyph_tree(ax, x, y, c):
    """A tiny two-level decision tree."""
    root = (x + 0.42, y + 0.17)
    kids = [(x + 0.2, y - 0.02), (x + 0.64, y - 0.02)]
    leaves = [(x + 0.08, y - 0.2), (x + 0.32, y - 0.2), (x + 0.52, y - 0.2), (x + 0.76, y - 0.2)]
    for k in kids:
        ax.plot([root[0], k[0]], [root[1], k[1]], color=c, lw=0.8)
    for i, l in enumerate(leaves):
        k = kids[i // 2]
        ax.plot([k[0], l[0]], [k[1], l[1]], color=c, lw=0.8)
    for p in [root] + kids:
        ax.add_patch(plt.Circle(p, 0.05, color=c))
    for p in leaves:
        ax.add_patch(Rectangle((p[0] - 0.045, p[1] - 0.045), 0.09, 0.09, color=c))


def glyph_series(ax, x, y, c):
    """A short sales line with the plan as small bars underneath."""
    xs = x + np.linspace(0, 0.84, 9)
    ys = y + 0.07 + 0.09 * np.array([0.0, 0.5, 0.2, 0.9, 0.4, 0.3, 1.0, 0.6, 0.7])
    ax.plot(xs, ys, color=c, lw=0.9, solid_capstyle="round")
    for xi, on in zip(xs, [0, 0, 1, 0, 0, 1, 1, 0, 0]):
        if on:
            ax.add_patch(Rectangle((xi - 0.035, y - 0.2), 0.07, 0.13, color=c, lw=0))
    ax.plot([x - 0.03, x + 0.87], [y - 0.2, y - 0.2], color=c, lw=0.5)


def fig_study():
    lay = layout()
    fig = plt.figure(figsize=(FULL, 2.6))
    # (a) timeline in windows before the end of the data (each row in its own window length)
    ax = fig.add_axes([0.13, 0.19, 0.43, 0.66])
    rows = [("dominicks", "Dominick's\n(six categories)"), ("frat", "Frat"), ("m5", "M5"), ("favorita", "Favorita")]
    bh = 0.5
    xmax = max(e / lay[k]["h"] for k, _ in rows for role in ("rolling", "development", "final")
               for _, e in lay[k][role])

    def above(i, sp, h, text, color=INK):
        x0, x1 = -max(e for _, e in sp) / h, -min(s for s, _ in sp) / h
        ax.text((x0 + x1) / 2, i - bh / 2 - 0.06, text, ha="center", va="bottom", fontsize=SMALL, color=color)

    for i, (k, lab) in enumerate(rows):
        L = lay[k]
        h = L["h"]

        def bar(s, e, role, hatch=None):
            x0, x1 = -e / h, -s / h
            ax.add_patch(Rectangle((x0 + 0.05, i - bh / 2), (x1 - x0) - 0.1, bh, facecolor=ROLE_FACE[role],
                                   edgecolor=EDGE if role == "development" else "none", lw=0.4, hatch=hatch,
                                   zorder=2))
        for s, e in L["rolling"]:
            bar(s, e, "rolling")
        dev = set(L["development"])
        for s, e in L["final"]:
            if (s, e) not in dev:
                bar(s, e, "final")
        for s, e in L["development"]:
            bar(s, e, "development", hatch="//////")
        if i == 0:  # direct labels for the three kinds of window, above the first row
            above(i, L["rolling"], h, f"rolling ({len(L['rolling'])})")
            above(i, L["development"], h, f"development ({len(L['development'])})")
            above(i, L["final"], h, "final")
        else:
            above(i, L["rolling"], h, f"rolling ({len(L['rolling'])})", color=MUTED)
        if k == "m5":
            fin = [w for w in L["final"] if w not in dev and w not in L["test"]]
            above(i, fin, h, "final", color=MUTED)
            above(i, L["development"], h, "practice", color=MUTED)
            above(i, L["test"], h, "test", color=MUTED)
        if not L["development"] and not L["final"]:
            x0 = -max(e for _, e in L["rolling"]) / h
            ax.text(x0 - 0.2, i, "no development or\nfinal window", ha="right", va="center", fontsize=SMALL,
                    color=MUTED, style="italic", linespacing=0.95)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows], fontsize=7.2)
    for i, (k, _) in enumerate(rows):
        ax.text(1.01, i, f"{lay[k]['h']} {lay[k]['unit']}\neach", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=SMALL, color=MUTED, linespacing=0.95)
    ax.set_ylim(len(rows) - 0.55, -0.8)
    ax.set_xlim(-np.ceil(xmax) - 0.15, 0.1)
    ticks = np.arange(-np.ceil(xmax), 0.5, 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(-t)}" for t in ticks])
    ax.set_xlabel("Windows before the end of the data")
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    despine(ax, keep=("bottom",))
    ax.grid(False)
    fig.text(0.005, 0.985, "(a)", fontsize=9.5, fontweight="bold", ha="left", va="top", color=INK)

    # (b) the ensemble, drawn in equal-aspect units
    left, width = 0.675, 0.32
    bx = fig.add_axes([left, 0.0, width, 1.0])
    W = 10.0
    Hh = W * 2.6 / (width * FULL)
    bx.set_xlim(0, W)
    bx.set_ylim(0, Hh)
    bx.axis("off")
    members = [("world", "World model", glyph_world), ("lgb", "LightGBM", glyph_tree),
               ("tfm_plan", "TimesFM + plan", glyph_series)]
    top = Hh - 1.6
    ys = [top, top - 1.75, top - 3.5]
    bw, bhh, x0 = 5.5, 1.3, 0.1
    mx, my = 7.15, ys[1]
    fig.text(left - 0.035, 0.985, "(b)", fontsize=9.5, fontweight="bold", ha="left", va="top", color=INK)
    for (m, name, g), y in zip(members, ys):
        bx.add_patch(FancyBboxPatch((x0, y - bhh / 2), bw, bhh, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    facecolor="white", edgecolor=COLOR[m], lw=0.8))
        if m == "world":
            g(bx, x0 + 0.4, y, COLOR[m], s=0.13)
        else:
            g(bx, x0 + 0.33, y + (0.02 if m == "lgb" else 0.0), COLOR[m])
        bx.text(x0 + 1.6, y, name, ha="left", va="center", fontsize=7.0, color=INK)
        end = (mx - 0.8, my + (y - my) * 0.22)
        bx.add_patch(FancyArrowPatch((x0 + bw, y), end, arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.7,
                                     shrinkA=1, shrinkB=1))
        bx.text(x0 + bw + 0.12, y + 0.12, "1/3", ha="left", va="bottom", fontsize=SMALL, color=MUTED)
    bx.add_patch(FancyBboxPatch((mx - 0.75, my - 0.5), 1.5, 1.0, boxstyle="round,pad=0.02,rounding_size=0.45",
                                facecolor=FILL, edgecolor=ACCENT, lw=0.8))
    bx.text(mx, my, "mean", ha="center", va="center", fontsize=6.8, color=ACCENT)
    bx.add_patch(FancyArrowPatch((mx + 0.75, my), (mx + 1.45, my), arrowstyle="-|>", mutation_scale=6,
                                 color=ACCENT, lw=0.8, shrinkA=0, shrinkB=0))
    bx.scatter([mx + 1.95], [my], marker=MARKER["ensemble"], s=90, color=COLOR["ensemble"], edgecolor="white",
               lw=0.5, zorder=4)
    bx.text(mx + 1.95, my - 0.6, "Ensemble", ha="center", va="top", fontsize=7.0, color=ACCENT)
    # the study A variants, from the ablation table
    ab = pd.concat([read("ablation_paired.csv"), read("favorita_ablation_paired.csv")])
    var = set(ab.a)
    drop = sorted(v for v in var if v.startswith("without_"))
    fit = sorted(v for v in var if v.startswith("fitted_"))
    lines = []
    if drop:
        lines.append(f"drop one member ({len(drop)} variants)")
    if "lift_for_world_model" in var:
        lines.append("lift model in place of the world model")
    if fit:
        lines.append(f"weights fitted ({len(fit)} variants)")
    vb_top = ys[2] - bhh / 2 - 0.55
    vb_h = 0.95 + 0.82 * len(lines)
    bx.add_patch(FancyBboxPatch((x0, vb_top - vb_h), W - 0.25, vb_h, boxstyle="round,pad=0.02,rounding_size=0.3",
                                facecolor=FILL, edgecolor="none"))
    bx.text(x0 + 0.35, vb_top - 0.3, "Variants tested", ha="left", va="top", fontsize=7.0, color=INK,
            fontweight="bold")
    for j, t in enumerate(lines):
        bx.text(x0 + 0.35, vb_top - 1.05 - j * 0.82, "–  " + t, ha="left", va="top", fontsize=SMALL + 0.3,
                color=INK)
    save(fig, OUT / "fig_study.pdf")
    return lay


# ================================================= 2. rolling accuracy ===
ROLL_MODELS = ["world_model", "lift", "lightgbm", "timesfm_plan", "timesfm_mean", "chronos", "tft", "timestral_plan"]


def fig_rolling():
    acc = R.rolling_accuracy()
    fig = plt.figure(figsize=(FULL, 3.0))
    axes = [fig.add_axes([0.155, 0.24, 0.355, 0.68]), fig.add_axes([0.600, 0.24, 0.355, 0.68])]
    ys = row_positions()
    lo, hi = 0.6, 3.0
    clipped = {}
    for p, (ax, level) in enumerate(zip(axes, [BOTTOM, "total"])):
        w = R.wide(acc, level)
        rel = w.div(w["ensemble"], axis=0)
        offs = dodge(len(ROLL_MODELS), 0.66)
        for yi, k in zip(ys, KEYS):
            over = []
            for m, o in zip(ROLL_MODELS, offs):
                v = rel.loc[k, m] if (k in rel.index and m in rel.columns) else np.nan
                if np.isnan(v):
                    continue
                if v > hi:
                    over.append((m, v, yi + o))
                    clipped[(level, k, m)] = v
                mk(ax, min(v, hi), yi + o, SK[m], s=15)
            for j, (m, v, yy) in enumerate(over):
                ax.annotate("", xy=(hi * 1.07, yy), xytext=(hi, yy), annotation_clip=False,
                            arrowprops=dict(arrowstyle="-|>", color=COLOR[SK[m]], lw=0.6, mutation_scale=5))
            if over:
                txt = ", ".join(f"{v:.1f}" if v < 10 else f"{v:.0f}" for _, v, _ in over)
                ax.text(hi * 1.09, np.mean([yy for _, _, yy in over]), txt, ha="left", va="center", fontsize=SMALL, color=COLOR[SK[over[0][0]]]
                        if len(over) == 1 else INK, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ticks = [0.7, 1, 1.5, 2, 3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        rows_axis(ax, ys, [NAME[k] for k in KEYS], show=(p == 0))
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel("WRMSSE relative to the ensemble")
        panel_label(ax, "ab"[p], dx=-0.02 if p else -0.33, dy=1.02)
        ax.text(0.0, 1.02, "Item by store" if level == BOTTOM else "Total", transform=ax.transAxes, fontsize=8,
                ha="left", va="bottom", color=INK)
    dom_bracket(axes[0], ys, x=-0.40)
    hs = [handle(SK[m]) for m in ROLL_MODELS]
    hs = [hs[i] for i in (0, 4, 1, 5, 2, 6, 3, 7)]  # legend fills by column; this reads row by row
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.55, -0.005), ncol=4, fontsize=7.0,
               handletextpad=0.3, columnspacing=1.4, borderaxespad=0.1)
    save(fig, OUT / "fig_rolling.pdf")
    return clipped


# ======================================================== 3. ablation ===
ABL = [("without_world_model", "Ensemble without the world model", INK, "D"),
       ("lift_for_world_model", "Lift model in place of the world model", COLOR["lift"], MARKER["lift"])]


def fig_ablation():
    ab = pd.concat([read("ablation_paired.csv"), read("favorita_ablation_paired.csv")])
    ab = ab[ab.b == "ensemble"]
    fig = plt.figure(figsize=(FULL, 2.6))
    axes = [fig.add_axes([0.135, 0.17, 0.235, 0.69]), fig.add_axes([0.40, 0.17, 0.235, 0.69])]
    ys = row_positions()
    lims = {BOTTOM: (-0.03, 0.03), "total": (-0.35, 0.25)}
    clipped = {}
    for p, (ax, level) in enumerate(zip(axes, [BOTTOM, "total"])):
        lo, hi = lims[level]
        offs = dodge(len(ABL), 0.38)
        for yi, k in zip(ys, KEYS):
            for (a, _, c, m), o in zip(ABL, offs):
                r = ab[(ab.key == k) & (ab.a == a) & (ab.level == level)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = yi + o
                excl = r.lo > 0 or r.hi < 0
                inside = lo <= r.mean_diff <= hi
                if inside:  # a mean off the axis is written out with its interval instead (below)
                    ax.plot([max(r.lo, lo), min(r.hi, hi)], [y, y], color=c, lw=0.7, zorder=2,
                            solid_capstyle="butt")
                    for end in (r.lo, r.hi):
                        if lo <= end <= hi:
                            ax.plot([end, end], [y - 0.09, y + 0.09], color=c, lw=0.6, zorder=2)
                if r.mean_diff < lo or r.mean_diff > hi or r.lo < lo or r.hi > hi:
                    clipped[(level, k, a)] = (r.mean_diff, r.lo, r.hi)
                if lo <= r.mean_diff <= hi:
                    ax.scatter([r.mean_diff], [y], s=15, marker=m, facecolor=c if excl else "white",
                               edgecolor=c if not excl else "white", lw=0.7 if not excl else 0.4, zorder=4)
                else:  # mean off the axis: an arrow at the edge and the value (interval low end in brackets)
                    span = hi - lo
                    yt = y + (0.07 if o > 0 else -0.07)  # a little apart from the other variant's label
                    ax.annotate("", xy=(lo, yt), xytext=(lo + 0.13 * span, yt), annotation_clip=False,
                                arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6), zorder=5)
                    ax.text(lo + 0.15 * span, yt, f"{r.mean_diff:.2f} ({r.lo:.2f} to {r.hi:.2f})".replace("-", "−"),
                            ha="left", va="center", fontsize=SMALL, color=c, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.03", fc="white", ec="none"))
        ax.set_xlim(lo, hi)
        rows_axis(ax, ys, [NAME[k] for k in KEYS], show=(p == 0))
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        ax.set_xlabel("Variant minus ensemble (WRMSSE)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        panel_label(ax, "ab"[p], dx=-0.02 if p else -0.44, dy=1.02)
        ax.text(0.0, 1.02, "Item by store" if level == BOTTOM else "Total", transform=ax.transAxes, fontsize=8,
                ha="left", va="bottom")
    dom_bracket(axes[0], ys, x=-0.53)
    hs = [Line2D([], [], color=c, marker=m, ls="-", lw=0.7, ms=4.2, markeredgecolor="white", markeredgewidth=0.4,
                 label=lab) for _, lab, c, m in ABL]
    hs.append(Line2D([], [], color=MUTED, marker="o", ls="none", ms=4.2, markerfacecolor="white",
                     markeredgecolor=MUTED, label="open: interval includes zero"))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.03, 1.0), ncol=3, fontsize=7.0, handletextpad=0.4,
               columnspacing=1.3, borderaxespad=0.1)

    # (c) weight sweep, item by store
    cx = fig.add_axes([0.745, 0.17, 0.215, 0.69])
    sw = read("ablation_sweep.csv")
    sw = sw[sw.level == BOTTOM].pivot(index="w_world_model", columns="key", values="wrmsse")
    # The sweep grid contains the equal-weight point itself (w_world_model = 0.33333), so the reference is
    # the nearest grid point, which is exactly 1/3 to five decimals; no interpolation.
    w0 = sw.index[np.argmin(np.abs(sw.index.to_numpy() - 1 / 3))]
    rel = sw / sw.loc[w0]
    ylo, yhi = 0.96, 1.14
    labelled = {"dominicks_bjc": None, "dominicks_che": None, "m5": None, "frat": None}
    for k in rel.columns:
        v = rel[k]
        hl = k in labelled
        cx.plot(v.index, v.values, color=INK if hl else MUTED, lw=0.9 if hl else 0.6, alpha=1 if hl else 0.75,
                zorder=3 if hl else 2, clip_on=True)
    # direct labels at w = 1 (or where a line leaves the axis)
    lab_pos = []
    for k in labelled:
        v = rel[k]
        if v.iloc[-1] <= yhi:
            lab_pos.append([k, v.iloc[-1]])
    lab_pos.sort(key=lambda t: t[1])
    for lp, y in zip(lab_pos, repel([t[1] for t in lab_pos], 0.0085)):
        lp[1] = y
    for k, y in lab_pos:
        cx.text(1.02, y, NAME[k], ha="left", va="center", fontsize=SMALL, color=INK, clip_on=False)
    if "dominicks_che" in rel and rel["dominicks_che"].max() > yhi:
        v = rel["dominicks_che"]
        xin = np.interp(yhi, v.loc[w0:].values, v.loc[w0:].index)
        cx.text(xin + 0.03, yhi - 0.004, f"Cheese: {v.min():.2f} at 0,\n{v.iloc[-1]:.2f} at 1", ha="left",
                va="top", fontsize=SMALL, color=INK, linespacing=0.95)
    cx.axvline(float(w0), color=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    cx.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
    cx.set_xlim(0, 1)
    cx.set_ylim(ylo, yhi)
    cx.set_xticks([0, 1 / 3, 2 / 3, 1])
    cx.set_xticklabels(["0", "1/3", "2/3", "1"])
    cx.set_xlabel("Weight on the world model")
    cx.set_ylabel("WRMSSE relative to equal weights")
    cx.set_yticks(np.arange(0.98, yhi + 1e-9, 0.04))
    cx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    cx.grid(axis="y")
    cx.set_axisbelow(True)
    despine(cx)
    panel_label(cx, "c", dx=-0.2, dy=1.02)
    cx.text(0.0, 1.02, "Item by store", transform=cx.transAxes, fontsize=8, ha="left", va="bottom")
    save(fig, OUT / "fig_ablation.pdf")
    return clipped, rel


# ======================================================= 4. mechanism ===
def fig_mechanism():
    me = read("mechanism.csv")
    H = [f"persist_h{h}" for h in range(1, 9)]
    sets = [("dominicks_che", "rolling"), ("dominicks_sna", "rolling"), ("dominicks_sdr", "rolling"),
            ("frat", "final")]
    fig = plt.figure(figsize=(FULL, 2.5))
    a1 = fig.add_axes([0.065, 0.19, 0.04, 0.70])
    a2 = fig.add_axes([0.128, 0.19, 0.17, 0.70])
    # (a) period 1 on its own scale, periods 2 to 8 on a finer one (broken axis)
    for k, ws in sets:
        for m in ("world_model", "lift"):
            r = me[(me.data == k) & (me.windows_set == ws) & (me.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            mk(a1, [0], [100 * r[H[0]]], SK[m], s=14)
    a1.set_xlim(-0.6, 0.6)
    a1.set_xticks([0])
    a1.set_xticklabels(["1"])
    a1.set_ylim(0, None)
    a1.grid(axis="y")
    a1.set_axisbelow(True)
    despine(a1)
    a1.set_ylabel("Change in forecast (%)")
    labels = []
    for k, ws in sets:
        r = me[(me.data == k) & (me.windows_set == ws) & (me.model == "world_model")]
        if r.empty:
            continue
        r = r.iloc[0]
        v = 100 * r[H[1:]].to_numpy(dtype=float)
        a2.plot(range(2, 9), v, color=COLOR["world"], lw=0.9, marker=MARKER["world"], ms=2.6,
                markeredgecolor="white", markeredgewidth=0.3, zorder=3)
        nm = "Frat (final)" if k == "frat" else NAME[k]
        if ws == "rolling" and int(r.windows) < 3:
            nm += f" ({int(r.windows)} windows)"
        labels.append([nm, v[-1]])
    lift = me[me.model == "lift"].dropna(subset=H)
    lv = 100 * lift[H[1:]].to_numpy(dtype=float)
    assert np.all(lv == 0), "lift model persistence expected zero after period 1"
    a2.plot(range(2, 9), lv[0], color=COLOR["lift"], lw=1.2, marker=MARKER["lift"], ms=3, markeredgecolor="white",
            markeredgewidth=0.3, zorder=4)
    labels.append([f"Lift model (all {len(lift)})", 0.0])
    # simple label repel
    labels.sort(key=lambda t: t[1])
    for lab, y in zip(labels, repel([t[1] for t in labels], 3.4)):
        lab[1] = y
    for nm, y in labels:
        a2.text(8.25, y, nm, ha="left", va="center", fontsize=SMALL,
                color=COLOR["lift"] if nm.startswith("Lift") else COLOR["world"], clip_on=False)
    a2.set_xlim(1.6, 8.2)
    a2.set_xticks(range(2, 9))
    a2.spines["bottom"].set_bounds(2, 8)
    a2.grid(axis="y")
    a2.set_axisbelow(True)
    despine(a2)
    a2.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    # shared x label across the broken axis
    fig.text((0.065 + 0.298) / 2, 0.035, "Period after the promotion", ha="center", fontsize=8)
    for axx, side in ((a1, "right"), (a2, "left")):
        d = 0.012
        x = 1 if side == "right" else 0
        axx.plot([x - d * (3 if side == "right" else 0.7), x + d * (3 if side == "right" else 0.7)],
                 [-0.02, 0.02], transform=axx.transAxes, color=EDGE, lw=0.6, clip_on=False)
    panel_label(a1, "a", dx=-0.6, dy=1.04)

    # (b) history occlusion, change in total WRMSSE
    # M5 (last 28 and 56 days kept) is left out of this panel: its relative changes (world model +615% and
    # +219%, lift model +255% and +42%) are on a different scale and the daily history is not comparable in length.
    bx = fig.add_axes([0.535, 0.19, 0.155, 0.70])
    pt = read("perturbation.csv")
    pt = pt[(pt.operation == "occlude") & pt.input.str.startswith("history_last_")]
    pt = pt.assign(keep=pt.input.str.replace("history_last_", "").astype(int),
                   data=np.where(pt.key.str.startswith("dominicks"), "dominicks", pt.key))
    occ = pt[pt.data.isin(["dominicks", "frat"])].groupby(["data", "model", "keep"]).total_change.mean()
    keeps = sorted(occ.index.get_level_values("keep").unique())
    xs = np.arange(len(keeps))
    for m in ("world_model", "lift"):
        for dset, ls, filled in (("dominicks", "-", True), ("frat", (0, (3, 2)), False)):
            # total_change is relative (perturbed / true - 1); drawn in percent
            v = [100 * occ.get((dset, m, kk), np.nan) for kk in keeps]
            bx.plot(xs, v, color=COLOR[SK[m]], lw=0.9, ls=ls, zorder=2)
            mk(bx, xs, v, SK[m], s=15, filled=filled)
    bx.set_xticks(xs)
    bx.set_xticklabels([str(k) for k in keeps])
    bx.set_xlim(-0.35, len(keeps) - 0.65)
    bx.set_xlabel("Weeks of history kept")
    bx.set_ylabel("Change in total WRMSSE (%)")
    bx.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    bx.grid(axis="y")
    bx.set_axisbelow(True)
    despine(bx)
    hs = [handle("world", label="World model"), handle("lift", label="Lift model"),
          Line2D([], [], color=MUTED, lw=0.9, ls="-", marker="o", ms=3.4, markeredgecolor="white",
                 markeredgewidth=0.4, label="Dominick's"),
          Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), marker="o", ms=3.4, markerfacecolor="white",
                 label="Frat")]
    bx.legend(handles=hs, loc="upper right", bbox_to_anchor=(1.06, 1.03), fontsize=SMALL, handlelength=1.5,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3, numpoints=1)
    panel_label(bx, "b", dx=-0.3, dy=1.04)

    # (c) Timestral's tails: sums over actual sales per window
    cx = fig.add_axes([0.785, 0.19, 0.205, 0.70])
    tl = read("studyF_tails.csv")
    groups = [("dominicks_ana", "Analgesics"), ("dominicks_sdr", "Soft drinks")]
    series = [("q99_ratio", "99% quantile", "^", True), ("forecast_ratio", "point", MARKER["timestral"], True),
              ("median_ratio", "median", "o", False)]
    x0 = 0
    centers = []
    tick_x, tick_l = [], []
    for k, nm in groups:
        g = tl[(tl.key == k) & (tl.model == "timestral_plan")].sort_values("origin")
        gm = tl[(tl.key == k) & (tl.model == "timestral_mean")].sort_values("origin")
        x = x0 + np.arange(len(g))
        for fld, _, mrk, filled in series:
            cx.plot(x, g[fld], color=COLOR["timestral"], lw=0.7, zorder=2)
            cx.scatter(x, g[fld], s=13, marker=mrk, facecolor=COLOR["timestral"] if filled else "white",
                       edgecolor="white" if filled else COLOR["timestral"], lw=0.4 if filled else 0.7, zorder=3)
        cx.plot(x, gm.set_index("origin").reindex(g.origin).forecast_ratio, color=MUTED, lw=0.8, ls="-",
                marker="x", ms=2.6, mew=0.7, zorder=2)
        centers.append((x.mean(), nm))
        tick_x += list(x)
        tick_l += [str(o + 1) for o in g.origin]
        x0 += len(g) + 1.2
    cx.set_yscale("log")
    cx.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    cx.set_xticks(tick_x)
    cx.set_xticklabels(tick_l, fontsize=SMALL)
    for xc, nm in centers:
        cx.text(xc, -0.17, nm, transform=cx.get_xaxis_transform(), ha="center", va="top", fontsize=7.0)
    cx.set_ylabel("Sum over actual sales")
    cx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    cx.grid(axis="y")
    cx.set_axisbelow(True)
    despine(cx)
    hs = [Line2D([], [], color=COLOR["timestral"], marker=mrk, lw=0.7, ms=3.6,
                 markerfacecolor=COLOR["timestral"] if f else "white",
                 markeredgecolor="white" if f else COLOR["timestral"], markeredgewidth=0.4 if f else 0.7, label=lab)
          for _, lab, mrk, f in series]
    hs.append(Line2D([], [], color=MUTED, marker="x", lw=0.8, ms=3, mew=0.7, label="point, history only"))
    cx.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=SMALL,
              handlelength=1.4, handletextpad=0.35, columnspacing=0.8, borderaxespad=0.0, labelspacing=0.25)
    panel_label(cx, "c", dx=-0.25, dy=1.04)
    save(fig, OUT / "fig_mechanism.pdf")
    return occ


# ======================================================= 5. decisions ===
def fig_decisions():
    fig = plt.figure(figsize=(FULL, 2.5))
    # (a) category planning cost relative to seasonal naive
    ax = fig.add_axes([0.125, 0.25, 0.235, 0.67])
    d = read("decisions.csv")
    d = d[(d.set == "rolling") & (d.decision == "category_planning")]
    models = ["world_model", "ensemble", "lightgbm", "lift"]
    keys = [k for k in KEYS if k in set(d.key)]
    ys = row_positions(n_dom=6)[: len(keys)]
    offs = dodge(len(models), 0.5)
    for yi, k in zip(ys, keys):
        q = d[d.key == k].set_index("model").ratio_to_naive
        for m, o in zip(models, offs):
            if m in q:
                mk(ax, [q[m]], [yi + o], SK[m], s=26 if m == "ensemble" else 15)
    rows_axis(ax, ys, [NAME[k] for k in keys])
    ax.set_xlim(0, max(1.05, float(d[d.model.isin(models)].ratio_to_naive.max()) * 1.05))
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("Planning cost relative to seasonal naive")
    dom_bracket(ax, ys, x=-0.52)
    panel_label(ax, "a", dx=-0.46, dy=1.02)

    # (b) interval coverage at the total, before and after the fix, mean over data sets in both tables
    bx = fig.add_axes([0.445, 0.25, 0.22, 0.67])
    new = read("intervals_fixed.csv")
    old = read("intervals.csv")
    cm = ["world_model", "lightgbm", "ensemble"]
    levels = [50, 80, 95]

    def pooled(t):
        t = t[(t.set == "rolling") & (t.level == "total") & t.model.isin(cm)]
        return t
    pn, po = pooled(new), pooled(old)
    common = sorted(set(pn.key) & set(po.key))  # Favorita has no "before" row, so it is left out of both
    pn, po = pn[pn.key.isin(common)], po[po.key.isin(common)]
    offs = dodge(len(cm), 0.42)
    cov = {}
    for li, lvl in enumerate(levels):
        col = f"coverage_{lvl}"
        bx.plot([li - 0.32, li + 0.32], [lvl / 100] * 2, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for m, o in zip(cm, offs):
            b = po[po.model == m][col].mean()
            a = pn[pn.model == m][col].mean()
            cov[(m, lvl)] = (b, a)
            x = li + o
            bx.annotate("", xy=(x, a), xytext=(x, b), arrowprops=dict(arrowstyle="-|>", color=COLOR[SK[m]], lw=0.6,
                                                                       mutation_scale=5, shrinkA=2.5, shrinkB=2.5))
            mk(bx, [x], [b], SK[m], s=22 if m == "ensemble" else 14, filled=False)
            mk(bx, [x], [a], SK[m], s=26 if m == "ensemble" else 15, filled=True)
    bx.set_xticks(range(len(levels)))
    bx.set_xticklabels([f"{l}%" for l in levels])
    bx.set_xlim(-0.5, len(levels) - 0.5)
    bx.set_ylim(0, 1.0)
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    bx.set_xlabel("Interval level")
    bx.set_ylabel("Coverage at the total")
    bx.grid(axis="y")
    bx.set_axisbelow(True)
    despine(bx)
    bx.legend(handles=[Line2D([], [], color=MUTED, marker="o", ls="none", ms=3.8, markerfacecolor="white",
                              markeredgecolor=MUTED, label="before fix"),
                       Line2D([], [], color=MUTED, marker="o", ls="none", ms=3.8, markeredgecolor="white",
                              label="after fix"),
                       Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="nominal")],
              loc="upper left", fontsize=SMALL, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.3,
              handlelength=1.4)
    panel_label(bx, "b", dx=-0.24, dy=1.02)

    # (c) ranking check: Spearman correlation of forecast change and actual change, item by store
    cx = fig.add_axes([0.775, 0.25, 0.215, 0.67])
    rk = pd.read_csv(T / "studyF_ranks.csv", header=[0, 1], index_col=[0, 1, 2])
    rk.index.names = ["data", "level", "model"]
    sp = rk[("spearman", "mean")].xs(BOTTOM, level="level")
    srcs = ["dominicks", "frat", "m5", "favorita"]
    snames = {"dominicks": "Dominick's", "frat": "Frat", "m5": "M5", "favorita": "Favorita"}
    rmodels = ["ensemble", "world_model", "lightgbm", "timesfm_plan", "timestral_plan"]
    offs = dodge(len(rmodels), 0.6)
    yy = list(range(len(srcs)))
    for yi, s in zip(yy, srcs):
        for m, o in zip(rmodels, offs):
            if (s, m) in sp.index:
                mk(cx, [sp[(s, m)]], [yi + o], SK[m], s=26 if m == "ensemble" else 15)
    rows_axis(cx, yy, [snames[s] for s in srcs])
    vals = sp[sp.index.get_level_values("model").isin(rmodels)]
    cx.set_xlim(np.floor(vals.min() * 10) / 10 - 0.02, np.ceil(vals.max() * 10) / 10 + 0.02)
    cx.set_xlabel("Rank correlation, item by store")
    panel_label(cx, "c", dx=-0.3, dy=1.02)

    allm = ["world_model", "ensemble", "lightgbm", "lift", "timesfm_plan", "timestral_plan"]
    fig.legend(handles=[handle(SK[m]) for m in allm], loc="lower center", bbox_to_anchor=(0.55, -0.01), ncol=6,
               fontsize=7.0, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1)
    save(fig, OUT / "fig_decisions.pdf")
    return cov, sp


def fig_grain():
    """Which model had the lowest error at each decision grain, and how many models could not be told apart."""
    gt = R.grain_table(R.rolling_accuracy())
    grains = [g for g, *_ in R.GRAIN]
    gname = {g: n for g, n, _, _ in R.GRAIN}
    fig = plt.figure(figsize=(FULL, 2.75))
    ax = fig.add_axes([0.14, 0.17, 0.53, 0.66])
    bx = fig.add_axes([0.765, 0.17, 0.225, 0.66])
    nk = len(KEYS)
    for i, g in enumerate(grains):
        for j, k in enumerate(KEYS):
            r = gt[(gt.grain == g) & (gt.key == k)].iloc[0]
            shade = 1.0 - 0.34 * (r.set_size - 1) / 8.0  # darker grey: more models in the confidence set
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=(shade, min(1, shade * 1.005), min(1, shade * 1.02)),
                                   edgecolor="white", lw=1.2, zorder=1))
            m = SK[r.best]
            ax.scatter([j - 0.13], [i], marker=MARKER[m], s=40 if m != "ensemble" else 70, color=COLOR[m],
                       edgecolor="white", lw=0.5, zorder=3)
            ax.text(j + 0.2, i, str(r.set_size), ha="left", va="center", fontsize=SMALL, color=INK, zorder=3)
    ax.set_xlim(-0.5, nk - 0.5)
    ax.set_ylim(len(grains) - 0.5, -0.5)
    ax.set_yticks(range(len(grains)))
    ax.set_yticklabels([gname[g] for g in grains])
    ax.set_xticks(range(nk))
    ax.set_xticklabels([NAME[k].replace("Snack crackers", "Snack\ncrackers").replace("Bottled juice", "Bottled\njuice")
                        .replace("Frozen entrees", "Frozen\nentrees").replace("Soft drinks", "Soft\ndrinks")
                        for k in KEYS], fontsize=SMALL)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.plot([5.5, 5.5], [-0.5, len(grains) - 0.5], color="white", lw=3, zorder=2)
    ax.text(2.5, len(grains) - 0.2, "Dominick's", ha="center", va="top", fontsize=SMALL, color=MUTED)
    panel_label(ax, "a", dx=-0.02, dy=1.16)
    used = [m for m in ("ensemble", "lightgbm", "world_model", "lift", "timesfm_plan") if m in set(gt.best)]
    hs = [Line2D([], [], color=COLOR[SK[m]], marker=MARKER[SK[m]], ls="none", ms=5 if m != "ensemble" else 7,
                 markeredgecolor="white", markeredgewidth=0.5, label=LABEL[SK[m]]) for m in used]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.14, -0.005), ncol=len(hs), frameon=False,
               fontsize=6.6, handletextpad=0.2, columnspacing=1.0)
    # (b) confidence-set size by grain
    rng = np.random.default_rng(0)
    for i, g in enumerate(grains):
        q = gt[gt.grain == g].set_size.to_numpy()
        bx.scatter(q, i + rng.uniform(-0.18, 0.18, len(q)), s=10, color=MUTED, alpha=0.7, edgecolor="none", zorder=2)
        med = np.median(q)
        bx.plot([med, med], [i - 0.3, i + 0.3], color=ACCENT, lw=2.0, zorder=3)
    bx.set_ylim(len(grains) - 0.5, -0.5)
    bx.set_yticks(range(len(grains)))
    bx.set_yticklabels([])
    bx.set_xlim(0.5, 9.5)
    bx.set_xticks([1, 3, 5, 7, 9])
    bx.set_xlabel("Models in the 90% confidence set")
    bx.grid(axis="x", color=GRID, lw=0.5)
    despine(bx)
    bx.text(9.4, len(grains) - 0.55, "bar: median", ha="right", va="bottom", fontsize=SMALL, color=ACCENT)
    panel_label(bx, "b", dx=-0.02, dy=1.16)
    save(fig, OUT / "fig_grain.pdf")


def main():
    use_style()
    fig_grain()
    lay = fig_study()
    clipped = fig_rolling()
    abl_clip, sweep = fig_ablation()
    fig_mechanism()
    cov, sp = fig_decisions()
    print("window layout (periods before the end of the data):")
    for k, v in lay.items():
        print(f"  {k}: " + "; ".join(f"{r}={v[r]}" for r in ("rolling", "development", "final") if v[r]))
    print("clipped in fig_rolling (level, key, model -> ratio):")
    for kk, v in sorted(clipped.items()):
        print(f"  {kk}: {v:.2f}")
    print("clipped in fig_ablation (level, key, variant -> mean, lo, hi):")
    for kk, v in sorted(abl_clip.items()):
        print(f"  {kk}: " + ", ".join(f"{x:.3f}" for x in v))


if __name__ == "__main__":
    main()
