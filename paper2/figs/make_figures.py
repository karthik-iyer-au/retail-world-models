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


def main():
    use_style()
    fig_audit()
    fig_zones()


if __name__ == "__main__":
    main()
