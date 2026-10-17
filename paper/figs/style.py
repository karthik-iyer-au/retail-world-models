"""Shared figure style for the manuscript (vector PDF, Pagella-matched serif).

One colour and one marker per model, fixed across every figure (Okabe-Ito,
colour-blind safe; the marker carries the model in greyscale too).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

ACCENT = "#264E78"
INK = "#1a1a1a"
MUTED = "#6b7280"
GRID = "#d9dde3"
EDGE = "#9aa2ad"
FILL = "#eef1f5"

# Column widths at final size (Elsevier): single 90 mm, full 190 mm.
SINGLE = 3.46
FULL = 7.1

COLOR = {
    "world": "#0072B2",
    "lift": "#56B4E9",
    "lgb": "#D55E00",
    "graph": "#009E73",
    "tfm": "#CC79A7",
    "tfm_plan": "#E69F00",
    "naive": "#000000",
    # revision models (added keys only)
    "ensemble": ACCENT,
    "chronos": "#8C8C8C",
    "tft": "#999933",
    "timestral": "#882255",
}
MARKER = {
    "world": "o",
    "lift": "^",
    "lgb": "s",
    "graph": "v",
    "tfm": "D",
    "tfm_plan": "P",
    "naive": "X",
    "ensemble": "*",
    "chronos": "h",
    "tft": "<",
    "timestral": "p",
}
LABEL = {
    "world": "World model",
    "lift": "Lift model",
    "lgb": "LightGBM",
    "graph": "Graph model",
    "tfm": "TimesFM",
    "tfm_plan": "TimesFM with the plan",
    "naive": "Seasonal naive",
    "ensemble": "Ensemble",
    "chronos": "Chronos-Bolt",
    "tft": "TFT",
    "timestral": "Timestral with the plan",
}


def use_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino", "URW Palladio L", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.edgecolor": EDGE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, letter, dx=-0.02, dy=1.06):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="right", color=INK)


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def model_handles(models, ms=5):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="none", ms=ms,
                   markeredgecolor="white", markeredgewidth=0.6, label=LABEL[m]) for m in models]


def save(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  wrote {path}")
