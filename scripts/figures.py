"""Paper figures from the scores recorded in results/README.md.

  python scripts/figures.py [--out docs/figures]

Every number below is copied from results/README.md (three-seed means unless
noted); the section each comes from is named next to it. No run is read here,
so the figures can be redrawn without the run folders.

Figure 1: error by M5 window and model, item by store and overall.
Figure 2: total forecast against total sales by M5 window and model (the level).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WINDOWS = ["29 Feb to 27 Mar", "28 Mar to 24 Apr", "25 Apr to 22 May\n(practice)", "23 May to 19 Jun\n(official test)"]

# "M5 confirmation windows", "TimesFM 2.5 zero-shot on the four M5 windows (mean output)",
# "Graph model over product attributes on the four M5 windows". Practice window: world model and
# LightGBM one seed each (version 1 of the world model, the settings carried to the final test).
ITEM_BY_STORE = {
    "World model": [0.851, 0.833, 0.835, 0.897],
    "LightGBM": [0.873, 0.846, 0.847, 0.905],
    "TimesFM (zero-shot, mean)": [0.836, 0.828, 0.828, 0.885],
    "Graph model": [0.878, 0.845, 0.851, 0.909],
}
OVERALL = {
    "World model": [0.642, 0.555, 0.646, 0.638],
    "LightGBM": [0.709, 0.557, 0.715, 0.547],
    "TimesFM (zero-shot, mean)": [0.854, 0.819, 0.896, 0.729],
    "Graph model": [0.779, 0.644, 0.641, 0.673],
}
# Forecast / actual over the window. Practice window: world model and LightGBM not recorded
# as three-seed means for the same runs, so left out (None).
LEVEL = {
    "World model": [0.991, 1.003, None, 1.026],
    "LightGBM": [0.943, 0.969, None, 0.980],
    "TimesFM (zero-shot, mean)": [0.920, 0.936, 0.904, 0.947],
    "Graph model": [1.058, 1.057, 0.962, 1.010],
}

# Three categorical slots that pass the all-pairs colour checks (reference palette slots 1 to 3),
# and a neutral grey for the fourth, a losing comparison. Shapes differ too, so identity is never colour alone.
STYLE = {
    "World model": dict(color="#2a78d6", marker="o"),
    "LightGBM": dict(color="#eb6834", marker="s"),
    "TimesFM (zero-shot, mean)": dict(color="#1baf7a", marker="D"),
    "Graph model": dict(color="#8a8984", marker="^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _axes(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def dot_panel(ax, data, title, xlabel, ref=None):
    n = len(data)
    for i, (name, vals) in enumerate(data.items()):
        offset = (i - (n - 1) / 2) * 0.16
        ys = [w + offset for w, v in enumerate(vals) if v is not None]
        xs = [v for v in vals if v is not None]
        ax.scatter(xs, ys, s=46, label=name, edgecolors="#fcfcfb", linewidths=1.5, zorder=3, **STYLE[name])
    if ref is not None:
        ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(range(len(WINDOWS)))
    ax.set_yticklabels(WINDOWS)
    if not ax.yaxis_inverted():  # shared axes: invert once
        ax.invert_yaxis()
    _axes(ax, title, xlabel)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/figures")
    out = Path(p.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    dot_panel(axes[0], ITEM_BY_STORE, "Item by store", "WRMSSE, lower is better")
    dot_panel(axes[1], OVERALL, "All 12 levels", "WRMSSE, lower is better")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"m5_error_by_window.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    dot_panel(ax, LEVEL, "Total forecast against total sales", "Forecast / actual (1.0 is right)", ref=1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"m5_level_by_window.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
