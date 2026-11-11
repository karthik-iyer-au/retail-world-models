"""Figures for the research-loop section, drawn from the committed loop logs
(results/research/<name>/log.jsonl). Vector PDF into paper/figs/.

  python paper/figs/make_loop_figures.py

Writes fig_loop_path.pdf (every try and the best so far, with the keep band), fig_loop_heat.pdf
(each change against each data set's starting settings, first seed), fig_loop_noise.pdf (seed
spread, one copy against five), fig_loop_heldout.pdf (search windows against the held-out
window, every try with three seeds) and fig_loop_slopes.pdf (starting against final settings,
both models, both windows). Every value drawn is read from a log at run time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLOR, EDGE, FULL, GRID, INK, MUTED, despine, panel_label, save, use_style  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "research"

DATA = ["m5", "favorita", "frat", "dominicks_ana", "dominicks_bjc", "dominicks_che", "dominicks_fre",
        "dominicks_sdr", "dominicks_sna"]
TITLE = {"m5": "M5", "favorita": "Favorita", "frat": "Frat", "dominicks_ana": "Analgesics",
         "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese", "dominicks_fre": "Frozen entrees",
         "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snacks"}
# each model's searches, in the order they ran (the paper's final settings come from the last stage)
WM = {"m5": ["pure_m5", "lossT_pure_m5", "hybridU_pure_m5"],
      "frat": ["avg5_world_model_frat", "lossT_avg5_world_model_frat", "hybridU_avg5_world_model_frat"],
      "favorita": ["avg5_world_model_favorita", "lossT_r1_avg5_world_model_favorita",
                   "hybridU_r1_avg5_world_model_favorita"],  # the rerun on the final loss was stopped early (11 November 2026)
      **{f"dominicks_{c}": [f"clean_avg5_world_model_dominicks_{c}", f"lossT_clean_avg5_world_model_dominicks_{c}",
                            f"hybridU_clean_avg5_world_model_dominicks_{c}"]
         for c in ("ana", "bjc", "che", "fre", "sdr", "sna")}}
LGB = {"m5": ["pure_lightgbm_m5"], "frat": ["avg5_lightgbm_frat"], "favorita": ["avg5_lightgbm_favorita"],
       **{f"dominicks_{c}": [f"clean_avg5_lightgbm_dominicks_{c}"] for c in ("ana", "bjc", "che", "fre", "sdr", "sna")}}
# one copy against five, same data and windows (the single-copy Dominick's loops ran on recorded prices)
ONE = {f"dominicks_{c}": f"pure_dominicks_{c}" for c in ("ana", "bjc", "che", "fre", "sdr", "sna")}
FIVE = {f"dominicks_{c}": f"avg5_world_model_dominicks_{c}" for c in ("ana", "bjc", "che", "fre", "sdr", "sna")}


def rows(name: str) -> list[dict]:
    p = LOGS / name / "log.jsonl"
    if not p.exists():
        return []
    out, seen = [], set()
    for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()):
        key = (r["kind"], r.get("round"), r.get("change"))
        if r["kind"] in ("rung", "try", "kept") and (key in seen or r.get("score") is None):
            continue  # a restart reads finished runs back; a failed run has no score
        seen.add(key)
        out.append(r)
    return out


def first_start(name: str) -> dict | None:
    return next((r for r in rows(name) if r["kind"] == "start"), None)


def chain_path(chain: list[str]) -> tuple[list, list, list]:
    """Tries in the order they ran across a chain of searches: (screens, three-seed tries, best so far).
    Each element is (x, score); best so far steps down when a change is kept, with its keep bar."""
    screens, tries, best = [], [], []
    x = 0
    for name in chain:
        rs = rows(name)
        if not rs:
            continue
        st = next(r for r in rs if r["kind"] == "start")
        best.append((x, st["score"], st["keep_bar"]))
        for r in rs:
            if r["kind"] == "rung":
                x += 1
                screens.append((x, r["score"]))
            elif r["kind"] == "try":
                tries.append((x, r["score"]))
            elif r["kind"] == "kept":
                best.append((x, r["score"], st["keep_bar"]))
        x += 1
    return screens, tries, best


def fig_path():
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 5.6))
    for ax, d in zip(axes.flat, DATA):
        for chain, model in ((WM[d], "world"), (LGB[d], "lgb")):
            screens, tries, best = chain_path(chain)
            if not best:
                continue
            c = COLOR[model]
            if screens:
                sx, sy = zip(*screens)
                ax.scatter(sx, sy, s=5, color=c, alpha=0.28, lw=0, zorder=2)
            if tries:
                tx, ty = zip(*tries)
                ax.scatter(tx, ty, s=13, facecolor="white", edgecolor=c, lw=0.8, zorder=3)
            end = max([p[0] for p in screens] + [b[0] for b in best]) + 1
            xs = [b[0] for b in best] + [end]
            ys = [b[1] for b in best] + [best[-1][1]]
            ax.step(xs, ys, where="post", color=c, lw=1.3, zorder=4)
            for (x0, y0, bar), x1 in zip(best, xs[1:]):
                ax.fill_between([x0, x1], y0 - bar, y0, step=None, color=c, alpha=0.10, lw=0, zorder=1)
        ax.set_title(TITLE[d])
        ax.grid(axis="y", lw=0.4, color=GRID)
        despine(ax)
        ax.tick_params(axis="x", labelsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("Try, in the order run")
    for ax in axes[:, 0]:
        ax.set_ylabel("WRMSSE, search windows")
    handles = [Line2D([], [], color=COLOR["world"], lw=1.3, label="World model, best so far"),
               Line2D([], [], color=COLOR["lgb"], lw=1.3, label="LightGBM, best so far"),
               Line2D([], [], color=MUTED, marker="o", ls="none", ms=2.5, alpha=0.5, label="First-seed screen"),
               Line2D([], [], color=MUTED, marker="o", ls="none", ms=4, mfc="white", label="Three-seed try"),
               plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.2, label="Keep band (seed spread)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, OUT / "fig_loop_path.pdf")


def fig_heat():
    """Change on the first seed against each data set's starting settings, main world-model loop, round 1."""
    vals = {}
    for d in DATA:
        rs = rows(WM[d][0])
        st = next((r for r in rs if r["kind"] == "start"), None)
        if st is None:
            continue
        base = st["seed_scores"][0]
        for r in rs:
            if r["kind"] == "rung" and r.get("round") == 1:
                vals[(r["change"], d)] = 100 * (r["score"] - base) / base
    changes = sorted({c for c, _ in vals}, key=lambda c: np.nanmedian([vals.get((c, d), np.nan) for d in DATA]))
    M = np.array([[vals.get((c, d), np.nan) for d in DATA] for c in changes])
    lim = float(np.nanpercentile(np.abs(M), 95))
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 0.17 * len(changes) + 0.9))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks(range(len(DATA)), [TITLE[d] for d in DATA], rotation=40, ha="right")
    ax.set_yticks(range(len(changes)), [c.replace("_", " ") for c in changes], fontsize=6.4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Change in WRMSSE, first seed (%)", fontsize=6.6)
    cb.outline.set_visible(False)
    ax.set_title("Each change tried alone on the world model (round 1)")
    save(fig, OUT / "fig_loop_heat.pdf")


def fig_noise():
    """Starting settings' seed scores as deviations from their mean: one copy against five."""
    from scipy.stats import gaussian_kde

    groups = {"One copy": [], "Five-copy average": []}
    for label, names in (("One copy", ONE), ("Five-copy average", FIVE)):
        for d, n in names.items():
            st = first_start(n)
            if st:
                s = np.array(st["seed_scores"])
                groups[label] += list(100 * (s - s.mean()) / s.mean())
    fig, ax = plt.subplots(figsize=(FULL * 0.48, 2.1))
    grid = np.linspace(-12, 12, 400)
    for (label, v), c in zip(groups.items(), (COLOR["lift"], COLOR["world"])):
        v = np.array(v)
        if len(v) < 3:
            continue
        k = gaussian_kde(v)
        ax.fill_between(grid, k(grid), color=c, alpha=0.25, lw=0)
        ax.plot(grid, k(grid), color=c, lw=1.2)
        ax.plot(v, np.full_like(v, -0.012 if label == "One copy" else -0.004), "|", color=c, ms=5, mew=0.8)
        peak = grid[np.argmax(k(grid))]
        ax.annotate(f"{label}\nsd {v.std(ddof=1):.1f}%", xy=(peak, k(grid).max()), xytext=((-11.5 if label == "One copy" else 4.5), k(grid).max() * (1.6 if label == "One copy" else 0.9)),
                    fontsize=6.6, color=c, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=c, lw=0.6))
    ax.set_xlabel("Seed score against the mean of the three seeds (%)")
    ax.set_yticks([])
    despine(ax, keep=("bottom",))
    ax.set_title("Seed noise of the world model, Dominick's categories")
    save(fig, OUT / "fig_loop_noise.pdf")


def fig_heldout():
    fig, ax = plt.subplots(figsize=(FULL * 0.48, 2.7))
    for model, chains in (("world", WM), ("lgb", LGB)):
        xs, ys = [], []
        for d in DATA:
            for name in chains[d]:
                st = first_start(name)
                if not st or st.get("held_out") is None:
                    continue
                for r in rows(name):
                    if r["kind"] == "try" and r.get("held_out") is not None:
                        xs.append(100 * (r["score"] - st["score"]) / st["score"])
                        ys.append(100 * (r["held_out"] - st["held_out"]) / st["held_out"])
        ax.scatter(xs, ys, s=11, color=COLOR[model], alpha=0.65, lw=0, label=("World model" if model == "world" else "LightGBM"))
        if len(xs) > 2:
            r = np.corrcoef(xs, ys)[0, 1]
            ax.text(0.02, 0.96 - 0.07 * (model == "lgb"), f"{'World model' if model == 'world' else 'LightGBM'}: r = {r:.2f}, {len(xs)} tries",
                    transform=ax.transAxes, fontsize=6.6, color=COLOR[model], va="top")
    ax.axhline(0, color=EDGE, lw=0.6)
    ax.axvline(0, color=EDGE, lw=0.6)
    ax.set_xlabel("Change on the search windows (%)")
    ax.set_ylabel("Change on the held-out window (%)")
    despine(ax)
    ax.set_title("Search windows against the held-out window")
    save(fig, OUT / "fig_loop_heldout.pdf")


def fig_slopes():
    """Starting settings to final settings, by data set: one arrow per model, both windows."""
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.9), sharex=True)
    for ax, field, title in ((axes[0], "score", "Search windows"), (axes[1], "held_out", "Held-out window")):
        for model, chains, dx in (("world", WM, -0.13), ("lgb", LGB, 0.13)):
            for i, d in enumerate(DATA):
                names = [n for n in chains[d] if rows(n)]
                if not names:
                    continue
                st = first_start(names[0])
                end = next((r for r in reversed(rows(names[-1])) if r["kind"] == "end"), None)
                if not st or not end or end.get(field) is None:
                    continue
                y0, y1 = st[field], end[field]
                ax.plot(i + dx, y0, "o", ms=3.2, mfc="white", mec=COLOR[model], mew=0.9, zorder=3)
                if abs(y1 - y0) > 1e-9:
                    ax.annotate("", xy=(i + dx, y1), xytext=(i + dx, y0), zorder=2,
                                arrowprops=dict(arrowstyle="-|>", color=COLOR[model], lw=1.1, mutation_scale=7, shrinkA=2, shrinkB=0))
                ax.plot(i + dx, y1, "o", ms=3.2, color=COLOR[model], zorder=4)
        ax.set_title(title)
        ax.set_ylabel("WRMSSE")
        ax.grid(axis="y", lw=0.4, color=GRID)
        despine(ax)
        for i in range(len(DATA) - 1):
            ax.axvline(i + 0.5, color=GRID, lw=0.4)
    axes[1].set_xticks(range(len(DATA)), [TITLE[d] for d in DATA])
    fig.legend(handles=[Line2D([], [], color=COLOR["world"], marker="o", lw=1.1, ms=3.2, label="World model"),
                        Line2D([], [], color=COLOR["lgb"], marker="o", lw=1.1, ms=3.2, label="LightGBM"),
                        Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", ms=3.2, label="Starting settings"),
                        Line2D([], [], color=MUTED, marker="o", ls="none", ms=3.2, label="Final settings")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, OUT / "fig_loop_slopes.pdf")


def main():
    use_style()
    fig_path()
    fig_heat()
    fig_noise()
    fig_heldout()
    fig_slopes()
    write_table()


def write_table():
    """paper/generated/tab_loop.tex: each search, from starting to final settings."""
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize",
             r"\caption{Every search, from starting to final settings. Scores are WRMSSE (lower is better), the mean of three seeds on the "
             r"search windows and one score on the held-out window, which never decided anything. The keep bar is the seed spread of "
             r"the starting settings; a change was kept only when its three-seed mean beat the best so far by at least the bar. "
             r"Tries counts every run, first-seed screens and three-seed confirmations.}",
             r"\label{tab:loop}",
             r"\begin{tabular}{llrrrrrrp{5.2cm}}", r"\toprule",
             r"Data & Model & Tries & Keep bar & \multicolumn{2}{c}{Starting} & \multicolumn{2}{c}{Final} & Kept, in order \\",
             r"& & & & Search & Held-out & Search & Held-out & \\", r"\midrule"]
    for d in DATA:
        for model, chains in (("World model", WM), ("LightGBM", LGB)):
            names = [n for n in chains[d] if rows(n)]
            if not names:
                continue
            st = first_start(names[0])
            end = next((r for r in reversed(rows(names[-1])) if r["kind"] == "end"), None)
            tries = sum(1 for n in names for r in rows(n) if r["kind"] in ("rung", "try"))
            kept = [r["change"] for n in names for r in rows(n) if r["kind"] == "kept"]
            fin = (f"{end['score']:.3f} & {end['held_out']:.3f}" if end else r"\multicolumn{2}{c}{running}")
            lines.append(f"{TITLE[d] if model == 'World model' else ''} & {model} & {tries} & {st['keep_bar']:.3f} & "
                         f"{st['score']:.3f} & {st['held_out']:.3f} & {fin} & "
                         + (", ".join(k.replace('_', ' ') for k in kept) or "nothing") + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = ROOT / "paper" / "generated" / "tab_loop.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
