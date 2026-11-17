"""Paper 3 figures, tables and numbers, all read from results/paper3 (scripts/paper3_analysis.py).

  cd paper3/figs && python make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import EDGE, FULL, GRID, INK, MUTED, SINGLE, despine, panel_label, save, use_style

ROOT = Path(__file__).resolve().parents[2]
RES, GEN, FIG = ROOT / "results/paper3", ROOT / "paper3/generated", ROOT / "paper3/figs"
WORDS = "none one two three four five six seven eight nine ten eleven twelve".split()

# the models shown, in order, with one colour each (Okabe-Ito)
M = {
    "wm_d32_s2000_h26": ("World model", "#0072B2", "o"),
    "wm_shape": ("World model with price rules", "#CC79A7", "^"),
    "loglinear_shape": ("Log-linear with price rules", "#009E73", "D"),
    "lgb_shape": ("LightGBM with price rules", "#D55E00", "s"),
}
PLAN_LABEL = {"deeper_50": "Cut 50%", "deeper_60": "Cut 60%", "regular_-10": "Regular $-$10%",
              "regular_-5": "Regular $-$5%", "regular_+5": "Regular +5%", "regular_+10": "Regular +10%",
              "off_pattern": "Off-pattern", "no_plan": "No plan"}
SHAPE = {"loglinear": "log-linear world", "bent": "bent world", "kinked": "kinked world"}
SHAPES = ("loglinear", "bent", "kinked")
REMEDY = [  # (version, label, group)
    ("wm_d32_s2000_h26", "World model", "World model"),
    ("wm_jepa", "+ JEPA pretraining", "World model"),
    ("wm_latent", "+ latent check", "World model"),
    ("wm_overshoot", "+ latent check 7 ahead", "World model"),
    ("wm_adajepa", "+ adaptation at the plan date", "World model"),
    ("wm_rr", "+ recorded regular price", "World model"),
    ("wm_ll", "Split: world model + log-linear", "Split"),
    ("wm_shape", "+ price rules", "Shape"),
    ("lgb_l31_r500_m20", "LightGBM", "LightGBM"),
    ("lgb_rr", "+ recorded regular price", "LightGBM"),
    ("lgb_ll", "Split: LightGBM + log-linear", "Split"),
    ("lgb_shape", "+ price rules", "Shape"),
    ("loglinear", "Log-linear", "Log-linear"),
    ("loglinear_rr", "+ recorded regular price", "Log-linear"),
    ("loglinear_shape", "+ price rules", "Shape"),
]


def med(d, cols, by=("shape", "version")):
    return d.groupby(list(by))[cols].median()


def fig_ratio(d):
    plans = list(PLAN_LABEL)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.9), sharey=True)
    x = np.arange(len(plans))
    for ax, shape, letter in zip(axes, SHAPES, "abc"):
        g = med(d[d["shape"] == shape], [f"ratio_{p}" for p in plans], by=("version",))
        ax.axhline(1, color=INK, lw=0.7)
        ax.axhspan(0.8, 1.25, color=GRID, alpha=0.5, lw=0)
        for i, (v, (lab, col, mk)) in enumerate(M.items()):
            y = g.loc[v].to_numpy().clip(0.03, 40)
            ax.plot(x + (i - 1.5) * 0.12, y, mk, color=col, ms=4.2, mec="white", mew=0.5, label=lab, ls="none")
        ax.set_yscale("log")
        ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4, 10, 20])
        ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "10", "20"])
        ax.set_xticks(x)
        ax.set_xticklabels([PLAN_LABEL[p] for p in plans], rotation=35, ha="right")
        ax.set_title(f"In the {SHAPE[shape]}", fontweight="normal")
        despine(ax)
        panel_label(ax, letter)
    axes[0].set_ylabel("Predicted change / true change")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    save(fig, FIG / "fig_ratio.pdf")


def fig_link(d, L):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.6), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    ax = axes[0]
    g = d[(d["shape"] == "bent") & (d.confounding == 1.0) & d.grid]
    for model, col, mk in (("LightGBM", "#D55E00", "s"), ("world model", "#0072B2", "o"), ("log-linear", "#009E73", "D")):
        h = g[g.model == model]
        ax.plot(h.rmsse, h.err_mean, mk, color=col, ms=3.4, mec="white", mew=0.4, ls="none",
                label=model[0].upper() + model[1:])
    ax.set_xlabel("Forecast error on the merchants' plans\n(item-by-store RMSSE)")
    ax.set_ylabel("Error on new plans")
    ax.set_title("Bent world, planning strength 1", fontweight="normal")
    despine(ax)
    panel_label(ax, "a")
    for ax, shape, letter in zip(axes[1:], ("loglinear", "bent"), "bc"):
        h = L[(L["shape"] == shape) & (L.measure == "err_mean")]
        for k, (who, col, mk) in enumerate((("all", INK, "D"), ("LightGBM", "#D55E00", "s"),
                                            ("world model", "#0072B2", "o"))):
            r = h[h.versions_of == who].sort_values("confounding")
            xs = r.confounding.to_numpy() + (k - 1) * 0.07
            ax.errorbar(xs, r.spearman, yerr=[r.spearman - r.lo, r.hi - r.spearman], fmt=mk, color=col, ms=3.6,
                        mec="white", mew=0.4, elinewidth=0.8, capsize=0,
                        label={"all": "All 25 versions", "LightGBM": "LightGBM versions",
                               "world model": "World model versions"}[who])
        ax.axhline(0, color=EDGE, lw=0.7)
        ax.set_ylim(-0.9, 1)
        ax.set_xticks([0, 0.5, 1, 1.5])
        ax.set_xlabel("Planning strength")
        ax.set_title(f"In the {SHAPE[shape]}", fontweight="normal")
        despine(ax)
        panel_label(ax, letter)
    axes[1].set_ylabel("Rank correlation, forecast error\nwith error on new plans")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, ["LightGBM version", "World model version", "Log-linear model"] + l1, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.02), handletextpad=0.1, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    save(fig, FIG / "fig_link.pdf")


def fig_remedies(d):
    plain, tl = d[d.explore == 0], d[d.explore > 0]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 3.6), sharey=True)
    y = np.arange(len(REMEDY))[::-1]
    for ax, shape, letter in zip(axes, SHAPES, "abc"):
        a = med(plain[plain["shape"] == shape], ["err_mean"], by=("version",))["err_mean"]
        b = med(tl[tl["shape"] == shape], ["err_mean"], by=("version",))["err_mean"]
        for yy, (v, lab, grp) in zip(y, REMEDY):
            col = {"World model": "#0072B2", "Split": "#E69F00", "Shape": "#CC79A7", "LightGBM": "#D55E00",
                   "Log-linear": "#009E73"}[grp]
            if v not in a.index:
                continue
            ax.plot(a[v], yy, "o", color=col, ms=4.4, mec="white", mew=0.5)
            if v in b.index:
                ax.plot(b[v], yy, "o", mfc="white", mec=col, ms=4.4, mew=0.9)
            ax.text(min(a[v], 2.5) + 0.06, yy, f"{a[v]:.2f}", va="center", fontsize=6.2, color=MUTED)
        ax.set_xlim(0, 2.8)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab, _ in REMEDY])
        ax.set_xlabel("Error on new plans")
        ax.set_title(f"In the {SHAPE[shape]}", fontweight="normal")
        ax.grid(axis="x")
        despine(ax)
        panel_label(ax, letter)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", color=INK, ls="none", ms=4, label="Trained on the merchants' history"),
                        Line2D([], [], marker="o", mfc="white", mec=INK, ls="none", ms=4,
                               label="Trained with 5% random test plans added")], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0)
    save(fig, FIG / "fig_remedies.pdf")


def tables(d, L):
    plain = d[d.explore == 0]
    cols = ["rmsse", "err_mean", "regular_loss", "promo_loss", "stock_mean"]
    t = med(plain, cols)
    lines = []
    for v, lab, _ in REMEDY:
        cells = []
        for shape in SHAPES:
            if (shape, v) in t.index:
                r = t.loc[(shape, v)]
                cells += [f"{r.rmsse:.3f}", f"{r.err_mean:.2f}", f"{100 * r.regular_loss:.1f}", f"{r.stock_mean:.2f}"]
            else:
                cells += ["", "", "", ""]
        lines.append(lab.replace("+ ", "\\quad + ") + " & " + " & ".join(cells) + r" \\")
    tab = r"""\begin{table*}[t]
\centering
\caption{\textbf{Every model and remedy in the three worlds.} Medians over the 12 worlds of each kind (four planning
strengths, three seeds). Forecast: item-by-store RMSSE on the merchants' own plan in the held-out weeks. New plans:
mean over the eight held-out plans of $|\log(\text{predicted change}/\text{true change})|$. Price choice: revenue
given up by the regular-price move the model picks, \% of the best. Stock: summed absolute error of ordering to the
forecast, over expected sales. Rows marked + change one thing in the row above; all rows after the first of each
model, and the split rows, were added after the first results (Appendix~\ref{app:plan}). The kinked world was run
for the versions shown.}
\label{tab:remedies}
\footnotesize\setlength{\tabcolsep}{3.2pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{Log-linear world} & \multicolumn{4}{c}{Bent world} & \multicolumn{4}{c}{Kinked world} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
Model & Forecast & New plans & Price & Stock & Forecast & New plans & Price & Stock & Forecast & New plans & Price & Stock \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""
    (GEN / "tab_remedies.tex").write_text(tab)
    # the planned test, all measures
    rows = []
    for shape in ("loglinear", "bent"):
        for c in (0.0, 0.5, 1.0, 1.5):
            cells = []
            for who in ("all", "world model", "LightGBM"):
                for m in ("err_mean", "regular_loss", "stock_mean"):
                    r = L[(L["shape"] == shape) & (L.confounding == c) & (L.versions_of == who) & (L.measure == m)].iloc[0]
                    s = f"{r.spearman:.2f}" if pd.notna(r.spearman) else "--"
                    if pd.notna(r.spearman) and (r.lo > 0 or r.hi < 0):
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
            rows.append(f"{SHAPE[shape].capitalize() if c == 0 else ''} & {c:g} & " + " & ".join(cells) + r" \\")
    tab = r"""\begin{table*}[t]
\centering
\caption{\textbf{Does better forecasting go with better simulation?} Rank correlation, within each world, between
forecast error on the merchants' plans and each measure (higher is worse for every measure, so a positive number
means better forecasters are also better at it), averaged over three seeds. Bold: the 95\% bootstrap interval excludes
zero. --: every version made the same choice, so there is nothing to rank. All versions: the test fixed in the plan (12 world model, 12 LightGBM and the log-linear model). Within a model
class: added after the first results.}
\label{tab:link}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrrrrrrr}
\toprule
 & & \multicolumn{3}{c}{All 25 versions} & \multicolumn{3}{c}{World model versions} & \multicolumn{3}{c}{LightGBM versions} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
World & Planning & New plans & Price choice & Stock & New plans & Price choice & Stock & New plans & Price choice & Stock \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""
    (GEN / "tab_link.tex").write_text(tab)


def table_h2h():
    h = pd.read_csv(RES / "head_to_head.csv")
    name = {"world model": "World model", "log-linear": "Log-linear", "LightGBM": "LightGBM"}
    rows = []
    for shape in SHAPES:
        for (a, b) in (("world model", "log-linear"), ("world model", "LightGBM"), ("log-linear", "LightGBM")):
            cells = []
            for how in ("base", "median", "best"):
                r = h[(h["shape"] == shape) & (h.a == a) & (h.b == b) & (h.how == how)]
                if not len(r):
                    cells.append("")
                    continue
                r = r.iloc[0]
                p = "$<$0.001" if r.sign_p < 0.001 else f"{r.sign_p:.3f}"
                f = lambda x: f"{x:+.2f}".replace("-", "$-$")
                cells.append(f"{int(r.a_ahead)}/{int(r.worlds)} ({p}); {f(r.median_diff)} [{f(r.lo)}, {f(r.hi)}]")
            lab = SHAPE[shape].replace(" world", "").capitalize() if a == "world model" and b == "log-linear" else ""
            rows.append(f"{lab} & {name[a]} vs {name[b]} & " + " & ".join(cells) + r" \\")
    tab = r"""\begin{table*}[t]
\centering
\caption{\textbf{The three models with the price rules, head to head, with the world as the unit} (added after the
first results). Each cell: worlds in which the first model has the lower error on new plans, of all worlds (two-sided
sign test), then the median paired difference in error (negative favours the first model) with a 95\% interval from
a bootstrap over worlds. Base: one version of each model. Median and best: each model's median and best version on
the held-out plans over 12 world model, 12 LightGBM and 6 log-linear versions. The kinked base comparison has 40
worlds (seeds 101 to 110); the tuned comparisons have 12.}
\label{tab:h2h}
\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lllll}
\toprule
World & Comparison & Base version & Median version & Best version \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""
    (GEN / "tab_h2h.tex").write_text(tab)
    X = {}
    for _, r in h.iterrows():
        tag = {"base": "Base", "median": "Med", "best": "Best"}[r.how] + {"loglinear": "L", "bent": "B", "kinked": "K"}[
            r["shape"]] + {("world model", "log-linear"): "WmLl", ("world model", "LightGBM"): "WmLgb",
                           ("log-linear", "LightGBM"): "LlLgb"}[(r.a, r.b)]
        X[f"hh{tag}Won"] = str(int(r.a_ahead))
        X[f"hh{tag}N"] = str(int(r.worlds))
        X[f"hh{tag}P"] = "below 0.001" if r.sign_p < 0.001 else f"{r.sign_p:.3f}"
        X[f"hh{tag}Diff"] = f"{r.median_diff:.2f}"
        X[f"hh{tag}Ci"] = f"{r.lo:.2f} to {r.hi:.2f}"
        X[f"hh{tag}A"], X[f"hh{tag}B"] = f"{r.a_median:.2f}", f"{r.b_median:.2f}"
    return X


REAL = [("wm_d32_s2000_h26", "World model"), ("wm_rr", "\\quad + recorded regular price"),
        ("wm_shape", "\\quad + price rules"), ("lgb_l31_r500_m20", "LightGBM"),
        ("lgb_rr", "\\quad + recorded regular price"), ("lgb_shape", "\\quad + price rules"),
        ("loglinear", "Log-linear"), ("loglinear_rr", "\\quad + recorded regular price"),
        ("loglinear_shape", "\\quad + price rules")]


def table_real():
    s = pd.read_csv(RES / "real_summary.csv")
    pr = pd.read_csv(RES / "real_pairs.csv")
    cols = [("frat", "deep"), ("frat", "regular_move"), ("frat", "all"), ("m5", "regular_move"), ("m5", "all")]
    lines = []
    for v, lab in REAL:
        cells = []
        for data, g in cols:
            r = s[(s.data == data) & (s.group == g) & (s.version == v)].iloc[0]
            cells.append(f"{r.abs_error:.3f}" + (f" ({r.bias:.2f})" if g != "all" else ""))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    n = lambda data, g: int(s[(s.data == data) & (s.group == g)].rows.max())
    tab = r"""\begin{table*}[t]
\centering
\caption{\textbf{Real data: error on the rarest plans that were actually run} (added after the first results;
Appendix~\ref{app:plan}). Summed absolute error over summed sales, median over four forecast origins; in brackets,
summed forecast over summed sales (1 is right). Breakfast at the Frat: every cut deeper than 30\% was removed from
training; X with such a cut; regular-price moves are
item-weeks whose recorded regular price differs by 10\% or more from its last value before the origin. M5 (FOODS\_3
in three stores, weekly): regular-price moves of 5\% or more on a constructed regular price. These are sales that
happened, not plans never run.}
\label{tab:real}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{3}{c}{Breakfast at the Frat} & \multicolumn{2}{c}{M5} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Model & Deep cuts & Regular moves & All & Regular moves & All \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""
    (GEN / "tab_real.tex").write_text(tab)
    X = {}
    for data, g in cols:
        for v, _ in REAL:
            r = s[(s.data == data) & (s.group == g) & (s.version == v)].iloc[0]
            tag = {"frat": "F", "m5": "M"}[data] + {"deep": "Deep", "regular_move": "Move", "all": "All"}[g] + \
                {"wm_d32_s2000_h26": "Wm", "wm_rr": "WmRr", "wm_shape": "WmShape", "lgb_l31_r500_m20": "Lgb",
                 "lgb_rr": "LgbRr", "lgb_shape": "LgbShape", "loglinear": "Ll", "loglinear_rr": "LlRr",
                 "loglinear_shape": "LlShape"}[v]
            X[f"re{tag}"], X[f"rb{tag}"] = f"{r.abs_error:.2f}", f"{r.bias:.2f}"
    X["reFDeepRows"] = f"{n('frat', 'deep'):,}"
    for _, r in pr.iterrows():
        tag = {"frat": "F", "m5": "M"}[r.data] + {"deep": "Deep", "regular_move": "Move"}[r.group] + \
            r.a.replace("_", "").replace("d32s2000h26", "").replace("l31r500m20", "") + "V" + \
            r.b.replace("_", "").replace("d32s2000h26", "").replace("l31r500m20", "")
        X[f"rp{tag}"] = f"{r['diff']:.3f}"
        X[f"rp{tag}Ci"] = f"{r.lo:.3f} to {r.hi:.3f}"
        X[f"rp{tag}Won"] = WORDS[int(r.origins_a_ahead)]
    return X


def macros(d, L):
    plain, tl = d[d.explore == 0], d[d.explore > 0]
    t = med(plain, ["rmsse", "err_mean", "regular_loss", "stock_mean", "promo_loss"] +
            [f"ratio_{p}" for p in PLAN_LABEL])
    e = lambda s, v, c="err_mean", f="{:.2f}": f.format(t.loc[(s, v), c])
    X = {}
    for s, k in (("loglinear", "L"), ("bent", "B"), ("kinked", "K")):
        for v, tag in (("loglinear_shape", "LlShape"), ("lgb_shape", "LgbShape"),("wm_d32_s2000_h26", "Wm"), ("lgb_l31_r500_m20", "Lgb"), ("loglinear", "Ll"),
                       ("loglinear_rr", "LlRr"), ("wm_rr", "WmRr"), ("lgb_rr", "LgbRr"), ("wm_ll", "Split"),
                       ("lgb_ll", "LgbSplit"), ("wm_shape", "Shape"), ("wm_jepa", "Jepa"), ("wm_latent", "Latent"),
                       ("wm_overshoot", "Over"), ("wm_adajepa", "Adapt")):
            if (s, v) not in t.index:
                continue
            X[f"err{k}{tag}"] = e(s, v)
            X[f"fc{k}{tag}"] = e(s, v, "rmsse", "{:.3f}")
            X[f"cutsix{k}{tag}"] = e(s, v, "ratio_deeper_60", "{:.1f}")
            X[f"regdown{k}{tag}"] = e(s, v, "ratio_regular_-10", "{:.1f}")
            X[f"regup{k}{tag}"] = e(s, v, "ratio_regular_+10", "{:.2f}")
            X[f"price{k}{tag}"] = f"{100 * t.loc[(s, v), 'regular_loss']:.1f}"
        w = plain[plain["shape"] == s].pivot_table(index=["confounding", "seed"], columns="version", values="err_mean")
        X[f"shapeWins{k}"] = WORDS[int((w["wm_shape"] < w.drop(columns="wm_shape").min(axis=1)).sum())]
        X[f"shapeBeatsLl{k}"] = WORDS[int((w["wm_shape"] < w["loglinear_shape"]).sum())]
        X[f"shapeBeatsLgb{k}"] = WORDS[int((w["wm_shape"] < w["lgb_shape"]).sum())]
        X[f"worlds{k}"] = WORDS[len(w)]
        if s == "kinked":
            continue
        tt = tl[tl["shape"] == s].groupby("version")["err_mean"].median()
        X[f"tlLlShape{k}"] = f"{tt['loglinear_shape']:.2f}"
        X[f"tlShape{k}"], X[f"tlLgb{k}"], X[f"tlWm{k}"] = (f"{tt['wm_shape']:.2f}", f"{tt['lgb_l31_r500_m20']:.2f}",
                                                         f"{tt['wm_d32_s2000_h26']:.2f}")
        a = L[(L["shape"] == s) & (L.measure == "err_mean")]
        for who, tag in (("all", "All"), ("world model", "Wm"), ("LightGBM", "Lgb")):
            r = a[a.versions_of == who].set_index("confounding")
            X[f"link{k}{tag}Lo"], X[f"link{k}{tag}Hi"] = f"{r.spearman.min():.2f}", f"{r.spearman.max():.2f}"
            for c, ct in ((0.0, "Zero"), (1.0, "One"), (1.5, "OneHalf")):
                X[f"link{k}{tag}{ct}"] = f"{r.loc[c, 'spearman']:.2f}"
                X[f"link{k}{tag}{ct}Ci"] = f"{r.loc[c, 'lo']:.2f} to {r.loc[c, 'hi']:.2f}"
    a = L[(L.measure == "err_mean") & (L.versions_of == "all")]
    X["linkAllMin"], X["linkAllMax"] = f"{a.spearman.min():.2f}", f"{a.spearman.max():.2f}"
    full = pd.read_csv(RES / "versions.csv")
    real = len(list((RES / "real").rglob("*.json")))
    X["nFits"] = f"{len(full) + real:,}"
    jepa_like = ["wm_jepa", "wm_latent", "wm_overshoot", "wm_adajepa"]
    X["errLActLo"] = f"{t.loc[[('loglinear', v) for v in jepa_like], 'err_mean'].min():.2f}"
    X["errLActHi"] = f"{t.loc[[('loglinear', v) for v in jepa_like], 'err_mean'].max():.2f}"
    X["errBActLo"] = f"{t.loc[[('bent', v) for v in jepa_like], 'err_mean'].min():.2f}"
    X["errBActHi"] = f"{t.loc[[('bent', v) for v in jepa_like], 'err_mean'].max():.2f}"
    X.update(table_h2h())
    X.update(table_real())
    import re
    num = re.compile(r"-(?=\d)")  # a typeset minus sign in every number
    (GEN / "p3_macros.tex").write_text("".join(
        f"\\newcommand{{\\{k}}}{{{num.sub(lambda m: chr(0x2212), str(v))}}}\n" for k, v in X.items()))
    return X


def main():
    use_style()
    d = pd.read_csv(RES / "versions.csv")
    d = d[d.seed <= 103]  # seeds 104 to 110 (amendment 4) enter only the head-to-head of tab_h2h
    L = pd.read_csv(RES / "link.csv")
    fig_ratio(d[d.explore == 0])
    fig_link(d[d.explore == 0], L)
    fig_remedies(d)
    tables(d, L)
    X = macros(d, L)
    print({k: X[k] for k in list(X)[:12]})


if __name__ == "__main__":
    main()
