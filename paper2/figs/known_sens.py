"""Known-answer sensitivity and the cost of a wrong lift (plan amendment of 16 November 2026).

Reads results/paper2/known_answer (the two main worlds) and results/paper2/known_answer_sens (the planned-into-demand
world with one setting changed at a time), and results/paper2/known_answer/decisions.csv and
known_answer_sens/decisions.csv (scripts/known_answer_decisions.py). Writes generated/tab_known_sens.tex and
generated/known_sens_macros.tex.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from known import rank
from make_figures import ols_slope

ROOT = Path(__file__).resolve().parents[2]
KA, KS = ROOT / "results/paper2/known_answer", ROOT / "results/paper2/known_answer_sens"
GEN = ROOT / "paper2/generated"
ROWS = [("confounded", KA, "As in the main study"), ("sens_conf0.5", KS, "Planning into demand, half as strong"),
        ("sens_conf1.5", KS, "Planning into demand, 1.5 times as strong"),
        ("sens_rho0.4", KS, "Demand shocks shorter (persistence 0.4)"),
        ("sens_rho0.9", KS, "Demand shocks longer (persistence 0.9)"),
        ("sens_reg-0.25", KS, "Regular-price elasticity $-0.25$"), ("sens_reg-1.0", KS, "Regular-price elasticity $-1.0$"),
        ("sens_cut-1.5", KS, "Temporary-cut elasticity $-1.5$"), ("sens_cut-3.5", KS, "Temporary-cut elasticity $-3.5$")]
WORDS = "none one two three four five six seven eight nine".split()


def checks(r, L, lgb) -> tuple[int, int]:
    """C1, C2, C4, C5 and C6 with the pass rules of 13 October (C3 needs three seeds and is not run here)."""
    ok = []
    if "construction" in r:
        ok.append(max(r["construction"].values()) < 1e-3)
    if "randomisation" in r:
        ok.append(r["randomisation"]["lift readout"] < 0.5)
    ok.append(r["monotonicity"]["share_lower_after_deeper_cut"] < 0.01)
    s = r["against_sales"]["any_plan"]
    ok.append((0.8 <= s["bias"] <= 1.25) and s["rank"] >= 0.3)
    if lgb is not None:
        ok.append(rank(L["total_lift"], lgb["total_lift"]) >= 0.5)
    return int(sum(ok)), len(ok)


def row(w: str, folder: Path, dec: pd.DataFrame) -> dict:
    world = json.loads((folder / f"{w}_world.json").read_text())
    true_reg = world["truth"]["regular_elasticity"]
    out = {"world": w}
    lgb_l = pd.read_csv(folder / f"{w}_lightgbm_lifts.csv.gz")
    for m, tag in (("world_seed1", "wm"), ("lightgbm", "lgb")):
        r = json.loads((folder / f"{w}_{m}.json").read_text())
        L = pd.read_csv(folder / f"{w}_{m}_lifts.csv.gz") if m != "lightgbm" else lgb_l
        out[f"{tag}_bias"] = r["truth"]["total_lift"]["bias"]
        eps = r["origin_episodes"]
        x = [e["log_ratio"] for e in eps]
        stated, _ = ols_slope([e["stated"] for e in eps], x)
        out[f"{tag}_reg_times"] = stated / true_reg
        out[f"{tag}_passed"], out[f"{tag}_applicable"] = checks(r, L, lgb_l if m != "lightgbm" else None)
        d = dec[(dec.world == w) & (dec.model == m)].iloc[0]
        out[f"{tag}_choice"], out[f"{tag}_stock"] = d["choice"], d["stock"]
    return out


def main():
    dec = pd.concat([pd.read_csv(KA / "decisions.csv"), pd.read_csv(KS / "decisions.csv")])
    t = pd.DataFrame([row(w, f, dec) for w, f, _ in ROWS])
    t["label"] = [lab for _, _, lab in ROWS]
    lines = []
    for _, r in t.iterrows():
        lines.append(f"{r.label} & {r.wm_bias:.2f} & {r.wm_reg_times:.1f} & {r.wm_passed}/{r.wm_applicable} & "
                     f"{r.wm_stock:.2f} & {r.lgb_bias:.2f} & {r.lgb_reg_times:.1f} & {r.lgb_passed}/{r.lgb_applicable} & "
                     f"{r.lgb_stock:.2f} \\\\")
    tab = r"""\begin{table*}[t]
\centering
\caption{\textbf{The world model's lift stays overstated when the generated world changes (added after the results,
plan amendment of 16 November 2026).} The world where merchants plan into high-demand weeks, regenerated with one
setting changed at a time; world model seed 1 and LightGBM, fitted and audited as in the main study. Lift / truth:
summed stated lift over the true lift on planned item-weeks. Regular price: stated response to a lasting
regular-price change over the true one. Checks: C1, C2, C4, C5 and C6 passed of those that apply (C3 needs three
seeds). Stock error: absolute difference between stated and true lift, summed, over the true lift.}
\label{tab:known-sens}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{World model} & \multicolumn{4}{c}{LightGBM} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
World & Lift / truth & Regular price & Checks & Stock error & Lift / truth & Regular price & Checks & Stock error \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (GEN / "tab_known_sens.tex").write_text(tab)
    s = t.iloc[1:]
    M = {
        "ksN": WORDS[len(s)],
        "ksWmBiasLo": f"{s.wm_bias.min():.2f}", "ksWmBiasHi": f"{s.wm_bias.max():.2f}",
        "ksWmOver": WORDS[int((s.wm_bias > 1.1).sum())],
        "ksWmRegLo": f"{s.wm_reg_times.min():.1f}", "ksWmRegHi": f"{s.wm_reg_times.max():.1f}",
        "ksLgbBiasLo": f"{s.lgb_bias.min():.2f}", "ksLgbBiasHi": f"{s.lgb_bias.max():.2f}",
        "ksLgbRegLo": f"{s.lgb_reg_times.min():.1f}", "ksLgbRegHi": f"{s.lgb_reg_times.max():.1f}",
        "ksWmPassedLo": str(int(s.wm_passed.min())), "ksWmPassedHi": str(int(s.wm_passed.max())),
        "ksWmApplicable": str(int(s.wm_applicable.max())),
        "ksWmStockLo": f"{s.wm_stock.min():.2f}", "ksWmStockHi": f"{s.wm_stock.max():.2f}",
    }
    base = dec[dec.world.isin(["clean", "confounded"])]
    for w, tag in (("clean", "Cl"), ("confounded", "Co")):
        b = base[base.world == w]
        wm = b[b.model.str.startswith("world_seed")]
        M[f"kd{tag}WmChoiceLo"], M[f"kd{tag}WmChoiceHi"] = f"{wm.choice.min():.3f}", f"{wm.choice.max():.3f}"
        M[f"kd{tag}WmStockLo"], M[f"kd{tag}WmStockHi"] = f"{wm.stock.min():.2f}", f"{wm.stock.max():.2f}"
        lg = b[b.model == "lightgbm"].iloc[0]
        M[f"kd{tag}LgbChoice"], M[f"kd{tag}LgbStock"] = f"{lg.choice:.3f}", f"{lg.stock:.2f}"
        M[f"kd{tag}Random"] = f"{b.choice_random.iloc[0]:.2f}"
        pb = b[b.model == "world_price_blind"].iloc[0]
        M[f"kd{tag}PriceBlindChoice"], M[f"kd{tag}PriceBlindStock"] = f"{pb.choice:.3f}", f"{pb.stock:.2f}"
    (GEN / "known_sens_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    print(t.round(3).to_string())
    print(M)


if __name__ == "__main__":
    main()
