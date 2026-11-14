"""Audit of paper 1's final tuned world models (plan amendment of 13 November 2026): pass rates by data set and
check, the zone-price comparison (C7), the plan read-back (C8) and checks against accuracy across the versions the
settings searches saved. Reads results/paper2/tuned/summary/ (scripts/paper2_tuned_collect.py) and writes
paper2/generated/tuned_*.tex and paper2/figs/fig_tuned_*.pdf.

Pass rules are those fixed on 13 October 2026, as in real.py; C1 is scored twice (the audit's nothing-planned plan,
as fixed, and the model's own, as amended 18 October); C3 for a final model is the rank agreement of lifts between
its five copies (amendment of 13 November).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import real
from style import ACCENT, FULL, GRID, INK, MUTED, despine, panel_label, save

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / "results" / "paper2" / "tuned" / "summary"
GEN = ROOT / "paper2" / "generated"
KEYS = {"dominicks_ana": "Analgesics", "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese",
        "dominicks_fre": "Frozen entrees", "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snacks",
        "frat": "Frat", "m5": "M5", "favorita": "Favorita"}
CHECKS = ["C1", "C1own", "C2", "C3", "C4", "C5", "C6"]
CHECK_LABEL = {"C1": "C1 (audit plan)", "C1own": "C1 (own plan)", "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5",
               "C6": "C6"}


def available() -> bool:
    return (SUM / "final" / "versions.csv").exists()


def key_of(config: str) -> str:
    for k in KEYS:
        if f"_{k}_" in f"_{config}_":
            return k
    return "other"


def final_scores() -> pd.DataFrame:
    v = pd.read_csv(SUM / "final" / "versions.csv")
    ag = pd.read_csv(SUM / "final" / "agreement.csv")
    d = v.copy()
    d["key"] = d["config"].map(key_of)
    audit = ["c1_zero_lift_when_nothing_planned", "c1_baseline_moves_with_plan", "c1_parts_do_not_add_up"]
    own = ["c1_zero_lift_own_plan", "c1_baseline_moves_own_plan", "c1_parts_do_not_add_up"]
    d["C1_v"], d["C1own_v"] = d[audit].max(axis=1), d[own].max(axis=1)
    d["C1"], d["C1own"] = d["C1_v"] < 1e-3, d["C1own_v"] < 1e-3
    d["C2_v"] = d["c2_lift_readout"]
    d["C2"] = d["C2_v"] < 0.5
    d["C3_v"] = d["c3_copy_lift"]
    d["C3"] = np.where(d["C3_v"].notna(), d["C3_v"] >= 0.7, np.nan)  # M5's final model is one copy
    d["C4_v"] = d["c4_share_lower_after_deeper_cut"]
    d["C4"] = d["C4_v"] < 0.01
    d["C5"] = d["c5_any_plan_bias"].between(0.8, 1.25) & (d["c5_any_plan_rank"] >= 0.3)
    a = ag.groupby(["model_config", "origin"])["total_lift_onoff"].mean()
    d["C6_v"] = [a.get((c, o), np.nan) for c, o in zip(d["config"], d["origin"])]
    d["C6"] = np.where(d["C6_v"].notna(), d["C6_v"] >= 0.5, np.nan)
    return d


def pass_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, name in KEYS.items():
        g = d[d["key"] == k]
        if not len(g):
            continue
        r = {"key": k, "data": name, "windows": len(g)}
        for c in CHECKS:
            x = g[c].astype(float)
            r[c] = float(x.mean()) if x.notna().any() else np.nan
        r["C2_v"], r["C3_v"], r["C6_v"] = g["C2_v"].mean(), g["C3_v"].mean(), g["C6_v"].mean()
        r["C5_bias"], r["C5_rank"] = g["c5_any_plan_bias"].mean(), g["c5_any_plan_rank"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def readback() -> pd.DataFrame:
    rows = []
    for f in sorted((SUM / "readback").glob("readback_*.json")):
        r = json.loads(f.read_text())
        k = f.stem.replace("readback_", "")
        for x in r["rows"]:
            if x.get("gain") is not None and np.isfinite(x["gain"]) and x.get("periods back") == 0:
                rows.append({"key": k, "input": x["plan input"], "before": x["from state before window"],
                             "rolled": x["from rolled state"], "gain": x["gain"]})
    return pd.DataFrame(rows)


SEARCHES = None


def _spec_of(config: str) -> str | None:
    global SEARCHES
    if SEARCHES is None:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from paper2_tuned_collect import SEARCHES as S
        SEARCHES = sorted(S, key=len, reverse=True)
    name = config[len("research_"):] if config.startswith("research_") else config
    for s in SEARCHES:
        if name.startswith(s + "_"):
            return s
    return None


def _window_group(spec: str) -> str:
    """Data and windows a search scored on: its base config's data set and its windows."""
    s = yaml.safe_load((ROOT / "configs" / "research" / f"{spec}.yaml").read_text())
    base = yaml.safe_load((ROOT / s["base"]).read_text())
    return json.dumps([base["dataset"], s["evaluation"]], sort_keys=True, default=str)


def search_test() -> list[dict]:
    v = pd.read_csv(SUM / "search" / "versions.csv")
    v["spec"] = v["config"].map(_spec_of)
    v = v[v["spec"].notna()].copy()
    groups = {s: _window_group(s) for s in v["spec"].unique()}
    v["dataset"] = v["config"].map(key_of)
    v["base"] = v["config"].map(real.base)
    d = real.score(v, pd.DataFrame(), pd.DataFrame())
    real_cat = real.category_of
    real.category_of = lambda c: groups[_spec_of(c)]  # windows by search, not by experiment config
    try:
        out = [real.within_window_test(d, col) for col in ("passed", "passed_no_c5", "passed_no_c1_c5")]
    finally:
        real.category_of = real_cat
    return out, d


def c7():
    z = pd.read_csv(SUM / "final" / "zones.csv")
    ep = pd.read_csv(ROOT / "results" / "paper2" / "zone_episodes.csv")
    keep = ep.loc[ep["scored"], ["category", "store", "first_week"]]
    z = z.merge(keep, on=["category", "store", "first_week"])
    from make_figures import ols_slope
    e = ep[ep["scored"]]
    m = ols_slope(e["estimate_8wk"], e["price_log_ratio"])
    rows = {"measured": m}
    for col, lab in (("world_implied", "world"), ("lightgbm_implied", "lightgbm")):
        rows[lab] = ols_slope(z[col], z["price_log_ratio"])
    return z, rows


def fig_checks(t: pd.DataFrame, rb: pd.DataFrame, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.2), gridspec_kw={"wspace": 0.55, "width_ratios": [1.5, 1]})
    m = t[CHECKS].to_numpy(dtype=float)
    a.imshow(np.where(np.isnan(m), 0.5, m), cmap="Blues", vmin=0, vmax=1.3, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a.text(j, i, "n/a" if np.isnan(m[i, j]) else f"{m[i, j]:.0%}".replace("%", ""), ha="center", va="center",
                   fontsize=6.5, color="white" if (not np.isnan(m[i, j]) and m[i, j] > 0.7) else INK)
    short = {"C1": "C1\naudit", "C1own": "C1\nown"}
    a.set_xticks(range(len(CHECKS)), [short.get(c, c) for c in CHECKS], fontsize=6.3)
    a.set_yticks(range(len(t)), t["data"], fontsize=7)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("Windows passing each check, %", fontsize=8)
    panel_label(a, "a")
    if len(rb):
        g = rb.groupby("key")[["before", "rolled"]].mean().reindex([k for k in KEYS if k in set(rb["key"])])
        y = np.arange(len(g))
        b.scatter(g["before"], y, s=14, color=MUTED, label="state before the window", zorder=3)
        b.scatter(g["rolled"], y, s=14, color=ACCENT, label="state rolled under the plan", zorder=3)
        for k in range(len(g)):
            b.plot([g["before"].iloc[k], g["rolled"].iloc[k]], [k, k], color=GRID, lw=1, zorder=1)
        b.set_yticks(y, [KEYS[k] for k in g.index], fontsize=7)
        b.set_ylim(len(g) - 0.5, -0.5)
        b.set_xlabel("Plan read back from the state (R squared)", fontsize=7)
        b.set_xlim(0, 1.05)
        b.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
        b.set_title("C8: the state carries the plan", fontsize=8)
        despine(b)
        panel_label(b, "b")
    fig.subplots_adjust(bottom=0.27)
    save(fig, out / "fig_tuned_checks.pdf")


def fig_c7(z, rows, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1.3, 1]})
    for col, lab, colr in (("world_implied", "World model", ACCENT), ("lightgbm_implied", "LightGBM", MUTED)):
        a.scatter(z["price_log_ratio"] * 100, z[col] * 100, s=12, color=colr, edgecolor="white", lw=0.3, label=lab)
    a.scatter(z["price_log_ratio"] * 100, z["did_estimate"] * 100, s=10, marker="x", color=INK, lw=0.7,
              label="measured")
    a.axhline(0, color=GRID, lw=0.8)
    a.axvline(0, color=GRID, lw=0.8)
    a.set_xlabel("Price against zone peers, %")
    a.set_ylabel("Change in category units, %")
    a.legend(fontsize=6, loc="upper right")
    a.set_title("Each episode", fontsize=8)
    despine(a)
    panel_label(a, "a")
    lab = {"measured": "Measured", "world": "World model", "lightgbm": "LightGBM"}
    for k, (name, (sl, se)) in enumerate(rows.items()):
        col = INK if name == "measured" else (ACCENT if name == "world" else MUTED)
        b.plot([sl - 1.96 * se, sl + 1.96 * se], [k, k], color=col, lw=1)
        b.scatter([sl], [k], s=16, color=col, zorder=3)
    b.axvline(0, color=INK, lw=0.5)
    b.set_yticks(range(len(rows)), [lab[n] for n in rows], fontsize=7)
    b.set_ylim(len(rows) - 0.4, -0.6)
    b.set_xlabel("Pooled slope, 95% interval")
    b.set_title("Pooled response", fontsize=8)
    b.tick_params(axis="y", length=0)
    despine(b, keep=("bottom",))
    panel_label(b, "b")
    fig.subplots_adjust(bottom=0.2)
    save(fig, out / "fig_tuned_c7.pdf")


def write(t, rb, tests, c7rows, nversions):
    def pct(x):
        return "n/a" if x != x else f"{100 * x:.0f}"

    lines = ["\\begin{tabular}{@{}lr" + "r" * len(CHECKS) + "@{}}", "\\toprule",
             "Data & Windows & " + " & ".join(CHECK_LABEL[c] for c in CHECKS) + " \\\\", "\\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r['data']} & {r['windows']} & " + " & ".join(pct(r[c]) for c in CHECKS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_tuned_pass.tex").write_text("\n".join(lines) + "\n")
    v = {}
    W = "none one two three four five six seven eight nine".split()
    n = len(t)
    v["tSets"] = W[n] if n < 10 else str(n)
    word = {"C1": "COne", "C1own": "COneOwn", "C2": "CTwo", "C3": "CThree", "C4": "CFour", "C5": "CFive",
            "C6": "CSix"}
    for c in CHECKS:
        full = int((t[c] == 1).sum())
        none = int((t[c] == 0).sum())
        v[f"t{word[c]}Full"] = W[full] if full < 10 else str(full)
        v[f"t{word[c]}None"] = W[none] if none < 10 else str(none)
        v[f"t{word[c]}Mean"] = pct(t[c].mean())
    v["tCThreeLo"], v["tCThreeHi"] = f"{t['C3_v'].min():.2f}", f"{t['C3_v'].max():.2f}"
    v["tCSixLo"], v["tCSixHi"] = f"{t['C6_v'].min():.2f}", f"{t['C6_v'].max():.2f}"
    if len(rb):
        g = rb.groupby("key")[["before", "rolled", "gain"]].mean()
        v["tReadGainLo"], v["tReadGainHi"] = f"{g['gain'].min():.2f}", f"{g['gain'].max():.2f}"
        v["tReadRolledLo"], v["tReadRolledHi"] = f"{g['rolled'].min():.2f}", f"{g['rolled'].max():.2f}"
    for name, (sl, se) in c7rows.items():
        tag = {"measured": "Meas", "world": "Wm", "lightgbm": "Lgb"}[name]
        v[f"tSlope{tag}"] = f"{sl:.2f}"
        v[f"tSlope{tag}Lo"], v[f"tSlope{tag}Hi"] = f"{sl - 1.96 * se:.2f}", f"{sl + 1.96 * se:.2f}"
    for r in tests:
        tag = {"passed": "All", "passed_no_c5": "NoFive", "passed_no_c1_c5": "NoOneFive"}[r["score"]]
        v[f"tRho{tag}"], v[f"tP{tag}"] = f"{r['rho']:.2f}", f"{r['p']:.3f}"
        v[f"tRho{tag}Lo"], v[f"tRho{tag}Hi"] = f"{r['lo']:.2f}", f"{r['hi']:.2f}"
        v[f"tConfigs{tag}"] = str(r["configs"])
    v["tSearchVersions"] = str(nversions)
    (GEN / "tuned_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in v.items()))
    t.to_csv(GEN / "tuned_pass.csv", index=False)
    pd.DataFrame(tests).to_csv(GEN / "tuned_search_test.csv", index=False)
    return v


def make(out: Path):
    d = final_scores()
    t = pass_table(d)
    rb = readback()
    z, c7rows = c7()
    tests, sd = search_test()
    fig_checks(t, rb, out)
    fig_c7(z, c7rows, out)
    v = write(t, rb, tests, c7rows, len(sd))
    c4 = d.groupby("key")["C4_v"].mean()
    dom = c4[[k for k in c4.index if k.startswith("dominicks")]]
    extra = {"tCFourDomLo": f"{100 * dom.min():.0f}", "tCFourDomHi": f"{100 * dom.max():.0f}",
             "tCFourMfive": f"{100 * c4.get('m5', np.nan):.0f}", "tCFourFav": f"{100 * c4.get('favorita', np.nan):.0f}",
             "tCFourFrat": f"{100 * c4.get('frat', np.nan):.2f}",
             "tCFiveRankMfive": f"{t.set_index('key').loc['m5', 'C5_rank']:.2f}",
             "tWindows": str(int(t["windows"].sum()))}
    with open(GEN / "tuned_macros.tex", "a") as f:
        f.write("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in extra.items()))
    return {**v, **extra}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    print(make(Path(__file__).parent))
