"""Manuscript tables and the inline-number macros, from figs/rwdata.py.

  python paper/figs/make_tables.py

Writes paper/generated/tab_*.tex and paper/generated/macros.tex. Every number
the prose quotes from a run is a macro defined here, so the text cannot drift
from the run records.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean

import rwdata as D

GEN = Path(__file__).resolve().parents[1] / "generated"
GEN.mkdir(exist_ok=True)

MCODE = {"world": "World", "lift": "Lift", "lgb": "Lgb", "graph": "Graph", "tfm": "Tfm", "naive": "Naive",
         "tfm_plan": "TfmPlan"}
WCODE = {"mar": "Mar", "apr": "Apr", "practice": "Prac", "test": "Test"}
DCODE = {"frat": "Frat", "che": "Che", "sna": "Sna", "sdr": "Sdr", "bjc": "Bjc", "ana": "Ana", "fre": "Fre"}
METRIC = {"overall": "Overall", "total": "Total", "ixs": "Ixs", "fa": "Fa"}

macros: dict[str, str] = {}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def put(name, value):
    if name in macros:
        raise ValueError(f"duplicate macro {name}")
    macros[name] = value


def m5_mean(model, window, key):
    if model == "tfm_plan":
        return D.TFM_PLAN_M5[window].get(key)
    xs = D.m5(model, window)[key]
    return None if any(x is None for x in xs) else mean(xs)


def final_mean(model, data, key):
    if model == "tfm":
        return D.TFM_FINAL[data].get(key)
    if model == "tfm_plan":
        return D.TFM_PLAN_FINAL[data].get(key)
    if data not in D.FINAL_CONFIG.get(model, {}):
        return None
    return mean(D.final(model, data)[key])


# ------------------------------------------------------------- macros ---
def build_macros():
    for model in ["world", "lift", "lgb", "graph", "tfm", "naive"]:
        for w in D.M5_WINDOWS:
            s = D.m5(model, w)
            for key, mc in METRIC.items():
                v = m5_mean(model, w, key)
                if v is not None:
                    put(f"m{MCODE[model]}{WCODE[w]}{mc}", f3(v))
            put(f"m{MCODE[model]}{WCODE[w]}Seeds", str(len(s["overall"])))
            if len(s["overall"]) > 1:
                put(f"m{MCODE[model]}{WCODE[w]}OverallLo", f3(min(s["overall"])))
                put(f"m{MCODE[model]}{WCODE[w]}OverallHi", f3(max(s["overall"])))
                put(f"m{MCODE[model]}{WCODE[w]}IxsLo", f3(min(s["ixs"])))
                put(f"m{MCODE[model]}{WCODE[w]}IxsHi", f3(max(s["ixs"])))
    for w in D.M5_WINDOWS:
        put(f"mTfmPlan{WCODE[w]}Overall", f3(D.TFM_PLAN_M5[w]["overall"]))
        put(f"mTfmPlan{WCODE[w]}Ixs", f3(D.TFM_PLAN_M5[w]["ixs"]))
    for i, v in enumerate(D.m5("lift", "apr")["overall"], 1):
        put("mLiftAprSeed" + "ABC"[i - 1] + "Overall", f3(v))
    for i, v in enumerate(D.m5("lift", "apr")["total"], 1):
        put("mLiftAprSeed" + "ABC"[i - 1] + "Total", f3(v))
    for d in DCODE:
        for model in ["world", "lgb", "lift", "tfm", "tfm_plan"]:
            for key in ("overall", "ixs", "fa"):
                v = final_mean(model, d, key)
                if v is not None:
                    put(f"f{MCODE[model]}{DCODE[d]}{METRIC[key]}", f3(v))
            if model in ("world", "lgb", "lift") and d in D.FINAL_CONFIG.get(model, {}):
                s = D.final(model, d)
                put(f"f{MCODE[model]}{DCODE[d]}OverallLo", f3(min(s["overall"])))
                put(f"f{MCODE[model]}{DCODE[d]}OverallHi", f3(max(s["overall"])))
                if s["peaks"][0]:
                    put(f"f{MCODE[model]}{DCODE[d]}Promo",
                        f3(mean(p["promoted"]["forecast_to_actual"] for p in s["peaks"])))
                    put(f"f{MCODE[model]}{DCODE[d]}After",
                        f3(mean(p["after_promo"]["forecast_to_actual"] for p in s["peaks"])))
                    put(f"f{MCODE[model]}{DCODE[d]}Ordinary",
                        f3(mean(p["ordinary"]["forecast_to_actual"] for p in s["peaks"])))
            if model in ("tfm", "tfm_plan") and "promo_fa" in (D.TFM_FINAL if model == "tfm" else D.TFM_PLAN_FINAL)[d]:
                put(f"f{MCODE[model]}{DCODE[d]}Promo",
                    f3((D.TFM_FINAL if model == "tfm" else D.TFM_PLAN_FINAL)[d]["promo_fa"]))
        if d != "frat":
            share = D.final("lgb", d)["peaks"][0]["promoted"]["share_of_units"]
            put(f"f{DCODE[d]}PromoShare", f"{share * 100:.0f}")

    # ranges quoted in the prose
    promo = {m: [mean(p["promoted"]["forecast_to_actual"] for p in D.final(m, c)["peaks"]) for c in D.CATS]
             for m in ("world", "lgb")}
    tfm_p = [D.TFM_FINAL[c]["promo_fa"] for c in D.CATS]
    tfmp_p = [D.TFM_PLAN_FINAL[c]["promo_fa"] for c in D.CATS]
    put("promoTfmLo", f"{min(tfm_p) * 100:.0f}")
    put("promoTfmHi", f"{max(tfm_p) * 100:.0f}")
    put("promoTfmPlanLo", f"{min(tfmp_p) * 100:.0f}")
    put("promoTfmPlanHi", f"{max(tfmp_p) * 100:.0f}")
    for m in ("world", "lgb"):
        put(f"promo{MCODE[m]}Lo", f"{min(promo[m]) * 100:.0f}")
        put(f"promo{MCODE[m]}Hi", f"{max(promo[m]) * 100:.0f}")
    shares = [D.final("lgb", c)["peaks"][0]["promoted"]["share_of_units"] for c in D.CATS]
    put("promoShareLo", f"{min(shares) * 100:.0f}")
    put("promoShareHi", f"{max(shares) * 100:.0f}")
    # M5 TimesFM against the others, item by store
    gaps_w = [m5_mean("world", w, "ixs") - m5_mean("tfm", w, "ixs") for w in D.M5_WINDOWS]
    gaps_l = [m5_mean("lgb", w, "ixs") - m5_mean("tfm", w, "ixs") for w in D.M5_WINDOWS]
    put("tfmIxsGapWorldLo", f3(min(gaps_w)))
    put("tfmIxsGapWorldHi", f3(max(gaps_w)))
    put("tfmIxsGapLgbLo", f3(min(gaps_l)))
    put("tfmIxsGapLgbHi", f3(max(gaps_l)))
    fa = [m5_mean("tfm", w, "fa") for w in D.M5_WINDOWS]
    put("tfmUnderLo", f"{(1 - max(fa)) * 100:.0f}")
    put("tfmUnderHi", f"{(1 - min(fa)) * 100:.0f}")
    # recorded
    put("cutSold", f"{D.PRICE_CUT['sold']}")
    put("cutWorld", ", ".join(f"{v:,.0f}" for v in D.PRICE_CUT["world"]))
    put("cutLift", ", ".join(f"{v:,.0f}" for v in D.PRICE_CUT["lift"]))
    put("cutGraph", ", ".join(f"{v:,.0f}" for v in D.PRICE_CUT["graph"]))
    put("cutLgb", ", ".join(f"{v:.1f}" for v in D.PRICE_CUT["lgb"]))
    put("cutTfm", f"{D.PRICE_CUT['tfm'][0]:.1f}")
    put("liftSeedTwoForecast", f"{D.LIFT_APR_SEED2['forecast']:,}")
    put("liftSeedTwoSold", f"{D.LIFT_APR_SEED2['sold']:,}")
    put("mPublishedWinner", f3(D.M5_PUBLISHED["winner"]))
    put("mPublishedTenth", f3(D.M5_PUBLISHED["tenth"]))
    put("mPublishedFiftieth", f3(D.M5_PUBLISHED["fiftieth"]))
    put("mPublishedStat", f3(D.M5_PUBLISHED["best_stat"]))
    put("nPromotedRuns", str(sum(1 for _ in D.PROMOTED.iterdir() if (_ / "metrics.json").exists())))

    lines = ["% Generated by figs/make_tables.py from results/promoted and figs/rwdata.py. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {GEN / 'macros.tex'} ({len(macros)} macros)")


# ------------------------------------------------------------- tables ---
def bold_min(vals):
    real = [v for v in vals if v is not None]
    best = min(real) if real else None
    return [("\\textbf{%s}" % f3(v)) if v is not None and v == best else f3(v) for v in vals]


def tab_m5():
    models = ["world", "lift", "lgb", "graph", "tfm", "tfm_plan", "naive"]
    rows = {m: [] for m in models}
    for w in D.M5_WINDOWS:
        for key in ("overall", "total", "ixs"):
            col = [m5_mean(m, w, key) for m in models]
            for m, cell in zip(models, bold_min(col)):
                rows[m].append(cell)
    seeds = {m: "/".join(str(len(D.m5(m, w)["overall"])) for w in D.M5_WINDOWS) if m != "tfm_plan" else "1/1/1/1"
             for m in models}
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{lbl}}}}}" for lbl in
                      ["29 Feb to 27 Mar", "28 Mar to 24 Apr", "25 Apr to 22 May (practice)", "23 May to 19 Jun (test)"])
    sub = " & ".join(["All & Total & I$\\times$S"] * 4)
    body = "\n".join(f"{D.MODEL_LABEL[m]} & {seeds[m]} & " + " & ".join(rows[m]) + " \\\\" for m in models)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{On M5 the best model depends on the level scored and on the month.}} WRMSSE on the four 28-day
windows (lower is better), mean over seeds; seeds per window in the second column. All: the 12 levels together;
Total: the chain total; I$\\times$S: item by store. The lowest value in each column is in bold. TimesFM with the
plan has no recorded total. The lift model's April mean includes one failed seed (2.603 overall;
\\S\\ref{{sec:failures}}).}}
\\label{{tab:m5}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.6pt}}
\\begin{{tabular}}{{lc*{{12}}{{r}}}}
\\toprule
 & & {head} \\\\
\\cmidrule(lr){{3-5}}\\cmidrule(lr){{6-8}}\\cmidrule(lr){{9-11}}\\cmidrule(lr){{12-14}}
\\textbf{{Model}} & \\textbf{{Seeds}} & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "tab_m5.tex").write_text(tex)


def tab_finals():
    models = ["world", "lift", "lgb", "tfm", "tfm_plan"]
    names = {"frat": "Frat", **D.CAT_LABEL}
    lines = []
    for d in ["frat"] + D.CATS:
        cells = []
        for key in ("overall", "ixs"):
            cells += bold_min([final_mean(m, d, key) for m in models])
        lines.append(f"{names[d]} & " + " & ".join(cells) + " \\\\")
        if d == "frat":
            lines.append("\\addlinespace[2pt]\n\\multicolumn{11}{l}{\\itshape Dominick's} \\\\")
    head = " & ".join(["World & Lift & LightGBM & TimesFM & TimesFM+plan"] * 2)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{On the promotion-driven weekly data, models that see the plan lead, and which one leads
changes by category.}} WRMSSE on the final 8 weeks (lower is better). World model, lift model and LightGBM:
mean of three seeds, settings fixed on the development windows. TimesFM and TimesFM with the plan: one run each,
added after the final tests. The lift model was run on Frat only. The lowest value in each row and group is in bold.}}
\\label{{tab:finals}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l*{{10}}{{r}}}}
\\toprule
 & \\multicolumn{{5}}{{c}}{{\\textbf{{All levels}}}} & \\multicolumn{{5}}{{c}}{{\\textbf{{Item by store}}}} \\\\
\\cmidrule(lr){{2-6}}\\cmidrule(lr){{7-11}}
\\textbf{{Data}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "tab_finals.tex").write_text(tex)


def tab_ledger():
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in D.LEDGER)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{Every change and comparison model that did not become a default.}} Overall WRMSSE unless
stated; ``against'' gives the default it was compared with on the same window. dev.: development windows before
the final 8 weeks. Practice: the M5 window 25 April to 22 May 2016. Each row is a section of
\\texttt{{results/README.md}} with its run folders.}}
\\label{{tab:ledger}}
\\footnotesize
\\begin{{tabular}}{{p{{0.30\\textwidth}}p{{0.28\\textwidth}}p{{0.34\\textwidth}}}}
\\toprule
\\textbf{{Change or model}} & \\textbf{{Where tested}} & \\textbf{{Result}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "tab_ledger.tex").write_text(tex)


def main():
    build_macros()
    tab_m5()
    tab_finals()
    tab_ledger()
    print(f"  wrote tables to {GEN}")


if __name__ == "__main__":
    main()
