"""Paper 1: Dominick's 28 categories and where the six used sit among them (Karthik, 16 November 2026: reviewers will
ask why these six). Reads results/dominicks_eda/categories.csv (scripts/dominicks_eda.py) and
results/dominicks_eda/early_7oct.csv (the 7 October run of all 28 categories with the first, untuned settings, one
seed; results/README.md). Writes paper/generated/tab_dom_categories.tex and dom_cat_values.tex.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results" / "dominicks_eda"
WORDS = "none one two three four five six seven eight nine ten".split()
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}


def main():
    d = pd.read_csv(E / "categories.csv")
    e = pd.read_csv(E / "early_7oct.csv")
    d = d.merge(e, on="code")
    # the six's gap after tuning: final runs on the rolling windows (results/analysis_tuned/scores.csv)
    t = pd.read_csv(ROOT / "results" / "analysis_tuned" / "scores.csv")
    if "set" in t:
        t = t[t["set"] == "rolling"]
    t = t[t.key.str.startswith("dominicks") & ~t.level.isin(["cat_id", "cat_idxstore_id"])]
    t = t.groupby(["key", "model", "origin"]).wrmsse.mean().groupby(["key", "model"]).mean().unstack()
    tuned = (t.world_model - t.lightgbm).rename(lambda k: k.replace("dominicks_", ""))
    d["tuned_gap"] = d.code.map(tuned)
    d["gap"] = d.world_model - d.lightgbm
    d["gap_rank"] = d.gap.rank(ascending=False, method="min").astype(int)
    d = d.sort_values("gap", ascending=False)
    used = d[d.used]
    v = {
        "domCatN": str(len(d)), "domCatRest": str(int((~d.used).sum())),
        "domCatDollarShare": f"{100 * used.dollars_per_week.sum() / d.dollars_per_week.sum():.0f}",
        "domCatBehindTop": WORDS[int((used.gap_rank <= 7).sum())],
        "domCatEarlyLgbWins": str(int((d.lightgbm < d[["lift", "world_model"]].min(axis=1)).sum())),
        "domCatEarlyWmWins": str(int((d.world_model < d[["lift", "lightgbm"]].min(axis=1)).sum())),
        "domCatDealSix": f"{100 * used.share_units_on_deal.median():.0f}",
        "domCatTunedBehind": WORDS[int((used.tuned_gap > 0).sum())],
        "domCatTunedGapLo": f"{used.tuned_gap.min():+.3f}".replace("-", "$-$"),
        "domCatTunedGapHi": f"{used.tuned_gap.max():+.3f}".replace("-", "$-$"),
        "domCatEarlyGapHardLo": f"{used[used.gap > 0].gap.min():+.3f}",
        "domCatEarlyGapHardHi": f"{used[used.gap > 0].gap.max():+.3f}",
        "domCatDealRest": f"{100 * d[~d.used].share_units_on_deal.median():.0f}",
    }
    for code, tag in (("sna", "Sna"), ("che", "Che"), ("bjc", "Bjc"), ("sdr", "Sdr")):
        v[f"domCatRank{tag}"] = ORD.get(int(d.set_index("code").gap_rank[code]), str(int(d.set_index("code").gap_rank[code])))
    rows = []
    for _, r in d.iterrows():
        name = rf"\textbf{{{r.category}}}" if r.used else r.category
        line = (f"{name} & {r['items']:,} & {r.units_per_week / 1000:,.0f} & {r.dollars_per_week / 1000:,.0f} & "
                f"{r.units_per_item_store_week:.1f} & {100 * r.share_units_on_deal:.0f} & "
                f"{100 * r.share_price_changed:.0f} & {r.lightgbm:.3f} & {r.world_model:.3f} & {r.gap:+.3f} & "
                + (f"{r.tuned_gap:+.3f}" if r.used else ""))
        rows.append(line.replace("-0.", "$-$0.") + r" \\")
    tab = (r"""\begin{table*}[t]
\centering
\caption{\textbf{Dominick's 28 categories, the six used in bold.} Weeks the Kilts Center flags as reliable. Units and
dollars per week with sales, in thousands; units per item-store-week, median over item-store pairs; on deal: share of
units sold with a deal code; price changed: share of item-store weeks whose price differs from the previous recorded
week. First run: all 28 categories with the first, untuned settings (one seed, development windows), WRMSSE of LightGBM
and the world model and the world model's gap; rows in order of that gap; the six were chosen from this run, before
any tuning. Tuned gap: the same gap for the final tuned models on the rolling windows. Descriptive columns added in
\thisrev{} (scripts/dominicks\_eda.py).}
\label{tab:domcats}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{Per week (000)} & Units per & On deal & Price & \multicolumn{3}{c}{First run, untuned} & Tuned \\
\cmidrule(lr){3-4}\cmidrule(lr){8-10}
Category & Items & Units & Dollars & item-store-week & (\%) & changed (\%) & LGB & WM & Gap & gap \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    gen = ROOT / "paper" / "generated"
    (gen / "tab_dom_categories.tex").write_text(tab)
    (gen / "dom_cat_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(v.items())))
    print(v)


if __name__ == "__main__":
    main()
