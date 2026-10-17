"""Revision tables and number lookups, from results/paper1_revision/tables.

  python paper/figs/make_revision.py

Writes paper/generated/rev_*.tex and paper/generated/rev_values.tex. The prose
quotes revision results only through the lookup commands defined in
rev_values.tex (\\rv, \\rpair, \\rab, ...), so every number traces to a row of
a committed result table; a missing value stops the LaTeX build.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results" / "paper1_revision" / "tables"
GEN = Path(__file__).resolve().parents[1] / "generated"
GEN.mkdir(exist_ok=True)

KEYS = ["dominicks_che", "dominicks_sna", "dominicks_sdr", "dominicks_bjc", "dominicks_ana",
        "dominicks_fre", "frat", "m5", "favorita"]
SHORT = {"dominicks_che": "che", "dominicks_sna": "sna", "dominicks_sdr": "sdr", "dominicks_bjc": "bjc",
         "dominicks_ana": "ana", "dominicks_fre": "fre", "frat": "frat", "m5": "mfive", "favorita": "fav"}
NAME = {"dominicks_che": "Cheese", "dominicks_sna": "Snack crackers", "dominicks_sdr": "Soft drinks",
        "dominicks_bjc": "Bottled juice", "dominicks_ana": "Analgesics", "dominicks_fre": "Frozen entrees",
        "frat": "Frat", "m5": "M5", "favorita": "Favorita"}
MODEL_LABEL = {"ensemble": "Ensemble", "world_model": "World model", "lift": "Lift model", "lightgbm": "LightGBM",
               "timesfm_plan": "TimesFM+plan", "timesfm_mean": "TimesFM", "chronos": "Chronos-Bolt",
               "tft": "TFT", "timestral_plan": "Timestral+plan", "seasonal_naive": "Seas.\\ naive"}
MCODE = {"ensemble": "ens", "world_model": "world", "lift": "lift", "lightgbm": "lgb", "timesfm_plan": "tfmplan",
         "timesfm_mean": "tfm", "chronos": "chronos", "tft": "tft", "timestral_plan": "tsplan",
         "timestral_mean": "tsmean", "seasonal_naive": "naive", "graph": "graph", "ensemble_timestral": "enstimestral",
         "ensemble_4": "ensfour", "timestral209_plan": "tsbig", "lift_stock": "liftstock"}
BOTTOM = "item_idxstore_id"


PROSE = {"lightgbm": "LightGBM", "lift": "the lift model", "world_model": "the world model",
         "timesfm_plan": "TimesFM with the plan", "ensemble": "the ensemble", "tft": "TFT", "chronos": "Chronos-Bolt",
         "timesfm_mean": "TimesFM", "seasonal_naive": "seasonal naive"}


def where(k: str) -> str:
    return f"in {lname(k)}" if k.startswith("dominicks") else f"on {NAME[k]}"


def join_and(xs) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


def lname(k: str) -> str:
    """Name for use inside a sentence: Dominick's categories in lower case."""
    return NAME[k].lower() if k.startswith("dominicks") else NAME[k]

vals: dict[str, str] = {}


def lv(level: str) -> str:
    return "ixs" if level == BOTTOM else ("total" if level == "total" else level.replace("_", ""))


def put(cmd: str, *keys, value) -> None:
    name = "@".join([cmd, *map(str, keys)])
    if name in vals and vals[name] != value:
        raise ValueError(f"conflicting value for {name}")
    vals[name] = value


def f3(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.3f}"
    return s.replace("-", "\\textminus ") if s.startswith("-") else s


def f2(x) -> str:
    s = f"{x:.2f}"
    return s.replace("-", "\\textminus ") if s.startswith("-") else s


def pct(x) -> str:
    v = round(100 * x)
    return f"{v:d}" if v >= 0 else f"\\textminus{{}}{-v:d}"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


# ------------------------------------------------------------- loading ---
def read_decisions() -> pd.DataFrame:
    """decisions.csv with the category-plan rows of the rolling windows replaced by
    the corrected cost in dollars (study G, category_cost.csv): the first cost let
    category-by-store nodes with no sales in a window dominate on Dominick's."""
    d = read("decisions.csv")
    d = d[~((d.set == "rolling") & (d.decision == "category_planning"))]
    cc = read("category_cost.csv")
    cc = cc[cc.variant == "dollars"].assign(set="rolling", decision="category_planning")
    return pd.concat([d, cc[["key", "set", "decision", "model", "cost", "windows", "ratio_to_naive"]]], ignore_index=True)


def rolling_accuracy() -> pd.DataFrame:
    """Mean WRMSSE over rolling windows, one row per data set, model and level."""
    a = read("accuracy.csv")
    a = a[a.set == "rolling"][["key", "model", "level", "wrmsse"]]
    fav = read("favorita_mcs.csv").rename(columns={"mean_wrmsse": "wrmsse"})[["key", "model", "level", "wrmsse"]]
    rows = [a, fav]
    # Timestral (study F): the paired table holds ensemble minus Timestral per data set and level; the ensemble's
    # own mean is in the tables above, so Timestral's mean follows (both are means over the same windows).
    sf = read("studyF_paired.csv")
    sf = sf[(sf.a == "ensemble") & sf.b.isin(["timestral_plan", "timestral_mean", "ensemble_timestral", "ensemble_4"])]
    base = pd.concat(rows)
    ens = base[base.model == "ensemble"].set_index(["key", "level"]).wrmsse
    t = sf.assign(wrmsse=[ens.get((k, l), np.nan) - d for k, l, d in zip(sf.key, sf.level, sf.mean_diff)])
    rows.append(t.rename(columns={"b": "model"})[["key", "model", "level", "wrmsse"]])
    return pd.concat(rows, ignore_index=True)


def wide(acc: pd.DataFrame, level: str) -> pd.DataFrame:
    return acc[acc.level == level].pivot_table(index="key", columns="model", values="wrmsse")


def all_levels(acc: pd.DataFrame) -> pd.DataFrame:
    """Mean over every level of the data set's hierarchy (the M5 score)."""
    a = acc[~acc.model.isin(["timestral_plan", "timestral_mean", "ensemble_timestral", "ensemble_4"])]
    return a.groupby(["key", "model"]).wrmsse.mean().unstack()


# ------------------------------------------------------------- values ----
def accuracy_values(acc):
    for level in [BOTTOM, "total"]:
        w = wide(acc, level)
        for k in w.index:
            for m in w.columns:
                v = w.loc[k, m]
                if not np.isnan(v):
                    put("rv", SHORT[k], MCODE[m], lv(level), value=f3(v))
    al = all_levels(acc)
    for k in al.index:
        for m in al.columns:
            if not np.isnan(al.loc[k, m]):
                put("rv", SHORT[k], MCODE[m], "all", value=f3(al.loc[k, m]))
    a = read("accuracy.csv")
    f = a[a.set == "final"]
    for (k, m, l), v in f.set_index(["key", "model", "level"]).wrmsse.items():
        if l in (BOTTOM, "total"):
            put("rf", SHORT[k], MCODE[m], lv(l), value=f3(v))
    for (k, m), v in f.groupby(["key", "model"]).wrmsse.mean().items():
        put("rf", SHORT[k], MCODE[m], "all", value=f3(v))


def mcs_sets():
    m = pd.concat([read("mcs.csv"), read("favorita_mcs.csv"), read("frat_mcs.csv")])
    m = m[m.level.isin([BOTTOM, "total"])]
    out = {}
    for (k, l), g in m.groupby(["key", "level"]):
        out[(k, l)] = set(g[g.in_set.astype(str) == "True"].model)
    return out


def paired_values():
    p = read("paired.csv")
    p = p[(p.set == "rolling") & p.level.isin([BOTTOM, "total"])]
    for r in p.itertuples():
        for fld in ("mean_diff", "lo", "hi"):
            put("rpair", SHORT[r.key], MCODE[r.a], MCODE[r.b], lv(r.level), fld.replace("_", ""), value=f3(getattr(r, fld)))
        put("rpair", SHORT[r.key], MCODE[r.a], MCODE[r.b], lv(r.level), "awins", value=str(int(r.a_better_windows)))
        put("rpair", SHORT[r.key], MCODE[r.a], MCODE[r.b], lv(r.level), "n", value=str(int(r.windows)))
    return p


def verdict(lo, hi):
    """+1: ensemble better (the variant's difference is above zero with its interval), -1: variant better."""
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def ablation_values():
    ab = pd.concat([read("ablation_paired.csv"), read("favorita_ablation_paired.csv")])
    ab = ab[ab.level.isin([BOTTOM, "total"])]
    for r in ab.itertuples():
        for fld in ("mean_diff", "lo", "hi"):
            put("rab", SHORT[r.key], r.a.replace("_", ""), lv(r.level), fld.replace("_", ""), value=f3(getattr(r, fld)))
    w = pd.concat([read("ablation_weights.csv"), read("favorita_ablation_weights.csv").assign(key="favorita")
                   if "key" not in read("favorita_ablation_weights.csv") else read("favorita_ablation_weights.csv")])
    g = w.groupby(["key", "target"])[["w_world_model", "w_lightgbm", "w_timesfm_plan"]].mean()
    for (k, t), r in g.iterrows():
        put("rw", SHORT[k], t.replace("_", ""), "world", value=f2(r.w_world_model))
        put("rw", SHORT[k], t.replace("_", ""), "lgb", value=f2(r.w_lightgbm))
        put("rw", SHORT[k], t.replace("_", ""), "tfmplan", value=f2(r.w_timesfm_plan))
    return ab


def other_values():
    d = read_decisions()
    for r in d[d.set == "rolling"].itertuples():
        if not np.isnan(r.ratio_to_naive):
            put("rdec", SHORT[r.key], r.decision.replace("_", "").replace(".", ""), MCODE[r.model], value=f2(r.ratio_to_naive))
    i = read("intervals_fixed.csv")
    i = i[(i.set == "rolling")]
    for r in i.itertuples():
        if r.level in (BOTTOM, "total"):
            for fld in ("spl", "coverage_50", "coverage_80", "coverage_95"):
                v = getattr(r, fld)
                if not np.isnan(v):
                    put("rint", SHORT[r.key], MCODE[r.model], lv(r.level), fld.replace("_", ""), value=f2(v))
        if r.level == "native_bottom" and not np.isnan(r.native_spl):
            put("rint", SHORT[r.key], MCODE[r.model], "native", "spl", value=f2(r.native_spl))
    old = read("intervals.csv")
    old = old[(old.set == "rolling") & (old.level.isin([BOTTOM, "total"]))]
    for r in old.itertuples():
        for fld in ("coverage_50", "coverage_80", "coverage_95", "spl"):
            v = getattr(r, fld, np.nan)
            if not np.isnan(v):
                put("rintold", SHORT[r.key], MCODE[r.model], lv(r.level), fld.replace("_", ""), value=f2(v))
    rc = read("reconcile.csv")
    for r in rc[(rc.set == "rolling") & rc.level.isin([BOTTOM, "total"])].itertuples():
        put("rrec", SHORT[r.key], r.method.replace("_", ""), lv(r.level), value=f3(r.wrmsse))
    pt = read("perturbation.csv")
    pt["data"] = pt.key.str.split("_").str[0]
    g = pt.groupby(["data", "model", "input", "operation"])[["bottom_change", "total_change", "units_change_on"]].mean()
    for (dset, m, inp, op), r in g.iterrows():
        dk = {"dominicks": "dom", "frat": "frat", "m5": "mfive"}[dset]
        put("rper", dk, MCODE[m], inp.replace("_", ""), op, "ixs", value=pct(r.bottom_change))
        put("rper", dk, MCODE[m], inp.replace("_", ""), op, "total", value=pct(r.total_change))
        if not np.isnan(r.units_change_on):
            put("rper", dk, MCODE[m], inp.replace("_", ""), op, "units", value=pct(r.units_change_on))
    me = read("mechanism.csv")
    for r in me.itertuples():
        k = SHORT.get(r.data, r.data)
        for fld in ["bottom_change", "total_change"] + [f"persist_h{h}" for h in range(1, 9)] + ["probe_before_r2", "probe_after_r2"]:
            v = getattr(r, fld)
            if not np.isnan(v):
                put("rmech", k, r.windows_set, MCODE[r.model], fld.replace("_", ""), value=f2(v) if "probe" in fld else pct(v))


def window_values():
    """Per-window scores of the saved world models (mechanism_weekly.csv): the worst and best window."""
    mw = pd.read_csv(T.parent / "mechanism_weekly.csv")
    mw = mw[mw.config.str.contains("rolling")]
    mw["k"] = mw.config.str.extract(r"dominicks_(\w{3})", expand=False)
    mw["model"] = np.where(mw.rollout, "world", "lift")
    for (k, m), g in mw.dropna(subset=["k"]).groupby(["k", "model"]):
        put("rwin", k, m, "ixsmax", value=f"{g.bottom_true.max():.2f}")
        put("rwin", k, m, "ixsmaxorigin", value=str(int(g.loc[g.bottom_true.idxmax(), "origin"]) + 1))
        put("rwin", k, m, "ixsmedian", value=f"{g.bottom_true.median():.2f}")
        put("rwin", k, m, "n", value=str(len(g)))


def regret_values():
    """Promotion-decision regret in paper 2's generated worlds (promo_regret.csv)."""
    r = pd.read_csv(T.parent / "promo_regret.csv")
    r = r[r.cost_percentile == 50]
    for w, g in r.groupby("world"):
        seeds = g[g.model.str.startswith("world_seed")].regret_share
        p1 = lambda x: f"{100 * x:.1f}"
        put("rreg", w, "world", "lo", value=p1(seeds.min()))
        put("rreg", w, "world", "hi", value=p1(seeds.max()))
        for m in ("lightgbm", "lightgbm_price_blind", "world_lever_blind", "world_price_blind"):
            v = g[g.model == m].regret_share.iloc[0]
            put("rreg", w, m.replace("_", ""), "v", value=p1(v) if v < 0.2 else pct(v))
        put("rreg", w, "n", "v", value=str(int(g.promotions.iloc[0])))


def studyf_values():
    acc = read("studyF_accuracy.csv")
    for r in acc.itertuples():
        for m in ["ensemble", "ensemble_4", "ensemble_timestral", "lightgbm", "timesfm_plan", "timestral209_plan",
                  "timestral_mean", "timestral_plan", "world_model"]:
            v = getattr(r, m)
            if not np.isnan(v):
                put("rsf", {"dominicks": "dom", "favorita": "fav", "frat": "frat", "m5": "mfive"}[r.data],
                    MCODE[m], lv(r.level), value=f3(v))
    tails = read("studyF_tails.csv")
    for (k, m), g in tails.groupby(["key", "model"]):
        for fld in ("forecast_ratio", "median_ratio", "q99_ratio"):
            fmt = (lambda x: f"{x:.0f}" if x >= 10 else f"{x:.1f}") if fld == "q99_ratio" else f2
            put("rtail", SHORT[k], MCODE[m], fld.replace("_ratio", ""), "lo", value=fmt(g[fld].min()))
            put("rtail", SHORT[k], MCODE[m], fld.replace("_ratio", ""), "hi", value=fmt(g[fld].max()))
        s = g.top1pct_share_of_excess.dropna()
        if len(s):
            put("rtail", SHORT[k], MCODE[m], "share", "lo", value=pct(s.min()))
            put("rtail", SHORT[k], MCODE[m], "share", "hi", value=pct(s.max()))
    rk = pd.read_csv(T / "studyF_ranks.csv", header=[0, 1], index_col=[0, 1, 2])
    rk.index.names = ["data", "level", "model"]
    for (dset, l, m), r in rk.iterrows():
        dk = {"dominicks": "dom", "favorita": "fav", "frat": "frat", "m5": "mfive"}[dset]
        for fld in ("spearman", "top_decile_hit", "promo_spearman", "total_ratio", "median_node_ratio"):
            v = r[(fld, "mean")]
            if not np.isnan(v):
                put("rrank", dk, MCODE[m], lv(l), fld.replace("_", ""), value=f2(v))


def pantry_values():
    p = read("pantry_stage2.csv")
    for r in p.itertuples():
        run = {"seed 1": "a", "seed 2": "b", "seed 3": "c", "mean of 3": "mean", "rolling run": "roll"}[r.run]
        for fld in ("overall", "total", "item_by_store", "after_promo_fa", "after_promo_err"):
            put("rpan", r.windows[:3], r.category, r.variant.replace("_", ""), run, fld.replace("_", ""),
                value=f3(getattr(r, fld)) if fld != "after_promo_fa" else f2(getattr(r, fld)))
    e = read("afterpromo_stage1.csv")
    for r in e.itertuples():
        for fld in ("overall", "after_promo_fa", "after_promo_err"):
            put("raft", r.category, r.variant.replace("_", ""), fld.replace("_", ""),
                value=f3(getattr(r, fld)) if fld != "after_promo_fa" else f2(getattr(r, fld)))


# ------------------------------------------------------------- counts ----
def decision_counts(word):
    d = read_decisions()
    d = d[(d.set == "rolling") & d.model.isin(["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan"])]
    for dec, code in (("category_planning", "cat"), ("replenish_0.8", "order")):
        best = d[d.decision == dec].loc[lambda x: x.groupby("key").ratio_to_naive.idxmin()]
        for m in ("world_model", "ensemble", "lightgbm", "lift"):
            put("rcount", f"dec{MCODE[m]}{code}", value=word[int((best.model == m).sum())])
            put("rcount", f"dec{MCODE[m]}{code}list", value=join_and(lname(k) for k in best[best.model == m].key) or "none")
    cp = d[d.decision == "category_planning"].pivot_table(index="key", columns="model", values="ratio_to_naive")
    put("rcount", "worldbeatsliftcat", value=word[int((cp.world_model < cp.lift).sum())])
    a = read("accuracy.csv")
    f = a[(a.set == "final") & (a.level == BOTTOM) & a.model.isin(["ensemble", "world_model", "lift", "lightgbm",
                                                                     "timesfm_plan", "chronos", "tft"])]
    gap = (f[f.model == "ensemble"].set_index("key").wrmsse - f.groupby("key").wrmsse.min()).max()
    put("rcount", "finalsgap", value=f"{np.ceil(gap * 100) / 100:.2f}")


def count_values(acc, sets, p, ab):
    """Counts the prose states ("in seven of nine data sets"), computed here."""
    word = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    decision_counts(word)
    bottom_in = [k for k in KEYS if "ensemble" in sets.get((k, BOTTOM), set())]
    bottom_alone = [k for k in KEYS if sets.get((k, BOTTOM), set()) == {"ensemble"}]
    total_in = [k for k in KEYS if "ensemble" in sets.get((k, "total"), set())]
    put("rcount", "ensinsetixs", value=word[len(bottom_in)])
    put("rcount", "ensaloneixs", value=word[len(bottom_alone)])
    put("rcount", "ensinsettotal", value=word[len(total_in)])
    put("rcount", "ensaloneixslist", value=join_and(lname(k) for k in bottom_alone))
    put("rcount", "ensnotinsetixs", value=join_and(lname(k) for k in KEYS if k not in bottom_in) or "none")
    # lowest mean item-by-store error among single models and the ensemble
    w = wide(acc, BOTTOM)[["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan", "timesfm_mean", "chronos", "tft",
                           "seasonal_naive"]]
    best = w.idxmin(axis=1)
    put("rcount", "enslowestixs", value=word[int((best == "ensemble").sum())])
    put("rcount", "enslowestixsnot", value=", ".join(f"{lname(k)} ({MODEL_LABEL[best[k]]})" for k in best.index if best[k] != "ensemble"))
    wt = wide(acc, "total")[w.columns]
    bt = wt.idxmin(axis=1)
    put("rcount", "enslowesttotal", value=word[int((bt == "ensemble").sum())])
    put("rcount", "lowesttotallist", value="; ".join(f"{PROSE[bt[k]]} {where(k)}" for k in bt.index if bt[k] != "ensemble"))
    # ensemble against LightGBM and the world model, item by store and total (rolling windows)
    for b in ("lightgbm", "world_model", "tft", "timesfm_mean", "seasonal_naive"):
        for level in (BOTTOM, "total"):
            q = p[(p.a == "ensemble") & (p.b == b) & (p.level == level)]
            better = [r.key for r in q.itertuples() if r.hi < 0]
            worse = [r.key for r in q.itertuples() if r.lo > 0]
            put("rcount", f"ensbeats{MCODE[b]}{lv(level)}", value=word[len(better)])
            put("rcount", f"ensbeats{MCODE[b]}{lv(level)}list", value=join_and(lname(k) for k in better) or "none")
            put("rcount", f"enslosesto{MCODE[b]}{lv(level)}", value=word[len(worse)])
            put("rcount", f"ensn{MCODE[b]}{lv(level)}", value=word[len(q)])
    for a_, b_ in (("lightgbm", "world_model"), ("lift", "world_model"), ("timesfm_plan", "world_model"),
                   ("lightgbm", "timesfm_plan"), ("lightgbm", "tft")):
        for level in (BOTTOM, "total"):
            q = p[(p.a == a_) & (p.b == b_) & (p.level == level)]
            put("rcount", f"{MCODE[a_]}beats{MCODE[b_]}{lv(level)}", value=word[int((q.hi < 0).sum())])
            put("rcount", f"{MCODE[a_]}beats{MCODE[b_]}{lv(level)}list", value=join_and(lname(k) for k in q[q.hi < 0].key) or "none")
            put("rcount", f"{MCODE[b_]}beats{MCODE[a_]}{lv(level)}", value=word[int((q.lo > 0).sum())])
            put("rcount", f"{MCODE[b_]}beats{MCODE[a_]}{lv(level)}list", value=join_and(lname(k) for k in q[q.lo > 0].key) or "none")
            put("rcount", f"{MCODE[a_]}{MCODE[b_]}{lv(level)}n", value=word[len(q)])
    for level in (BOTTOM, "total"):
        w_ = set(ab[(ab.a == "without_world_model") & (ab.level == level) & (ab.lo > 0)].key)
        l_ = set(ab[(ab.a == "lift_for_world_model") & (ab.level == level) & (ab.lo > 0)].key)
        put("rcount", f"bothadd{lv(level)}", value=word[len(w_ & l_)])
        put("rcount", f"bothadd{lv(level)}list", value=join_and(lname(k) for k in KEYS if k in w_ & l_) or "none")
        put("rcount", f"addsnotlift{lv(level)}list", value=join_and(lname(k) for k in KEYS if k in w_ - l_) or "none")
        put("rcount", f"liftonly{lv(level)}list", value=join_and(lname(k) for k in KEYS if k in l_ - w_) or "none")
    rc = read("reconcile.csv")
    rc = rc[(rc.set == "rolling") & (rc.level == "total")]
    dom = [k for k in KEYS if k.startswith("dominicks")]
    worse = 0
    for k in dom:
        r_ = rc[(rc.key == k) & (rc.method == "lightgbm_with_ets_wls_var")].wrmsse.iloc[0]
        u_ = acc[(acc.key == k) & (acc.model == "lightgbm") & (acc.level == "total")].wrmsse.iloc[0]
        worse += r_ > u_
    put("rcount", "reclgbworsedom", value=word[int(worse)])
    tt = read("studyF_timestral_vs_timesfm.csv")
    for form, code in (("timestral_plan", "plan"), ("timestral_mean", "mean")):
        q = tt[(tt.a == form) & tt.level.isin([BOTTOM, "total"])]
        put("rcount", f"tfmbeatsts{code}", value=str(int((q.lo > 0).sum())))
        put("rcount", f"tsbeatstfm{code}", value=word.get(int((q.hi < 0).sum()), str(int((q.hi < 0).sum()))))
        put("rcount", f"tsvstfm{code}n", value=str(len(q)))
        put("rcount", f"tsvstfm{code}level", value=join_and(f"the {lv(r.level).replace('ixs', 'item-by-store level').replace('total', 'total')} {where(r.key)}"
                                                       for r in q.itertuples() if r.lo <= 0 <= r.hi) or "none")
    i = read("intervals_fixed.csv")
    i = i[i.set == "rolling"]
    nat = i[(i.level == "native_bottom") & (i.model == "tft")].set_index("key").native_spl
    ens = i[(i.level == BOTTOM) & (i.model == "ensemble")].set_index("key").spl.reindex(nat.index)
    put("rcount", "tftnativebetter", value=word[int((nat < ens).sum())])
    put("rcount", "tftnativen", value=word[len(nat)])
    fit = ab[ab.a.str.startswith("fitted") & ab.level.isin([BOTTOM, "total"])]
    put("rcount", "fittedwins", value=word.get(int((fit.hi < 0).sum()), str(int((fit.hi < 0).sum()))))
    put("rcount", "fittedlosses", value=str(int((fit.lo > 0).sum())))
    put("rcount", "fittedn", value=str(len(fit)))
    # what each member adds: the ensemble beats the pair without it
    for variant in ("without_world_model", "without_lightgbm", "without_timesfm", "lift_for_world_model",
                    "fitted_all_levels", "fitted_bottom", "fitted_total"):
        for level in (BOTTOM, "total"):
            q = ab[(ab.a == variant) & (ab.level == level)]
            adds = [r.key for r in q.itertuples() if r.lo > 0]
            hurts = [r.key for r in q.itertuples() if r.hi < 0]
            v = variant.replace("_", "")
            put("rcount", f"{v}{lv(level)}ens", value=word[len(adds)])
            put("rcount", f"{v}{lv(level)}enslist", value=join_and(lname(k) for k in adds) or "none")
            put("rcount", f"{v}{lv(level)}var", value=word[len(hurts)])
            put("rcount", f"{v}{lv(level)}varlist", value=join_and(lname(k) for k in hurts) or "none")


# ------------------------------------------------------------- tables ----
def bold_min(cells):
    real = [c for c in cells if c is not None and not np.isnan(c)]
    lo = min(real) if real else None
    return lo


def tab_rolling(acc, sets):
    models = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan", "timesfm_mean", "chronos", "tft",
              "timestral_plan", "seasonal_naive"]
    lines = []
    for level, title in [(BOTTOM, "Item by store"), ("total", "Total")]:
        w = wide(acc, level).reindex(index=KEYS, columns=models)
        lines.append(f"\\addlinespace[2pt]\\multicolumn{{{len(models) + 2}}}{{l}}{{\\itshape {title}}} \\\\")
        for k in KEYS:
            row = w.loc[k]
            lo = bold_min(list(row.values))
            inset = sets.get((k, level), set())
            cells = []
            for m in models:
                v = row[m]
                s = f3(v)
                if s != "--" and v == lo:
                    s = f"\\textbf{{{s}}}"
                if m in inset:
                    s += "$^{\\circ}$"
                cells.append(s)
            n = {"frat": 4, "m5": 4, "favorita": 4}.get(k, 6)
            lines.append(f"{NAME[k]} & {n} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\rotatebox{{70}}{{{MODEL_LABEL[m]}}}" for m in models)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{Over the rolling windows the equal-weight ensemble has the lowest item-by-store error on
\\rcount{{enslowestixs}} of nine data sets, is in the 90\\% model confidence set item by store on \\rcount{{ensinsetixs}}
and at the total on \\rcount{{ensinsettotal}}.}} Mean WRMSSE over the rolling
windows (lower is better); W: windows. Bold: the lowest in the row. $^{{\\circ}}$: in the 90\\% model confidence set of
the row (Hansen, Lunde and Nason, range statistic, bootstrap over windows; Timestral, added in study F, is not
in the comparison). Ensemble: the equal-weight mean of the world model, LightGBM and TimesFM with the plan,
fixed before any rolling result. Trained models are one fit each (\\S\\ref{{sec:setup}}). The Temporal Fusion Transformer was not run on M5
or Favorita, Chronos-Bolt, TimesFM and TFT not on Favorita. Timestral+plan is Timestral-28M with the plan (study F).}}
\\label{{tab:rolling}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.4pt}}
\\begin{{tabular}}{{lc*{{{len(models)}}}{{r}}}}
\\toprule
\\textbf{{Data}} & \\textbf{{W}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_rolling.tex").write_text(tex)


def cell(r):
    if r is None:
        return "--"
    s = f3(r.mean_diff)
    if r.lo > 0:
        return f"\\textbf{{{s}}}"
    if r.hi < 0:
        return f"\\underline{{{s}}}"
    return s


def tab_ablation(ab):
    variants = [("without_world_model", "$-$World"), ("without_lightgbm", "$-$LGB"),
                ("without_timesfm", "$-$TFM+p"), ("lift_for_world_model", "Lift"),
                ("fitted_all_levels", "Fitted")]
    lines = []
    for k in KEYS:
        cells = []
        for level in (BOTTOM, "total"):
            for v, _ in variants:
                q = ab[(ab.key == k) & (ab.a == v) & (ab.level == level)]
                cells.append(cell(next(q.itertuples(), None)))
        lines.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(lbl for _, lbl in variants)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{Removing the world model makes the ensemble less accurate item by store on \\rcount{{withoutworldmodelixsens}}
data sets and at the total on \\rcount{{withoutworldmodeltotalens}}; replacing it with the lift model, the same network
without the carried state, does so on \\rcount{{liftforworldmodelixsens}} and \\rcount{{liftforworldmodeltotalens}}.}}
Mean WRMSSE of each variant minus the ensemble's over the rolling windows (study A, added after the rolling results);
above zero the ensemble is more accurate. Bold: the ensemble is more accurate and the 95\\% interval from the bootstrap
over windows excludes zero; underlined: the variant is, with the same test. No X: the equal-weight mean of the other two
members ($-$World: without the world model; $-$LGB: without LightGBM; $-$TFM+p: without TimesFM with the plan). Lift:
the world model replaced by the lift model. Fitted: weights chosen on the earlier
windows of the data set (lowest mean WRMSSE over all levels), scored on the next, so one window fewer.}}
\\label{{tab:ablation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.2pt}}
\\begin{{tabular}}{{l*{{10}}{{r}}}}
\\toprule
 & \\multicolumn{{5}}{{c}}{{\\textbf{{Item by store}}}} & \\multicolumn{{5}}{{c}}{{\\textbf{{Total}}}} \\\\
\\cmidrule(lr){{2-6}}\\cmidrule(lr){{7-11}}
\\textbf{{Data}} & {head} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_ablation.tex").write_text(tex)


def tab_decisions():
    d = read_decisions()
    d = d[d.set == "rolling"]
    models = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan"]
    lines = []
    for k in KEYS:
        if k not in set(d.key):
            continue
        cells = []
        for dec in ("category_planning", "replenish_0.8"):
            q = d[(d.key == k) & (d.decision == dec)].set_index("model").ratio_to_naive.reindex(models)
            lo = q.min()
            cells += [(f"\\textbf{{{f2(v)}}}" if v == lo else f2(v)) for v in q.values]
        lines.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(["Ens. & World & Lift & LGB & TFM+p"] * 2)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{For category plans and for store orders the ensemble or LightGBM has the lowest cost on every
data set except the M5 category plan.}} Cost relative to seasonal naive (below 1 is better), mean over the rolling
windows. Category plan: absolute error of each category-by-store total in each period, in dollars, over sales in
dollars (corrected in study G; Table~\\ref{{tab:catcost}}). Store orders: order the model's 80\\% interval
quantile for each item, store and period; each unit over costs 1 and each unit short costs 4. Simulations under
stated costs, not business results. Bold: the lowest in the group.}}
\\label{{tab:decisions}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.6pt}}
\\begin{{tabular}}{{l*{{10}}{{r}}}}
\\toprule
 & \\multicolumn{{5}}{{c}}{{\\textbf{{Category plan}}}} & \\multicolumn{{5}}{{c}}{{\\textbf{{Store orders (ratio 0.8)}}}} \\\\
\\cmidrule(lr){{2-6}}\\cmidrule(lr){{7-11}}
\\textbf{{Data}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_decisions.tex").write_text(tex)


def tab_perturbation():
    pt = read("perturbation.csv")
    pt["data"] = pt.key.str.split("_").str[0]
    g = pt.groupby(["data", "model", "input", "operation"])[["bottom_change", "total_change", "units_change_on"]].mean()
    rows = [("dominicks", "Dominick's", ["whole_plan", "price", "promo"]),
            ("frat", "Frat", ["whole_plan", "price", "promo", "feature", "display", "tpr_only"]),
            ("m5", "M5", ["whole_plan", "price", "event", "snap"])]
    lab = {"whole_plan": "Whole plan", "price": "Price", "promo": "Promotion", "feature": "Feature",
           "display": "Display", "tpr_only": "Price reduction only", "event": "Event days", "snap": "SNAP days"}
    models = ["world_model", "lift", "lightgbm"]
    lines = []
    for dset, dname, inputs in rows:
        lines.append(f"\\addlinespace[2pt]\\multicolumn{{10}}{{l}}{{\\itshape {dname}}} \\\\")
        for inp in inputs:
            cells = []
            for fld in ("bottom_change", "total_change", "units_change_on"):
                for m in models:
                    try:
                        v = g.loc[(dset, m, inp, "occlude"), fld]
                    except KeyError:
                        v = np.nan
                    cells.append("--" if np.isnan(v) else pct(v).replace("-", "\\textminus "))
            lines.append(f"{lab[inp]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(["World & Lift & LGB"] * 3)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{All three trained models use the plan. On the weekly data price carries most of it; on M5 the world
model relies on event and SNAP days more than LightGBM does.}} Input occlusion on the saved models (study B): one plan input is set to
``nothing planned'' (regular price, lever off) in the forecast periods and the forecast recomputed, with no refitting.
Change in WRMSSE item by store and at the total, in percent (above zero: the forecast gets worse without the input),
and the change in forecast units, in percent, in the periods where the input was on. Mean over the rolling windows (LightGBM on M5: the
last two).}}
\\label{{tab:perturbation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.6pt}}
\\begin{{tabular}}{{l*{{9}}{{r}}}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Item by store (\\%)}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Total (\\%)}}}}
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Units where on (\\%)}}}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}
\\textbf{{Input occluded}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_perturbation.tex").write_text(tex)


def tab_intervals():
    i = read("intervals_fixed.csv")
    i = i[i.set == "rolling"]
    models = ["ensemble", "world_model", "lightgbm"]
    lines = []
    for k in KEYS:
        cells = []
        for level in (BOTTOM, "total"):
            q = i[(i.key == k) & (i.level == level)].set_index("model")
            for m in models:
                cells += [f2(q.loc[m, "spl"]), f2(q.loc[m, "coverage_80"]), f2(q.loc[m, "coverage_95"])]
        lines.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    sub = " & ".join(["SPL & 80 & 95"] * 6)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{Rolling conformal intervals cover close to their level item by store, from 0.72 to 1.00 at the
95\\% level at the total, and their pinball losses item by store are within 0.03 of each other.}} Scaled pinball loss (SPL, lower is better) and the share of actual sales
inside the 80\\% and 95\\% intervals, mean over the rolling windows after the first. Intervals from each model's own
scaled errors in the earlier windows of the data set, pooled over horizons where a level has fewer than 50 earlier
errors per horizon, ends rounded outwards to whole units except on Favorita (interval check of 16 October).}}
\\label{{tab:intervals}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.6pt}}
\\begin{{tabular}}{{l*{{18}}{{r}}}}
\\toprule
 & \\multicolumn{{9}}{{c}}{{\\textbf{{Item by store}}}} & \\multicolumn{{9}}{{c}}{{\\textbf{{Total}}}} \\\\
\\cmidrule(lr){{2-10}}\\cmidrule(lr){{11-19}}
 & \\multicolumn{{3}}{{c}}{{Ensemble}} & \\multicolumn{{3}}{{c}}{{World}} & \\multicolumn{{3}}{{c}}{{LightGBM}}
 & \\multicolumn{{3}}{{c}}{{Ensemble}} & \\multicolumn{{3}}{{c}}{{World}} & \\multicolumn{{3}}{{c}}{{LightGBM}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}\\cmidrule(lr){{11-13}}\\cmidrule(lr){{14-16}}\\cmidrule(lr){{17-19}}
\\textbf{{Data}} & {sub} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_intervals.tex").write_text(tex)


def tab_reconcile():
    rc = read("reconcile.csv")
    rc = rc[(rc.set == "rolling") & rc.level.isin([BOTTOM, "total"])]
    acc = rolling_accuracy()
    methods = [("ets_bottom_up", "ETS, bottom-up"), ("ets_top_down", "ETS, top-down"), ("ets_ols", "ETS, OLS"),
               ("ets_wls_struct", "ETS, WLS structural"), ("ets_wls_var", "ETS, WLS variance"),
               ("ets_mint_shrink", "ETS, MinT shrink"), ("lightgbm_with_ets_wls_var", "LightGBM + ETS, WLS variance"),
               ("world_model_with_ets_wls_var", "World model + ETS, WLS variance")]
    lines = []
    dom = [k for k in KEYS if k.startswith("dominicks")]

    def val(frame, col, k, level, what):
        if k == "dom":
            q = frame[frame.key.isin(dom) & (frame[col] == what) & (frame.level == level)].wrmsse
            return q.mean() if len(q) == len(dom) else None
        q = frame[(frame.key == k) & (frame[col] == what) & (frame.level == level)].wrmsse
        return q.iloc[0] if len(q) else None

    for m, lbl in methods:
        cells = [f3(val(rc, "method", k, level, m)) for k in ("dom", "frat", "m5") for level in (BOTTOM, "total")]
        lines.append(f"{lbl} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for m in ("lightgbm", "world_model", "ensemble"):
        cells = [f3(val(acc, "model", k, level, m)) for k in ("dom", "frat", "m5") for level in (BOTTOM, "total")]
        lines.append(f"{MODEL_LABEL[m]}, unreconciled & " + " & ".join(cells) + " \\\\")
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{Reconciled ETS forecasts are far behind the trained models; reconciling a trained model's item
forecasts with ETS above them helps at the Frat total and hurts on Dominick's and M5.}} Mean WRMSSE over the rolling
windows (Dominick's: mean of the six categories). MinT with a shrunk
covariance was computed only where the hierarchy has at most 6,000 nodes (Frat), after the fix for nodes with no
in-sample error; its first run gave a total WRMSSE of \\rrec{{frat}}{{etsmintshrinkbeforefix}}{{total}}
(\\S\\ref{{sec:res-recon}}).}}
\\label{{tab:reconcile}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l*{{6}}{{r}}}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{\\textbf{{Dominick's}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Frat}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{M5}}}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}
\\textbf{{Method}} & I$\\times$S & Total & I$\\times$S & Total & I$\\times$S & Total \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_reconcile.tex").write_text(tex)


def tab_finals_new():
    a = read("accuracy.csv")
    f = a[(a.set == "final") & a.level.isin([BOTTOM, "total"])]
    models = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan", "chronos", "tft"]
    lines = []
    for k in [k for k in KEYS if k in set(f.key)]:
        cells = []
        for level in (BOTTOM, "total"):
            q = f[(f.key == k) & (f.level == level)].set_index("model").wrmsse.reindex(models)
            lo = q.min()
            cells += [(f"\\textbf{{{f3(v)}}}" if v == lo else f3(v)) for v in q.values]
        lines.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(["Ens. & World & Lift & LGB & TFM+p & Chr. & TFT"] * 2)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{On the single final windows the ensemble is lowest or within \\rcount{{finalsgap}} of the lowest item by
store on every data set, and the order at the total changes from data set to data set.}} WRMSSE on the final
window of each data set (lower is better; on M5 the official test month). One fit per trained model, so the world
model and LightGBM differ slightly from the three-seed means of Tables~\\ref{{tab:m5}} and~\\ref{{tab:finals}}. Bold:
lowest in the group.}}
\\label{{tab:finalsnew}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l*{{14}}{{r}}}}
\\toprule
 & \\multicolumn{{7}}{{c}}{{\\textbf{{Item by store}}}} & \\multicolumn{{7}}{{c}}{{\\textbf{{Total}}}} \\\\
\\cmidrule(lr){{2-8}}\\cmidrule(lr){{9-15}}
\\textbf{{Data}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_finalsnew.tex").write_text(tex)


def tab_timestral():
    acc = read("studyF_accuracy.csv")
    models = [("ensemble", "Ens."), ("ensemble_timestral", "Ens.\\,T"), ("ensemble_4", "Ens.\\,4"),
              ("timesfm_plan", "TFM+p"), ("timestral_plan", "TS+p"), ("timestral_mean", "TS")]
    lines = []
    for dset, dname in [("dominicks", "Dominick's (six)"), ("frat", "Frat"), ("m5", "M5"), ("favorita", "Favorita")]:
        cells = []
        for level in (BOTTOM, "total"):
            q = acc[(acc.data == dset) & (acc.level == level)].iloc[0]
            vs = [q[m] for m, _ in models]
            lo = min(vs)
            cells += [(f"\\textbf{{{f3(v)}}}" if v == lo else f3(v)) for v in vs]
        lines.append(f"{dname} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(lbl for _, lbl in models)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{Timestral, released two days after this revision began, is behind the ensemble on every data set, and
putting it in place of TimesFM helps only at the M5 total.}} Mean WRMSSE over the rolling windows (Dominick's: mean
of the six categories). Timestral-28M, zero-shot, the mean of its 21 quantiles as the point forecast; with the plan as
known future inputs (TS+p) and from history alone (TS); TFM+p: TimesFM with the plan. Ens.: the ensemble; Ens.\\,T:
Timestral with the plan in place of TimesFM with the plan; Ens.\\,4: all four together. Study F, written down before it ran.}}
\\label{{tab:timestral}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l*{{12}}{{r}}}}
\\toprule
 & \\multicolumn{{6}}{{c}}{{\\textbf{{Item by store}}}} & \\multicolumn{{6}}{{c}}{{\\textbf{{Total}}}} \\\\
\\cmidrule(lr){{2-7}}\\cmidrule(lr){{8-13}}
\\textbf{{Data}} & {head} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_timestral.tex").write_text(tex)


# ------------------------------------------------------------- grain ----
GRAIN = [("ixs", "Item by store", "Store orders and inventory", {"*": "item_idxstore_id"}),
         ("item", "Item, all stores", "Supplier funding and deals", {"*": "item_id"}),
         ("catstore", "Category by store", "Store-level category performance",
          {"dom": "cat_idxstore_id", "frat": "cat_idxstore_id", "m5": "store_idxcat_id", "favorita": "store_idxfamily"}),
         ("cat", "Category, all stores", "Category sales and margin plans",
          {"dom": "cat_id", "frat": "cat_id", "m5": "cat_id", "favorita": "family"}),
         ("total", "Chain total", "Financial plan", {"*": "total"})]
GRAIN_MODELS = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan", "timesfm_mean", "chronos", "tft",
                "seasonal_naive"]
SHORTM = {"ensemble": "Ens.", "world_model": "World", "lift": "Lift", "lightgbm": "LGB", "timesfm_plan": "TFM+p",
          "timesfm_mean": "TFM", "chronos": "Chr.", "tft": "TFT", "seasonal_naive": "Naive"}


def grain_level(k: str, m: dict) -> str:
    d = "dom" if k.startswith("dominicks") else k
    return m.get(d, m.get("*"))


def grain_table(acc):
    """Lowest mean rolling WRMSSE and the 90% confidence set at each decision grain."""
    mcs = pd.concat([read("mcs.csv"), read("favorita_mcs.csv"), read("frat_mcs.csv")])
    rows = []
    for g, _, _, m in GRAIN:
        for k in KEYS:
            lv = grain_level(k, m)
            w = acc[(acc.key == k) & (acc.level == lv) & acc.model.isin(GRAIN_MODELS)].set_index("model").wrmsse
            st = mcs[(mcs.key == k) & (mcs.level == lv)]
            inset = set(st[st.in_set.astype(str) == "True"].model)
            rows.append({"grain": g, "key": k, "level": lv, "best": w.idxmin(), "best_wrmsse": w.min(),
                         "set": inset, "set_size": len(inset), "n_models": len(w)})
    return pd.DataFrame(rows)


def grain_values(acc):
    word = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    gt = grain_table(acc)
    gt.assign(set=gt["set"].map(lambda x: ";".join(sorted(x)))).to_csv(T / "grain.csv", index=False, float_format="%.5g")
    for g, *_ in GRAIN:
        q = gt[gt.grain == g]
        for mod in ("ensemble", "lightgbm", "world_model", "lift", "timesfm_plan"):
            n = int((q.best == mod).sum())
            put("rgrain", g, MCODE[mod], "best", value=word[n])
            put("rgrain", g, MCODE[mod], "bestlist", value=join_and(lname(k) for k in q[q.best == mod].key) or "none")
            put("rgrain", g, MCODE[mod], "inset", value=word[int(q.set.map(lambda x: mod in x).sum())])
        put("rgrain", g, "set", "median", value=f"{q.set_size.median():g}")
        put("rgrain", g, "set", "min", value=str(int(q.set_size.min())))
        put("rgrain", g, "set", "max", value=str(int(q.set_size.max())))
    return gt


def tab_grain(gt):
    lines = []
    for g, name, decision, _ in GRAIN:
        cells = []
        for k in KEYS:
            r = gt[(gt.grain == g) & (gt.key == k)].iloc[0]
            cells.append(f"{SHORTM[r.best]} ({r.set_size})")
        lines.append(f"{name} & {decision} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\rotatebox{{70}}{{{NAME[k]}}}" for k in KEYS)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{On these data, which model had the lowest error depended on the grain, and the coarser the grain the
more models the windows did not separate.}} For each decision grain and data set: the model with the lowest mean WRMSSE
over the rolling windows, and in brackets how many of the nine models compared are in the 90\\% model confidence set
(models four to six windows did not separate from the best; not a finding that they are equally good). Ens.: the equal-weight ensemble of the world model, LightGBM and
TimesFM with the plan; World: world model; Lift: lift model; LGB: LightGBM; TFM+p: TimesFM with the plan. On
Dominick's each data set is one category, so the category and the chain total are the same series. Favorita: the
category is the product family. Daily on M5 and Favorita, weekly elsewhere.}}
\\label{{tab:grain}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.2pt}}
\\begin{{tabular}}{{p{{0.13\\textwidth}}p{{0.17\\textwidth}}*{{9}}{{c}}}}
\\toprule
\\textbf{{Grain}} & \\textbf{{Decision made there}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_grain.tex").write_text(tex)


def studyg_values():
    """Study G (plan amendment of 17 October 2026): the Frat lift model rerun with the
    world model's regularisation, and the category-plan cost with paired uncertainty."""
    m = read("studyG_frat_lift_matched.csv")
    for r in m.itertuples():
        put("rsg", "liftmatched", lv(r.level) if r.level != "overall" else "overall", value=f3(r.wrmsse))
    if (T / "category_cost.csv").exists():
        cc = read("category_cost.csv")
        for r in cc.itertuples():
            k, v, mm = SHORT[r.key], r.variant.replace("_", ""), MCODE.get(r.model, r.model.replace("_", ""))
            put("rcc", k, v, mm, "cost", value=f3(r.cost))
            if pd.notna(getattr(r, "world_minus", np.nan)):
                for fld in ("world_minus", "lo", "hi"):
                    put("rcc", k, v, mm, fld.replace("_", ""), value=f3(getattr(r, fld)))
                put("rcc", k, v, mm, "lowerwin", value=str(int(r.world_lower_windows)))
                put("rcc", k, v, mm, "windows", value=str(int(r.windows)))


CC_VARIANTS = [("dollars", "By period"), ("first_half", "First half"), ("window_total", "Window total"),
               ("units", "Units"), ("node_mean", "Equal nodes")]
CORE5 = ["ensemble", "world_model", "lift", "lightgbm", "timesfm_plan"]
ABBR = {"ensemble": "Ens.", "world_model": "World", "lift": "Lift", "lightgbm": "LGB", "timesfm_plan": "TFM+p"}


def catcost_values():
    """Study G: which model has the lowest category-plan cost under each variant, and
    the world model against the lowest other model with its paired interval."""
    word = {i: w for i, w in enumerate(["none", "one", "two", "three", "four", "five", "six", "seven", "eight"])}
    cc = read("category_cost.csv")
    lines = []
    for v, _ in CC_VARIANTS:
        x = cc[cc.variant == v].pivot(index="key", columns="model", values="cost")[CORE5]
        best = x.idxmin(axis=1)
        for m in ("world_model", "ensemble", "lightgbm"):
            put("rcount", f"cc{v.replace('_', '')}{MCODE[m]}", value=word[int((best == m).sum())])
            put("rcount", f"cc{v.replace('_', '')}{MCODE[m]}list",
                value=join_and(lname(k) for k in best[best == m].index) or "none")
    z = read("category_cost_nodes_zero.csv")
    dz = z[z.key.str.startswith("dominicks")]
    put("rcount", "cczerolo", value=str(int(dz.zero_node_windows.min())))
    put("rcount", "cczerohi", value=str(int(dz.zero_node_windows.max())))
    put("rcount", "cczeron", value=str(int(dz.node_windows.iloc[0])))
    d = cc[cc.variant == "dollars"]
    clear_other, clear_world = [], []
    for k in KEYS:
        g = d[d.key == k].set_index("model")
        if g.empty:
            continue
        others = g.loc[[m for m in CORE5 if m != "world_model"]]
        b = others.cost.idxmin()
        r = g.loc[b]
        cells = []
        for v, _ in CC_VARIANTS:
            x = cc[(cc.key == k) & (cc.variant == v)].set_index("model").cost.reindex(CORE5)
            cells.append(ABBR[x.idxmin()])
        if g.loc["world_model"].cost < others.cost.min():
            lead = "World"
        else:
            lead = ABBR[b]
        diff = f"{f3(r.world_minus)} ({f3(r.lo)} to {f3(r.hi)})"
        wins = f"{int(r.world_lower_windows)}/{int(r.windows)}"
        if r.lo > 0:
            clear_other.append(k)
        if r.hi < 0:
            clear_world.append(k)
        lines.append(f"{NAME[k]} & {ABBR[b]} & {diff} & {wins} & " + " & ".join(cells) + " \\\\")
    put("rcount", "ccclearother", value=word[len(clear_other)])
    put("rcount", "ccclearotherlist", value=join_and(lname(k) for k in clear_other) or "none")
    put("rcount", "ccclearworld", value=word[len(clear_world)])
    head = " & ".join(lbl for _, lbl in CC_VARIANTS)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{With the cost corrected, the world model had the lowest category-plan cost only on M5; the
ensemble or LightGBM was lowest elsewhere under every rule but one.}} On M5 the world model is not separated from the
ensemble by period and is lower in all four windows over the whole window; the exception elsewhere is cheese over the
first half, where TimesFM with the plan is lowest.
Category-by-store planning cost, mean over the rolling windows. Lowest other: the model other than the world model with
the lowest cost by period, in dollars. World minus it: the world model's cost minus that model's, with a 95\\% interval
from 5,000 resamples of category-by-store nodes within each window, the same nodes for both models (positive: the
world model costs more); this interval does not cover variation between windows, for which the next column counts the
windows in which the world model cost less. The last five columns give the lowest-cost model of the five under each
rule: absolute error of each period's total in dollars (the cost used in Table~\\ref{{tab:decisions}}); the same over
the first half of the window; error of the total over the whole window; each unit of error counted alike; and the mean
over nodes of each node's error over its sales. Study G.}}
\\label{{tab:catcost}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.4pt}}
\\begin{{tabular}}{{lllc*{{5}}{{c}}}}
\\toprule
 & & & & \\multicolumn{{5}}{{c}}{{\\textbf{{Lowest cost under each rule}}}} \\\\
\\cmidrule(l){{5-9}}
\\textbf{{Data}} & \\textbf{{Lowest other}} & \\textbf{{World minus it}} & \\textbf{{World lower}} & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / "rev_tab_catcost.tex").write_text(tex)


def core_values():
    """Per-window comparison of the core models with LightGBM (results/paper1_revision/scores)."""
    word = {i: w for i, w in enumerate(["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}
    sc = pd.concat([pd.read_csv(f) for f in sorted((T.parent / "scores").glob("studyF_scores_*.csv"))])
    sc = sc[sc.set == "rolling"].drop_duplicates(["key", "origin", "model", "level"])
    for level in (BOTTOM, "total"):
        w = sc[sc.level == level].pivot_table(index=["key", "origin"], columns="model", values="wrmsse")
        for m in ("world_model", "timesfm_plan", "ensemble"):
            r = (w[m] / w["lightgbm"]).dropna()
            put("rcore", MCODE[m], lv(level), "median", value=f"{r.median():.2f}")
            put("rcore", MCODE[m], lv(level), "better", value=str(int((r < 1).sum())))
            put("rcore", MCODE[m], lv(level), "n", value=str(len(r)))
            put("rcore", MCODE[m], lv(level), "max", value=f"{r.max():.1f}")


def write_values():
    lines = ["% Generated by figs/make_revision.py from results/paper1_revision/tables. Do not edit.",
             "\\makeatletter",
             "\\newcommand{\\revget}[1]{\\ifcsname rev@#1\\endcsname\\csname rev@#1\\endcsname"
             "\\else\\errmessage{Revision value #1 is not in the result tables}\\fi}"]
    for k, v in sorted(vals.items()):
        lines.append(f"\\expandafter\\def\\csname rev@{k}\\endcsname{{{v}}}")
    lines += [
        "\\newcommand{\\rv}[3]{\\revget{rv@#1@#2@#3}}",
        "\\newcommand{\\rf}[3]{\\revget{rf@#1@#2@#3}}",
        "\\newcommand{\\rpair}[5]{\\revget{rpair@#1@#2@#3@#4@#5}}",
        "\\newcommand{\\rab}[4]{\\revget{rab@#1@#2@#3@#4}}",
        "\\newcommand{\\rw}[3]{\\revget{rw@#1@#2@#3}}",
        "\\newcommand{\\rdec}[3]{\\revget{rdec@#1@#2@#3}}",
        "\\newcommand{\\rint}[4]{\\revget{rint@#1@#2@#3@#4}}",
        "\\newcommand{\\rintold}[4]{\\revget{rintold@#1@#2@#3@#4}}",
        "\\newcommand{\\rrec}[3]{\\revget{rrec@#1@#2@#3}}",
        "\\newcommand{\\rper}[5]{\\revget{rper@#1@#2@#3@#4@#5}}",
        "\\newcommand{\\rmech}[4]{\\revget{rmech@#1@#2@#3@#4}}",
        "\\newcommand{\\rsf}[3]{\\revget{rsf@#1@#2@#3}}",
        "\\newcommand{\\rtail}[4]{\\revget{rtail@#1@#2@#3@#4}}",
        "\\newcommand{\\rrank}[4]{\\revget{rrank@#1@#2@#3@#4}}",
        "\\newcommand{\\rpan}[5]{\\revget{rpan@#1@#2@#3@#4@#5}}",
        "\\newcommand{\\raft}[3]{\\revget{raft@#1@#2@#3}}",
        "\\newcommand{\\rcount}[1]{\\revget{rcount@#1}}",
        "\\newcommand{\\rwin}[3]{\\revget{rwin@#1@#2@#3}}",
        "\\newcommand{\\rreg}[3]{\\revget{rreg@#1@#2@#3}}",
        "\\newcommand{\\rgrain}[3]{\\revget{rgrain@#1@#2@#3}}",
        "\\newcommand{\\rcore}[3]{\\revget{rcore@#1@#2@#3}}",
        "\\newcommand{\\rsg}[2]{\\revget{rsg@#1@#2}}",
        "\\newcommand{\\rcc}[4]{\\revget{rcc@#1@#2@#3@#4}}",
        "\\makeatother",
    ]
    (GEN / "rev_values.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {GEN / 'rev_values.tex'} ({len(vals)} values)")


def main():
    acc = rolling_accuracy()
    sets = mcs_sets()
    accuracy_values(acc)
    p = paired_values()
    ab = ablation_values()
    other_values()
    studyf_values()
    regret_values()
    window_values()
    pantry_values()
    count_values(acc, sets, p, ab)
    gt = grain_values(acc)
    core_values()
    studyg_values()
    catcost_values()
    tab_grain(gt)
    tab_rolling(acc, sets)
    tab_ablation(ab)
    tab_decisions()
    tab_perturbation()
    tab_intervals()
    tab_reconcile()
    tab_finals_new()
    tab_timestral()
    write_values()


if __name__ == "__main__":
    main()
