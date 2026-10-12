"""Every number in the manuscript, in one place.

Two sources, and nothing else:

1. RUNS: the committed run records in results/promoted/<run>/metrics.json
   (one folder per run and seed). Scores are read from them directly.
2. RECORDED: numbers for runs whose folders are not in results/promoted
   (TimesFM on Frat and Dominick's, TimesFM with the plan, and the analyses
   of saved forecasts run on Colab). Each is copied from the section of
   results/README.md named next to it, which records the run folder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PROMOTED = ROOT / "results" / "promoted"

# ------------------------------------------------------------------ runs ---
# Config names as they appear in the run folder names, per model and window.
M5_WINDOWS = ["mar", "apr", "practice", "test"]
M5_WINDOW_LABEL = {
    "mar": "29 Feb to 27 Mar",
    "apr": "28 Mar to 24 Apr",
    "practice": "25 Apr to 22 May (practice)",
    "test": "23 May to 19 Jun (official test)",
}
M5_CONFIG = {
    "world": {"mar": "m5_confirm_mar_world_model_v1", "apr": "m5_confirm_apr_world_model_v1",
              "practice": "m5_validation_world_model_roll_norm", "test": "m5_test_world_model_roll_norm"},
    "lift": {"mar": "m5_confirm_mar_lift", "apr": "m5_confirm_apr_lift",
             "practice": "m5_validation_lift", "test": "m5_test_lift"},
    "lgb": {"mar": "m5_confirm_mar_lightgbm", "apr": "m5_confirm_apr_lightgbm",
            "practice": "m5_validation_lightgbm", "test": "m5_test_lightgbm"},
    "graph": {"mar": "m5_confirm_mar_graph", "apr": "m5_confirm_apr_graph",
              "practice": "m5_validation_graph", "test": "m5_test_graph"},
    "tfm": {"mar": "m5_confirm_mar_timesfm_mean", "apr": "m5_confirm_apr_timesfm_mean",
            "practice": "m5_validation_timesfm_mean", "test": "m5_test_timesfm_mean"},
    "naive": {"mar": "m5_confirm_mar_seasonal_naive", "apr": "m5_confirm_apr_seasonal_naive",
              "practice": "m5_validation_seasonal_naive", "test": "m5_test_seasonal_naive"},
}
CATS = ["che", "sna", "sdr", "bjc", "ana", "fre"]
CAT_LABEL = {"che": "Cheese", "sna": "Snack crackers", "sdr": "Soft drinks",
             "bjc": "Bottled juice", "ana": "Analgesics", "fre": "Frozen entrees"}
FINAL_CONFIG = {
    "world": {**{c: f"dominicks_{c}_final_world_model_switching_avg3_roll_norm" for c in CATS},
              "frat": "frat_final_world_model"},
    "lgb": {**{c: f"dominicks_{c}_final_lightgbm" for c in CATS}, "frat": "frat_final_lightgbm"},
    "lift": {"frat": "frat_final_lift"},
}

MODEL_LABEL = {
    "world": "World model",
    "lift": "Lift model",
    "lgb": "LightGBM",
    "graph": "Graph model",
    "tfm": "TimesFM",
    "tfm_plan": "TimesFM with the plan",
    "naive": "Seasonal naive",
}

M5_LEVELS = ["total", "state_id", "store_id", "cat_id", "dept_id", "state_idxcat_id", "state_idxdept_id",
             "store_idxcat_id", "store_idxdept_id", "item_id", "item_idxstate_id", "item_idxstore_id"]
M5_LEVEL_LABEL = ["Total", "State", "Store", "Category", "Department", "State ×\ncategory",
                  "State ×\ndepartment", "Store ×\ncategory", "Store ×\ndepartment", "Item",
                  "Item ×\nstate", "Item ×\nstore"]


def _runs(config: str) -> list[dict]:
    """metrics.json of every seed of a config, seed order. A config with no
    seeded run (one early practice-window run) falls back to its single run."""
    pat = re.compile(rf"^\d{{8}}T\d{{6}}_{re.escape(config)}_seed(\d+)_[0-9a-f]{{8}}$|^{re.escape(config)}_seed(\d+)_[0-9a-f]{{8}}$")
    found = []
    for d in PROMOTED.iterdir():
        m = pat.match(d.name)
        if m and (d / "metrics.json").exists():
            found.append((int(m.group(1) or m.group(2)), d))
    if not found:
        pat1 = re.compile(rf"^\d{{8}}T\d{{6}}_{re.escape(config)}_[0-9a-f]{{8}}$")
        found = [(0, d) for d in PROMOTED.iterdir() if pat1.match(d.name)]
    if not found:
        raise FileNotFoundError(f"no promoted run for {config}")
    return [json.load(open(d / "metrics.json")) | {"_folder": d.name} for _, d in sorted(found)]


def _level(metrics: dict, level: str) -> float:
    for b in metrics["splits"][0]["by_level"]:
        if b["level"] == level:
            return b["wrmsse"]
    raise KeyError(level)


def scores(config: str) -> dict:
    """Per-seed scores of a config: overall, total, item by store, forecast / actual, every level."""
    out = {"overall": [], "total": [], "ixs": [], "fa": [], "levels": [], "peaks": [], "folders": []}
    for m in _runs(config):
        s = m["splits"][0]
        out["overall"].append(m["wrmsse"])
        out["total"].append(_level(m, "total"))
        out["ixs"].append(_level(m, "item_idxstore_id"))
        out["fa"].append(s.get("forecast_to_actual"))
        out["levels"].append({b["level"]: b["wrmsse"] for b in s["by_level"]})
        out["peaks"].append(s.get("peaks"))
        out["folders"].append(m["_folder"])
    return out


def m5(model: str, window: str) -> dict:
    return scores(M5_CONFIG[model][window])


def final(model: str, data: str) -> dict:
    return scores(FINAL_CONFIG[model][data])


def avg(xs):
    return mean(xs)


# -------------------------------------------------------------- recorded ---
# results/README.md, "TimesFM 2.5 zero-shot on the Frat and Dominick's final
# windows (mean output)" and "TimesFM with the plan, every window".
# Keys: overall, ixs (item by store), fa (forecast / actual), promo_fa
# (forecast / actual on promoted weeks, Dominick's).
TFM_FINAL = {
    "frat": dict(overall=0.965, ixs=0.827, fa=0.847),
    "che": dict(overall=0.829, ixs=0.792, fa=0.855, promo_fa=0.343),
    "sna": dict(overall=0.468, ixs=0.640, fa=0.927, promo_fa=0.630),
    "sdr": dict(overall=1.070, ixs=0.895, fa=0.731, promo_fa=0.462),
    "bjc": dict(overall=0.940, ixs=0.782, fa=0.866, promo_fa=0.327),
    "ana": dict(overall=1.009, ixs=0.790, fa=0.973, promo_fa=0.306),
    "fre": dict(overall=0.563, ixs=0.682, fa=0.869, promo_fa=0.290),
}
TFM_PLAN_FINAL = {
    "frat": dict(overall=0.686, ixs=0.641, fa=0.977),
    "che": dict(overall=0.545, ixs=0.699, promo_fa=0.685),
    "sna": dict(overall=0.475, ixs=0.614, promo_fa=1.129),
    "sdr": dict(overall=0.974, ixs=0.780, promo_fa=0.932),
    "bjc": dict(overall=0.990, ixs=0.754, promo_fa=0.603),
    "ana": dict(overall=0.889, ixs=0.742, promo_fa=0.779),
    "fre": dict(overall=0.566, ixs=0.599, promo_fa=0.942),
}
TFM_PLAN_M5 = {  # overall, item by store
    "mar": dict(overall=0.858, ixs=0.837),
    "apr": dict(overall=0.772, ixs=0.829),
    "practice": dict(overall=0.929, ixs=0.830),
    "test": dict(overall=0.719, ixs=0.883),
}
# Frat promoted weeks (promotion on), WRMSSE: TimesFM 1.284 history only, 0.869 with the plan.
TFM_FRAT_PROMO_ON = dict(history=1.284, plan=0.869)

# "April: the five series where lift seed 2 failed, every model".
# HOUSEHOLD_1_146 in four stores, units over the 28-day April window.
PRICE_CUT = {
    "sold": 120,
    "world": [121, 600, 243],
    "lift": [79, 206886, 316],
    "graph": [59, 45, 55],
    "tfm": [7.8],
    "lgb": [3.1, 2.9, 3.3],
}
PRICE_CUT_FOODS = {"sold": 4, "world": [26, 1, 18], "lift": [346, 25348, 366], "lgb": [3.4, 3.4, 3.2],
                   "tfm": [0.4], "graph": [10, 10, 10]}
LIFT_APR_SEED2 = dict(forecast=1413724, sold=1183626, top5_share=1.01, ceiling=22000)

# "Frat final test: where the world model loses to LightGBM".
FRAT_WEEKS = ["16 Nov", "23 Nov", "30 Nov", "7 Dec", "14 Dec", "21 Dec", "28 Dec", "4 Jan"]
FRAT_ACTUAL = [53725, 73950, 60048, 82656, 76231, 66469, 50941, 64489]
FRAT_BIAS = {"world": [2.3, -0.6, 15.8, -17.0, -15.3, -1.4, 23.9, 6.1],
             "lgb": [-4.4, -11.6, 1.8, -10.9, -11.9, -11.5, -3.1, 6.2]}
FRAT_DEV = dict(world=0.408, lgb=0.421, lift=0.406)

# "April banded by March sales".
BANDS = ["Sold nothing", "Under 0.5", "0.5 to 1", "1 to 3", "3 to 10", "Over 10"]
BAND_UNITS = [14102, 106500, 133684, 332022, 358060, 239258]
BAND_BIAS = {"world": [62.1, -4.5, -3.1, -1.3, 1.5, 1.3], "lgb": [-5.4, -5.6, -3.3, -3.3, -1.7, -3.4]}
BAND_CLOSER = [65.9, 50.6, 52.1, 53.8, 53.7, 54.2]

# "Plan read-back: does the world model's state take in the plan". Discount,
# R squared by days back: (control, rolled state).
READBACK = {
    "mar": {0: (0.00, 0.83), 1: (0.00, 0.81), 7: (0.00, 0.59)},
    "apr": {0: (0.02, 0.58), 1: (0.02, 0.65), 7: (0.02, 0.48)},
    "test": {0: (0.00, 0.80), 1: (0.00, 0.79), 7: (0.00, 0.46)},
}

# Published M5 scores on the official test period ("Final test on the official M5 test period").
M5_PUBLISHED = dict(winner=0.520, tenth=0.547, fiftieth=0.576, best_stat=0.671)

# "What did not work" ledger (docs/PAPER_NOTES.md, reporting rule; results/README.md sections named).
LEDGER = [
    ("Twice the training steps", "Dominick's (6) and Frat, dev., 3 seeds", "Better in 2, worse in 3, level or mixed in 2"),
    ("Weekday as seven inputs", "M5 practice, seed 1", "0.653 against 0.646"),
    ("Event kinds (with weekday)", "M5 practice, seed 1", "0.646 against 0.646"),
    ("60,000 training steps", "M5 practice, seed 1", "0.642 against 0.646; gain below seed range 0.029"),
    ("Item level against past-year level", "M5 practice, seed 1", "0.680 against 0.646"),
    ("Totals term in the loss", "Dominick's analgesics, dev.", "0.819 against 0.683; total 9\\% under"),
    ("Negative binomial likelihood", "Frat, analgesics, dev.", "Level"),
    ("Plan split, latent check, near-week weighting", "Frat, dev., 5 seeds", "0.417 to 0.427 against 0.408"),
    ("Own forecast fed into the roll-forward", "Dominick's (6), dev.", "Worse overall in 4 of 6"),
    ("Item role labels", "Dominick's (6), dev.", "No effect on LightGBM; snack crackers 2.422 for the world model"),
    ("Averaging three fits", "M5 practice, seed 1", "0.665 against 0.656 (kept on Dominick's)"),
    ("Graph model over product attributes", "M5, four windows, 3 seeds", "Level with LightGBM item by store"),
    ("TimesFM, median output", "M5, four windows", "Overall 1.71 to 1.88; totals 27\\% to 31\\% under"),
    ("Lift model (no roll-forward)", "M5, four windows, 3 seeds", "One seed failed in April (2.603)"),
]
