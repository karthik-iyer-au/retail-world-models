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


# ===================================================== development study ===
# All copied from results/README.md; the section heading is named above each.
# Development windows: the two 8-week windows before the final 8 weeks
# (Dominick's, Frat), or the M5 practice window. Overall WRMSSE unless stated.

# "Breakfast at the Frat benchmarks" (three 8-week windows, before the final window was locked).
FRAT_BENCH = [("Seasonal naive", 0.832), ("Average of last 8 weeks", 0.756),
              ("LightGBM, price only", 0.554), ("LightGBM, all recorded levers", 0.447)]

# "Development runs over several seeds": Frat, five seeds: (label, group, mean, lo, hi).
# group: what the row tests. Reference rows are the lift model (0.406) and LightGBM (0.421).
FRAT_DEV = [
    ("LightGBM, all recorded levers", "ref", 0.421, 0.416, 0.427),
    ("Item encoder and readout", "base", 0.424, 0.407, 0.438),
    ("+ item-to-item, conditional similarity", "neighbours", 0.476, 0.430, 0.508),
    ("+ item-to-item, attention", "neighbours", 0.465, 0.416, 0.505),
    ("+ item-to-item, both", "neighbours", 0.453, 0.419, 0.498),
    ("+ item-to-item, same category only", "neighbours", 0.443, 0.412, 0.470),
    ("+ regular price", "lever", 0.429, 0.419, 0.442),
    ("Lift model: regular price, baseline times lift", "ref", 0.406, 0.401, 0.414),
    ("Lift + plan of the four weeks before", "inputs", 0.411, 0.406, 0.418),
    ("Lift + same week last year", "inputs", 0.408, 0.394, 0.420),
    ("Lift + recorded base price as regular price", "inputs", 0.434, 0.418, 0.460),
    ("Lift + US seasonal calendar", "inputs", 0.425, 0.416, 0.434),
    ("Lift + promotion crowding", "inputs", 0.406, 0.400, 0.414),
    ("Lift + base price, calendar and crowding", "inputs", 0.446, 0.425, 0.459),
    ("Lift + latent loss alongside, weight 0.1", "jepa", 0.411, 0.406, 0.417),
    ("Lift + latent loss alongside, weight 2", "jepa", 0.410, 0.407, 0.415),
    ("Lift, latent pretraining, then all trained", "jepa", 0.407, 0.400, 0.412),
    ("Lift, pretraining, encoder at a tenth of the rate", "jepa", 0.414, 0.402, 0.425),
    ("Lift, pretraining, encoder frozen", "jepa", 0.428, 0.416, 0.441),
    ("Lift, long pretraining, encoder at a tenth", "jepa", 0.408, 0.400, 0.422),
    ("Lift, stopped on validation weeks", "training", 0.425, 0.414, 0.437),
    ("Lift + totals term in the loss", "training", 0.414, 0.395, 0.433),
    ("Lift + readout dropout and weight decay", "training", 0.409, 0.405, 0.412),
    ("World model, baseline and lift", "world", 0.408, 0.396, 0.418),
    ("World model + latent check alongside", "world", 0.417, 0.401, 0.426),
    ("World model, long latent pretraining", "world", 0.411, 0.392, 0.423),
    ("World model + plan split (after DWM)", "world", 0.417, None, None),
    ("World model + latent check, VICReg, near-week weighting", "world", 0.427, None, None),
    ("World model + all three", "world", 0.424, None, None),
]
# "Combination with LightGBM, and negative binomial likelihood"; "Overnight part 1".
COMBO = {"frat": dict(lgb=0.421, lift=0.406, world=0.408, lift_lgb=0.388, world_lgb=0.390, negbin=0.410),
         "ana": dict(lgb=0.715, lift=0.676, lift_lgb=0.670, world_lgb=0.668, negbin=0.665)}
# "New-item test" and "New items with a LightGBM route": new-item error, item by store.
NEW_ITEMS = {"frat": dict(zero=1.510, lgb=1.310, borrow=0.890, borrow_hidden=0.879),
             "ana": dict(lgb=0.923, borrow=2.999)}
# "Plan A against plan B": share picking the better week, miss on size (log).
PLAN_PAIRS = [("LightGBM", 78.2, 0.448), ("Lift model", 77.4, 0.460), ("World model", 77.5, 0.459),
              ("World model + plan split", 77.5, 0.458)]
# "New items with a LightGBM route, and unusual plans judged item by item".
UNUSUAL = [("LightGBM", 0.696, 0.697), ("Lift model", 0.743, 0.725), ("World model", 0.730, 0.730)]

# "Dominick's analgesics" development rows (one run or three seeds).
ANA_DEV = [("Seasonal naive", 1.274), ("Average of last 8 weeks", 0.723), ("LightGBM", 0.715),
           ("Item encoder and readout", 0.692), ("Lift model", 0.676),
           ("Lift + item-to-item", 0.697), ("Lift, held price as regular", 0.674),
           ("Lift + held price, calendar, crowding", 0.919), ("Lift, 24,000 steps", 1.473),
           ("Lift, stop on validation", 0.683), ("Lift, totals in the loss", 0.819),
           ("World model, totals in the loss", 0.732), ("World model, no totals loss", 0.667),
           ("World model + plan split", 0.704), ("World model + latent check, VICReg, near weeks", 0.685)]

# "All 28 Dominick's categories" and "World model with an 8-week level, all 28".
DOM28 = {  # code: (LightGBM, lift, world 26-week level, world 8-week level)
    "bat": (1.188, 1.568, 1.465, 1.195), "oat": (1.084, 1.026, 1.060, 1.036), "ptw": (0.486, 0.445, 0.485, 0.425),
    "tti": (0.394, 0.413, 0.465, 0.427), "ber": (0.454, 0.460, 0.470, 0.505), "frd": (0.930, 0.905, 0.873, 0.884),
    "soa": (0.517, 0.521, 0.516, 0.518), "cra": (0.709, 0.967, 0.687, 0.749), "did": (0.344, 0.436, 0.393, 0.406),
    "tbr": (0.647, 0.671, 0.699, 0.686), "fsf": (0.650, 0.748, 0.832, 0.742), "tna": (1.163, 1.030, 0.959, 1.004),
    "cig": (0.534, 0.918, 0.690, 0.643), "frj": (0.314, 0.360, 0.376, 0.342), "sna": (0.587, 0.959, 1.463, 0.874),
    "tpa": (0.731, 0.802, 0.779, 0.857), "lnd": (0.768, 0.735, 0.687, 0.729), "fec": (0.414, 0.358, 0.397, 0.417),
    "bjc": (0.992, 1.248, 1.229, 1.100), "cer": (0.753, 0.795, 0.673, 0.805), "cso": (0.613, 0.532, 0.583, 0.500),
    "gro": (0.901, 1.208, 0.809, 0.874), "sha": (0.994, 0.938, 0.906, 0.921), "coo": (0.487, 0.503, 0.522, 0.491),
    "che": (0.742, 0.900, 0.996, 1.116), "fre": (0.417, 0.374, 0.382, 0.393), "sdr": (0.747, 1.406, 0.864, 0.747),
    "ana": (0.702, 0.718, 0.666, 0.679),
}
DOM28_NAME = {
    "bat": "Bath soap", "oat": "Oatmeal", "ptw": "Paper towels", "tti": "Toilet tissue", "ber": "Beer",
    "frd": "Frozen dinners", "soa": "Soaps", "cra": "Crackers", "did": "Dish detergent", "tbr": "Toothbrushes",
    "fsf": "Front-end candies", "tna": "Canned tuna", "cig": "Cigarettes", "frj": "Frozen juices",
    "sna": "Snack crackers", "tpa": "Toothpaste", "lnd": "Laundry detergent", "fec": "Fabric softener",
    "bjc": "Bottled juice", "cer": "Cereals", "cso": "Canned soup", "gro": "Grooming products",
    "sha": "Shampoos", "coo": "Cookies", "che": "Cheese", "fre": "Frozen entrees", "sdr": "Soft drinks",
    "ana": "Analgesics",
}

# "Peaks and dips" (six categories, one seed): forecast / actual by week type.
PEAKS = {  # cat: {variant: (promoted, after, ordinary, overall)}
    "che": {"lgb": (0.800, 0.969, 0.977, 0.742), "world": (1.207, 1.035, 1.025, 1.116),
            "feedback": (1.081, 1.085, 1.050, 1.072), "weight": (1.001, 1.009, 1.055, 0.814)},
    "sna": {"lgb": (0.908, 0.921, 1.022, 0.587), "world": (1.195, 1.329, 1.132, 0.874),
            "feedback": (1.277, 1.348, 1.173, 0.941), "weight": (1.317, 1.179, 1.111, 0.908)},
    "sdr": {"lgb": (0.768, 0.904, 0.863, 0.747), "world": (1.010, 1.316, 0.964, 0.747),
            "feedback": (0.974, 0.997, 0.890, 1.050), "weight": (1.102, 1.235, 1.089, 0.777)},
    "bjc": {"lgb": (0.714, 0.957, 0.946, 0.992), "world": (0.924, 1.035, 0.918, 1.100),
            "feedback": (0.936, 1.153, 0.958, 1.046), "weight": (1.136, 1.050, 0.961, 1.063)},
    "ana": {"lgb": (0.887, 0.927, 1.028, 0.701), "world": (0.923, 0.966, 1.047, 0.679),
            "feedback": (0.978, 0.978, 1.099, 0.838), "weight": (1.027, 0.966, 1.115, 0.827)},
    "fre": {"lgb": (0.840, 0.679, 0.948, 0.417), "world": (0.881, 0.705, 0.887, 0.393),
            "feedback": (0.820, 0.696, 0.892, 0.411), "weight": (0.878, 0.596, 0.855, 0.400)},
}
PEAK_VARIANT = {"lgb": "LightGBM", "world": "World model", "feedback": "+ own sales fed forward",
                "weight": "+ dollar and peak weighting"}

# Dominick's development ladder, six categories: the versions of the world model in order,
# three-seed means where recorded ("LightGBM against the world model on three seeds",
# "Brand switching", "Brand switching with the normalised roll-forward", "Averaged switching
# with the normalised roll-forward, three seeds", "Change C").
LADDER_STEPS = ["8-week level", "+ brand switching", "+ normalised roll-forward", "+ average of three fits",
                "+ twice the steps"]
LADDER = {  # cat: [values per step], LightGBM
    "che": ([1.088, 1.111, 1.028, 0.971, 0.901], 0.749),
    "sna": ([0.813, 0.752, 0.830, 1.092, 0.810], 0.586),
    "sdr": ([0.902, 0.789, 0.827, 0.743, 0.765], 0.759),
    "bjc": ([1.074, 1.066, 1.047, 1.048, 1.015], 0.987),
    "ana": ([0.711, 0.707, 0.696, 0.622, 0.647], 0.701),
    "fre": ([0.406, 0.416, 0.404, 0.391, 0.393], 0.419),
}
LADDER_RANGE = {  # seed range of the first and the chosen (average of three fits) step
    "che": ((0.981, 1.167), (0.943, 0.988)), "sna": ((0.751, 0.874), (0.730, 1.806)),
    "sdr": ((0.747, 1.123), (0.713, 0.770)), "bjc": ((1.033, 1.100), (1.019, 1.065)),
    "ana": ((0.655, 0.800), (0.618, 0.625)), "fre": ((0.391, 0.434), (0.386, 0.398)),
}
# "Brand switching": learned share of the summed lift that is new sales, range over runs.
SWITCH_SHARE = {"che": (0.70, 1.00), "sna": (0.00, 0.66), "sdr": (0.64, 1.00), "bjc": (0.69, 1.00),
                "ana": (0.71, 1.00), "fre": (0.68, 1.00)}
# "Item roles": overall, (LightGBM, LightGBM + role, world, world + role, world + role picks level).
ROLES = {"che": (0.742, 0.754, 1.116, 1.107, 1.074), "sna": (0.587, 0.577, 0.874, 1.163, 2.422),
         "sdr": (0.747, 0.734, 0.747, 0.919, 1.046), "bjc": (0.992, 0.987, 1.100, 1.062, 1.140),
         "ana": (0.701, 0.694, 0.679, 0.639, 0.655), "fre": (0.417, 0.423, 0.393, 0.411, 0.415)}
# "Recent and longer level together": (26-week, 8-week, both levels as inputs), LightGBM.
LEVEL_VIEWS = {"che": (0.996, 1.116, 0.883, 0.742), "sna": (1.463, 0.874, 1.515, 0.587),
               "sdr": (0.864, 0.747, 1.069, 0.747), "bjc": (1.229, 1.100, 1.159, 0.992),
               "ana": (0.666, 0.679, 0.648, 0.702), "fre": (0.382, 0.393, 0.399, 0.417)}

# M5 practice window development ("M5 practice window: roll-forward normalisation, three seeds",
# "Lift model on M5, three seeds", "Averaging three fits on M5", "Revised world model").
M5_DEV = [
    ("Seasonal naive", 0.870, None, None),
    ("LightGBM, one model per store", 0.715, None, None),
    ("Lift model (3 seeds)", 0.713, 0.683, 0.734),
    ("Item encoder and readout (5 seeds)", 0.699, 0.669, 0.725),
    ("World model (3 seeds)", 0.707, 0.680, 0.750),
    ("World model + normalised roll-forward (3 seeds)", 0.656, 0.646, 0.675),
    ("+ average of three fits (1 run)", 0.665, None, None),
    ("+ weekday as seven inputs (seed 1)", 0.653, None, None),
    ("+ weekday and event kinds (seed 1)", 0.646, None, None),
    ("+ 60,000 training steps (seed 1)", 0.642, None, None),
    ("+ item level against past-year level (seed 1)", 0.680, None, None),
]
