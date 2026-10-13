"""Every number in the manuscript, in one place.

Two sources of run results, and nothing else:

1. RUNS: the committed run records in results/promoted/<run>/metrics.json
   (one folder per run and seed). Scores are read from them directly.
2. RECORDS: numbers for runs whose folders are not in results/promoted
   (TimesFM on Frat and Dominick's, TimesFM with the plan, the development
   study, and the analyses of saved forecasts). They are read from
   results/records/snapshot.json, which scripts/records_snapshot.py builds from
   the run folders on Drive and which names the run folder behind every value.

Until snapshot.json exists, setting RWDATA_ALLOW_TRANSCRIBED=1 loads
results/records/snapshot_transcribed.json instead: the values once copied by
hand from results/README.md, kept for comparison only. Without the variable a
missing snapshot is an error. paper/Makefile sets it only while the snapshot is
missing.

The only literals below are labels and external constants (published M5
competition scores, a model constant), each marked.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PROMOTED = ROOT / "results" / "promoted"
RECORDS = ROOT / "results" / "records"

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


# --------------------------------------------------------------- records ---
def _load_records() -> tuple[dict, Path]:
    snap = RECORDS / "snapshot.json"
    if snap.exists():
        path = snap
    elif os.environ.get("RWDATA_ALLOW_TRANSCRIBED") == "1":
        path = RECORDS / "snapshot_transcribed.json"
        print(f"WARNING: {snap.relative_to(ROOT)} not found; using the hand-transcribed values in "
              f"{path.relative_to(ROOT)} (RWDATA_ALLOW_TRANSCRIBED=1). Not for submission: run "
              f"scripts/records_snapshot.py on the run folders.", file=sys.stderr)
    else:
        raise FileNotFoundError(
            f"{snap} is missing. Build it with scripts/records_snapshot.py --results <run folders>, "
            f"or set RWDATA_ALLOW_TRANSCRIBED=1 to build from the hand-transcribed values for now.")
    doc = json.loads(path.read_text())
    return {k: b["value"] for k, b in doc["blocks"].items()}, path


_R, RECORDS_FILE = _load_records()
RECORDS_KIND = "transcribed" if RECORDS_FILE.name == "snapshot_transcribed.json" else "records"


def _tuples(d: dict) -> dict:
    return {k: tuple(v) for k, v in d.items()}


# TimesFM 2.5 zero-shot (mean output) and TimesFM with the plan on the final
# windows; keys: overall, ixs (item by store), fa (forecast / actual), promo_fa
# (forecast / actual on promoted weeks, Dominick's).
TFM_FINAL = _R["TFM_FINAL"]
TFM_PLAN_FINAL = _R["TFM_PLAN_FINAL"]
TFM_PLAN_M5 = _R["TFM_PLAN_M5"]  # overall, item by store
# Frat promoted weeks (promotion on), WRMSSE: TimesFM history only, and with the plan.
TFM_FRAT_PROMO_ON = _R["TFM_FRAT_PROMO_ON"]

# April: the five series where lift seed 2 failed. HOUSEHOLD_1_146 in four
# stores, and FOODS_3_443 in WI_1: units over the 28-day April window, per seed.
PRICE_CUT = _R["PRICE_CUT"]
PRICE_CUT_FOODS = _R["PRICE_CUT_FOODS"]
# The lift model's ceiling is a model constant (log forecast clamped at 10,
# e^10 = 22,026, src/rwm/model/state_model.py), not a run result.
LIFT_APR_SEED2 = {**_R["LIFT_APR_SEED2"], "ceiling": 22000}

# Frat final test: weekly totals and bias, world model and LightGBM.
FRAT_WEEKS = _R["FRAT_WEEKS"]
FRAT_ACTUAL = _R["FRAT_ACTUAL"]
FRAT_BIAS = _R["FRAT_BIAS"]
# Development-window references on Frat (this dict used to be defined as
# FRAT_DEV and was then overwritten by the list of the same name below).
FRAT_DEV_REF = _R["FRAT_DEV_REF"]

# April banded by March sales. Band labels are display text.
BANDS = ["Sold nothing", "Under 0.5", "0.5 to 1", "1 to 3", "3 to 10", "Over 10"]
BAND_UNITS = _R["BAND_UNITS"]
BAND_BIAS = _R["BAND_BIAS"]
BAND_CLOSER = _R["BAND_CLOSER"]

# Plan read-back: discount, R squared by days back: (control, rolled state).
READBACK = {w: {int(d): tuple(v) for d, v in days.items()} for w, days in _R["READBACK"].items()}

# EXTERNAL, not a run result: published M5 scores on the official test period,
# from the organisers' files in data/raw/m5_reference/ (Makridakis, Spiliotis and
# Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions",
# International Journal of Forecasting 38(4), 2022; references.tex: m5accuracy).
M5_PUBLISHED = dict(winner=0.520, tenth=0.547, fiftieth=0.576, best_stat=0.671)

# "What did not work" ledger: (change, where tested, outcome).
LEDGER = [tuple(r) for r in _R["LEDGER"]]


# ===================================================== development study ===
# Development windows: the two 8-week windows before the final 8 weeks
# (Dominick's, Frat), or the M5 practice window. Overall WRMSSE unless stated.
FRAT_BENCH = [tuple(r) for r in _R["FRAT_BENCH"]]
# Frat, five seeds: (label, group, mean, lo, hi).
FRAT_DEV = [tuple(r) for r in _R["FRAT_DEV"]]
COMBO = _R["COMBO"]
NEW_ITEMS = _R["NEW_ITEMS"]  # new-item error, item by store
PLAN_PAIRS = [tuple(r) for r in _R["PLAN_PAIRS"]]  # share picking the better week (%), miss on size (log)
UNUSUAL = [tuple(r) for r in _R["UNUSUAL"]]  # unusual plans, usual plans
ANA_DEV = [tuple(r) for r in _R["ANA_DEV"]]

DOM28 = _tuples(_R["DOM28"])  # code: (LightGBM, lift, world 26-week level, world 8-week level)
DOM28_NAME = {  # labels
    "bat": "Bath soap", "oat": "Oatmeal", "ptw": "Paper towels", "tti": "Toilet tissue", "ber": "Beer",
    "frd": "Frozen dinners", "soa": "Soaps", "cra": "Crackers", "did": "Dish detergent", "tbr": "Toothbrushes",
    "fsf": "Front-end candies", "tna": "Canned tuna", "cig": "Cigarettes", "frj": "Frozen juices",
    "sna": "Snack crackers", "tpa": "Toothpaste", "lnd": "Laundry detergent", "fec": "Fabric softener",
    "bjc": "Bottled juice", "cer": "Cereals", "cso": "Canned soup", "gro": "Grooming products",
    "sha": "Shampoos", "coo": "Cookies", "che": "Cheese", "fre": "Frozen entrees", "sdr": "Soft drinks",
    "ana": "Analgesics",
}

# Peaks and dips (six categories, one seed): cat: {variant: (promoted, after, ordinary, overall)}.
PEAKS = {c: _tuples(v) for c, v in _R["PEAKS"].items()}
PEAK_VARIANT = {"lgb": "LightGBM", "world": "World model", "feedback": "+ own sales fed forward",
                "weight": "+ dollar and peak weighting"}  # labels

# Dominick's development ladder, six categories, three-seed means.
LADDER_STEPS = ["8-week level", "+ brand switching", "+ normalised roll-forward", "+ average of three fits",
                "+ twice the steps"]  # labels
LADDER = {c: (list(v[0]), v[1]) for c, v in _R["LADDER"].items()}  # cat: ([value per step], LightGBM)
# seed range of the first and the chosen (average of three fits) step
LADDER_RANGE = {c: (tuple(v[0]), tuple(v[1])) for c, v in _R["LADDER_RANGE"].items()}
# Learned share of the summed lift that is new sales, range over runs.
SWITCH_SHARE = _tuples(_R["SWITCH_SHARE"])
# Item roles: (LightGBM, LightGBM + role, world, world + role, world + role picks level).
ROLES = _tuples(_R["ROLES"])
# Recent and longer level together: (26-week, 8-week, both levels as inputs, LightGBM).
LEVEL_VIEWS = _tuples(_R["LEVEL_VIEWS"])
# M5 practice window development: (label, mean, lo, hi).
M5_DEV = [tuple(r) for r in _R["M5_DEV"]]
# Change C (twice the training steps): data: (normal, twice the steps), three-seed means.
STEPS2X = _tuples(_R["STEPS2X"])
