"""Every recorded number of paper 1, regenerated from the run records.

  python scripts/records_snapshot.py --results <folder of run folders> [--results <another>] \
      --out results/records/snapshot.json

The paper's figures read paper/figs/rwdata.py. The blocks there that are not
scored from results/promoted/ (TimesFM on the weekly finals, the development
study, the analyses of saved forecasts, the plan read-back) are built here from
the run folders themselves, so no number in the paper is copied by hand.

A run folder is "<UTC stamp>_<config name>_<8 hex>"; job runs add _seed<N> to the
config name. Each holds manifest.json (the full config under "config"),
metrics.json (scores per window), forecasts.csv and, for models that save,
checkpoint_origin<k>.pt. A run is found by the config name in its manifest. When
several folders hold the same config and seed (a repeat), the one that finished
last is used; the folder name of every run used is written next to each value.

Scores are averaged over a run's windows the same way the job summary does
(src/rwm/experiments/job.py), then over seeds. Values are written unrounded;
results/README.md shows them rounded.

Needs, besides the metrics: forecasts.csv for the M5 April and Frat final runs,
checkpoint_origin*.pt for the brand-switching runs (the learned share) and, for the
plan read-back, the world-model checkpoints and the M5 data (through
scripts/plan_readback.py; a plan_readback.json already written by that script
next to the run, or under <results>/analysis/plan_readback/, is used instead).

Exits with an error, writing nothing, if any run is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CATS = ["che", "sna", "sdr", "bjc", "ana", "fre"]
DOM28 = ["bat", "oat", "ptw", "tti", "ber", "frd", "soa", "cra", "did", "tbr", "fsf", "tna", "cig", "frj", "sna",
         "tpa", "lnd", "fec", "bjc", "cer", "cso", "gro", "sha", "coo", "che", "fre", "sdr", "ana"]
M5_WINDOW_CONFIG = {"mar": "m5_confirm_mar", "apr": "m5_confirm_apr", "practice": "m5_validation", "test": "m5_test"}
ITEM_STORE = "item_idxstore_id"
S3, S5 = (1, 2, 3), (1, 2, 3, 4, 5)


class MissingRun(LookupError):
    pass


# ------------------------------------------------------------------ records ---
class Records:
    """The run folders under one or more results folders, found by config name."""

    def __init__(self, roots):
        self.runs = defaultdict(list)  # config name as run (with _seed<N>) -> [(finished, folder name, path)]
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                raise FileNotFoundError(root)
            for d in sorted(root.iterdir()):
                if not (d.is_dir() and (d / "manifest.json").exists() and (d / "metrics.json").exists()):
                    continue
                man = json.loads((d / "manifest.json").read_text())
                name = (man.get("config") or {}).get("name")
                if not name:
                    continue
                self.runs[name].append((man.get("finished_utc") or "", d.name, d))
        self._metrics = {}

    def folder(self, config: str, seed: int | None = None, need: str | None = None) -> Path:
        """The run of a config and seed (None: the run made without --seed) that
        finished last; with `need`, only runs holding that file."""
        name = config if seed is None else f"{config}_seed{seed}"
        found = [r for r in self.runs.get(name, []) if need is None or (r[2] / need).exists()]
        if not found:
            raise MissingRun(name + (f" (with {need})" if need else ""))
        return max(found)[2]

    def metrics(self, folder: Path) -> dict:
        if folder not in self._metrics:
            self._metrics[folder] = json.loads((folder / "metrics.json").read_text())
        return self._metrics[folder]


# --------------------------------------------------------------- measures ---
# Each takes a run's metrics.json and returns one number: averaged over the
# run's windows, as in rwm.experiments.job.summary_row.
def overall(m: dict) -> float:
    return m["wrmsse"]


def _per_split(m: dict, fn) -> float:
    return mean(fn(s) for s in m["splits"])


def level(name: str):
    def get(m):
        def one(s):
            for b in s["by_level"]:
                if b["level"] == name:
                    return b["wrmsse"]
            raise KeyError(name)
        return _per_split(m, one)
    return get


def fa(m: dict) -> float:
    return _per_split(m, lambda s: s["forecast_to_actual"])


def condition_on(c: str):
    return lambda m: _per_split(m, lambda s: s["by_condition"][c]["on"]["wrmsse"])


def peak_fa(group: str):
    return lambda m: _per_split(m, lambda s: s["peaks"][group]["forecast_to_actual"])


def key(k: str, scale: float = 1.0):
    return lambda m: scale * _per_split(m, lambda s: s[k])


ixs = level(ITEM_STORE)


# ------------------------------------------------------------------ values ---
class Builder:
    """Collects values and, for each, the run folders it came from."""

    def __init__(self, rec: Records):
        self.rec = rec
        self.sources = {}

    def _note(self, path: str, folders):
        self.sources.setdefault(path, [])
        for f in folders:
            name = Path(f).name
            if name not in self.sources[path]:
                self.sources[path].append(name)

    def seeds(self, path: str, config: str, seeds, measure=overall) -> dict:
        """A measure for each seed of a config: mean, lowest, highest, per seed."""
        seeds = list(seeds) if seeds is not None else [None]
        folders = [self.rec.folder(config, s) for s in seeds]
        vals = [measure(self.rec.metrics(f)) for f in folders]
        self._note(path, folders)
        return {"mean": mean(vals), "lo": min(vals), "hi": max(vals), "values": vals}

    def one(self, path: str, config: str, seeds=(1,), measure=overall) -> float:
        return self.seeds(path, config, seeds, measure)["mean"]

    def range3(self, path, config, seeds, measure=overall):
        r = self.seeds(path, config, seeds, measure)
        return (r["mean"], r["lo"], r["hi"])


def _block(b: Builder, value, computation: str) -> dict:
    out = {"value": value, "computation": computation, "sources": b.sources}
    b.sources = {}
    return out


# Each block function returns {"value", "computation", "sources"}; the value has
# the shape rwdata.py exposes (tuples become lists, integer keys strings).

def tfm_final(b):
    v = {}
    for d in ["frat"] + CATS:
        cfg = "frat_final_timesfm_mean" if d == "frat" else f"dominicks_{d}_final_timesfm_mean"
        v[d] = {"overall": b.one(f"{d}.overall", cfg), "ixs": b.one(f"{d}.ixs", cfg, measure=ixs),
                "fa": b.one(f"{d}.fa", cfg, measure=fa)}
        if d != "frat":
            v[d]["promo_fa"] = b.one(f"{d}.promo_fa", cfg, measure=peak_fa("promoted"))
    return _block(b, v, "<data>_final_timesfm_mean seed 1: overall WRMSSE, item-by-store WRMSSE, forecast/actual, "
                         "promoted-week forecast/actual (peaks.promoted)")


def tfm_plan_final(b):
    v = {}
    for d in ["frat"] + CATS:
        cfg = "frat_final_timesfm_plan" if d == "frat" else f"dominicks_{d}_final_timesfm_plan"
        v[d] = {"overall": b.one(f"{d}.overall", cfg), "ixs": b.one(f"{d}.ixs", cfg, measure=ixs)}
        if d == "frat":
            v[d]["fa"] = b.one(f"{d}.fa", cfg, measure=fa)
        else:
            v[d]["promo_fa"] = b.one(f"{d}.promo_fa", cfg, measure=peak_fa("promoted"))
    return _block(b, v, "<data>_final_timesfm_plan seed 1, measures as TFM_FINAL")


def tfm_plan_m5(b):
    v = {w: {"overall": b.one(f"{w}.overall", f"{c}_timesfm_plan"),
             "ixs": b.one(f"{w}.ixs", f"{c}_timesfm_plan", measure=ixs)} for w, c in M5_WINDOW_CONFIG.items()}
    return _block(b, v, "m5_<window>_timesfm_plan seed 1: overall and item-by-store WRMSSE")


def tfm_frat_promo_on(b):
    v = {"history": b.one("history", "frat_final_timesfm_mean", measure=condition_on("promo")),
         "plan": b.one("plan", "frat_final_timesfm_plan", measure=condition_on("promo"))}
    return _block(b, v, "frat_final_timesfm_mean / _plan seed 1: by_condition.promo.on WRMSSE")


APR_MODELS = {"world": ("m5_confirm_apr_world_model_v1", S3), "lift": ("m5_confirm_apr_lift", S3),
              "graph": ("m5_confirm_apr_graph", S3), "tfm": ("m5_confirm_apr_timesfm_mean", (1,)),
              "lgb": ("m5_confirm_apr_lightgbm", S3)}


def _price_cut(b, series: list[str], label: str):
    ac = _script("after_confirm")
    v, sold, sold_from = {}, None, None
    for m, (cfg, seeds) in APR_MODELS.items():
        v[m] = []
        for s in seeds:
            f = b.rec.folder(cfg, s, need="forecasts.csv")
            sums = ac.series_sums(str(f), series)
            if set(sums.index) != set(series):
                raise MissingRun(f"{f.name}: forecasts.csv lacks some of {series}")
            units = float(sums["units"].sum())
            if sold is None:
                sold, sold_from = units, f
            elif not math.isclose(sold, units):
                raise ValueError(f"{f.name}: units sold differ between runs ({units} against {sold})")
            v[m].append(float(sums["forecast"].sum()))
            b._note(m, [f])
    v = {"sold": int(round(sold)), **v}
    b._note("sold", [sold_from])
    return _block(b, v, f"forecasts.csv of the April confirmation runs (each seed): units and forecast summed over "
                         f"the 28 days and over {label} (scripts/after_confirm.py series_sums)")


def price_cut(b):
    return _price_cut(b, [s for s in _script("after_confirm").FAILED if s.startswith("HOUSEHOLD_1_146")],
                      "HOUSEHOLD_1_146 in WI_2, CA_3, TX_3, CA_2")


def price_cut_foods(b):
    return _price_cut(b, ["FOODS_3_443_WI_1"], "FOODS_3_443_WI_1")


def lift_apr_seed2(b):
    ac = _script("after_confirm")
    s2 = b.rec.folder("m5_confirm_apr_lift", 2, need="forecasts.csv")
    s1 = b.rec.folder("m5_confirm_apr_lift", 1, need="forecasts.csv")
    g = ac.seed_excess(str(s2), str(s1))
    excess = g["seed2"].sum() - g["units"].sum()
    v = {"forecast": int(round(g["seed2"].sum())), "sold": int(round(g["units"].sum())),
         "top5_share": float(g["excess_seed2"].head(5).sum() / excess)}
    for k in v:
        b._note(k, [s2, s1])
    return _block(b, v, "m5_confirm_apr_lift seed 2 against seed 1 (scripts/after_confirm.py seed_excess): "
                         "summed forecast, summed units, share of the excess held by the five series with most excess")


def frat_weeks(b):
    cf = _script("compare_forecasts")
    world = [b.rec.folder("frat_final_world_model", s, need="forecasts.csv") for s in S3]
    lgb = [b.rec.folder("frat_final_lightgbm", s, need="forecasts.csv") for s in S3]
    df = cf.frame([str(f) for f in world], [str(f) for f in lgb])
    tot = df.groupby("date")[["units", "fa", "fb"]].sum().sort_index()
    weeks = [f"{d.day} {d.strftime('%b')}" for d in tot.index]
    out = {
        "FRAT_WEEKS": weeks,
        "FRAT_ACTUAL": [int(round(u)) for u in tot["units"]],
        "FRAT_BIAS": {"world": [float(100 * (x / u - 1)) for x, u in zip(tot["fa"], tot["units"])],
                      "lgb": [float(100 * (x / u - 1)) for x, u in zip(tot["fb"], tot["units"])]},
    }
    blocks = {}
    for k, val in out.items():
        b._note("world", world)
        b._note("lgb", lgb)
        blocks[k] = _block(b, val, "scripts/compare_forecasts.py frame() on frat_final_world_model (a) and "
                                   "frat_final_lightgbm (b), seeds 1-3 averaged: weekly totals over all item-stores, "
                                   "bias = 100 (forecast / actual - 1); week label is the date in forecasts.csv")
    return blocks


def bands(b):
    cf = _script("compare_forecasts")
    world = [b.rec.folder("m5_confirm_apr_world_model_v1", s, need="forecasts.csv") for s in S3]
    lgb = [b.rec.folder("m5_confirm_apr_lightgbm", s, need="forecasts.csv") for s in S3]
    march = b.rec.folder("m5_confirm_mar_world_model_v1", 1, need="forecasts.csv")
    df = cf.frame([str(f) for f in world], [str(f) for f in lgb], band_from=str(march))
    t = cf.table(df, "sales band", ["World model", "LightGBM"])
    order = ["no sales", "under 0.5 a day", "0.5 to 1", "1 to 3", "3 to 10", "over 10"]
    t = t.reindex(order)
    out = {
        "BAND_UNITS": [int(round(x)) for x in t["actual units"]],
        "BAND_BIAS": {"world": [float(x) for x in t["World model bias %"]],
                      "lgb": [float(x) for x in t["LightGBM bias %"]]},
        "BAND_CLOSER": [float(x) for x in t["pairs where World model is closer %"]],
    }
    blocks = {}
    for k, val in out.items():
        b._note("world", world)
        b._note("lgb", lgb)
        b._note("band_from", [march])
        blocks[k] = _block(b, val, "scripts/compare_forecasts.py frame(band_from=March world model seed 1) and "
                                   "table(by 'sales band') on the April world model and LightGBM, seeds 1-3 averaged")
    return blocks


READBACK_RUNS = {"mar": "m5_confirm_mar_world_model_v1", "apr": "m5_confirm_apr_world_model_v1",
                 "test": "m5_test_world_model_roll_norm"}
DISCOUNT = "price / regular price"


def _readback_rows(rec: Records, folder: Path) -> list[dict]:
    """Rows of scripts/plan_readback.py for a run: saved by that script, or made now."""
    saved = [folder / "plan_readback.json", *sorted((folder.parent / "analysis" / "plan_readback").glob("*.json"))]
    for f in saved:
        if f.exists():
            got = json.loads(f.read_text())
            if got.get("run") == folder.name:
                return got["rows"]
    probe = _script("plan_readback")
    config = json.loads((folder / "manifest.json").read_text())["config"]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.yaml"
        import yaml

        cfg.write_text(yaml.safe_dump(config))
        out = Path(tmp) / "readback.json"
        probe.main(["--config", str(cfg), "--run", str(folder), "--out", str(out)])
        return json.loads(out.read_text())["rows"]


def readback(b):
    v = {}
    for w, cfg in READBACK_RUNS.items():
        f = b.rec.folder(cfg, 1, need="checkpoint_origin0.pt")
        rows = [r for r in _readback_rows(b.rec, f) if r["plan input"] == DISCOUNT]
        v[w] = {str(r["periods back"]): [r["from state before window"], r["from rolled state"]]
                for r in rows if r["periods back"] in (0, 1, 7)}
        if set(v[w]) != {"0", "1", "7"}:
            raise MissingRun(f"{f.name}: read-back lacks the discount at 0, 1 and 7 days back")
        b._note(w, [f])
    return _block(b, v, "scripts/plan_readback.py (6,000 pairs, seed 0) on seed 1 of the world model: R squared of "
                        "the discount (price / regular price) at 0, 1, 7 days back, (control, rolled state)")


# ------------------------------------------------------ development study ---
FRAT_BENCH_RUNS = [("Seasonal naive", "frat_seasonal_naive"), ("Average of last 8 weeks", "frat_recent_average"),
                   ("LightGBM, price only", "frat_lightgbm_price_only"),
                   ("LightGBM, all recorded levers", "frat_lightgbm")]


def frat_bench(b):
    v = [[label, b.one(label, cfg, seeds=None)] for label, cfg in FRAT_BENCH_RUNS]
    return _block(b, v, "single run (no seed) of each config: overall WRMSSE, mean of three windows")


# (label, group, config, whether the range is shown)
FRAT_DEV_RUNS = [
    ("LightGBM, all recorded levers", "ref", "frat_dev_lightgbm", True),
    ("Item encoder and readout", "base", "frat_dev_state_model", True),
    ("+ item-to-item, conditional similarity", "neighbours", "frat_dev_state_model_similarity", True),
    ("+ item-to-item, attention", "neighbours", "frat_dev_state_model_attention", True),
    ("+ item-to-item, both", "neighbours", "frat_dev_state_model_both", True),
    ("+ item-to-item, same category only", "neighbours", "frat_dev_state_model_both_within_category", True),
    ("+ regular price", "lever", "frat_dev_state_model_regular_price", True),
    ("Lift model: regular price, baseline times lift", "ref", "frat_dev_state_model_lift", True),
    ("Lift + plan of the four weeks before", "inputs", "frat_dev_state_model_lift_earlier_weeks", True),
    ("Lift + same week last year", "inputs", "frat_dev_state_model_lift_year_ago", True),
    ("Lift + recorded base price as regular price", "inputs", "frat_dev_state_model_lift_base_price", True),
    ("Lift + US seasonal calendar", "inputs", "frat_dev_state_model_lift_calendar", True),
    ("Lift + promotion crowding", "inputs", "frat_dev_state_model_lift_crowding", True),
    ("Lift + base price, calendar and crowding", "inputs", "frat_dev_state_model_lift_merchant_inputs", True),
    ("Lift + latent loss alongside, weight 0.1", "jepa", "frat_dev_state_model_lift_jepa_joint_light", True),
    ("Lift + latent loss alongside, weight 2", "jepa", "frat_dev_state_model_lift_jepa_joint_heavy", True),
    ("Lift, latent pretraining, then all trained", "jepa", "frat_dev_state_model_lift_jepa_pretrain_full", True),
    ("Lift, pretraining, encoder at a tenth of the rate", "jepa", "frat_dev_state_model_lift_jepa_pretrain_low_lr", True),
    ("Lift, pretraining, encoder frozen", "jepa", "frat_dev_state_model_lift_jepa_pretrain_frozen", True),
    ("Lift, long pretraining, encoder at a tenth", "jepa", "frat_dev_state_model_lift_jepa_pretrain_long_low_lr", True),
    ("Lift, stopped on validation weeks", "training", "frat_dev_lift_validation_stop", True),
    ("Lift + totals term in the loss", "training", "frat_dev_lift_totals_loss", True),
    ("Lift + readout dropout and weight decay", "training", "frat_dev_lift_dropout_decay", True),
    ("World model, baseline and lift", "world", "frat_dev_world_model_plain", True),
    ("World model + latent check alongside", "world", "frat_dev_world_model_latent", True),
    ("World model, long latent pretraining", "world", "frat_dev_world_model_best_recipe", True),
    # results/README.md records no range for these three; the figure draws none
    ("World model + plan split (after DWM)", "world", "frat_dev_world_model_plan_split", False),
    ("World model + latent check, VICReg, near-week weighting", "world", "frat_dev_world_model_checked", False),
    ("World model + all three", "world", "frat_dev_world_model_plan_split_checked", False),
]


def frat_dev(b):
    v = []
    for label, group, cfg, ranged in FRAT_DEV_RUNS:
        mu, lo, hi = b.range3(label, cfg, S5)
        v.append([label, group, mu, lo if ranged else None, hi if ranged else None])
    return _block(b, v, "Frat development windows, seeds 1-5: mean, lowest and highest overall WRMSSE")


def frat_dev_ref(b):
    v = {"world": b.one("world", "frat_dev_world_model_plain", S5), "lgb": b.one("lgb", "frat_dev_lightgbm", S5),
         "lift": b.one("lift", "frat_dev_state_model_lift", S5)}
    return _block(b, v, "Frat development windows, seeds 1-5, mean overall WRMSSE")


def combo(b):
    v = {"frat": {"lgb": b.one("frat.lgb", "frat_dev_lightgbm", S5),
                  "lift": b.one("frat.lift", "frat_dev_state_model_lift", S5),
                  "world": b.one("frat.world", "frat_dev_world_model_plain", S5),
                  "lift_lgb": b.one("frat.lift_lgb", "frat_dev_combination_lift_lightgbm", S5),
                  "world_lgb": b.one("frat.world_lgb", "frat_dev_combination_world_model_lightgbm", S5),
                  "negbin": b.one("frat.negbin", "frat_dev_lift_negative_binomial", S5)},
         "ana": {"lgb": b.one("ana.lgb", "dominicks_ana_dev_lightgbm", None),
                 "lift": b.one("ana.lift", "dominicks_ana_dev_state_model_lift", S3),
                 "lift_lgb": b.one("ana.lift_lgb", "dominicks_ana_dev_combination_lift_lightgbm", S3),
                 "world_lgb": b.one("ana.world_lgb", "dominicks_ana_dev_combination_world_model_lightgbm", S3),
                 "negbin": b.one("ana.negbin", "dominicks_ana_dev_lift_negative_binomial", S3)}}
    return _block(b, v, "mean overall WRMSSE over seeds (Frat 1-5; analgesics 1-3, LightGBM the single unseeded run)")


def new_items(b):
    m = key("new_items")
    v = {"frat": {"zero": b.one("frat.zero", "frat_dev_new_items_lift_no_cold_start", S3, m),
                  "lgb": b.one("frat.lgb", "frat_dev_new_items_lightgbm_cold_start", S3, m),
                  "borrow": b.one("frat.borrow", "frat_dev_new_items_lift_borrow", S3, m),
                  "borrow_hidden": b.one("frat.borrow_hidden", "frat_dev_new_items_lift_borrow_hidden_label", S3, m)},
         "ana": {"lgb": b.one("ana.lgb", "dominicks_ana_dev_new_items_lightgbm_cold_start", S3, m),
                 "borrow": b.one("ana.borrow", "dominicks_ana_dev_new_items_lift_borrow", S3, m)}}
    return _block(b, v, "splits[].new_items (item-by-store WRMSSE on held-out products), mean over windows and seeds 1-3")


PLAN_PAIR_RUNS = [("LightGBM", "frat_dev_plan_pairs_lightgbm"), ("Lift model", "frat_dev_plan_pairs_lift"),
                  ("World model", "frat_dev_plan_pairs_world_model"),
                  ("World model + plan split", "frat_dev_plan_pairs_world_model_plan_split")]


def plan_pairs(b):
    v = [[label, b.one(label, cfg, S3, key("plan_pair_order", 100.0)),
          b.one(label, cfg, S3, key("plan_pair_change_error"))] for label, cfg in PLAN_PAIR_RUNS]
    return _block(b, v, "splits[].plan_pair_order (as %) and plan_pair_change_error, mean over windows and seeds 1-3")


UNUSUAL_RUNS = [("LightGBM", "frat_dev_rare_for_item_lightgbm"), ("Lift model", "frat_dev_rare_for_item_lift"),
                ("World model", "frat_dev_rare_for_item_world_model")]


def unusual(b):
    v = [[label, b.one(label, cfg, S3, key("rare_plans")), b.one(label, cfg, S3, key("usual_plans"))]
         for label, cfg in UNUSUAL_RUNS]
    return _block(b, v, "splits[].rare_plans and usual_plans (lever mix under 2% of the item's training weeks), "
                        "mean over windows and seeds 1-3")


ANA_DEV_RUNS = [
    ("Seasonal naive", "dominicks_ana_dev_seasonal_naive", None),
    ("Average of last 8 weeks", "dominicks_ana_dev_recent_average", None),
    ("LightGBM", "dominicks_ana_dev_lightgbm", None),
    ("Item encoder and readout", "dominicks_ana_dev_state_model", S3),
    ("Lift model", "dominicks_ana_dev_state_model_lift", S3),
    ("Lift + item-to-item", "dominicks_ana_dev_state_model_item_to_item", S3),
    ("Lift, held price as regular", "dominicks_ana_dev_state_model_lift_held_price", S3),
    ("Lift + held price, calendar, crowding", "dominicks_ana_dev_state_model_lift_merchant_inputs", S3),
    ("Lift, 24,000 steps", "dominicks_ana_dev_state_model_lift_longer", S3),
    ("Lift, stop on validation", "dominicks_ana_dev_lift_validation_stop", S3),
    ("Lift, totals in the loss", "dominicks_ana_dev_lift_totals_loss", S3),
    ("World model, totals in the loss", "dominicks_ana_dev_world_model_fixes", S3),
    ("World model, no totals loss", "dominicks_ana_dev_world_model_plain", S3),
    ("World model + plan split", "dominicks_ana_dev_world_model_plan_split", S3),
    ("World model + latent check, VICReg, near weeks", "dominicks_ana_dev_world_model_checked", S3),
]


def ana_dev(b):
    v = [[label, b.one(label, cfg, seeds)] for label, cfg, seeds in ANA_DEV_RUNS]
    return _block(b, v, "Dominick's analgesics development windows: mean overall WRMSSE over seeds 1-3 "
                        "(benchmarks: the single unseeded run)")


def dom28(b):
    v = {}
    for c in DOM28:
        if c == "ana":  # the five-seed means of the overnight runs
            row = [b.one("ana.lgb", "dominicks_ana_dev_lightgbm", S5),
                   b.one("ana.lift", "dominicks_ana_dev_state_model_lift", S5),
                   b.one("ana.world26", "dominicks_ana_dev_world_model_plain", S5)]
        else:
            row = [b.one(f"{c}.lgb", f"dominicks_{c}_dev_lightgbm"), b.one(f"{c}.lift", f"dominicks_{c}_dev_lift"),
                   b.one(f"{c}.world26", f"dominicks_{c}_dev_world_model")]
        row.append(b.one(f"{c}.world8", f"dominicks_{c}_dev_world_model_level8"))
        v[c] = row
    return _block(b, v, "Dominick's development windows, seed 1, overall WRMSSE: (LightGBM, lift, world model "
                        "26-week level, world model 8-week level); analgesics first three: seeds 1-5 mean")


PEAK_RUNS = {"lgb": "peaks_lightgbm", "world": "peaks_world_model", "feedback": "peaks_feedback",
             "weight": "peaks_weighted"}


def peaks(b):
    v = {}
    for c in CATS:
        v[c] = {}
        for k, suffix in PEAK_RUNS.items():
            cfg = f"dominicks_{c}_dev_{suffix}"
            v[c][k] = [b.one(f"{c}.{k}", cfg, measure=peak_fa(g)) for g in ("promoted", "after_promo", "ordinary")]
            v[c][k].append(b.one(f"{c}.{k}", cfg))
    return _block(b, v, "dominicks_<cat>_dev_peaks_* seed 1: forecast/actual in promoted, after-promotion and "
                        "ordinary weeks (splits[].peaks, mean over windows), and overall WRMSSE")


LADDER_RUNS = ["peaks_world_model", "peaks_world_model_switching", "peaks_world_model_switching_roll_norm",
               "peaks_world_model_switching_avg3_roll_norm", "peaks_world_model_switching_avg3_roll_norm_steps2x"]


def ladder(b):
    lad, rng = {}, {}
    for c in CATS:
        steps = [b.seeds(f"{c}.{s}", f"dominicks_{c}_dev_{s}", S3) for s in LADDER_RUNS]
        lad[c] = [[s["mean"] for s in steps], b.one(f"{c}.lgb", f"dominicks_{c}_dev_peaks_lightgbm", S3)]
        rng[c] = [[steps[0]["lo"], steps[0]["hi"]], [steps[3]["lo"], steps[3]["hi"]]]
    src = dict(b.sources)
    out = {"LADDER": _block(b, lad, "dominicks_<cat>_dev_<step> seeds 1-3, mean overall WRMSSE per step; "
                                    "LightGBM dominicks_<cat>_dev_peaks_lightgbm seeds 1-3")}
    b.sources = {k: v for k, v in src.items() if k.endswith((LADDER_RUNS[0], LADDER_RUNS[3]))}
    out["LADDER_RANGE"] = _block(b, rng, "lowest and highest seed (1-3) of the first step (8-week level) and of "
                                         "the chosen step (average of three fits)")
    return out


def switch_share(b):
    import torch

    v = {}
    for c in CATS:
        shares = []
        for s in S3:
            f = b.rec.folder(f"dominicks_{c}_dev_peaks_world_model_switching", s, need="checkpoint_origin0.pt")
            for ck in sorted(f.glob("checkpoint_origin*.pt")):
                share = torch.load(ck, map_location="cpu", weights_only=True).get("new_share")
                if share is None:
                    raise MissingRun(f"{f.name}/{ck.name}: no learned share")
                shares.append(float(share))
            b._note(c, [f])
        v[c] = [min(shares), max(shares)]
    return _block(b, v, "new_share saved in every checkpoint_origin*.pt of dominicks_<cat>_dev_peaks_world_model_"
                        "switching seeds 1-3: lowest and highest over runs and windows")


ROLE_RUNS = ["peaks_lightgbm", "peaks_lightgbm_roles", "peaks_world_model", "peaks_world_model_roles",
             "peaks_world_model_role_level"]


def roles(b):
    v = {c: [b.one(f"{c}.{r}", f"dominicks_{c}_dev_{r}") for r in ROLE_RUNS] for c in CATS}
    return _block(b, v, "dominicks_<cat>_dev_<run> seed 1, overall WRMSSE")


def level_views(b, dom):
    v = {}
    for c in CATS:
        lgb, _, w26, w8 = dom["value"][c]
        v[c] = [w26, w8, b.one(f"{c}.both", f"dominicks_{c}_dev_peaks_level_views"), lgb]
        for k in ("lgb", "world26", "world8"):
            b._note(f"{c}.{k}", dom["sources"][f"{c}.{k}"])
    return _block(b, v, "(26-week, 8-week, both levels, LightGBM): the DOM28 values and dominicks_<cat>_dev_peaks_"
                        "level_views seed 1, overall WRMSSE")


M5_DEV_RUNS = [
    ("Seasonal naive", "m5_validation_seasonal_naive", None, False),
    ("LightGBM, one model per store", "m5_validation_lightgbm", None, False),
    ("Lift model (3 seeds)", "m5_validation_lift", S3, True),
    ("Item encoder and readout (5 seeds)", "m5_validation_state_model_mixed_precision", (None, 1, 2, 3, 4), True),
    ("World model (3 seeds)", "m5_validation_world_model", S3, True),
    ("World model + normalised roll-forward (3 seeds)", "m5_validation_world_model_roll_norm", S3, True),
    ("+ average of three fits (1 run)", "m5_validation_world_model_roll_norm_avg3", (1,), False),
    ("+ weekday as seven inputs (seed 1)", "m5_validation_world_model_v2_weekday", (1,), False),
    ("+ weekday and event kinds (seed 1)", "m5_validation_world_model_v2_weekday_events", (1,), False),
    ("+ 60,000 training steps (seed 1)", "m5_validation_world_model_v2_weekday_events_60k", (1,), False),
    ("+ item level against past-year level (seed 1)", "m5_validation_world_model_v2_long_level", (1,), False),
]


def m5_dev(b):
    v = []
    for label, cfg, seeds, ranged in M5_DEV_RUNS:
        mu, lo, hi = b.range3(label, cfg, seeds)
        v.append([label, mu, lo if ranged else None, hi if ranged else None])
    return _block(b, v, "M5 practice window, overall WRMSSE: mean, lowest and highest over the seeds named "
                        "(the item encoder's five runs are the unseeded run and seeds 1-4)")


STEPS2X = {**{c: (f"dominicks_{c}_dev_peaks_world_model_switching_avg3_roll_norm",
                  f"dominicks_{c}_dev_peaks_world_model_switching_avg3_roll_norm_steps2x") for c in CATS},
           "frat": ("frat_dev_world_model_plain", "frat_dev_world_model_plain_steps2x")}


def steps2x(b):
    v = {d: [b.one(f"{d}.normal", n, S3), b.one(f"{d}.steps2x", t, S3)] for d, (n, t) in STEPS2X.items()}
    return _block(b, v, "Change C: (normal training, twice the steps), mean overall WRMSSE over seeds 1-3")


def ledger(b, blocks):
    """The 'what did not work' rows; the numbers in them come from the runs."""
    m5 = {r[0]: r for r in blocks["M5_DEV"]["value"]}
    v1 = b.one("v1", "m5_validation_world_model_roll_norm", (1,))
    rn = m5["World model + normalised roll-forward (3 seeds)"]
    tl = b.seeds("totals", "dominicks_ana_dev_lift_totals_loss", S3)
    tl_fa = b.one("totals", "dominicks_ana_dev_lift_totals_loss", S3, fa)
    vs = b.one("totals", "dominicks_ana_dev_lift_validation_stop", S3)
    tricks = [b.one("tricks", c, S5) for c in ("frat_dev_world_model_plan_split", "frat_dev_world_model_checked",
                                               "frat_dev_world_model_plan_split_checked")]
    wm = b.one("tricks", "frat_dev_world_model_plain", S5)
    pk = blocks["PEAKS"]["value"]
    worse = sum(pk[c]["feedback"][3] > pk[c]["world"][3] for c in CATS)
    tfm_o = [b.one("timesfm", f"{c}_timesfm") for c in M5_WINDOW_CONFIG.values()]
    tfm_u = [round(100 * (1 - b.one("timesfm", f"{c}_timesfm", measure=fa))) for c in M5_WINDOW_CONFIG.values()]
    failed = b.one("lift", "m5_confirm_apr_lift", (2,))
    for src in ("M5_DEV", "PEAKS", "ROLES", "STEPS2X"):
        b._note(src, [f for fs in blocks[src]["sources"].values() for f in fs])
    rows = [
        ["Twice the training steps", "Dominick's (6) and Frat, dev., 3 seeds",
         "Better in 2, worse in 3, level or mixed in 2"],
        ["Weekday as seven inputs", "M5 practice, seed 1", f"{m5['+ weekday as seven inputs (seed 1)'][1]:.3f} against {v1:.3f}"],
        ["Event kinds (with weekday)", "M5 practice, seed 1",
         f"{m5['+ weekday and event kinds (seed 1)'][1]:.3f} against {v1:.3f}"],
        ["60,000 training steps", "M5 practice, seed 1",
         f"{m5['+ 60,000 training steps (seed 1)'][1]:.3f} against {v1:.3f}; gain below seed range {rn[3] - rn[2]:.3f}"],
        ["Item level against past-year level", "M5 practice, seed 1",
         f"{m5['+ item level against past-year level (seed 1)'][1]:.3f} against {v1:.3f}"],
        ["Totals term in the loss", "Dominick's analgesics, dev.",
         f"{tl['mean']:.3f} against {vs:.3f}; total {round(100 * (1 - tl_fa))}\\% under"],
        ["Negative binomial likelihood", "Frat, analgesics, dev.", "Level"],
        ["Plan split, latent check, near-week weighting", "Frat, dev., 5 seeds",
         f"{min(tricks):.3f} to {max(tricks):.3f} against {wm:.3f}"],
        ["Own forecast fed into the roll-forward", "Dominick's (6), dev.", f"Worse overall in {worse} of 6"],
        ["Item role labels", "Dominick's (6), dev.",
         f"No effect on LightGBM; snack crackers {blocks['ROLES']['value']['sna'][4]:.3f} for the world model"],
        ["Averaging three fits", "M5 practice, seed 1",
         f"{m5['+ average of three fits (1 run)'][1]:.3f} against {rn[1]:.3f} (kept on Dominick's)"],
        ["Graph model over product attributes", "M5, four windows, 3 seeds", "Level with LightGBM item by store"],
        ["TimesFM, median output", "M5, four windows",
         f"Overall {min(tfm_o):.2f} to {max(tfm_o):.2f}; totals {min(tfm_u)}\\% to {max(tfm_u)}\\% under"],
        ["Lift model (no roll-forward)", "M5, four windows, 3 seeds", f"One seed failed in April ({failed:.3f})"],
    ]
    return _block(b, rows, "rows of the 'what did not work' table: numbers from the runs named in sources; the wording "
                           "of the verdicts ('Level', the Change C count, see STEPS2X) is editorial")


# ------------------------------------------------------------------ build ---
SIMPLE = {
    "TFM_FINAL": tfm_final, "TFM_PLAN_FINAL": tfm_plan_final, "TFM_PLAN_M5": tfm_plan_m5,
    "TFM_FRAT_PROMO_ON": tfm_frat_promo_on, "PRICE_CUT": price_cut, "PRICE_CUT_FOODS": price_cut_foods,
    "LIFT_APR_SEED2": lift_apr_seed2, "READBACK": readback, "FRAT_BENCH": frat_bench, "FRAT_DEV": frat_dev,
    "FRAT_DEV_REF": frat_dev_ref, "COMBO": combo, "NEW_ITEMS": new_items, "PLAN_PAIRS": plan_pairs,
    "UNUSUAL": unusual, "ANA_DEV": ana_dev, "DOM28": dom28, "PEAKS": peaks, "SWITCH_SHARE": switch_share,
    "ROLES": roles, "M5_DEV": m5_dev, "STEPS2X": steps2x,
}
MULTI = {"frat_weeks": frat_weeks, "bands": bands, "ladder": ladder}
BLOCKS = ["TFM_FINAL", "TFM_PLAN_FINAL", "TFM_PLAN_M5", "TFM_FRAT_PROMO_ON", "PRICE_CUT", "PRICE_CUT_FOODS",
          "LIFT_APR_SEED2", "FRAT_WEEKS", "FRAT_ACTUAL", "FRAT_BIAS", "FRAT_DEV_REF", "BAND_UNITS", "BAND_BIAS",
          "BAND_CLOSER", "READBACK", "LEDGER", "FRAT_BENCH", "FRAT_DEV", "COMBO", "NEW_ITEMS", "PLAN_PAIRS",
          "UNUSUAL", "ANA_DEV", "DOM28", "PEAKS", "LADDER", "LADDER_RANGE", "SWITCH_SHARE", "ROLES", "LEVEL_VIEWS",
          "M5_DEV", "STEPS2X"]


def build(rec: Records, only=None, skip=()) -> tuple[dict, list[str]]:
    """Every block that can be built, and the errors of those that cannot."""
    b, blocks, errors = Builder(rec), {}, []
    want = lambda n: (only is None or n in only) and n not in skip

    def attempt(name, fn):
        try:
            return fn()
        except (MissingRun, FileNotFoundError, KeyError, ValueError) as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            b.sources = {}
            return None

    for name, fn in SIMPLE.items():
        if want(name):
            got = attempt(name, lambda: fn(b))
            if got is not None:
                blocks[name] = got
    for name, fn in MULTI.items():
        produced = {"frat_weeks": ("FRAT_WEEKS", "FRAT_ACTUAL", "FRAT_BIAS"),
                    "bands": ("BAND_UNITS", "BAND_BIAS", "BAND_CLOSER"), "ladder": ("LADDER", "LADDER_RANGE")}[name]
        if any(want(p) for p in produced):
            got = attempt(name, lambda: fn(b))
            if got is not None:
                blocks.update({k: v for k, v in got.items() if want(k)})
    if want("LEVEL_VIEWS") and "DOM28" in blocks:
        got = attempt("LEVEL_VIEWS", lambda: level_views(b, blocks["DOM28"]))
        if got is not None:
            blocks["LEVEL_VIEWS"] = got
    if want("LEDGER") and all(k in blocks for k in ("M5_DEV", "PEAKS", "ROLES", "STEPS2X")):
        got = attempt("LEDGER", lambda: ledger(b, blocks))
        if got is not None:
            blocks["LEDGER"] = got
    for name in BLOCKS:
        if want(name) and name not in blocks and not any(e.startswith(name) for e in errors):
            errors.append(f"{name}: not built (a block it depends on failed)")
    return {k: blocks[k] for k in BLOCKS if k in blocks}, errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", required=True, action="append", help="folder of run folders (repeatable)")
    p.add_argument("--out", default=str(ROOT / "results" / "records" / "snapshot.json"))
    p.add_argument("--only", nargs="+", default=None, help="build only these blocks (for checking; not a full snapshot)")
    args = p.parse_args(argv)
    if args.only is not None and Path(args.out).resolve() == (ROOT / "results" / "records" / "snapshot.json").resolve():
        p.error("--only builds part of the snapshot; give another --out")
    rec = Records(args.results)
    blocks, errors = build(rec, only=args.only)
    if errors:
        print("Cannot build the snapshot:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "kind": "records" if args.only is None else "records (partial: " + ", ".join(args.only) + ")",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/records_snapshot.py",
        "results": [str(Path(r).resolve()) for r in args.results],
        "selection": "per config and seed, the run that finished last (manifest finished_utc)",
        "blocks": blocks,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out} ({len(blocks)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
