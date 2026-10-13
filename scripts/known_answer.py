"""Paper 2 known-answer study (docs/PAPER2_PLAN.md, amendment of 13 October 2026).

Runs every check on generated data where the effect of each part of the plan
is known, for correct models and for models with planted failures, and scores
each model's explanations against the truth.

  python scripts/known_answer.py --out results/paper2/known_answer [--worlds clean confounded]

Writes, per world and model, `<world>_<model>.json` (checks and truth scores)
and `<world>_<model>_lifts.csv.gz` (per item-week contributions, model and
true). Finished models are skipped on a rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_runs import audit_model  # noqa: E402

from rwm.audit import lever_contributions, off_plan, rank_agreement, regular_prices  # noqa: E402
from rwm.audit.zones import Episode, did  # noqa: E402
from rwm.data import load_dataset  # noqa: E402
from rwm.data.schema import DATE, PRICE, SERIES, STORE, UNITS  # noqa: E402
from rwm.data.synthetic_levers import Truth, expected_units, planned  # noqa: E402
from rwm.forecaster import build_model  # noqa: E402

ORIGIN = 253          # first forecast week (1-based); weeks 253 to 260 are scored
HORIZON = 8
WORLDS = {"clean": 0.0, "confounded": 1.0}
DATA = dict(n_zones=3, stores_per_zone=6, n_items=20, n_periods=260, seed=7, test_origin=ORIGIN)
LEVERS = ["promo", "display"]

WORLD_MODEL = dict(
    horizon=HORIZON, history=26, train_periods=156, categorical=["item_id", "store_id"], extra=LEVERS,
    levers=LEVERS, d_model=32, layers=2, heads=4, steps=2000, batch=256, lr=1e-3, regular_price=True,
    lift_readout=True, rollout=True, validation_periods=8, scale_window=8, roll_norm=True, device="cpu",
)
LIGHTGBM = dict(
    horizon=HORIZON, train_periods=156, categorical=["item_id", "store_id"], extra=LEVERS, lags=[0, 1, 2, 4],
    windows=[4, 8, 13, 26], spread_window=13, price_window=12, price_lag=1, rounds=500, seed=0, threads=2,
    params={"num_leaves": 63, "min_data_in_leaf": 50},
)
# name: (model, settings, columns shuffled within each series in training)
MODELS = {
    "world_seed1": ("state_model", {**WORLD_MODEL, "seed": 1}, []),
    "world_seed2": ("state_model", {**WORLD_MODEL, "seed": 2}, []),
    "world_seed3": ("state_model", {**WORLD_MODEL, "seed": 3}, []),
    "world_price_blind": ("state_model", {**WORLD_MODEL, "seed": 1}, [PRICE]),
    "world_lever_blind": ("state_model", {**WORLD_MODEL, "seed": 1}, LEVERS),
    "lightgbm": ("lightgbm_direct", LIGHTGBM, []),
    "lightgbm_price_blind": ("lightgbm_direct", LIGHTGBM, [PRICE]),
}
HIDDEN = ["true_level", "true_regular", "true_zone", "true_prev_planned"]


def world(confounding: float):
    panel = load_dataset("synthetic_levers", confounding=confounding, **DATA).panel
    dates = np.sort(panel[DATE].unique())
    train = panel[panel[DATE] < dates[ORIGIN - 1]].copy()
    test = panel[panel[DATE] >= dates[ORIGIN - 1]].copy()
    last = train[train[DATE] == train[DATE].max()]
    prev = pd.Series(planned(last), index=last[SERIES].to_numpy())
    test["true_prev_planned"] = test[SERIES].map(prev).astype(int)
    return panel, train.reset_index(drop=True), test.reset_index(drop=True)


def shuffled(train: pd.DataFrame, cols: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = train.copy()
    for c in cols:
        out[c] = out.groupby(SERIES)[c].transform(lambda s: rng.permutation(s.to_numpy()))
    return out


def truth_parts(test: pd.DataFrame, regular: pd.Series, t: Truth) -> pd.DataFrame:
    """The true version of `lever_contributions`: the same counterfactual plans,
    scored by the expected sales the data were generated from."""
    hidden = test[HIDDEN]
    ev = lambda plan: expected_units(pd.concat([plan.reset_index(drop=True), hidden], axis=1), t)
    plan = test.drop(columns=[UNITS] + HIDDEN)
    out = pd.DataFrame({"forecast": ev(plan)})
    out["nothing_planned"] = ev(off_plan(plan, regular, LEVERS))
    out["total_lift"] = out["forecast"] - out["nothing_planned"]
    for c in LEVERS:
        out[f"lift_{c}"] = out["forecast"] - ev(plan.assign(**{c: 0}))
    reg = plan[SERIES].map(regular)
    out["lift_price"] = out["forecast"] - ev(plan.assign(**{PRICE: np.where(reg.notna(), reg, plan[PRICE])}))
    return out


def against_truth(model_parts: pd.DataFrame, true: pd.DataFrame, on: np.ndarray) -> dict:
    out = {}
    for c in ["total_lift", "lift_price"] + [f"lift_{c}" for c in LEVERS]:
        m, t = model_parts[c].to_numpy()[on], true[c].to_numpy()[on]
        out[c] = {"rank": rank_agreement(m, t), "bias": float(m.sum() / t.sum()) if t.sum() else float("nan")}
    lm = np.log(np.maximum(model_parts["forecast"], 1e-9) / np.maximum(model_parts["nothing_planned"], 1e-9))
    lt = np.log(true["forecast"] / true["nothing_planned"])
    out["lift_ratio_median_abs_log_error"] = float(np.median(np.abs(lm - lt)[on]))
    return out


def accuracy(train: pd.DataFrame, test: pd.DataFrame, fc: np.ndarray) -> dict:
    """RMSSE per series (scale: mean squared one-week change in training), averaged; and
    the same on store totals."""
    def rmsse(tr, te, f, key):
        scale = tr.sort_values(DATE).groupby(key)[UNITS].apply(lambda s: np.mean(np.diff(s.to_numpy()) ** 2))
        err = te.assign(f=f).groupby(key).apply(lambda d: np.mean((d[UNITS] - d["f"]) ** 2), include_groups=False)
        return float(np.mean(np.sqrt(err / scale.reindex(err.index).clip(lower=1e-9))))
    st_tr = train.groupby([STORE, DATE])[UNITS].sum().reset_index()
    st_te = test.assign(f=fc).groupby([STORE, DATE])[[UNITS, "f"]].sum().reset_index()
    return {"item_by_store_rmsse": rmsse(train, test, fc, SERIES),
            "store_rmsse": rmsse(st_tr, st_te, st_te["f"].to_numpy(), STORE)}


def origin_episodes(panel, test, model, t: Truth) -> list[dict]:
    """C7 on generated data: stated response to the price spells that start at the origin,
    against the truth and against difference in differences on the generated sales."""
    units = panel[[STORE, DATE, UNITS]].copy()
    dates = np.sort(panel[DATE].unique())
    units["week"] = units[DATE].map({d: i + 1 for i, d in enumerate(dates)})
    units["store"] = units[STORE].str[1:].astype(int)
    eps = panel.attrs["episodes"]
    plan = test.drop(columns=[UNITS] + HIDDEN)
    base = model.predict(plan)
    out = []
    for e in [e for e in eps if e["at_origin"]]:
        busy = {x["store"] for x in eps if x["zone"] == e["zone"] and x["first_week"] <= ORIGIN + HORIZON - 1
                and x["last_week"] >= ORIGIN - 26}
        zone_stores = range(e["zone"] * DATA["stores_per_zone"], (e["zone"] + 1) * DATA["stores_per_zone"])
        controls = [s for s in zone_stores if s not in busy]
        ep = Episode(e["zone"], e["store"], e["first_week"], e["last_week"], float(e["log_ratio"]), controls)
        mine = (plan[STORE] == f"S{e['store']:02d}").to_numpy()
        alt = model.predict(plan.assign(**{PRICE: np.where(mine, plan[PRICE] / np.exp(ep.log_ratio), plan[PRICE])}))
        out.append({"store": e["store"], "zone": e["zone"], "log_ratio": ep.log_ratio, "controls": len(controls),
                    "stated": float(np.log(base[mine].sum() / max(alt[mine].sum(), 1e-9))),
                    "true": t.regular_elasticity * ep.log_ratio,
                    "did": did(units[["store", "week", "units"]], ep,
                               last_week=ORIGIN + HORIZON - 1)})
    return out


def run(world_name: str, out: Path):
    panel, train, test = world(WORLDS[world_name])
    t = Truth(**panel.attrs["truth"])
    regular = regular_prices(train, 12)
    true = truth_parts(test, regular, t)
    on = planned(test) | (true["total_lift"].abs() > 1e-6 * true["forecast"]).to_numpy()
    seen = train.drop(columns=HIDDEN)
    future = test.drop(columns=HIDDEN)
    # what planned weeks sold over the true nothing-planned sales: the observed lift a
    # merchant would read, which confounding inflates
    obs = test[UNITS].to_numpy()[on].sum() / true["nothing_planned"].to_numpy()[on].sum()
    tru = true["forecast"].to_numpy()[on].sum() / true["nothing_planned"].to_numpy()[on].sum()
    (out / f"{world_name}_world.json").write_text(json.dumps(
        {"truth": panel.attrs["truth"], "planned_item_weeks": int(on.sum()), "rows": len(test),
         "observed_over_true_nothing_planned": obs, "true_forecast_over_true_nothing_planned": tru,
         "episodes": panel.attrs["episodes"]}, indent=1, default=float))
    for name, (kind, params, scramble) in MODELS.items():
        path = out / f"{world_name}_{name}.json"
        if path.exists():
            continue
        t0 = time.time()
        model = build_model(kind, **params).fit(shuffled(seen, scramble))
        fit_s = time.time() - t0
        cfg = {"model": {"name": kind, "params": params}}
        rec, parts = audit_model(model, train, future, cfg, randomise=kind == "state_model")
        rec.update({"world": world_name, "model": name, "kind": kind, "shuffled": scramble, "fit_seconds": round(fit_s, 1),
                    "accuracy": accuracy(train, test, parts["forecast"].to_numpy()),
                    "truth": against_truth(parts.reset_index(drop=True), true, on),
                    "origin_episodes": origin_episodes(panel, test, model, t)})
        if kind == "lightgbm_direct":
            shap = model.shap(future.drop(columns=[UNITS]))
            parts["shap_price"] = shap[[c for c in shap.columns if c.startswith("price")]].sum(axis=1).to_numpy()
            for c in LEVERS:
                parts[f"shap_{c}"] = shap[c].to_numpy()
        lifts = pd.concat([future[[SERIES, DATE, UNITS]].reset_index(drop=True), parts.reset_index(drop=True),
                           true.add_prefix("true_")], axis=1)
        lifts["planned"] = on
        lifts.to_csv(out / f"{world_name}_{name}_lifts.csv.gz", index=False)
        path.write_text(json.dumps(rec, indent=1, default=float))
        print(f"{world_name} {name}: fit {fit_s:.0f} s, audit {rec['seconds']} s, "
              f"item-by-store RMSSE {rec['accuracy']['item_by_store_rmsse']:.3f}, "
              f"total-lift rank vs truth {rec['truth']['total_lift']['rank']:.2f}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--worlds", nargs="*", default=list(WORLDS))
    p.add_argument("--steps", type=int)
    a = p.parse_args(argv)
    if a.steps:
        WORLD_MODEL["steps"] = a.steps
        for k, (kind, params, s) in MODELS.items():
            if kind == "state_model":
                params["steps"] = a.steps
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for w in a.worlds:
        run(w, out)


if __name__ == "__main__":
    main()
