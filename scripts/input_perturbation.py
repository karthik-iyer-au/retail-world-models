"""Paper 1 revision, study B of the amendment of 15 October 2026
(docs/PAPER1_REVISION_PLAN.md): which plan inputs each model uses, and how
much history the state uses.

  python scripts/input_perturbation.py --results <runs folder> --out <csv> [--keys ...] \
      [--lightgbm-last 2 --lightgbm-keys m5] [--threads 8]

For the world model and the lift model (saved checkpoints) and LightGBM
(refitted on the same training data with its recorded settings), in every
rolling window, the forecast is recomputed with one plan input changed:
  occlude   the input set to nothing planned (price at the regular price, lever off)
  permute   the input shuffled within each series across the window
and, for the world model and the lift model, with the history before the last
k periods set to missing (as for an item not yet on sale). One row per model,
window, input and operation, saved after every model and window; rows already
in the CSV are skipped.
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
from revision_analysis import index  # noqa: E402

from rwm.audit.runs import load_checkpoint, model_params, plan_levers, windows  # noqa: E402
from rwm.data.schema import DATE, PRICE, SERIES, UNITS  # noqa: E402
from rwm.evaluation import compare as C  # noqa: E402

HISTORY = {"m5": (28, 56), "default": (4, 13, 26)}
MODELS = ("world_model", "lift", "lightgbm")


def permute_within_series(plan: pd.DataFrame, col: str, seed: int) -> pd.DataFrame:
    """`col` shuffled across the periods of each series (vectorised: the
    positions of each series, in a random order within the series)."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(plan[SERIES])[0]
    grouped = np.argsort(codes, kind="stable")
    shuffled = np.lexsort((rng.random(len(codes)), codes))
    out = plan.copy()
    vals = out[col].to_numpy().copy()
    vals[grouped] = plan[col].to_numpy()[shuffled]
    out[col] = vals
    return out


def regular_price(train: pd.DataFrame) -> pd.Series:
    """The highest price of each series over its last 12 periods of training."""
    last = np.sort(train[DATE].unique())[-12:]
    return train[train[DATE].isin(last)].groupby(SERIES)[PRICE].max()


def switched_on(plan: pd.DataFrame, col: str, regular: pd.Series) -> np.ndarray:
    if col == PRICE:
        reg = plan[SERIES].map(regular).to_numpy()
        return np.nan_to_num(plan[PRICE].to_numpy()) < 0.95 * np.nan_to_num(reg, nan=np.inf)
    return np.nan_to_num(plan[col].to_numpy(dtype=float)) > 0


def occlude(plan: pd.DataFrame, col: str, regular: pd.Series) -> pd.DataFrame:
    out = plan.copy()
    on = switched_on(plan, col, regular)
    if col == PRICE:
        out.loc[on, PRICE] = plan.loc[on, SERIES].map(regular).to_numpy()
    else:
        out.loc[on, col] = 0
    return out


def masked_history(model, keep: int):
    """The model's stored history with every period before the last `keep` set to missing."""
    past = model._past
    cut = past["price"].shape[1] - keep
    if cut <= 0:
        return past
    new = dict(past)
    new["units"] = past["units"].clone()
    new["units"][:, :cut] = 0
    for k in ("price", "regular"):
        if k in past:
            new[k] = past[k].clone()
            new[k][:, :cut] = float("nan")
    new["extra"] = []
    for e in past["extra"]:
        e = e.clone()
        e[:, :cut] = float("nan")
        new["extra"].append(e)
    return new


def lightgbm_model(cfg: dict, train: pd.DataFrame, threads: int | None):
    import rwm.model  # noqa: F401  (fills the model registry)
    from rwm.forecaster import build_model

    params = dict(cfg["model"].get("params", {}))
    if threads:
        params["threads"] = threads
    return build_model(cfg["model"]["name"], **params).fit(train)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keys", nargs="*")
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--lightgbm-last", type=int, default=2, help="LightGBM on this many latest windows of --lightgbm-keys")
    p.add_argument("--lightgbm-keys", nargs="*", default=["m5"])
    p.add_argument("--threads", type=int)
    a = p.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["key"], r["model"], r["origin"]) for r in rows}
    idx = index(res)
    idx = idx[(idx["set"] == "rolling") & idx["model"].isin(a.models)]
    for key in a.keys or sorted(idx["key"].unique()):
        for _, run in idx[idx["key"] == key].iterrows():
            d = res / run["run"]
            cfg = json.loads((d / "manifest.json").read_text())["config"]
            model_name = run["model"]
            fr = C.frame(cfg)
            all_origins = [s.origin for s in fr.splits]
            if model_name == "lightgbm" and key in a.lightgbm_keys:
                wanted = set(all_origins[-a.lightgbm_last:])
            else:
                wanted = set(all_origins)
            # the inputs of the plan: price and the levers of this data set
            if model_name == "lightgbm":  # the same levers as the world model's config (base_price is not part of the plan)
                levers = [c for c in model_params(cfg).get("extra", []) if c not in ("cost", "base_price")]
            else:
                levers = [c for c in plan_levers(cfg) if c != "cost"]
            saved = pd.read_csv(d / "forecasts.csv")
            for origin, train, test in windows(cfg):
                if origin not in wanted or (key, model_name, origin) in done:
                    continue
                t_start = time.time()
                if model_name == "lightgbm":
                    model = lightgbm_model(cfg, train, a.threads)
                else:
                    if not (d / f"checkpoint_origin{origin}.pt").exists():
                        continue
                    model = load_checkpoint(d, origin)
                plan = test.drop(columns=[UNITS])
                dates = list(np.sort(test[DATE].unique()))
                fc = lambda f: C.to_matrix(test.assign(forecast=f), SERIES, DATE, "forecast", dates, fr.names)
                f0 = np.asarray(model.predict(plan), dtype=float)
                s0 = C.level_scores(fr, origin, fc(f0))
                top, bottom = list(s0)[0], list(s0)[-1]  # the total and item by store
                # how closely this forecast matches the run's recorded one (a refit for LightGBM)
                rec = C.forecast_matrix(saved, fr, origin)
                match = float(np.nanmax(np.abs(fc(f0) - rec)) / max(np.nanmean(np.abs(rec)), 1e-9))
                regular = regular_price(train)
                base = {"key": key, "model": model_name, "run": run["run"], "origin": origin,
                        "bottom_true": s0[bottom], "total_true": s0[top], "refit_gap": match}

                def record(inp, op, f, on=None):
                    s = C.level_scores(fr, origin, fc(f))
                    r = {**base, "input": inp, "operation": op, "bottom": s[bottom], "total": s[top]}
                    if on is not None and on.any():
                        r["rows_on"] = int(on.sum())
                        r["units_change_on"] = float(f[on].sum() / max(f0[on].sum(), 1e-9) - 1)
                    rows.append(r)

                inputs = [PRICE] + [c for c in levers if c in plan]
                for col in inputs:
                    on = switched_on(plan, col, regular)
                    record(col, "occlude", np.asarray(model.predict(occlude(plan, col, regular)), dtype=float), on)
                    record(col, "permute", np.asarray(model.predict(permute_within_series(plan, col, seed=origin)),
                                                      dtype=float), on)
                every = plan
                for col in inputs:
                    every = occlude(every, col, regular)
                anyon = np.any([switched_on(plan, c, regular) for c in inputs], axis=0)
                record("whole_plan", "occlude", np.asarray(model.predict(every), dtype=float), anyon)
                if model_name != "lightgbm":
                    keep_past = model._past
                    for k in HISTORY["m5" if key == "m5" else "default"]:
                        model._past = masked_history(model, k)
                        try:
                            record(f"history_last_{k}", "occlude", np.asarray(model.predict(plan), dtype=float))
                        finally:
                            model._past = keep_past
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{key} {model_name} window {origin}: refit gap {match:.2e}, {time.time() - t_start:.0f} s",
                      flush=True)
                del model
    print("done", flush=True)


if __name__ == "__main__":
    main()
