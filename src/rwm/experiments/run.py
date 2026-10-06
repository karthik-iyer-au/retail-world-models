"""Run one experiment from one config file.

  python -m rwm.experiments.run --config configs/experiments/smoke.yaml

Writes results/<run_id>/ with:
  manifest.json   what was run: code commit, config, data checksums, seed, packages
  metrics.json    the scores, per split and overall
  forecasts.csv   every forecast next to its actual
  checkpoint_origin<k>.pt   the fitted model for each window, for models that can be saved

With --strict the run refuses to start unless the code is committed and the
raw data matches its registered checksums. Reported numbers come from strict
runs only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rwm.data import load_dataset
from rwm.data import manifest as data_manifest
from rwm.data.calendar import add_calendar
from rwm.data.crowding import add_crowding
from rwm.data.schema import ITEM, DATE, PRICE, SERIES, UNITS
from rwm.evaluation.hierarchy import rmsse_where, to_matrix, wrmsse
from rwm.evaluation.splits import rolling_origins
from rwm.forecaster import build_model
from rwm.utils.hashing import sha256_file
from rwm.utils.paths import DATA_RAW, RESULTS
from rwm.utils.run_manifest import build_manifest
from rwm.utils.seed import set_seed


def _data_files(ds, strict: bool) -> list[dict]:
    if not ds.files:
        return []  # generated data, nothing on disk
    if strict or data_manifest.manifest_path(ds.name).exists():
        return data_manifest.verify(ds.name, files=ds.files)
    return data_manifest.describe(ds.files, DATA_RAW / ds.name)


def run(config: dict, strict: bool = False, out_root: Path = RESULTS) -> Path:
    set_seed(config["seed"])
    ds = load_dataset(config["dataset"]["name"], **config["dataset"].get("params", {}))
    manifest = build_manifest(config, _data_files(ds, strict))
    if strict and (manifest["git"]["commit"] is None or manifest["git"]["dirty"]):
        raise RuntimeError("strict run needs a clean, committed working tree")

    ev = config["evaluation"]
    panel = ds.panel
    # optional columns worked out from the plan and the calendar; both are
    # known for future periods
    extras = config["dataset"]
    if "calendar" in extras:
        add_calendar(panel, **extras["calendar"])
    if "crowding" in extras:
        add_crowding(panel, **extras["crowding"])
    dates = np.sort(panel[DATE].unique())
    if ev.get("holdout_periods"):
        # The last periods are set aside for the final test and are not loaded into this run.
        dates = dates[: -ev["holdout_periods"]]
        panel = panel[panel[DATE] <= dates[-1]].reset_index(drop=True)
    levels = ds.hierarchy or [[SERIES]]
    cols = sorted({c for lv in levels for c in lv} | {SERIES})
    attrs = panel[cols].drop_duplicates(SERIES).sort_values(SERIES).reset_index(drop=True)
    names = pd.Index(attrs[SERIES].astype(str))
    units = to_matrix(panel, SERIES, DATE, UNITS, dates, names)
    if PRICE in panel:
        price = to_matrix(panel, SERIES, DATE, PRICE, dates, names)
    else:
        price = np.ones_like(units)

    splits = rolling_origins(dates, ev["horizon"], ev["n_origins"], ev.get("step"))
    window = ev.get("weight_window", 28)
    forecasts, per_split, fitted = [], [], []
    # Latest window first. After each window the table is cut back to that
    # window's training rows, so only one copy of the data is held at a time.
    ds.panel = None
    # New-item test: a random share of products is kept out of training
    # altogether, in every window, and scored on its own.
    held = None
    if ev.get("new_items"):
        products = np.sort(panel[ITEM].astype(str).unique())
        rng = np.random.default_rng(ev["new_items"].get("seed", 0))
        n_held = max(1, int(round(ev["new_items"]["share"] * len(products))))
        held = set(rng.choice(products, n_held, replace=False))
        is_new = panel.drop_duplicates(SERIES).set_index(SERIES)[ITEM].astype(str).isin(held).reindex(names).to_numpy()
    for sp in reversed(splits):
        test = panel[panel[DATE].isin(sp.test_dates)].reset_index(drop=True)
        panel = train = panel[panel[DATE] <= sp.train_end]
        model = build_model(config["model"]["name"], **config["model"].get("params", {}))
        model.fit(train if held is None else train[~train[ITEM].astype(str).isin(held)])
        # Unusual plans: lever mixes that made up a small share of training rows.
        rare_mix = None
        if ev.get("rare_plans"):
            lv = ev["rare_plans"]["levers"]
            mix = lambda f: (f[lv].fillna(0).to_numpy() > 0).astype(int) @ (2 ** np.arange(len(lv)))
            below, by = ev["rare_plans"].get("below", 0.02), ev["rare_plans"].get("by")
            code = mix(test)
            if by:  # how often this item (or other label) got this mix in training
                seen = pd.DataFrame({"k": train[by].astype(str).to_numpy(), "m": mix(train)})
                share = (seen.groupby(["k", "m"]).size() / seen.groupby("k").size()).rename("s").reset_index()
                mine = pd.DataFrame({"k": test[by].astype(str).to_numpy(), "m": code}).merge(share, how="left", on=["k", "m"])
                freq = mine["s"].fillna(0.0).to_numpy()
            else:
                freq = (np.bincount(mix(train), minlength=2 ** len(lv)) / len(train))[code]
            rare_mix = test.assign(
                _rare=((code > 0) & (freq < below)).astype(float),
                _usual=((code > 0) & (freq >= below)).astype(float),
            )
        del train
        fitted.append((sp.origin, model))
        pred = np.asarray(model.predict(test.drop(columns=[UNITS])), dtype=float)
        if len(pred) != len(test):
            raise RuntimeError("model returned the wrong number of forecasts")
        out = test[[SERIES, DATE, UNITS]].copy()
        out["forecast"] = pred
        out["origin"] = sp.origin
        forecasts.append(out)

        t0 = int(np.searchsorted(dates, np.datetime64(sp.train_end))) + 1  # training columns
        t1 = t0 + len(sp.test_dates)
        revenue = (units[:, t0 - window : t0] * price[:, t0 - window : t0]).sum(axis=1, dtype=np.float64)
        score = wrmsse(
            units[:, :t0],
            units[:, t0:t1],
            to_matrix(out, SERIES, DATE, "forecast", dates[t0:t1], names),
            revenue,
            attrs,
            levels,
            ev.get("scale_lag", 1),
        )
        # Accuracy split by whether each named lever was on, at item-by-store level.
        by_condition = {}
        if ev.get("conditions"):
            cols = dates[t0:t1]
            recorded = to_matrix(test.assign(_row=1.0), SERIES, DATE, "_row", cols, names) > 0
            forecast = to_matrix(out, SERIES, DATE, "forecast", cols, names)
            for c in ev["conditions"]:
                on = recorded & (to_matrix(test, SERIES, DATE, c, cols, names) > 0)
                args = (units[:, :t0], units[:, t0:t1], forecast)
                by_condition[c] = {
                    "on": rmsse_where(*args, on, revenue, ev.get("scale_lag", 1)),
                    "off": rmsse_where(*args, recorded & ~on, revenue, ev.get("scale_lag", 1)),
                }
        # Accuracy at item-by-store level for each period ahead, to show whether a
        # model weakens further out.
        cols = dates[t0:t1]
        there = to_matrix(test.assign(_row=1.0), SERIES, DATE, "_row", cols, names) > 0
        made = to_matrix(out, SERIES, DATE, "forecast", cols, names)
        by_horizon = []
        for h in range(len(cols)):
            only = np.zeros_like(there)
            only[:, h] = there[:, h]
            by_horizon.append(rmsse_where(units[:, :t0], units[:, t0:t1], made, only, revenue, ev.get("scale_lag", 1))["wrmsse"])
        # Plan A against plan B: for the same item and store, two weeks in the
        # window with different lever mixes. Does the forecast say which week
        # sells more, and by how much?
        plan_pairs = {}
        if ev.get("plan_pairs"):
            lv = ev["plan_pairs"]["levers"]
            code = sum((to_matrix(test, SERIES, DATE, c, cols, names) > 0) * 2**i for i, c in enumerate(lv))
            actual = units[:, t0:t1]
            right = wrong = n_pairs = 0
            gap = 0.0
            for a in range(len(cols)):
                for b in range(a + 1, len(cols)):
                    m = there[:, a] & there[:, b] & (code[:, a] != code[:, b])
                    da, df = actual[m, a] - actual[m, b], made[m, a] - made[m, b]
                    right += int(((da * df) > 0).sum())
                    wrong += int(((da * df) < 0).sum()) + int(((da != 0) & (df == 0)).sum())
                    gap += float(np.abs(np.log1p(made[m, a]) - np.log1p(made[m, b]) - np.log1p(actual[m, a]) + np.log1p(actual[m, b])).sum())
                    n_pairs += int(m.sum())
            plan_pairs = {
                "plan_pairs": n_pairs,
                # share of pairs where the forecast picks the week that really sold more
                "plan_pair_order": right / max(right + wrong, 1),
                # average miss on the size of the change between the two weeks (log scale)
                "plan_pair_change_error": gap / max(n_pairs, 1),
            }
        per_split.append(
            {
                "origin": sp.origin,
                "train_end": str(pd.Timestamp(sp.train_end).date()),
                "test_start": str(pd.Timestamp(sp.test_dates[0]).date()),
                "test_end": str(pd.Timestamp(sp.test_dates[-1]).date()),
                **score,
                # all forecast units over all actual units: above 1 is over-forecasting
                "forecast_to_actual": float(out["forecast"].sum() / max(float(out[UNITS].sum()), 1e-9)),
                **({"by_condition": by_condition} if by_condition else {}),
                "by_horizon": by_horizon,
                **plan_pairs,
                **(
                    {
                        k: rmsse_where(units[:, :t0], units[:, t0:t1], made, there & (to_matrix(rare_mix, SERIES, DATE, "_" + k.split("_")[0], cols, names) > 0), revenue, ev.get("scale_lag", 1))["wrmsse"]
                        for k in ("rare_plans", "usual_plans")
                    }
                    if rare_mix is not None
                    else {}
                ),
                **(
                    {
                        "new_items": rmsse_where(units[:, :t0], units[:, t0:t1], made, there & is_new[:, None], revenue, ev.get("scale_lag", 1))["wrmsse"],
                        "known_items": rmsse_where(units[:, :t0], units[:, t0:t1], made, there & ~is_new[:, None], revenue, ev.get("scale_lag", 1))["wrmsse"],
                    }
                    if held is not None
                    else {}
                ),
            }
        )

    per_split.sort(key=lambda p: p["origin"])
    forecasts.sort(key=lambda f: f["origin"].iloc[0])

    metrics = {
        "model": config["model"]["name"],
        "dataset": ds.name,
        "levers_available": ds.levers,
        "levels_scored": len(levels),
        "wrmsse": float(np.mean([p["wrmsse"] for p in per_split])),
        "splits": per_split,
    }

    stamp = manifest["created_utc"].replace(":", "").replace("-", "")[:15]
    run_dir = out_root / f"{stamp}_{config['name']}_{manifest['config_sha256'][:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # Models that can be saved are saved, one file per window, with their checksums.
    checkpoints = []
    for origin, model in sorted(fitted, key=lambda f: f[0]):
        if hasattr(model, "save"):
            path = run_dir / f"checkpoint_origin{origin}.pt"
            model.save(path)
            checkpoints.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    metrics["checkpoints"] = checkpoints
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.concat(forecasts).to_csv(run_dir / "forecasts.csv", index=False)
    return run_dir


def with_seed(config: dict, seed: int | None) -> dict:
    """The same experiment with another seed. The run is renamed so its
    folder and manifest show which seed it used."""
    if seed is None:
        return config
    config = {**config, "seed": seed, "name": f"{config['name']}_seed{seed}"}
    params = config["model"].get("params", {})
    if "seed" in params:
        config["model"] = {**config["model"], "params": {**params, "seed": seed}}
    return config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", default=None, help="folder for run outputs (default: results/)")
    ap.add_argument("--seed", type=int, default=None, help="repeat the config with a different seed")
    args = ap.parse_args()
    config = with_seed(yaml.safe_load(Path(args.config).read_text()), args.seed)
    run_dir = run(config, strict=args.strict, out_root=Path(args.out) if args.out else RESULTS)
    print(f"wrote {run_dir}")
    print((run_dir / "metrics.json").read_text())


if __name__ == "__main__":
    main()
