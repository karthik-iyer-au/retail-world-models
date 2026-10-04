"""Run one experiment from one config file.

  python -m rwm.experiments.run --config configs/experiments/smoke.yaml

Writes results/<run_id>/ with:
  manifest.json   what was run: code commit, config, data checksums, seed, packages
  metrics.json    the scores, per split and overall
  forecasts.csv   every forecast next to its actual

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
from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.evaluation.hierarchy import to_matrix, wrmsse
from rwm.evaluation.splits import rolling_origins
from rwm.forecaster import build_model
from rwm.utils.paths import DATA_RAW, RESULTS
from rwm.utils.run_manifest import build_manifest
from rwm.utils.seed import set_seed


def _data_files(ds, strict: bool) -> list[dict]:
    if not ds.files:
        return []  # generated data, nothing on disk
    if strict or data_manifest.manifest_path(ds.name).exists():
        return data_manifest.verify(ds.name)
    return data_manifest.describe(ds.files, DATA_RAW / ds.name)


def run(config: dict, strict: bool = False, out_root: Path = RESULTS) -> Path:
    set_seed(config["seed"])
    ds = load_dataset(config["dataset"]["name"], **config["dataset"].get("params", {}))
    manifest = build_manifest(config, _data_files(ds, strict))
    if strict and (manifest["git"]["commit"] is None or manifest["git"]["dirty"]):
        raise RuntimeError("strict run needs a clean, committed working tree")

    ev = config["evaluation"]
    panel = ds.panel
    dates = np.sort(panel[DATE].unique())
    levels = ds.hierarchy or [[SERIES]]
    cols = sorted({c for lv in levels for c in lv} | {SERIES})
    attrs = panel[cols].drop_duplicates(SERIES).sort_values(SERIES).reset_index(drop=True)
    units = to_matrix(panel, SERIES, DATE, UNITS, dates)
    if PRICE in panel:
        price = to_matrix(panel, SERIES, DATE, PRICE, dates)
    else:
        price = np.ones_like(units)

    splits = rolling_origins(dates, ev["horizon"], ev["n_origins"], ev.get("step"))
    window = ev.get("weight_window", 28)
    forecasts, per_split = [], []
    # Latest window first. After each window the table is cut back to that
    # window's training rows, so only one copy of the data is held at a time.
    ds.panel = None
    for sp in reversed(splits):
        test = panel[panel[DATE].isin(sp.test_dates)].reset_index(drop=True)
        panel = train = panel[panel[DATE] <= sp.train_end]
        model = build_model(config["model"]["name"], **config["model"].get("params", {}))
        model.fit(train)
        del train
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
            to_matrix(out, SERIES, DATE, "forecast", dates[t0:t1]),
            revenue,
            attrs,
            levels,
            ev.get("scale_lag", 1),
        )
        per_split.append(
            {
                "origin": sp.origin,
                "train_end": str(pd.Timestamp(sp.train_end).date()),
                "test_start": str(pd.Timestamp(sp.test_dates[0]).date()),
                "test_end": str(pd.Timestamp(sp.test_dates[-1]).date()),
                **score,
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
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.concat(forecasts).to_csv(run_dir / "forecasts.csv", index=False)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run_dir = run(config, strict=args.strict)
    print(f"wrote {run_dir}")
    print((run_dir / "metrics.json").read_text())


if __name__ == "__main__":
    main()
