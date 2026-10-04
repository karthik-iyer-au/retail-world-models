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
from rwm.evaluation.metrics import rmsse_by_series, weighted_rmsse
from rwm.evaluation.splits import rolling_origins
from rwm.forecaster import build_model
from rwm.utils.panel import last_k_rows
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
    splits = rolling_origins(panel[DATE], ev["horizon"], ev["n_origins"], ev.get("step"))
    forecasts, per_split = [], []
    for sp in splits:
        train = panel[panel[DATE] <= sp.train_end]
        test = panel[panel[DATE].isin(sp.test_dates)].reset_index(drop=True)
        model = build_model(config["model"]["name"], **config["model"].get("params", {}))
        model.fit(train)
        pred = np.asarray(model.predict(test.drop(columns=[UNITS])), dtype=float)
        if len(pred) != len(test):
            raise RuntimeError("model returned the wrong number of forecasts")
        out = test[[SERIES, DATE, UNITS]].copy()
        out["forecast"] = pred
        out["origin"] = sp.origin
        forecasts.append(out)

        recent = train.iloc[last_k_rows(train[SERIES], ev.get("weight_window", 28))]
        value = recent[UNITS].astype("float64") * (recent[PRICE] if PRICE in recent else 1.0)
        weight = value.groupby(recent[SERIES].astype(str).to_numpy()).sum()
        s = rmsse_by_series(train, out, SERIES, UNITS, ev.get("scale_lag", 1))
        del train, test, recent
        per_split.append(
            {
                "origin": sp.origin,
                "train_end": str(pd.Timestamp(sp.train_end).date()),
                "test_start": str(pd.Timestamp(sp.test_dates[0]).date()),
                "test_end": str(pd.Timestamp(sp.test_dates[-1]).date()),
                "series_scored": int(s.notna().sum()),
                "mean_rmsse": float(s.mean()),
                "weighted_rmsse": weighted_rmsse(s.to_numpy(), weight.reindex(s.index).to_numpy()),
            }
        )

    metrics = {
        "model": config["model"]["name"],
        "dataset": ds.name,
        "levers_available": ds.levers,
        "splits": per_split,
        "mean_rmsse": float(np.mean([p["mean_rmsse"] for p in per_split])),
        "weighted_rmsse": float(np.mean([p["weighted_rmsse"] for p in per_split])),
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
