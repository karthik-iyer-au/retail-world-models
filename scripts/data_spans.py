"""What each data set holds, what the paper loads, and what each run used (paper 1, data section; 15 November 2026).

  python scripts/data_spans.py --out <folder>

For each data set, from the loaded panels and the final configs: the first and last period of the panel the paper loads,
the number of series and stores, the training length of the final runs, the dates of the first and last test window,
and the held-out periods at the end that no run sees. Favorita is also loaded from 1 January 2013 (the start of the
source file), to show what the 2015 start leaves out. Writes data_spans.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rwm.data.registry import load_dataset
from rwm.evaluation.splits import rolling_origins

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["m5", "dominicks_ana", "dominicks_bjc", "dominicks_che", "dominicks_fre", "dominicks_sdr", "dominicks_sna",
        "frat", "favorita"]


def train_periods(model: dict):
    if isinstance(model, dict):
        for k, v in model.items():
            if k == "train_periods":
                return v
            r = train_periods(v)
            if r is not None:
                return r
    return None


def row(key: str, cfg: dict, extra: dict | None = None, label: str = "") -> dict:
    params = {**cfg["dataset"].get("params", {}), **(extra or {})}
    ds = load_dataset(cfg["dataset"]["name"], **params)
    p = ds.panel
    dates = np.sort(p["date"].unique())
    ev = cfg["evaluation"]
    hold = ev.get("holdout_periods") or 0
    used = dates[: len(dates) - hold] if hold else dates
    sp = rolling_origins(used, ev["horizon"], ev["n_origins"], ev.get("step"))
    first_test = min(min(s.test_dates) for s in sp)
    last_test = max(max(s.test_dates) for s in sp)
    tp = train_periods(cfg["model"])
    first_train = used[max(0, int(np.searchsorted(used, np.datetime64(first_test))) - (tp or 0))]
    return dict(key=key + label, first=str(pd.Timestamp(dates[0]).date()), last=str(pd.Timestamp(dates[-1]).date()),
                periods=len(dates), series=p["series_id"].nunique(),
                stores=p["store_id"].nunique() if "store_id" in p else np.nan,
                train_periods=tp, first_train=str(pd.Timestamp(first_train).date()),
                first_test=str(pd.Timestamp(first_test).date()), last_test=str(pd.Timestamp(last_test).date()),
                windows=len(sp), horizon=ev["horizon"], holdout=hold,
                holdout_from=str(pd.Timestamp(dates[len(dates) - hold]).date()) if hold else "")


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    out = Path(a.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in KEYS:
        cfg = yaml.safe_load((ROOT / f"configs/experiments/final_{key}_lightgbm.yaml").read_text())
        rows.append(row(key, cfg))
        print(rows[-1], flush=True)
        if key == "favorita":
            rows.append(row(key, cfg, {"start": "2013-01-01"}, "_from_2013"))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "data_spans.csv", index=False)


if __name__ == "__main__":
    main()
