"""Study P (docs/PAPER1_REVISION_PLAN.md, entry of 1 November 2026): each
model's item-by-store forecasts reconciled with the same model's own forecasts
of the store totals, by WLS (rwm.evaluation.reconcile.gls_bottom) on the
two-level hierarchy store over item-by-store. Weights are each node's mean
squared one-step change over its training history (its RMSSE scale).

Inputs, all from the results volume: the study O starting runs
(research/avg5_<model>_dominicks_<cat>/..._start_s<seed>_seed<seed>_*) and the
study P store-total runs (study_p/agg_store_<model>_dominicks_<cat>_seed<seed>_*),
fetched into RESULTS_DIR. Scores every level with the run code's WRMSSE.

  RESULTS_DIR=... python scripts/study_p_reconcile.py che sna ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwm.data import load_dataset  # noqa: E402
from rwm.evaluation.hierarchy import series_scale, to_matrix, wrmsse  # noqa: E402
from rwm.evaluation.reconcile import gls_bottom  # noqa: E402
from rwm.evaluation.splits import rolling_origins  # noqa: E402
from rwm.data.schema import DATE, PRICE, SERIES, UNITS  # noqa: E402

RES = Path(os.environ.get("RESULTS_DIR", "study_p"))
SEEDS = [1, 2, 3]


def one(path_glob: str) -> Path:
    found = sorted(RES.glob(path_glob))
    if not found:
        raise FileNotFoundError(path_glob)
    return found[-1]


def main(cats: list[str]) -> None:
    rows = []
    for cat in cats:
        spec = yaml.safe_load((ROOT / f"configs/research/avg5_world_model_dominicks_{cat}.yaml").read_text())
        ev = spec["evaluation"]
        ds = load_dataset("dominicks", categories=[cat])
        panel = ds.panel
        dates = np.sort(panel[DATE].unique())[: -ev["holdout_periods"]]
        panel = panel[panel[DATE] <= dates[-1]]
        levels = ds.hierarchy
        cols = sorted({c for lv in levels for c in lv} | {SERIES})
        attrs = panel[cols].drop_duplicates(SERIES).sort_values(SERIES).reset_index(drop=True)
        names = pd.Index(attrs[SERIES].astype(str))
        units = to_matrix(panel, SERIES, DATE, UNITS, dates, names)
        price = to_matrix(panel, SERIES, DATE, PRICE, dates, names)
        stores = pd.Index(sorted(attrs["store_id"].astype(str).unique()))
        store_of = stores.get_indexer(attrs["store_id"].astype(str))
        A = sp.csr_matrix((np.ones(len(names)), (store_of, np.arange(len(names)))), shape=(len(stores), len(names)))
        S = sp.vstack([A, sp.identity(len(names), format="csr")]).tocsr()
        splits = rolling_origins(dates, ev["horizon"], ev["n_origins"], ev.get("step"))
        for model in ("world_model", "lightgbm"):
            for seed in SEEDS:
                bottom_f = pd.read_csv(one(f"research/avg5_{model}_dominicks_{cat}/*_start_s{seed}_seed{seed}_*/forecasts.csv"), parse_dates=[DATE])
                store_f = pd.read_csv(one(f"study_p/*agg_store_{model}_dominicks_{cat}_seed{seed}_*/forecasts.csv"), parse_dates=[DATE])
                store_f[SERIES] = store_f[SERIES].astype(str).str.zfill(3)
                for spl in splits:
                    t0 = int(np.searchsorted(dates, np.datetime64(spl.train_end))) + 1
                    t1 = t0 + len(spl.test_dates)
                    cols_t = dates[t0:t1]
                    b = to_matrix(bottom_f[bottom_f["origin"] == spl.origin], SERIES, DATE, "forecast", cols_t, names).astype(np.float64)
                    s = to_matrix(store_f[store_f["origin"] == spl.origin], SERIES, DATE, "forecast", cols_t, stores).astype(np.float64)
                    hist = units[:, :t0].astype(np.float64)
                    v = np.concatenate([series_scale(np.asarray(A @ hist)), series_scale(hist)])
                    v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
                    v = np.where(np.isnan(v), np.nanmax(v) * 1e6, v)  # a node with no history barely counts
                    rec = np.clip(gls_bottom(S, np.vstack([s, b]), v), 0, None)
                    window = ev.get("weight_window", 28)
                    revenue = (units[:, t0 - window:t0] * price[:, t0 - window:t0]).sum(axis=1, dtype=np.float64)
                    for name, f in (("as is", b), ("reconciled", rec)):
                        sc = wrmsse(hist, units[:, t0:t1], f, revenue, attrs, levels, ev.get("scale_lag", 1))
                        row = {"category": cat, "model": model, "seed": seed, "origin": spl.origin,
                               "held_out": spl.origin == max(x.origin for x in splits), "forecast": name, "wrmsse": sc["wrmsse"]}
                        row.update({lv["level"]: lv["wrmsse"] for lv in sc["by_level"]})
                        rows.append(row)
        print(cat, "done", flush=True)
    out = pd.DataFrame(rows)
    path = ROOT / "results" / "paper1_revision" / "tables" / f"study_p_{'_'.join(cats)}.csv"
    out.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
