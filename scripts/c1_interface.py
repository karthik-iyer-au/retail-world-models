"""Paper 2, C1 resolved: where the audit's "nothing planned" plan differs from the
model's own (docs/PAPER2_PLAN.md, amendment of 18 October 2026).

  python scripts/c1_interface.py --out results/paper2/diagnose/c1_interface.csv [--only <config> ...]

C1 asks that a model's lift be zero when nothing is planned. Given the plan the
model itself treats as nothing planned (StateModel.nothing_planned), the lift is
zero by construction for every series with history: the plan path and the
nothing-planned path then receive identical inputs. A C1 violation therefore
measures how far the audit's plan (levers off, price at the audit's regular
price) is from the model's. This script finds, for every audited configuration
and window, the series-periods where the two plans differ and why. The
difference depends on the data and the settings only, not on the fitted
weights, so each model is fitted for a single step to build its inputs.

Causes:
  audit_no_regular   the series has no price in the audit's last
                     `regular_window` training periods, so the audit keeps
                     the planned price; the model uses its fallback (its mean
                     price over its history window, or 1.0 when it has none;
                     model_price_one counts the second)
  rule_differs       both have a regular price and they differ (a recorded
                     regular column, a held price, or a shorter history)
  new_item           the series has no history and the model borrows a forecast
                     for it (new-item tests); without borrowing such a series
                     is forecast as zero and its lift is zero
  lever_differs      a lever is set differently

Writes one row per configuration and window.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from loss_ledger import params  # noqa: E402
from rwm.audit import regular_prices  # noqa: E402
from rwm.audit.lifts import off_plan  # noqa: E402
from rwm.audit.runs import model_params, plan_levers, windows  # noqa: E402
from rwm.data.schema import PRICE, SERIES, UNITS  # noqa: E402
from rwm.utils.frames import series_rows  # noqa: E402

KEYS = ["history", "regular_window", "regular_column", "regular_hold", "cold_start", "extra", "levers",
        "train_periods", "horizon", "categorical"]


def groups(audited: set[str]) -> dict[str, list[str]]:
    """Audited configurations grouped by everything that decides the two plans."""
    g: dict[str, list[str]] = {}
    for f in sorted(glob.glob(str(ROOT / "configs" / "experiments" / "*.yaml"))):
        c = yaml.safe_load(open(f))
        if c["name"] not in audited:
            continue
        p = params(c)
        key = json.dumps([c["dataset"], c["evaluation"], {k: p.get(k) for k in KEYS}], sort_keys=True, default=str)
        g.setdefault(key, []).append(c["name"])
    return g


def one_step_model(cfg: dict, train: pd.DataFrame):
    """The configuration's state model, fitted for one step: enough to build its
    inputs and its "nothing planned" plan."""
    import rwm.model  # noqa: F401
    from rwm.model.state_model import StateModel

    p = {k: v for k, v in model_params(cfg).items() if k not in ("seed",)}
    p.update(steps=1, batch=min(int(p.get("batch", 256)), 64), validation_periods=0, mixed_precision=False)
    p.pop("switching", None)  # rescales forecasts across a store; does not change either plan
    m = StateModel(**p)
    m.fit(train)
    return m


def compare(cfg: dict, m, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    prm = model_params(cfg)
    plan = test.drop(columns=[UNITS], errors="ignore")
    levers = [c for c in plan_levers(cfg) if c in plan.columns]
    reg = regular_prices(train, prm.get("regular_window", 12))
    audit = off_plan(plan, reg, levers)
    model = m.nothing_planned(plan)
    known = series_rows(plan[SERIES], m._names) >= 0
    # a series with no history is forecast as zero (lift zero) unless the model
    # borrows a forecast for it (new-item tests)
    new = ~known & bool(getattr(m, "cold_start", False))
    a, b = audit[PRICE].to_numpy(float), model[PRICE].to_numpy(float)
    price_diff = ~(np.isclose(a, b, rtol=1e-6, atol=1e-9) | (np.isnan(a) & np.isnan(b)))
    lever_diff = np.zeros(len(plan), dtype=bool)
    for c in set(levers) | set(m.levers):
        if c in plan.columns:
            x, y = audit[c].to_numpy(float), model[c].to_numpy(float)
            lever_diff |= ~(np.isclose(x, y) | (np.isnan(x) & np.isnan(y)))
    no_reg = plan[SERIES].map(reg).isna().to_numpy()
    out = {
        "rows": len(plan),
        "differ": float(((price_diff | lever_diff) & known | new).mean()),
        "audit_no_regular": float((price_diff & known & no_reg).mean()),
        "rule_differs": float((price_diff & known & ~no_reg).mean()),
        "new_item": float(new.mean()),
        "lever_differs": float((lever_diff & known).mean()),
        "series_differ": float(plan.loc[(price_diff | lever_diff) & known | new, SERIES].nunique()
                               / max(plan[SERIES].nunique(), 1)),
    }
    # the model's fallback when its history window records no price for the series
    out["model_price_one"] = float((price_diff & known & np.isclose(b, 1.0)).mean())
    # how far apart the two prices are where they differ
    r = np.abs(np.log(np.where(price_diff & known & (a > 0) & (b > 0), a / b, 1.0)))
    out["median_log_gap"] = float(np.median(r[price_diff & known])) if (price_diff & known).any() else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stores", nargs="*", help="M5 only: load these stores (for memory)")
    ap.add_argument("--sample-series", type=int, default=0,
                    help="keep this many series drawn at random (for memory); recorded in the output")
    a = ap.parse_args(argv)
    v = pd.read_csv(ROOT / "results" / "paper2" / "summary" / "versions.csv")
    audited = set(v["config"].str.replace(r"_seed\d+$", "", regex=True))
    rows = []
    out = Path(a.out)
    done = set(pd.read_csv(out)["group"]) if out.exists() else set()
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    for key, names in groups(audited).items():
        if a.only and not set(a.only) & set(names):
            continue
        if names[0] in done:
            continue
        cfg = yaml.safe_load(open(ROOT / "configs" / "experiments" / f"{names[0]}.yaml"))
        sample = ""
        if a.stores and cfg["dataset"]["name"] == "m5":
            cfg["dataset"].setdefault("params", {})["stores"] = a.stores
            sample = "stores " + " ".join(a.stores)
        for origin, train, test in windows(cfg):
            if a.sample_series:
                keep = pd.Series(train[SERIES].unique()).sample(
                    min(a.sample_series, train[SERIES].nunique()), random_state=0)
                train, test = train[train[SERIES].isin(keep)], test[test[SERIES].isin(keep)].reset_index(drop=True)
                sample = (sample + f" {len(keep)} series").strip()
            m = one_step_model(cfg, train)
            res = compare(cfg, m, train, test)
            for n in names:
                rows.append({"group": names[0], "config": n, "origin": origin, "sample": sample, **res})
            print(names[0], origin, {k: round(x, 4) for k, x in res.items()}, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
