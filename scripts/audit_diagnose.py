"""Paper 2, diagnosis of the C1 and C4 results (docs/PAPER2_PLAN.md, amendment of
17 October 2026). Reads saved models; changes nothing about them or about the
pre-registered scores, which are reported as computed.

  python scripts/audit_diagnose.py --results <runs folder> --runs <run folder name> ... --out <csv>

For each run and window:
  C1 as audited, and again with the audit's regular price taken by the model's
  own rule (highest price in the last `regular_window` periods of the series'
  own history, else its mean price), and the share of rows where the audit's
  regular price was undefined.
  C4 as audited (every price in the plan cut at once), and with only a random
  tenth of series cut (so that cuts to other items barely enter), each with the
  share of rows lower by more than 0.1% and by more than 5%, and the share by
  cut depth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.audit import construction_checks, price_monotonicity, regular_prices
from rwm.audit.runs import load_checkpoint, model_params, plan_levers, windows
from rwm.data.schema import DATE, PRICE, SERIES, UNITS

CUTS = (0.05, 0.10, 0.20, 0.30)


def model_rule_regular(model, window: int) -> pd.Series:
    """The state model's own regular price, from the price history it stores: the
    highest price in the last `window` periods, else the mean of its recorded
    prices (StateModel._forward)."""
    P = model._past["price"].detach().cpu().numpy()
    recent = np.where(np.isnan(P[:, -window:]), -np.inf, P[:, -window:]).max(axis=1)
    level = np.nansum(P, axis=1) / np.maximum((~np.isnan(P)).sum(axis=1), 1)
    return pd.Series(np.where(np.isfinite(recent) & (recent > 0), recent, level), index=model._names.to_numpy())


def c1_keep_missing(model, test: pd.DataFrame, regular: pd.Series, levers: list) -> dict:
    """C1 with 'nothing planned' as the model itself defines it: levers off, and
    the regular price only where the plan records a price (a missing price stays
    missing, as in the model's own baseline)."""
    plan = test.drop(columns=[UNITS], errors="ignore")
    off = plan.copy()
    for lv in levers:
        off[lv] = 0
    reg = off[SERIES].map(regular)
    off[PRICE] = np.where(off[PRICE].notna() & reg.notna(), reg, off[PRICE])
    on, nb = model.breakdown(plan), model.breakdown(off)
    top = max(float(on["forecast"].max()), 1e-9)
    big = np.abs(nb["lift"].to_numpy()) / top
    return {"zero_lift_when_nothing_planned": float(big.max()),
            "baseline_moves_with_plan": float(np.abs(on["baseline"] - nb["baseline"]).max() / top),
            "share_rows_over_0.001": float((big > 1e-3).mean()),
            "price_missing_share": float(plan[PRICE].isna().mean())}


def monotone(model, plan: pd.DataFrame, mask: np.ndarray) -> dict:
    prev = model.predict(plan)
    out = {}
    lower, lower5, n = 0, 0, 0
    for c in CUTS:
        cur = model.predict(plan.assign(**{PRICE: np.where(mask, plan[PRICE] * (1 - c), plan[PRICE])}))
        d = (cur < prev - 1e-3 * np.maximum(prev, 1e-9))[mask]
        d5 = (cur < prev - 5e-2 * np.maximum(prev, 1e-9))[mask]
        out[f"lower_at_{int(c * 100)}"] = float(d.mean())
        lower += int(d.sum())
        lower5 += int(d5.sum())
        n += int(mask.sum())
        prev = cur
    out["lower"] = lower / max(n, 1)
    out["lower_5pct"] = lower5 / max(n, 1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = []
    for name in a.runs:
        run = Path(a.results) / name
        cfg = json.loads((run / "manifest.json").read_text())["config"]
        prm = model_params(cfg)
        win = prm.get("regular_window", 12)
        for origin, train, test in windows(cfg):
            ck = run / f"checkpoint_origin{origin}.pt"
            if not ck.exists():
                continue
            m = load_checkpoint(run, origin)
            levers = [c for c in plan_levers(cfg) if c in test.columns]
            reg_audit = regular_prices(train, win)
            try:
                reg_model = model_rule_regular(m, win)
            except AttributeError:  # an averaged model keeps its history in each member
                reg_model = model_rule_regular(m.models[0], win) if hasattr(m, "models") else reg_audit
            undefined = float(test[SERIES].map(reg_audit).isna().mean())
            c1a = construction_checks(m, test, reg_audit, levers) if hasattr(m, "breakdown") else {}
            c1m = construction_checks(m, test, reg_model, levers) if hasattr(m, "breakdown") else {}
            c1k = c1_keep_missing(m, test, reg_model, levers) if hasattr(m, "breakdown") else {}
            plan = test.drop(columns=[UNITS], errors="ignore")
            rng = np.random.default_rng(origin)
            series = plan[SERIES].unique()
            tenth = set(rng.choice(series, size=max(1, len(series) // 10), replace=False))
            every = monotone(m, plan, np.ones(len(plan), dtype=bool))
            some = monotone(m, plan, plan[SERIES].isin(tenth).to_numpy())
            rows.append({"run": name, "config": cfg["name"], "origin": origin, "rows": len(test),
                         "regular_undefined_share": undefined,
                         **{f"c1_audit_{k}": v for k, v in c1a.items()},
                         **{f"c1_model_rule_{k}": v for k, v in c1m.items()},
                         **{f"c1_keep_missing_{k}": v for k, v in c1k.items()},
                         "c4_audit": price_monotonicity(m, test)["share_lower_after_deeper_cut"],
                         **{f"c4_all_{k}": v for k, v in every.items()},
                         **{f"c4_tenth_{k}": v for k, v in some.items()}})
            pd.DataFrame(rows).to_csv(a.out, index=False, float_format="%.5g")
            print(name, origin, "done", flush=True)


if __name__ == "__main__":
    main()
