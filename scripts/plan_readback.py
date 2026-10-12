"""Can the plan be read back from the world model's state?

  python scripts/plan_readback.py --config <experiment config> --run <run folder> [--out <file>]

The world model carries each item's state forward one period at a time under
the plan (price, events, SNAP days). If the state really takes the plan in,
a linear read-out of the state after period h should recover what was
planned for period h, and, if the model remembers, what was planned a few
periods earlier (the week after a promotion is when stock-up shows).

The model is frozen; nothing is retrained. For the test window of the run,
the script collects the state after each period and fits a ridge read-out on
half of the items, scored on the other half (R squared). The same read-out
from the state before the window (which has not seen the plan) is the
control: what the plan can be guessed from history alone. The gap between
the two is what the state learned from the plan.

A model built without the roll-forward (the lift model) has no such state;
the script reports the control only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS

LAGS = (0, 1, 7)


def ridge_r2(x_fit, y_fit, x_test, y_test, alpha=1.0):
    """R squared on held-out rows of a ridge read-out with an intercept."""
    mu, sd = x_fit.mean(0), x_fit.std(0) + 1e-6
    a, b = (x_fit - mu) / sd, (x_test - mu) / sd
    a, b = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ y_fit)
    resid = y_test - b @ w
    total = ((y_test - y_test.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / total) if total > 0 else float("nan")


def plan_columns(model) -> list[str]:
    """Names of the first columns of one period's plan, as the model builds them."""
    names = ["price / usual price", "price known"]
    for e in model.extra:
        names += [e, f"{e} known"]
    if model.regular_price:
        names.append("price / regular price")
    return names


def collect(model, test: pd.DataFrame, max_items: int, seed: int):
    """State before the window, state after each period, and the plan, for a sample of items."""
    import torch

    fut_dates, fut = model._future(test.drop(columns=[UNITS], errors="ignore"))
    h = len(fut_dates)
    states, rolled, plans = [], [], []
    model._net.eval()
    with torch.no_grad():
        for rows, picked in model._batches():
            start = torch.full_like(rows, model._past["price"].shape[1])
            got, _, on_sale, _ = model._forward(model._past, {**fut, "start": torch.zeros_like(rows)}, rows, start, picked, full=True)
            real = torch.ones_like(rows, dtype=torch.bool) if picked is None else picked.reshape(-1) >= 0
            states.append(got["state"][real].float().cpu().numpy())
            plans.append(got["plan"][real][:, :h].float().cpu().numpy())
            rolled.append(None if got["rolled"] is None else got["rolled"][real][:, :h].float().cpu().numpy())
    state = np.concatenate(states)
    plan = np.concatenate(plans)
    roll = None if rolled[0] is None else np.concatenate(rolled)
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(state), min(max_items, len(state)), replace=False)
    return state[keep], (None if roll is None else roll[keep]), plan[keep]


def readback(state, roll, plan, names, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    n, h, _ = plan.shape
    fit = rng.random(n) < 0.5
    out = []
    for j, name in enumerate(names):
        for lag in LAGS:
            if lag >= h:
                continue
            steps = np.arange(lag, h)
            y = plan[:, steps - lag, j]  # what was planned `lag` periods before each step
            if np.nanstd(y) == 0:
                continue
            row = {"plan input": name, "periods back": lag}
            x0 = np.repeat(state[:, None, :], len(steps), axis=1)
            yf, yt = y[fit].reshape(-1), y[~fit].reshape(-1)
            row["from state before window"] = ridge_r2(x0[fit].reshape(len(yf), -1), yf, x0[~fit].reshape(len(yt), -1), yt)
            if roll is not None:
                xr = roll[:, steps]
                row["from rolled state"] = ridge_r2(xr[fit].reshape(len(yf), -1), yf, xr[~fit].reshape(len(yt), -1), yt)
                row["gain"] = row["from rolled state"] - row["from state before window"]
            out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="the experiment config the run was made from")
    p.add_argument("--run", required=True, help="run folder holding checkpoint_origin0.pt")
    p.add_argument("--max-items", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from rwm.model.state_model import StateModel

    config = yaml.safe_load(Path(args.config).read_text())
    ev = config["evaluation"]
    ds = load_dataset(config["dataset"]["name"], **config["dataset"].get("params", {}))
    panel = ds.panel
    dates = np.sort(panel[DATE].unique())
    if ev.get("holdout_periods"):
        dates = dates[: -ev["holdout_periods"]]
    test = panel[panel[DATE].isin(dates[-ev["horizon"]:])].reset_index(drop=True)
    del panel, ds

    model = StateModel.load(Path(args.run) / "checkpoint_origin0.pt")
    state, roll, plan = collect(model, test, args.max_items, args.seed)
    rows = readback(state, roll, plan, plan_columns(model), args.seed)
    table = pd.DataFrame(rows)
    print(f"{config['name']}: {len(state)} items, {plan.shape[1]} periods, rolled state: {roll is not None}")
    print(table.round(3).to_string(index=False))
    out = Path(args.out) if args.out else Path(args.run) / "plan_readback.json"
    out.write_text(json.dumps({"config": config["name"], "run": Path(args.run).name, "items": int(len(state)),
                               "rolled_state": roll is not None, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
