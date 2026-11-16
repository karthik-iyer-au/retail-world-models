"""Paper 3 (ICWM 2027): forecasting the plans merchants make against simulating the plans they never make
(docs/PAPER3_PLAN.md, fixed 16 November 2026 before any run).

  python scripts/paper3_benchmark.py --out <folder> --confounding 1.0 --seed 101 [--versions wm_d32_s2000_h26 ...]

For one generated world (planning strength, seed), fits each model version on weeks 1 to 252, forecasts weeks 253 to
260 under the observed plan and under the held-out plans, and scores every forecast against the expected sales the
data were generated from. Writes <out>/c<confounding>_s<seed>_<version>.json; finished versions are skipped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from known_answer import HIDDEN, LEVERS, LIGHTGBM, ORIGIN, WORLD_MODEL, accuracy  # noqa: E402

from rwm.data import load_dataset  # noqa: E402
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS  # noqa: E402
from rwm.data.synthetic_levers import Truth, expected_units, planned  # noqa: E402
from rwm.forecaster import build_model  # noqa: E402

DATA = dict(n_zones=3, stores_per_zone=6, n_items=20, n_periods=260, test_origin=ORIGIN)
DEEP = (0.50, 0.60)
RR = "regular_price"  # the recorded regular price, given to the `_rr` versions only
REGULAR_MOVES = (-0.10, -0.05, 0.0, 0.05, 0.10)


def versions() -> dict:
    out = {}
    for d, steps, hist in itertools.product((16, 32, 64), (500, 2000), (13, 26)):
        out[f"wm_d{d}_s{steps}_h{hist}"] = ("state_model", {**WORLD_MODEL, "d_model": d, "steps": steps,
                                                            "history": hist, "seed": 1})
    for leaves, rounds, mdl in itertools.product((7, 31, 127), (100, 500), (20, 100)):
        out[f"lgb_l{leaves}_r{rounds}_m{mdl}"] = ("lightgbm_direct", {
            **LIGHTGBM, "rounds": rounds, "params": {"num_leaves": leaves, "min_data_in_leaf": mdl}})
    out["loglinear"] = ("loglinear", {})
    # amendment 1 (16 November 2026): world model versions trained to learn the effect of actions
    base = out["wm_d32_s2000_h26"][1]
    out["wm_jepa"] = ("state_model", {**base, "pretrain_steps": 500, "latent_weight": 0.1})
    out["wm_latent"] = ("state_model", {**base, "latent_weight": 0.1})
    out["wm_overshoot"] = ("state_model", {**base, "latent_weight": 0.1, "latent_checks": 7})
    out["wm_adajepa"] = ("state_model", {**base, "adapt_steps": 3, "adapt_optimizer": "adam", "adapt_lr": 1e-4,
                                         "adapt_periods": 4})
    # amendment 1: every model given the planned (recorded) regular price
    out["wm_rr"] = ("state_model", {**base, "regular_column": RR})
    lgb = out["lgb_l31_r500_m20"][1]
    out["lgb_rr"] = ("lightgbm_direct", {**lgb, "extra": list(lgb["extra"]) + [RR]})
    out["loglinear_rr"] = ("loglinear", {"recorded_regular": True})
    return out


# ---------- the world ----------

def world(confounding: float, seed: int, shape: str = "loglinear"):
    bent = dict(shape="bent", both_effect=-0.10) if shape == "bent" else {}
    panel = load_dataset("synthetic_levers", confounding=confounding, seed=seed, **bent, **DATA).panel
    panel[RR] = panel["true_regular"]
    dates = np.sort(panel[DATE].unique())
    train = panel[panel[DATE] < dates[ORIGIN - 1]].copy()
    test = panel[panel[DATE] >= dates[ORIGIN - 1]].copy()
    last = train[train[DATE] == train[DATE].max()]
    prev = pd.Series(planned(last), index=last[SERIES].to_numpy())
    test["true_prev_planned"] = test[SERIES].map(prev).astype(int)
    return panel, train.reset_index(drop=True), test.reset_index(drop=True)


def truth(plan: pd.DataFrame, hidden: pd.DataFrame, t: Truth, regular_move: float = 0.0) -> np.ndarray:
    """Expected sales of a plan. A lasting regular-price move changes the regular price and, through the regular-price
    elasticity, the level, as in the generator."""
    h = hidden.copy()
    if regular_move:
        h["true_regular"] = h["true_regular"] * (1 + regular_move)
        # the level's response to a lasting move of the regular price, about the current regular price
        f = (1 + t.regular_elasticity * regular_move) if t.shape == "bent" else (1 + regular_move) ** t.regular_elasticity
        h["true_level"] = h["true_level"] * f
    return expected_units(pd.concat([plan.reset_index(drop=True), h.reset_index(drop=True)], axis=1), t)


def plans(test: pd.DataFrame) -> dict:
    """The observed plan and the held-out plans (docs/PAPER3_PLAN.md). Each entry: (plan, regular move)."""
    plan = test.drop(columns=[UNITS] + HIDDEN)
    reg = test["true_regular"].to_numpy()
    cut = plan[PRICE].to_numpy() < reg * (1 - 1e-6)
    out = {"observed": (plan, 0.0)}
    for d in DEEP:
        out[f"deeper_{int(d * 100)}"] = (plan.assign(**{PRICE: np.where(cut, reg * (1 - d), plan[PRICE])}), 0.0)
    for m in REGULAR_MOVES:
        if m:
            out[f"regular_{int(round(m * 100)):+d}"] = (plan.assign(**{PRICE: plan[PRICE] * (1 + m), RR: plan[RR] * (1 + m)}), m)
    # promotions and displays moved from the weeks they were planned to the series' lowest-demand weeks
    moved = plan.copy()
    for c in LEVERS:
        new = np.zeros(len(plan), dtype=int)
        for _, idx in test.groupby(SERIES).groups.items():
            idx = np.asarray(idx)
            k = int(plan.loc[idx, c].sum())
            if k:
                low = idx[np.argsort(test.loc[idx, "true_level"].to_numpy(), kind="stable")[:k]]
                new[low] = 1
        moved[c] = new
    out["off_pattern"] = (moved, 0.0)
    out["no_plan"] = (plan.assign(**{PRICE: np.maximum(plan[PRICE], reg)}, **{c: 0 for c in LEVERS}), 0.0)
    return out


# ---------- the log-linear model ----------

class LogLinear:
    """Poisson regression of sales on the series, the season, log price over the 12-week highest price, log regular
    price over the series' mean regular price, promotion, display, both, and the week after a planned week."""

    def __init__(self, recorded_regular: bool = False):
        self.recorded = recorded_regular

    def fit(self, train: pd.DataFrame):
        from sklearn.linear_model import PoissonRegressor
        self.hist = train.sort_values([SERIES, DATE]).reset_index(drop=True)
        X, self.cols = self._features(self.hist, self.hist, fit=True)
        self.m = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(X, self.hist[UNITS].to_numpy())
        return self

    def _features(self, rows: pd.DataFrame, hist: pd.DataFrame, fit=False):
        keep = [SERIES, DATE, PRICE, *LEVERS] + ([RR] if self.recorded else [])
        allp = pd.concat([hist[keep], rows[keep]])
        allp = allp.drop_duplicates([SERIES, DATE], keep="last").sort_values([SERIES, DATE])
        if self.recorded:
            allp["reg"] = allp[RR]
        else:
            allp["reg"] = allp.groupby(SERIES)[PRICE].transform(lambda s: s.rolling(12, min_periods=1).max())
        on = (allp[PRICE] < allp["reg"] * (1 - 1e-6)) | (allp["promo"] > 0) | (allp["display"] > 0)
        allp["after"] = (on.groupby(allp[SERIES]).shift(1).fillna(False).astype(bool) & ~on).astype(float)
        f = rows[[SERIES, DATE]].merge(allp[[SERIES, DATE, "reg", "after"]], on=[SERIES, DATE], how="left")
        if fit:
            self.mean_reg = f.groupby(SERIES)["reg"].mean()
            self.series = sorted(rows[SERIES].unique())
            self.t0 = rows[DATE].min()
        week = ((pd.to_datetime(rows[DATE]) - pd.Timestamp(self.t0)).dt.days / 7).to_numpy()
        num = pd.DataFrame({
            "lp": np.log(rows[PRICE].to_numpy() / f["reg"].to_numpy()),
            "lr": np.log(f["reg"].to_numpy() / rows[SERIES].map(self.mean_reg).to_numpy()),
            "promo": rows["promo"].to_numpy(), "display": rows["display"].to_numpy(),
            "both": rows["promo"].to_numpy() * rows["display"].to_numpy(), "after": f["after"].to_numpy(),
        })
        dummies = pd.get_dummies(pd.Categorical(rows[SERIES], categories=self.series), dtype=float)
        season = pd.DataFrame(index=range(len(rows)))
        # season per item (sine and cosine of the year, one pair per item)
        items = sorted({s.split("_")[1] for s in self.series})
        it = rows[SERIES].str.split("_").str[1].to_numpy()
        ang = 2 * np.pi * week / 52
        for i in items:
            m = (it == i).astype(float)
            season[f"sin_{i}"] = m * np.sin(ang)
            season[f"cos_{i}"] = m * np.cos(ang)
        X = pd.concat([num, dummies.reset_index(drop=True), season], axis=1)
        return X.to_numpy(), list(X.columns)

    def predict(self, plan: pd.DataFrame) -> np.ndarray:
        X, _ = self._features(plan.reset_index(drop=True), self.hist)
        return self.m.predict(X)


# ---------- scoring ----------

def promo_choice(model, plan, hidden, t) -> float:
    hidden = hidden.reset_index(drop=True)
    on, off = plan.assign(promo=1), plan.assign(promo=0)
    pred = model.predict(on) - model.predict(off)
    true = truth(on, hidden, t) - truth(off, hidden, t)
    k = len(true) // 2
    best = np.sort(true)[::-1][:k].sum()
    return float(true[np.argsort(-pred, kind="stable")[:k]].sum() / best)


def regular_choice(preds: dict, trues: dict, plan: pd.DataFrame, plans_: dict) -> float:
    series = plan[SERIES].to_numpy()
    rev_p, rev_t = {}, {}
    for m in REGULAR_MOVES:
        key = "observed" if not m else f"regular_{int(round(m * 100)):+d}"
        price = plans_[key][0][PRICE].to_numpy()
        rev_p[m] = pd.Series(price * preds[key]).groupby(series).sum()
        rev_t[m] = pd.Series(price * trues[key]).groupby(series).sum()
    P, T = pd.DataFrame(rev_p), pd.DataFrame(rev_t)
    chosen = P.idxmax(axis=1)
    got = np.array([T.loc[s, chosen[s]] for s in T.index])
    best = T.max(axis=1).to_numpy()
    return float((best - got).sum() / best.sum())


def run(model, train, test, t: Truth, keep_rr: bool) -> dict:
    hidden = test[HIDDEN]
    ps = plans(test)
    if not keep_rr:
        ps = {k: (p.drop(columns=[RR]), m) for k, (p, m) in ps.items()}
    preds = {k: np.asarray(model.predict(p)) for k, (p, _) in ps.items()}
    trues = {k: truth(p, hidden, t, m) for k, (p, m) in ps.items()}
    out = {"accuracy": accuracy(train, test, preds["observed"])}
    obs_p, obs_t = preds["observed"], trues["observed"]
    for k in ps:
        p, tr = preds[k], trues[k]
        rec = {"stock": float(np.abs(p - tr).sum() / tr.sum()), "sum_pred_over_true": float(p.sum() / tr.sum())}
        if k != "observed":
            dp, dt = p - obs_p, tr - obs_t
            rec["change_ratio"] = float(dp.sum() / dt.sum()) if abs(dt.sum()) > 1e-9 else float("nan")
            rec["change_rank"] = float(pd.Series(dp).corr(pd.Series(dt), method="spearman"))
        out[k] = rec
    out["promo_choice"] = promo_choice(model, ps["observed"][0], hidden, t)
    out["regular_choice_loss"] = regular_choice(preds, trues, ps["observed"][0], ps)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--confounding", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--versions", nargs="*")
    p.add_argument("--steps", type=int, help="override training steps (smoke test only)")
    p.add_argument("--shape", default="loglinear", choices=["loglinear", "bent"])
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panel, train, test = world(a.confounding, a.seed, a.shape)
    tag = "" if a.shape == "loglinear" else "bent_"
    t = Truth(**panel.attrs["truth"])
    V = versions()
    for name in a.versions or list(V):
        f = out / f"{tag}c{a.confounding}_s{a.seed}_{name}.json"
        if f.exists():
            continue
        kind, settings = V[name][0], dict(V[name][1])
        if a.steps and kind == "state_model":
            settings["steps"] = a.steps
        t0 = time.time()
        keep_rr = name.endswith("_rr")
        fit_on = train.drop(columns=HIDDEN + ([] if keep_rr else [RR]))
        model = (LogLinear(**settings).fit(fit_on) if kind == "loglinear"
                 else build_model(kind, **settings).fit(fit_on))
        rec = {"confounding": a.confounding, "seed": a.seed, "shape": a.shape, "version": name, "kind": kind,
               **run(model, train, test, t, keep_rr), "seconds": round(time.time() - t0, 1)}
        f.write_text(json.dumps(rec, indent=1))
        print(name, json.dumps({k: rec[k] for k in ("accuracy", "promo_choice", "regular_choice_loss")}), flush=True)


if __name__ == "__main__":
    main()
