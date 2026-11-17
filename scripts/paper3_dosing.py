"""Paper 3, amendment 8: the test in a second domain, a generated dosing world (docs/PAPER3_PLAN.md).

  python scripts/paper3_dosing.py --out <folder> --strength 1.0 --seed 101 [--versions ...] [--steps N]

360 patients in 18 clinics over 260 days. A persistent severity state lowers each patient's daily response count;
the drug accumulates (half-life 3 days) and raises the response along a saturating curve; doctors dose by
indication (more often, and larger doses, when severity is high). Models are fitted on days 1 to 252 and forecast
days 253 to 260 under the doctors' own doses and five held-out dosing plans, scored against the known expected
response. The dose reaches every model through the price columns (price = 1 - 0.95 x dose, regular price 1), so the
models and the price rules of the retail study apply unchanged. Writes <out>/dose_s<strength>_<seed>_<version>.json.
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
from paper3_benchmark import RR, LogLinear, versions  # noqa: E402

from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS  # noqa: E402
from rwm.forecaster import build_model  # noqa: E402

DAYS, ORIGIN, HORIZON = 260, 253, 8
CLINICS, PER_CLINIC = 18, 20
HALF_LIFE, EC50, EMAX = 3.0, 0.5, 1.0
RHO, SEV_EFFECT = 0.8, 0.5
PLANS = ["dose_50", "dose_60", "steady_20", "off_pattern", "no_dose"]
GRID = [f"wm_d{d}_s{s}_h{h}" for d in (16, 32, 64) for s in (500, 2000) for h in (13, 26)] + \
    [f"lgb_l{l}_r{r}_m{m}" for l in (7, 31, 127) for r in (100, 500) for m in (20, 100)] + ["loglinear"]
RULES = ["wm_shape", "lgb_shape", "loglinear_shape"]


class World:
    def __init__(self, strength: float, seed: int):
        rng = np.random.default_rng(seed)
        n = CLINICS * PER_CLINIC
        self.n = n
        self.base = np.exp(rng.normal(np.log(8.0), 0.5, n))
        sev = np.zeros((n, DAYS))
        for t in range(1, DAYS):
            sev[:, t] = RHO * sev[:, t - 1] + np.sqrt(1 - RHO ** 2) * rng.normal(size=n)
        self.sev = sev
        give = rng.random((n, DAYS)) < 1 / (1 + np.exp(-(-1.4 + strength * sev)))
        size = 0.10 + 0.30 / (1 + np.exp(-(strength * sev + rng.normal(0, 0.5, (n, DAYS)))))
        self.dose = np.where(give, size, 0.0)
        self.week = 1 + 0.1 * np.sin(2 * np.pi * np.arange(DAYS) / 7)
        self.units = rng.poisson(self.expected(self.dose)).astype(float)

    def expected(self, dose: np.ndarray) -> np.ndarray:
        k = np.exp(-np.log(2) / HALF_LIFE)
        c = np.zeros_like(dose)
        for t in range(dose.shape[1]):
            c[:, t] = (c[:, t - 1] * k if t else 0) + dose[:, t]
        return self.base[:, None] * np.exp(-SEV_EFFECT * self.sev) * (1 + EMAX * c / (c + EC50)) * self.week

    def level(self, dose: np.ndarray) -> np.ndarray:
        """The built-up drug level from the doses (amendment 9)."""
        k = np.exp(-np.log(2) / HALF_LIFE)
        c = np.zeros_like(dose)
        for t in range(dose.shape[1]):
            c[:, t] = (c[:, t - 1] * k if t else 0) + dose[:, t]
        return c

    def frame(self, dose: np.ndarray | None = None, exposure: bool = False) -> pd.DataFrame:
        dose = self.dose if dose is None else dose
        signal = np.minimum(self.level(dose), 3.0) / 3.0 if exposure else dose
        dates = pd.date_range("2020-01-01", periods=DAYS, freq="D")
        clinic = np.repeat(np.arange(CLINICS), PER_CLINIC)
        sid = [f"c{c}_p{i}" for i, c in enumerate(clinic)]
        return pd.DataFrame({
            SERIES: np.repeat(sid, DAYS), STORE: np.repeat([f"c{c}" for c in clinic], DAYS),
            ITEM: np.repeat([f"p{i}" for i in range(self.n)], DAYS), DATE: np.tile(dates, self.n),
            UNITS: self.units.ravel(), PRICE: (1 - 0.95 * signal).ravel(), "promo": 0, "display": 0, RR: 1.0})

    def plan(self, kind: str) -> np.ndarray:
        d = self.dose.copy()
        w = slice(ORIGIN - 1, ORIGIN - 1 + HORIZON)
        test = d[:, w]
        if kind in ("dose_50", "dose_60"):
            test = np.where(test > 0, 0.5 if kind == "dose_50" else 0.6, 0.0)
        elif kind == "steady_20":
            test = np.full_like(test, 0.2)
        elif kind == "no_dose":
            test = np.zeros_like(test)
        elif kind == "off_pattern":  # the same doses, on each patient's days of lowest severity
            out = np.zeros_like(test)
            for i in range(self.n):
                k = int((test[i] > 0).sum())
                if k:
                    out[i, np.argsort(self.sev[i, w])[:k]] = np.sort(test[i][test[i] > 0])[::-1]
            test = out
        d[:, w] = test
        return d


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--strength", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--versions", nargs="*")
    p.add_argument("--steps", type=int)
    p.add_argument("--exposure", action="store_true", help="amendment 9: the built-up level in place of the dose")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = World(a.strength, a.seed)
    full = w.frame(exposure=a.exposure)
    dates = np.sort(full[DATE].unique())
    train = full[full[DATE] < dates[ORIGIN - 1]]
    window = dates[ORIGIN - 1:ORIGIN - 1 + HORIZON]
    truth_obs = w.expected(w.dose)[:, ORIGIN - 1:ORIGIN - 1 + HORIZON]
    plans = {k: w.plan(k) for k in PLANS}
    truth = {k: w.expected(v)[:, ORIGIN - 1:ORIGIN - 1 + HORIZON] for k, v in plans.items()}
    V = versions()
    y_tr = w.units[:, :ORIGIN - 1]
    scale = np.mean(np.diff(y_tr, axis=1) ** 2, axis=1)
    for name in a.versions or GRID + RULES:
        f = out / f"dose{'exp' if a.exposure else ''}_s{a.strength}_{a.seed}_{name}.json"
        if f.exists():
            continue
        kind, settings = V[name][0], dict(V[name][1])
        if a.steps and kind == "state_model":
            settings["steps"] = a.steps
        keep_rr = name.endswith("_rr") or "_shape" in name
        fit_on = train if keep_rr else train.drop(columns=[RR])
        t0 = time.time()
        model = LogLinear(**settings).fit(fit_on) if kind == "loglinear" else build_model(kind, **settings).fit(fit_on)

        def predict(dose: np.ndarray) -> np.ndarray:
            fr = w.frame(dose, exposure=a.exposure)
            fr = fr[fr[DATE].isin(window)].drop(columns=[UNITS] + ([] if keep_rr else [RR]))
            return np.clip(np.asarray(model.predict(fr), dtype=float), 0, None).reshape(w.n, HORIZON)

        p_obs = predict(w.dose)
        y_te = w.units[:, ORIGIN - 1:ORIGIN - 1 + HORIZON]
        rec = {"strength": a.strength, "seed": a.seed, "version": name, "kind": kind, "exposure": a.exposure,
               "rmsse": float(np.mean(np.sqrt(np.mean((y_te - p_obs) ** 2, axis=1) / np.maximum(scale, 1e-9))))}
        for k in PLANS:
            p_k = predict(plans[k])
            true_change = float((truth[k] - truth_obs).sum())
            rec[k] = {"change_ratio": float((p_k - p_obs).sum()) / true_change if true_change else float("nan"),
                      "true_change": true_change}
        rec["seconds"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(rec, indent=1))
        print(name, round(rec["rmsse"], 3), {k: round(rec[k]["change_ratio"], 2) for k in PLANS}, flush=True)


if __name__ == "__main__":
    main()
