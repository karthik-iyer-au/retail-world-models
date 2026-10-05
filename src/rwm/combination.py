"""A weighted average of the forecasts of different models.

Standard forecast combination. The weights are chosen on the last
`validation_periods` of the training data, which the members do not see
while the weights are being chosen; the members are then fitted again on
all of the training data. A combination is reported as a combination, never
as any one of its members.

  model:
    name: combination
    params:
      validation_periods: 8
      seed: 0
      models:
        - {name: state_model, params: {...}}
        - {name: lightgbm_direct, params: {...}}
"""
import inspect
import itertools

import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES, STORE, UNITS
from rwm.forecaster import _MODELS, Forecaster, build_model, register_model

STEP = 0.05  # weights are searched on this grid


def _score(frame: pd.DataFrame, forecast: np.ndarray) -> float:
    """Error at three levels (single series, store, everything), each as a
    share of its average sales, so a bias that adds up counts."""
    f = frame[[SERIES, STORE, DATE]].assign(y=frame[UNITS].to_numpy(), f=forecast)
    out = []
    for keys in ([SERIES, DATE], [STORE, DATE], [DATE]):
        g = f.groupby(keys, observed=True)[["y", "f"]].sum()
        out.append(np.sqrt(((g["f"] - g["y"]) ** 2).mean()) / max(g["y"].abs().mean(), 1e-9))
    return float(np.mean(out))


@register_model("combination")
class Combination(Forecaster):
    def __init__(self, models: list[dict], validation_periods: int, seed: int = 0):
        if len(models) < 2:
            raise ValueError("a combination needs at least two models")
        if len(models) > 3:
            raise ValueError("at most three models, to keep the weight search small")
        self.specs, self.validation_periods, self.seed = models, validation_periods, seed
        self.weights: list[float] | None = None

    def _build(self) -> list[Forecaster]:
        out = []
        for spec in self.specs:
            params = dict(spec.get("params") or {})
            build_model(spec["name"], **params)  # also makes sure the model is registered
            if "seed" in inspect.signature(_MODELS[spec["name"]].__init__).parameters:
                params["seed"] = self.seed
            out.append(build_model(spec["name"], **params))
        return out

    def fit(self, train: pd.DataFrame) -> "Combination":
        dates = np.sort(train[DATE].unique())
        cut = dates[-self.validation_periods - 1]
        inner, held = train[train[DATE] <= cut], train[train[DATE] > cut].reset_index(drop=True)
        plan = held.drop(columns=[UNITS])
        forecasts = [m.fit(inner).predict(plan) for m in self._build()]
        grid = np.arange(0, 1 + 1e-9, STEP)
        best = (float("inf"), None)
        for w in itertools.product(grid, repeat=len(forecasts) - 1):
            if sum(w) > 1 + 1e-9:
                continue
            w = [*w, 1 - sum(w)]
            score = _score(held, sum(a * f for a, f in zip(w, forecasts)))
            if score < best[0]:
                best = (score, w)
        self.weights = [round(float(a), 4) for a in best[1]]
        self.validation_scores = [_score(held, f) for f in forecasts] + [best[0]]
        print(f"combination weights {self.weights}, validation scores {[round(s, 4) for s in self.validation_scores]}", flush=True)
        self.models = [m.fit(train) for m in self._build()]
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return sum(w * m.predict(future) for w, m in zip(self.weights, self.models))
