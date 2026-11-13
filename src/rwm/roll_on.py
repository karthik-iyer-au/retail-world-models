"""Forecasting further than the horizon trained for by rolling the world model's state on.

The world model is fitted to forecast `params.horizon` periods ahead, as tuned. To forecast a longer window it reads
its state from the latest sales once and rolls that state forward under the plan for every period of the window, its
own forecasts never taken in as sales. Past the trained horizon the transition is applied as trained and the readout
uses the last trained step's embedding. Added on 14 November 2026 after the 52-week test (see
docs/PAPER1_REVISION_PLAN.md): there, taking each 13-week block's forecast in as sales made the world model drift
upward block after block.

  model:
    name: roll_on
    params:
      model: seed_average   # the world model, or an average of copies of it
      params: {...}
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE
from rwm.forecaster import Forecaster, build_model, register_model


def _set_horizon(m, h: int) -> None:
    for sub in getattr(m, "models", None) or [m]:
        if sub is m:
            if not hasattr(m, "horizon"):
                raise TypeError(f"{type(m).__name__} cannot roll on")
            m.horizon = h
        else:
            _set_horizon(sub, h)


@register_model("roll_on")
class RollOn(Forecaster):
    def __init__(self, model: str, params: dict | None = None):
        self.inner = build_model(model, **(params or {}))

    def fit(self, train: pd.DataFrame) -> "RollOn":
        self.inner.fit(train)
        return self

    def refresh(self, history: pd.DataFrame) -> "RollOn":
        self.inner.refresh(history)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        _set_horizon(self.inner, int(future[DATE].nunique()))
        return np.asarray(self.inner.predict(future), dtype=float)
