"""Forecasting far ahead with one model for each block of periods, no forecast taken in.

Model k (k = 1, 2, ...) is fitted to forecast `block * k` periods ahead and forecasts the k-th block of the window.
For LightGBM, whose lags start one horizon back, block k is then forecast from the freshest sales a model that far
ahead can use, and no forecast is taken in as sales: the direct strategy with one model per distance. Added on
14 November 2026 as LightGBM's counterpart of rolling the world model's state on (docs/PAPER1_REVISION_PLAN.md).

  model:
    name: by_block
    params:
      block: 13
      blocks: 4
      model: seed_average
      params: {...}        # with a `horizon` (inside `params` for seed_average), set to block * k for model k
"""
import copy

import numpy as np
import pandas as pd

from rwm.data.schema import DATE
from rwm.forecaster import Forecaster, build_model, register_model


def _with_horizon(model: str, params: dict, h: int) -> dict:
    p = copy.deepcopy(params)
    inner = p["params"] if model == "seed_average" else p
    inner["horizon"] = h
    if inner.get("validation_periods") and inner["validation_periods"] < h:
        inner["validation_periods"] = h
    return p


@register_model("by_block")
class ByBlock(Forecaster):
    def __init__(self, model: str, block: int, blocks: int, params: dict | None = None):
        self.block, self.blocks = block, blocks
        self.models = [build_model(model, **_with_horizon(model, params or {}, block * (k + 1))) for k in range(blocks)]

    def fit(self, train: pd.DataFrame) -> "ByBlock":
        for m in self.models:
            m.fit(train)
        return self

    def refresh(self, history: pd.DataFrame) -> "ByBlock":
        for m in self.models:
            m.refresh(history)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        dates = np.sort(future[DATE].unique())
        if len(dates) > self.block * self.blocks:
            raise ValueError("window longer than block * blocks")
        out = np.full(len(future), np.nan)
        for k, m in enumerate(self.models):
            these = dates[k * self.block:(k + 1) * self.block]
            if not len(these):
                break
            # model k forecasts from the window's start; its own horizon covers blocks 1 to k
            reach = future[DATE].isin(dates[:(k + 1) * self.block]).to_numpy()
            pred = np.asarray(m.predict(future[reach]), dtype=float)
            mine = future[reach][DATE].isin(these).to_numpy()
            out[np.flatnonzero(reach)[mine]] = pred[mine]
        return out
