"""Forecasting far ahead in blocks, each block's forecast taken in as sales.

A model fitted to forecast `block` periods ahead forecasts a longer window in
blocks. After each block its forecast is added to the history as if it were the
sales of those periods, with the plan as recorded, and the next block is
forecast from the end of that history through the model's `refresh`, without
refitting. Used by the 52-week test (docs/PAPER1_REVISION_PLAN.md, 13 November
2026), where both models forecast 52 weeks in four blocks of 13.

  model:
    name: blocks
    params:
      block: 13               # periods per block; the wrapped model's horizon
      model: seed_average     # the model to wrap
      params: {...}           # its own settings
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, UNITS
from rwm.forecaster import Forecaster, build_model, register_model


@register_model("blocks")
class Blocks(Forecaster):
    def __init__(self, model: str, block: int, params: dict | None = None):
        if block < 1:
            raise ValueError("block must be at least 1")
        self.block = block
        self.inner = build_model(model, **(params or {}))
        self._history: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> "Blocks":
        self.inner.fit(train)
        self._history = train
        return self

    def refresh(self, history: pd.DataFrame) -> "Blocks":
        self.inner.refresh(history)
        self._history = history
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("fit before predict")
        dates = np.sort(future[DATE].unique())
        out = np.full(len(future), np.nan)
        history = self._history
        for b in range(0, len(dates), self.block):
            these = dates[b:b + self.block]
            rows = np.flatnonzero(future[DATE].isin(these).to_numpy())
            part = future.iloc[rows]
            pred = np.asarray(self.inner.predict(part), dtype=float)
            out[rows] = pred
            if b + self.block < len(dates):
                taken = part.assign(**{UNITS: pred})
                history = pd.concat([history, taken[history.columns.intersection(taken.columns)]],
                                    ignore_index=True)
                self.inner.refresh(history)
        # leave the wrapped model where fit put it, so a second predict starts from the same place
        if len(dates) > self.block:
            self.inner.refresh(self._history)
        return out
