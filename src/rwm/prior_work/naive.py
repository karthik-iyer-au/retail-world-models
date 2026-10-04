"""Reference forecasts: repeat last season, and average of recent periods.

These are the standard simple benchmarks (Makridakis et al., M5). Any model
that cannot beat them is not worth reporting.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model
from rwm.utils.panel import last_k_rows


@register_model("seasonal_naive")
class SeasonalNaive(Forecaster):
    """Forecast = the value one season ago. season=7 for daily data."""

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        tail = train.iloc[last_k_rows(train[SERIES], self.season)][[SERIES, UNITS]].copy()
        tail["pos"] = tail.groupby(SERIES, observed=True).cumcount()
        self._tail = tail
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        fut = future[[SERIES, DATE]].copy()
        fut["row"] = np.arange(len(fut))
        fut = fut.sort_values([SERIES, DATE])
        fut["pos"] = fut.groupby(SERIES, observed=True).cumcount() % self.season
        out = fut.merge(self._tail, on=[SERIES, "pos"], how="left").sort_values("row")
        return out[UNITS].fillna(0.0).to_numpy()


@register_model("recent_average")
class RecentAverage(Forecaster):
    """Forecast = average of the last `window` periods, held flat."""

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, train: pd.DataFrame) -> "RecentAverage":
        tail = train.iloc[last_k_rows(train[SERIES], self.window)]
        self._mean = tail.groupby(SERIES, observed=True)[UNITS].mean()
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return future[SERIES].map(self._mean).fillna(0.0).to_numpy()
