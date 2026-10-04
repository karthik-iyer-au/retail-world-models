"""Reference forecasts: repeat last season, and average of recent periods.

These are the standard simple benchmarks (Makridakis et al., M5). Any model
that cannot beat them is not worth reporting.

Both work from dates, not row positions, so gaps in a series do not shift
the forecast onto the wrong week or weekday.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model


def _period(dates: np.ndarray) -> np.timedelta64:
    """Length of one period: the smallest gap between dates in the data."""
    return np.diff(np.sort(dates)).min()


@register_model("seasonal_naive")
class SeasonalNaive(Forecaster):
    """Forecast = sales one season earlier. season=7 for daily data, 52 for weekly.
    Beyond one season ahead it repeats the last season. No record means zero."""

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        dates = train[DATE].unique()
        self._end = dates.max()
        self._span = _period(dates) * self.season
        last = train[train[DATE] > self._end - self._span]
        self._last = last[[SERIES, DATE, UNITS]].rename(columns={DATE: "source"})
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        seasons_back = np.ceil((future[DATE] - self._end) / self._span).astype(int)
        fut = pd.DataFrame(
            {SERIES: future[SERIES].to_numpy(), "source": (future[DATE] - seasons_back * self._span).to_numpy()}
        )
        out = fut.merge(self._last, on=[SERIES, "source"], how="left")
        return out[UNITS].fillna(0.0).to_numpy()


@register_model("recent_average")
class RecentAverage(Forecaster):
    """Forecast = average over the last `window` periods, held flat."""

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, train: pd.DataFrame) -> "RecentAverage":
        dates = train[DATE].unique()
        recent = train[train[DATE] > dates.max() - _period(dates) * self.window]
        self._mean = recent.groupby(SERIES, observed=True)[UNITS].sum() / self.window
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return future[SERIES].map(self._mean).astype("float64").fillna(0.0).to_numpy()
