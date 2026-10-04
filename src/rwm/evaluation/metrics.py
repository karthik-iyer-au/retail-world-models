"""RMSSE for one series, written out plainly.

This is the M5 measure: forecast error divided by the error of a one-step
naive forecast on the series' own history, counted from its first sale.
Below 1 is better than naive. `hierarchy.py` computes the same thing for all
series and levels at once; the tests check the two agree.
"""
import numpy as np


def rmsse(train: np.ndarray, actual: np.ndarray, forecast: np.ndarray, lag: int = 1) -> float:
    train = np.asarray(train, dtype=float)
    nz = np.flatnonzero(train)
    if len(nz) == 0:
        return float("nan")
    train = train[nz[0]:]  # M5 rule: history starts at the first sale
    if len(train) <= lag:
        return float("nan")
    scale = np.mean((train[lag:] - train[:-lag]) ** 2)
    if scale == 0:
        return float("nan")
    err = np.mean((np.asarray(actual, float) - np.asarray(forecast, float)) ** 2)
    return float(np.sqrt(err / scale))

