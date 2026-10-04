"""Accuracy measures.

RMSSE is the M5 measure for one series: forecast error divided by the error
of a one-step naive forecast on that series' own history. Below 1 is better
than naive.

`weighted_rmsse` weights each series by its recent revenue, as M5 does, but
at the item-store level only. The full M5 WRMSSE also scores 11 aggregate
levels. That version is not implemented yet, so numbers from this module are
not comparable with the M5 leaderboard.
"""
import numpy as np
import pandas as pd


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


def weighted_rmsse(scores: np.ndarray, weights: np.ndarray) -> float:
    scores, weights = np.asarray(scores, float), np.asarray(weights, float)
    ok = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(scores[ok] * weights[ok]) / np.sum(weights[ok]))


def rmsse_by_series(
    train: pd.DataFrame, scored: pd.DataFrame, series: str, units: str, lag: int = 1
) -> pd.Series:
    """`rmsse` for every series at once. `train` must be ordered by series
    then date. `scored` holds actuals in `units` and a `forecast` column."""
    y = train[units].to_numpy(dtype=np.float64)
    codes, names = pd.factorize(train[series], sort=True)
    codes = codes.astype(np.int32)
    n = len(names)
    # count of sales so far in the series, used to drop history before the first sale
    sold = np.cumsum(y != 0, dtype=np.int32)
    start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    sold -= np.repeat(np.r_[0, sold[start[1:] - 1]], np.diff(np.r_[start, len(y)])).astype(np.int32)
    valid = (codes[lag:] == codes[:-lag]) & (sold[:-lag] > 0)
    del sold
    sq = (y[lag:] - y[:-lag])[valid] ** 2
    kept = codes[lag:][valid]
    del y, valid
    total = np.bincount(kept, weights=sq, minlength=n)
    count = np.bincount(kept, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = pd.Series(np.where(count > 0, total / count, np.nan), index=names)
    scale = scale.where(scale > 0)
    err = (scored[units].astype("float64") - scored["forecast"]) ** 2
    mse = err.groupby(scored[series].astype(str).to_numpy()).mean()
    scale.index = scale.index.astype(str)
    return np.sqrt(mse / scale.reindex(mse.index))
