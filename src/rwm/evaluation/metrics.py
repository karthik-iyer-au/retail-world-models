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
