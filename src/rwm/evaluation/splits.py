"""Rolling-origin splits.

The data is cut at several points in time. For each cut the model is trained
on everything before it and tested on the `horizon` periods after it. Test
periods never appear in training.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    origin: int  # 0 is the earliest cut
    train_end: pd.Timestamp  # last period in training
    test_dates: tuple  # the periods forecast


def rolling_origins(dates, horizon: int, n_origins: int, step: int | None = None) -> list[Split]:
    d = sorted(pd.unique(pd.Series(dates)))
    step = step or horizon
    splits = []
    for k in range(n_origins - 1, -1, -1):
        test_end = len(d) - k * step
        test_start = test_end - horizon
        if test_start <= 0:
            raise ValueError("not enough history for the requested splits")
        splits.append(
            Split(
                origin=n_origins - 1 - k,
                train_end=d[test_start - 1],
                test_dates=tuple(d[test_start:test_end]),
            )
        )
    return splits
