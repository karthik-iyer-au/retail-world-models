"""Helpers for panels that are ordered by series then date."""
import numpy as np
import pandas as pd


def last_k_rows(series: pd.Series, k: int) -> np.ndarray:
    """Row positions of the last `k` rows of each series, in panel order.
    Works on the block boundaries, so it does not copy or sort the panel."""
    codes = pd.factorize(series, sort=True)[0].astype(np.int32)
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(codes)]
    pos = ends[:, None] - np.arange(k, 0, -1)[None, :]
    return pos[pos >= starts[:, None]]
