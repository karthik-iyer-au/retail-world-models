"""Turning the long table into series-by-date arrays for models."""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES


def series_rows(col: pd.Series, names: pd.Index) -> np.ndarray:
    """Position of each row's series in `names`, or -1 if it is not there."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        lookup = names.get_indexer(col.cat.categories.astype(str))
        return lookup[col.cat.codes.to_numpy()]
    return names.get_indexer(col.astype(str))


def frame_to_matrix(frame: pd.DataFrame, names: pd.Index, dates: np.ndarray, col: str) -> np.ndarray:
    """Array of `col` with one row per series in `names` and one column per
    date. Empty where the table has no row or no value."""
    out = np.full((len(names), len(dates)), np.nan, dtype=np.float32)
    r = series_rows(frame[SERIES], names)
    c = pd.Index(dates).get_indexer(frame[DATE])
    ok = (r >= 0) & (c >= 0)
    out[r[ok], c[ok]] = frame[col].to_numpy(dtype=np.float32)[ok]
    return out
