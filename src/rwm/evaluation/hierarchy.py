"""WRMSSE: the M5 accuracy measure.

Sales are added up to each level of the hierarchy (for M5: total, state,
store, category, department, their combinations, item, item by state, item
by store). Every series at every level gets an RMSSE. Each is weighted by
its share of revenue in the last `weight_window` periods of training, and
every level counts equally.

Follows the organisers' reference code ("Estimate WRMSSE.R"): the history of
each series, including aggregated ones, starts at its first non-zero value.
"""
import numpy as np
import pandas as pd


def _sum_by_group(x: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_groups))
    return np.add.reduceat(x[order].astype(np.float64), starts, axis=0)


def series_scale(history: np.ndarray, lag: int = 1) -> np.ndarray:
    """Mean squared one-step change of each row, from its first non-zero value."""
    started = np.cumsum(history != 0, axis=1) > 0
    sq = (history[:, lag:] - history[:, :-lag]) ** 2
    valid = started[:, :-lag]
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = (sq * valid).sum(axis=1) / count
    scale[(count == 0) | (scale == 0)] = np.nan
    return scale


def wrmsse(
    history: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    revenue: np.ndarray,
    attrs: pd.DataFrame,
    levels: list[list[str]],
    lag: int = 1,
) -> dict:
    """All arrays have one row per bottom-level series, in the row order of
    `attrs`. `levels` lists the columns of `attrs` that define each level; an
    empty list means the grand total."""
    by_level = []
    for cols in levels:
        if cols:
            codes = attrs.groupby(cols, observed=True, sort=True).ngroup().to_numpy()
        else:
            codes = np.zeros(len(attrs), dtype=np.int64)
        g = int(codes.max()) + 1
        h, a, f = (_sum_by_group(x, codes, g) for x in (history, actual, forecast))
        w = _sum_by_group(revenue[:, None], codes, g)[:, 0]
        w = w / w.sum()
        score = np.sqrt(((a - f) ** 2).mean(axis=1) / series_scale(h, lag))
        ok = np.isfinite(score) & (w > 0)
        by_level.append(
            {
                "level": "x".join(cols) if cols else "total",
                "series": g,
                "wrmsse": float((score[ok] * w[ok]).sum()),
                "mean_rmsse": float(np.nanmean(score)),
            }
        )
    return {
        "wrmsse": float(np.mean([lv["wrmsse"] for lv in by_level])),
        "by_level": by_level,
    }


def rmsse_where(
    history: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    where: np.ndarray,
    revenue: np.ndarray,
    lag: int = 1,
) -> dict:
    """Item-by-store accuracy over a chosen set of periods, for example the
    weeks an item was on display. Each series is scored on its chosen periods
    only, scaled as in RMSSE, and weighted by its share of revenue among the
    series that have any chosen period."""
    n = where.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = (((actual - forecast) ** 2) * where).sum(axis=1) / n
        score = np.sqrt(mse / series_scale(history.astype(np.float64), lag))
    ok = np.isfinite(score) & (n > 0) & (revenue > 0)
    if not ok.any():
        return {"periods": int(where.sum()), "series": 0, "wrmsse": float("nan")}
    w = revenue[ok] / revenue[ok].sum()
    return {"periods": int(where.sum()), "series": int(ok.sum()), "wrmsse": float((score[ok] * w).sum())}


def to_matrix(
    panel: pd.DataFrame, series: str, date: str, value: str, dates, names: pd.Index | None = None
) -> np.ndarray:
    """Series-by-date array of `value`. Missing rows and missing values are 0.
    Rows follow `names` (default: the series present, sorted); columns follow `dates`."""
    s = panel[series]
    if names is None:
        names = pd.Index(s.astype(str).unique()).sort_values()
    if isinstance(s.dtype, pd.CategoricalDtype) and s.cat.categories.astype(str).equals(names):
        rows = s.cat.codes.to_numpy()
    else:
        rows = names.get_indexer(s.astype(str))
    cols = pd.Index(dates).get_indexer(panel[date])
    out = np.zeros((len(names), len(dates)), dtype=np.float32)
    keep = (cols >= 0) & (rows >= 0)
    out[rows[keep], cols[keep]] = np.nan_to_num(panel[value].to_numpy(dtype=np.float32)[keep])
    return out
