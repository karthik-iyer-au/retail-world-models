"""Forecast reconciliation for the paper 1 revision (docs/PAPER1_REVISION_PLAN.md).

Base forecasts for every node of the hierarchy come from AutoETS
(statsforecast) fitted on each node's history before the window. They are
reconciled by bottom-up, top-down with historical proportions, and the MinT
family (Wickramasuriya, Athanasopoulos and Hyndman 2019): OLS, WLS with
structural scaling, WLS with in-sample residual variances, and MinT-shrink
when the hierarchy is small enough to hold a full covariance.

The generalised least squares step, bottom = (S' W^-1 S)^-1 S' W^-1 y, is
solved by conjugate gradients on a sparse S, so it scales to M5's 42,840
nodes for any diagonal W.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from rwm.evaluation.compare import Frame
from rwm.evaluation.hierarchy import _sum_by_group

MAX_DENSE = 6000
# worker processes for the ETS fits; each holds a copy of its share of the
# histories, so many workers on a large hierarchy can run out of memory
ETS_JOBS = int(os.environ.get("RWM_ETS_JOBS", "4"))


def summing_matrix(fr: Frame) -> tuple[sp.csr_matrix, list[np.ndarray]]:
    """S (nodes x bottom series) with the levels in order, and each level's codes."""
    blocks, codes = [], []
    n = len(fr.attrs)
    for cols in fr.levels:
        c = fr.codes(cols)
        g = int(c.max()) + 1
        blocks.append(sp.csr_matrix((np.ones(n), (c, np.arange(n))), shape=(g, n)))
        codes.append(c)
    return sp.vstack(blocks).tocsr(), codes


def ets_base(history: np.ndarray, horizon: int, season: int, keep: int = 104, seasonal: np.ndarray | None = None):
    """AutoETS forecasts, in-sample one-step residual variances, and the last
    `keep` in-sample residuals (node x period, NaN before a node's start) for
    each row of `history` (node x period), from its first non-zero value.
    `seasonal` marks the rows that may take a seasonal model (all by default)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive

    rows, lengths = [], np.zeros(len(history), dtype=int)
    for i, h in enumerate(history):
        nz = np.flatnonzero(h)
        start = nz[0] if len(nz) else len(h) - 2
        y = h[max(0, min(start, len(h) - 3)):]  # at least three periods for every node
        lengths[i] = len(y)
        rows.append(pd.DataFrame({"unique_id": i, "ds": np.arange(len(y)), "y": y.astype(float)}))
    df = pd.concat(rows, ignore_index=True)
    # a seasonal ETS needs at least two seasons and a little more; shorter nodes get a non-seasonal one
    long = lengths >= 2 * season + 4 if season > 1 else np.zeros(len(lengths), dtype=bool)
    if seasonal is not None:
        long &= seasonal
    parts_fc, parts_fit = [], []
    for mask, m in ((long, season), (~long, 1)):
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        # a node where ETS cannot be fitted (for example a constant history) falls back to the last value
        sf = StatsForecast(models=[AutoETS(season_length=m)], freq=1, n_jobs=ETS_JOBS, fallback_model=Naive())
        part = df[df["unique_id"].isin(ids)]
        parts_fc.append(sf.forecast(df=part, h=horizon, fitted=True))
        parts_fit.append(sf.forecast_fitted_values())
    fc, fit = pd.concat(parts_fc), pd.concat(parts_fit)
    # each node's own time index differs (nodes start at different periods): number the forecast steps 0..h-1
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    fc = fc.assign(step=fc.groupby("unique_id").cumcount())
    f = fc.pivot(index="unique_id", columns="step", values="AutoETS").reindex(range(len(history))).to_numpy()
    resid = (fit["y"] - fit["AutoETS"]).groupby(fit["unique_id"]).var().reindex(range(len(history))).to_numpy()
    f = np.nan_to_num(np.clip(f, 0, None))
    resid = np.where(np.isfinite(resid) & (resid > 0), resid, np.nanmedian(resid[resid > 0]) if (resid > 0).any() else 1.0)
    fit = fit.assign(e=fit["y"] - fit["AutoETS"])
    fit["back"] = fit.groupby("unique_id")["ds"].transform("max") - fit["ds"]
    fit = fit[fit["back"] < keep]
    R = np.full((len(history), keep), np.nan)
    R[fit["unique_id"].to_numpy(int), keep - 1 - fit["back"].to_numpy(int)] = fit["e"].to_numpy()
    return f, resid, R


def gls_bottom(S: sp.csr_matrix, y: np.ndarray, wdiag: np.ndarray) -> np.ndarray:
    """Bottom-level forecasts from all-node forecasts y (node x horizon) with a
    diagonal W, by conjugate gradients."""
    Winv = sp.diags(1.0 / wdiag)
    A = (S.T @ Winv @ S).tocsr()
    rhs = S.T @ (Winv @ y)
    out = np.zeros((S.shape[1], y.shape[1]))
    for h in range(y.shape[1]):
        x, _ = cg(A, rhs[:, h], rtol=1e-8, maxiter=2000)
        out[:, h] = x
    return out


def mint_shrink_bottom(S: sp.csr_matrix, y: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """MinT with the shrinkage covariance of Schafer and Strimmer, for small
    hierarchies. `residuals` is node x period of in-sample errors."""
    r = residuals - np.nanmean(residuals, axis=1, keepdims=True)
    r = np.nan_to_num(r)
    T = r.shape[1]
    cov = r @ r.T / T
    sd = np.sqrt(np.diag(cov))
    sd[sd == 0] = 1e-9
    corr = cov / np.outer(sd, sd)
    xs = r / sd[:, None]
    v = ((xs[:, None, :] * xs[None, :, :]) ** 2).mean(axis=2) * T / (T - 1) ** 2 if len(r) < 1500 else None
    if v is not None:
        off = ~np.eye(len(r), dtype=bool)
        lam = float(np.clip(v[off].sum() / (corr[off] ** 2).sum(), 0, 1))
    else:
        lam = 0.5
    shrunk = lam * np.diag(np.diag(cov)) + (1 - lam) * cov
    Sd = S.toarray()
    Winv = np.linalg.pinv(shrunk)
    G = np.linalg.solve(Sd.T @ Winv @ Sd, Sd.T @ Winv)
    return G @ y


def reconcile_all(fr: Frame, origin: int, season: int, bottoms: dict[str, np.ndarray] | None = None) -> dict:
    """Bottom-level forecasts from each reconciliation method, plus each given
    model's bottom level combined with ETS above it (WLS with residual variances)."""
    t0, t1, _ = fr.cut(origin)
    H = t1 - t0
    S, codes = summing_matrix(fr)
    hist = np.vstack([_sum_by_group(fr.units[:, :t0], c, int(c.max()) + 1) for c in codes])
    nb = S.shape[1]
    # seasonal ETS at the aggregate nodes only; the item-by-store nodes take a
    # non-seasonal one (docs/PAPER1_REVISION_PLAN.md, amendment of 14 October 2026)
    base, var, R = ets_base(hist, H, season, seasonal=np.arange(S.shape[0]) < S.shape[0] - nb)
    last = codes[-1]  # the bottom level: one node per series, nodes sorted by the level's columns
    if int(last.max()) + 1 != nb or len(np.unique(last)) != nb:
        raise ValueError("the last level of the hierarchy must be one node per series")
    first = S.shape[0] - nb
    to_series = lambda nodes: nodes[last]  # bottom-node rows in series order
    out = {"ets_bottom_up": to_series(base[first:])}
    # bottom level is the last level (item by store); the first is the total
    top = base[0]
    share = fr.units[:, t0 - min(t0, 52 if season == 52 else 365):t0].sum(axis=1)
    share = share / max(share.sum(), 1e-9)
    out["ets_top_down"] = share[:, None] * top[None, :]
    out["ets_ols"] = gls_bottom(S, base, np.ones(S.shape[0]))
    out["ets_wls_struct"] = gls_bottom(S, base, np.asarray(S.sum(axis=1)).ravel())
    out["ets_wls_var"] = gls_bottom(S, base, var)
    if S.shape[0] <= MAX_DENSE:
        out["ets_mint_shrink"] = mint_shrink_bottom(S, base, R)
    for name, b in (bottoms or {}).items():
        y = base.copy()
        y[first + last] = b  # series i is bottom node last[i]
        out[f"{name}_with_ets_wls_var"] = gls_bottom(S, y, var)
    return {k: np.clip(v, 0, None) for k, v in out.items()}
