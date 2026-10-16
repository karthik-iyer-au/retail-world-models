"""Paper 1 revision analyses on saved forecasts (docs/PAPER1_REVISION_PLAN.md):
paired comparisons with bootstrap intervals, interval forecasts by rolling
conformal calibration, decision costs, equal-weight ensembles, and
reconciliation. Nothing here fits a model except the ETS base forecasts used
for reconciliation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from rwm.data import load_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, UNITS
from rwm.evaluation.hierarchy import _sum_by_group, series_scale, to_matrix, wrmsse
from rwm.evaluation.splits import rolling_origins

QUANTILES = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


@dataclass
class Frame:
    """The data of one run config as its run scored it."""
    attrs: pd.DataFrame
    names: pd.Index
    units: np.ndarray
    price: np.ndarray
    dates: np.ndarray
    levels: list
    splits: list
    window: int
    lag: int

    def cut(self, origin: int):
        sp = next(s for s in self.splits if s.origin == origin)
        t0 = int(np.searchsorted(self.dates, np.datetime64(sp.train_end))) + 1
        t1 = t0 + len(sp.test_dates)
        rev = (self.units[:, t0 - self.window:t0] * self.price[:, t0 - self.window:t0]).sum(axis=1, dtype=np.float64)
        return t0, t1, rev

    def codes(self, cols):
        if not cols:
            return np.zeros(len(self.attrs), dtype=np.int64)
        return self.attrs.groupby(cols, observed=True, sort=True).ngroup().to_numpy()


def frame(config: dict) -> Frame:
    ds = load_dataset(config["dataset"]["name"], **config["dataset"].get("params", {}))
    panel, ev = ds.panel, config["evaluation"]
    dates = np.sort(panel[DATE].unique())
    if ev.get("holdout_periods"):
        dates = dates[: -ev["holdout_periods"]]
        panel = panel[panel[DATE] <= dates[-1]].reset_index(drop=True)
    levels = ds.hierarchy or [[SERIES]]
    cols = sorted({c for lv in levels for c in lv} | {SERIES} | ({ITEM} if ITEM in panel else set()))
    attrs = panel[cols].drop_duplicates(SERIES).sort_values(SERIES).reset_index(drop=True)
    names = pd.Index(attrs[SERIES].astype(str))
    units = to_matrix(panel, SERIES, DATE, UNITS, dates, names)
    price = to_matrix(panel, SERIES, DATE, PRICE, dates, names) if PRICE in panel else np.ones_like(units)
    splits = rolling_origins(dates, ev["horizon"], ev["n_origins"], ev.get("step"))
    return Frame(attrs, names, units, price, dates, levels, splits, ev.get("weight_window", 28), ev.get("scale_lag", 1))


def forecast_matrix(forecasts: pd.DataFrame, fr: Frame, origin: int, col: str = "forecast") -> np.ndarray:
    sp = next(s for s in fr.splits if s.origin == origin)
    f = forecasts[forecasts["origin"] == origin]
    return to_matrix(f.assign(**{DATE: pd.to_datetime(f[DATE])}), SERIES, DATE, col, list(sp.test_dates), fr.names)


def level_scores(fr: Frame, origin: int, fc: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """WRMSSE by level; `weights` (one per bottom series, for example bootstrap
    counts) multiplies every bottom series before aggregation."""
    t0, t1, rev = fr.cut(origin)
    w = np.ones(len(fr.attrs)) if weights is None else weights
    sc = wrmsse(fr.units[:, :t0] * w[:, None], fr.units[:, t0:t1] * w[:, None], fc * w[:, None], rev * w,
                fr.attrs, fr.levels, fr.lag)
    return {lv["level"]: lv["wrmsse"] for lv in sc["by_level"]}


class _Mixer:
    """Products of resample counts (k x items) with item-by-node sums
    (items x nodes x periods), on the GPU when there is one."""

    def __init__(self, X: np.ndarray):
        self.shape = X.shape[1:]
        flat = X.reshape(X.shape[0], -1)
        self.torch = None
        try:
            import torch
            if torch.cuda.is_available():
                self.torch, self.X = torch, torch.as_tensor(flat, device="cuda", dtype=torch.float64)
        except ImportError:
            pass
        if self.torch is None:
            self.X = flat

    def __call__(self, c: np.ndarray) -> np.ndarray:
        if self.torch is not None:
            out = (self.torch.as_tensor(c, device="cuda", dtype=self.torch.float64) @ self.X).cpu().numpy()
        else:
            out = c @ self.X
        return out.reshape(len(c), *self.shape)


class Bootstrap:
    """Item bootstrap of WRMSSE at every level: items are drawn with replacement,
    every series of a drawn item counts as often as the item was drawn, and the
    aggregates and revenue weights are recomputed from the drawn set. The
    aggregated histories (the costly part) are shared by all models."""

    def __init__(self, fr: Frame, origin: int, n: int = 1000, seed: int = 0):
        self.fr, self.origin, self.n = fr, origin, n
        item = fr.attrs[ITEM].astype(str).to_numpy() if ITEM in fr.attrs else fr.names.to_numpy()
        self.items, self.item_of = np.unique(item, return_inverse=True)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(len(self.items), np.full(len(self.items), 1 / len(self.items)), size=n)
        t0, t1, rev = fr.cut(origin)
        self.t0, self.t1, self.rev = t0, t1, rev
        self.level_codes = [fr.codes(c) for c in fr.levels]
        self.level_names = ["x".join(c) if c else "total" for c in fr.levels]

    def scores(self, forecasts: dict[str, np.ndarray], chunk: int = 50) -> dict[str, np.ndarray]:
        """For each model: an array resamples x levels of WRMSSE.

        Exact and fast. At a level whose every node holds one item (item,
        item x store), a resample multiplies a node's history, actuals and
        forecasts by its item's count, which leaves the node's RMSSE unchanged:
        only the revenue weights change. At a coarser level, the node sums of
        every resample are one product of the resample counts with the
        item-by-node sums (on the GPU when there is one)."""
        fr = self.fr
        hist, act = fr.units[:, :self.t0].astype(np.float64), fr.units[:, self.t0:self.t1].astype(np.float64)
        n_items = len(self.items)
        out = {m: np.zeros((self.n, len(self.level_codes))) for m in forecasts}
        counts = self.counts.astype(np.float64)
        for j, codes in enumerate(self.level_codes):
            g = int(codes.max()) + 1
            pair = np.unique(codes * n_items + self.item_of)
            if len(pair) == g:  # one item per node
                node_item = np.zeros(g, dtype=int)
                node_item[pair // n_items] = pair % n_items
                h, a = _sum_by_group(hist, codes, g), _sum_by_group(act, codes, g)
                rev = _sum_by_group(self.rev[:, None], codes, g)[:, 0]
                scale = series_scale(h, fr.lag)
                rw = counts[:, node_item] * rev[None, :]  # resamples x nodes
                total = rw.sum(axis=1)
                for m, fc in forecasts.items():
                    f = _sum_by_group(fc.astype(np.float64), codes, g)
                    sc = np.sqrt(((a - f) ** 2).mean(axis=1) / scale)
                    ok = np.isfinite(sc)
                    out[m][:, j] = (rw[:, ok] * sc[ok]).sum(axis=1) / total
                continue
            # item-by-node sums: (items, nodes, periods)
            key = self.item_of * g + codes
            M = sp.csr_matrix((np.ones(len(key)), (key, np.arange(len(key)))), shape=(n_items * g, len(key)))

            def by_item(x):
                return np.asarray(M @ x).reshape(n_items, g, x.shape[1])
            H, A = _Mixer(by_item(hist)), _Mixer(by_item(act))
            R = by_item(self.rev[:, None].astype(np.float64))[:, :, 0]
            F = {m: _Mixer(by_item(fc.astype(np.float64))) for m, fc in forecasts.items()}
            for b0 in range(0, self.n, chunk):
                c = counts[b0:b0 + chunk]
                h, a = H(c), A(c)  # chunk x nodes x periods
                rw = c @ R
                rw = rw / rw.sum(axis=1, keepdims=True)
                k = len(c)
                scale = series_scale(h.reshape(k * g, -1), fr.lag).reshape(k, g)
                for m in forecasts:
                    f = F[m](c)
                    sc = np.sqrt(((a - f) ** 2).mean(axis=2) / scale)
                    ok = np.isfinite(sc) & (rw > 0)
                    out[m][b0:b0 + k, j] = np.where(ok, sc * rw, 0.0).sum(axis=1)
        return out

    def scores_reference(self, forecasts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The direct computation, resample by resample (slow; kept to check `scores`)."""
        fr = self.fr
        hist, act = fr.units[:, :self.t0].astype(np.float32), fr.units[:, self.t0:self.t1].astype(np.float32)
        out = {m: np.zeros((self.n, len(self.level_codes))) for m in forecasts}
        for b in range(self.n):
            w = self.counts[b][self.item_of].astype(np.float32)
            for j, codes in enumerate(self.level_codes):
                g = int(codes.max()) + 1
                h = _sum_by_group(hist * w[:, None], codes, g)
                a = _sum_by_group(act * w[:, None], codes, g)
                rw = _sum_by_group((self.rev * w)[:, None], codes, g)[:, 0]
                rw = rw / rw.sum()
                scale = series_scale(h, fr.lag)
                for m, fc in forecasts.items():
                    f = _sum_by_group(fc * w[:, None], codes, g)
                    s = np.sqrt(((a - f) ** 2).mean(axis=1) / scale)
                    ok = np.isfinite(s) & (rw > 0)
                    out[m][b, j] = (s[ok] * rw[ok]).sum()
        return out


def paired(boot: dict[str, np.ndarray], point: dict[str, dict], level_names: list[str], pairs) -> pd.DataFrame:
    """Difference A minus B in WRMSSE by level, with its 95% bootstrap interval."""
    rows = []
    for a, b in pairs:
        if a not in boot or b not in boot:
            continue
        d = boot[a] - boot[b]
        for j, lv in enumerate(level_names):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            rows.append({"a": a, "b": b, "level": lv, "diff": point[a][lv] - point[b][lv], "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, n: int = 5000, seed: int = 0) -> list[str]:
    """Hansen, Lunde and Nason (2011) with the range statistic, bootstrap over
    windows (rows of `losses`; one column per model). Returns the models kept."""
    rng = np.random.default_rng(seed)
    keep = list(losses.columns)
    L = losses.to_numpy(dtype=float)
    idx = {m: i for i, m in enumerate(losses.columns)}
    T = len(L)
    draws = rng.integers(0, T, size=(n, T))
    while len(keep) > 1:
        sub = L[:, [idx[m] for m in keep]]
        d = sub[:, :, None] - sub[:, None, :]  # T x k x k
        dbar = d.mean(axis=0)
        boot = d[draws].mean(axis=1)  # n x k x k
        se = boot.std(axis=0) + 1e-12
        t = np.abs(dbar) / se
        stat = t.max()
        tb = (np.abs(boot - dbar) / se).max(axis=(1, 2))
        p = (tb >= stat).mean()
        if p >= alpha:
            break
        worst = np.argmax((dbar / se).sum(axis=1))  # the model with the largest losses against the rest
        keep.pop(worst)
    return keep


# ---------------------------------------------------------------- intervals

def scaled_errors(fr: Frame, origin: int, fc: np.ndarray, cols) -> np.ndarray:
    """Forecast errors of the nodes of one level, each divided by its RMSSE scale
    (square root of the mean squared one-step change), node x horizon."""
    t0, t1, _ = fr.cut(origin)
    codes = fr.codes(cols)
    g = int(codes.max()) + 1
    h = _sum_by_group(fr.units[:, :t0], codes, g)
    a = _sum_by_group(fr.units[:, t0:t1], codes, g)
    f = _sum_by_group(fc, codes, g)
    return (a - f) / np.sqrt(series_scale(h, fr.lag))[:, None]


def conformal_quantiles(errors: list[np.ndarray], qs=QUANTILES, min_per_horizon: int = 0) -> np.ndarray:
    """Quantiles of pooled scaled errors from earlier windows, one row per horizon.

    With `min_per_horizon`, a level with fewer earlier errors than that for each
    horizon (the total has one per window) pools all horizons into one set of
    quantiles: the 2.5% quantile of two or three numbers is no interval at all."""
    e = np.concatenate(errors, axis=0)
    if np.isfinite(e).sum(axis=0).min() < min_per_horizon:
        return np.tile(np.nanquantile(e, qs), (e.shape[1], 1))
    return np.vstack([np.nanquantile(e[:, h], qs) for h in range(e.shape[1])])  # horizon x quantile


def interval_scores(fr: Frame, origin: int, fc: np.ndarray, qerr: np.ndarray, cols, qs=QUANTILES,
                    whole_units: bool = False) -> dict:
    """Scaled pinball loss averaged over quantiles (as in the M5 uncertainty
    competition), weighted by revenue, and coverage of the 50, 80 and 95%
    intervals, for one level."""
    t0, t1, rev = fr.cut(origin)
    codes = fr.codes(cols)
    g = int(codes.max()) + 1
    h = _sum_by_group(fr.units[:, :t0], codes, g)
    a = _sum_by_group(fr.units[:, t0:t1], codes, g)
    f = _sum_by_group(fc, codes, g)
    w = _sum_by_group(rev[:, None], codes, g)[:, 0]
    w = w / w.sum()
    s = np.sqrt(series_scale(h, fr.lag))
    started = np.cumsum(h != 0, axis=1) > 0
    abs_scale = (np.abs(np.diff(h, axis=1)) * started[:, 1:]).sum(axis=1) / np.maximum(started[:, 1:].sum(axis=1), 1)
    q = np.clip(f[:, :, None] + qerr[None, :, :] * s[:, None, None], 0.0, None)  # node x horizon x quantile
    qa = np.asarray(qs)[None, None, :]
    pin = np.where(a[:, :, None] >= q, qa * (a[:, :, None] - q), (1 - qa) * (q - a[:, :, None]))
    spl = pin.mean(axis=(1, 2)) / abs_scale
    ok = np.isfinite(spl) & (w > 0) & (abs_scale > 0)
    out = {"spl": float((spl[ok] * w[ok]).sum() / w[ok].sum())}
    for lev, (lo, hi) in {50: (0.25, 0.75), 80: (0.1, 0.9), 95: (0.025, 0.975)}.items():
        i, j = qs.index(lo), qs.index(hi)
        lo_b, hi_b = q[:, :, i], q[:, :, j]
        if whole_units:  # sales come in whole units: an interval from 0.3 to 0.8 holds no possible value
            lo_b, hi_b = np.floor(lo_b), np.ceil(hi_b)
        inside = (a >= lo_b) & (a <= hi_b)
        out[f"coverage_{lev}"] = float(inside[ok].mean())
    return out


def native_scores(fr: Frame, origin: int, forecasts: pd.DataFrame) -> dict:
    """Scaled pinball loss of a model's own quantiles at the bottom level, on the
    quantiles it gives."""
    t0, t1, rev = fr.cut(origin)
    qcols = sorted([c for c in forecasts.columns if c.startswith("q")], key=lambda c: float(c[1:]))
    if not qcols:
        return {}
    h, a = fr.units[:, :t0], fr.units[:, t0:t1]
    started = np.cumsum(h != 0, axis=1) > 0
    abs_scale = (np.abs(np.diff(h, axis=1)) * started[:, 1:]).sum(axis=1) / np.maximum(started[:, 1:].sum(axis=1), 1)
    w = rev / rev.sum()
    losses = []
    for c in qcols:
        qv = float(c[1:])
        q = forecast_matrix(forecasts, fr, origin, c)
        losses.append(np.where(a >= q, qv * (a - q), (1 - qv) * (q - a)).mean(axis=1))
    spl = np.mean(losses, axis=0) / abs_scale
    ok = np.isfinite(spl) & (w > 0) & (abs_scale > 0)
    return {"native_spl": float((spl[ok] * w[ok]).sum() / w[ok].sum()), "native_quantiles": ",".join(c[1:] for c in qcols)}


# ---------------------------------------------------------------- decisions

def replenishment_cost(fr: Frame, origin: int, fc: np.ndarray, qerr_bottom: np.ndarray, ratio: float) -> float:
    """Order the forecast's quantile at the critical ratio for each item-store-period;
    cost per unit over-ordered 1, per unit short ratio / (1 - ratio); total over
    the window, in units."""
    t0, t1, _ = fr.cut(origin)
    a = fr.units[:, t0:t1]
    s = np.sqrt(series_scale(fr.units[:, :t0], fr.lag))
    s = np.where(np.isfinite(s), s, 0.0)
    z = np.array([np.interp(ratio, QUANTILES, row) for row in qerr_bottom])  # per horizon
    order = np.clip(fc + z[None, :] * s[:, None], 0.0, None)
    under = ratio / (1 - ratio)
    return float((np.clip(order - a, 0, None) + under * np.clip(a - order, 0, None)).sum())


def category_planning_loss(fr: Frame, origin: int, fc: np.ndarray, cols) -> float:
    """Absolute error of category-by-store totals, weighted by each node's dollar
    sales in the window before, as a share of those dollar sales."""
    t0, t1, rev = fr.cut(origin)
    codes = fr.codes(cols)
    g = int(codes.max()) + 1
    a = _sum_by_group(fr.units[:, t0:t1], codes, g)
    f = _sum_by_group(fc, codes, g)
    w = _sum_by_group(rev[:, None], codes, g)[:, 0]
    err = np.abs(a - f).sum(axis=1) / np.maximum(a.sum(axis=1), 1e-9)
    return float((err * w).sum() / w.sum())


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    m = json.loads((Path(run_dir) / "manifest.json").read_text())
    return m["config"], pd.read_csv(Path(run_dir) / "forecasts.csv")
