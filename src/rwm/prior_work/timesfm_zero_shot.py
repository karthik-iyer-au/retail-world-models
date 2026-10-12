"""TimesFM 2.5, a pretrained time-series foundation model (Das et al., 2024),
used zero-shot: it is not trained or tuned on our data.

By default it sees each series' unit history only: no prices, promotions or
calendar, so it answers "what will sell if nothing changes". With `plan` (and
`price_plan`) it also gets the plan, through the package's own covariate
method (`forecast_with_covariates`, "xreg + timesfm"): a linear model on the
plan inputs, fitted in context over the series forecast together, and
TimesFM on what that model leaves. History plus plan in one forecaster.

Needs the optional `foundation` dependencies (`pip install -e ".[foundation]"`)
and downloads the checkpoint from Hugging Face on first use.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model

REPO = "google/timesfm-2.5-200m-pytorch"


def _load(context: int, horizon: int, batch_size: int, backcast: bool = False):
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO)
    model.compile(timesfm.ForecastConfig(
        max_context=context,
        max_horizon=horizon,
        normalize_inputs=True,
        per_core_batch_size=batch_size,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=backcast,  # needed by forecast_with_covariates
    ))
    return model


@register_model("timesfm")
class TimesFMZeroShot(Forecaster):
    """Forecast of TimesFM 2.5 from the last `context` periods of units.

    `point` picks which output is the forecast: "median" (the model's default
    point forecast, its 0.5 quantile) or "mean" (its separate mean output).
    For items that often sell nothing in a day the median sits well below the
    mean, so the median under-forecasts sales totals.

    Dates before a series' first record are left out of its history, so a new
    item is not read as a long run of zeros. Dates after its first record with
    no row count as zero sales, as in the other models. A series with no
    record in the context window is forecast at zero.
    """

    def __init__(self, context: int = 512, batch_size: int = 256, point: str = "median",
                 plan: list[str] | None = None, price_plan: bool = False, ridge: float = 1.0,
                 chunk: int = 2048, loader=None):
        # plan: columns given for every period (history and future), e.g. promo, event, snap.
        # price_plan: also give each period's price against the series' highest price in the context window.
        if point not in ("median", "mean"):
            raise ValueError("point must be 'median' or 'mean'")
        self.point = point
        self.plan = list(plan or [])
        self.price_plan = price_plan
        self.ridge = ridge
        # series per call of the covariate method: its linear model is shared within a call,
        # and one call over every series runs out of memory on large categories
        self.chunk = chunk
        self.context = context
        self.batch_size = batch_size
        self._loader = loader or _load

    def fit(self, train: pd.DataFrame) -> "TimesFMZeroShot":
        dates = np.sort(train[DATE].unique())
        self._end = dates[-1]
        self._step = np.diff(dates).min() if len(dates) > 1 else np.timedelta64(1, "D")
        # Every period in the window, even one with no row in the data at all.
        first = max(dates[0], self._end - (self.context - 1) * self._step)
        self._dates = np.arange(first, self._end + self._step, self._step)
        recent = train[train[DATE] >= self._dates[0]]
        grid = recent.pivot_table(index=SERIES, columns=DATE, values=UNITS, aggfunc="sum", observed=True)
        grid = grid.reindex(columns=self._dates)
        values = grid.to_numpy(dtype="float64")
        seen = np.isfinite(values)
        started = np.cumsum(seen, axis=1) > 0
        values = np.where(started & ~seen, 0.0, values)  # after the first record, a missing day sold nothing
        self._series = grid.index.to_numpy()
        self._history = values
        self._plan_past = {}
        if self.plan or self.price_plan:
            for c in self.plan:
                g = recent.pivot_table(index=SERIES, columns=DATE, values=c, aggfunc="max", observed=True)
                self._plan_past[c] = g.reindex(index=grid.index, columns=self._dates).to_numpy(dtype="float64")
            if self.price_plan:
                g = recent.pivot_table(index=SERIES, columns=DATE, values=PRICE, aggfunc="mean", observed=True)
                price = g.reindex(index=grid.index, columns=self._dates).to_numpy(dtype="float64")
                with np.errstate(all="ignore"):
                    self._top = np.nanmax(price, axis=1)
                self._plan_past["price"] = price
        return self

    def _covariates(self, future: pd.DataFrame, horizon: int, keep: list[np.ndarray]) -> dict:
        """Each plan input over each series' history (as kept) and the horizon."""
        fut_dates = self._end + np.arange(1, horizon + 1) * self._step
        out = {}
        for c, past in self._plan_past.items():
            col = PRICE if c == "price" else c
            g = future.pivot_table(index=SERIES, columns=DATE, values=col, aggfunc="max", observed=True)
            fut = g.reindex(index=self._series, columns=fut_dates).to_numpy(dtype="float64")
            both = np.concatenate([past, fut], axis=1)
            if c == "price":  # price against the highest in the context window; unknown counts as regular
                with np.errstate(all="ignore"):
                    both = both / self._top[:, None]
                both = np.where(np.isfinite(both) & (both > 0), both, 1.0)
            else:
                both = np.nan_to_num(both)
            series = []
            for i, k in enumerate(keep):
                hist = both[i, : len(self._dates)][k] if k.any() else np.zeros(3)
                series.append(np.concatenate([hist, both[i, len(self._dates):]]).tolist())
            out[f"plan_{c}"] = series
        return out

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        ahead = ((future[DATE].to_numpy() - self._end) / self._step).astype(int)
        horizon = int(ahead.max())
        rows = {s: i for i, s in enumerate(self._series)}
        inputs, keep, has = [], [], np.zeros(len(self._series), dtype=bool)
        for i, h in enumerate(self._history):
            k = np.isfinite(h)
            keep.append(k)
            has[i] = k.any()
            inputs.append(h[k] if has[i] else np.zeros(3))
        if self._plan_past:
            model = self._loader(self.context, max(horizon, 1), self.batch_size, True)
            cov = self._covariates(future, horizon, keep)
            median, full = [], []
            for lo in range(0, len(inputs), self.chunk):
                part = slice(lo, lo + self.chunk)
                m, f = model.forecast_with_covariates(
                    inputs=inputs[part], dynamic_numerical_covariates={k: v[part] for k, v in cov.items()},
                    xreg_mode="xreg + timesfm", ridge=self.ridge,
                )
                median += list(m)
                full += list(f)
            median, full = np.stack(median), np.stack(full)
        else:
            model = self._loader(self.context, max(horizon, 1), self.batch_size)
            median, full = model.forecast(horizon=horizon, inputs=inputs)
        # `full` holds the mean first, then the 0.1 to 0.9 quantiles
        point = median if self.point == "median" else np.asarray(full)[..., 0]
        point = np.clip(np.asarray(point, dtype="float64")[:, :horizon], 0.0, None)
        point[~has] = 0.0
        idx = future[SERIES].map(rows)
        out = np.zeros(len(future))
        known = idx.notna().to_numpy()
        out[known] = point[idx[known].astype(int).to_numpy(), ahead[known] - 1]
        return out
