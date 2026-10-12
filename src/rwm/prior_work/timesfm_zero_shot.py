"""TimesFM 2.5, a pretrained time-series foundation model (Das et al., 2024),
used zero-shot: it is not trained or tuned on our data.

It sees each series' unit history only. It gets no prices, promotions or
calendar, so it answers "what will sell if nothing changes", not "what will
sell under this plan". It is reported as a comparison on accuracy only.

Needs the optional `foundation` dependencies (`pip install -e ".[foundation]"`)
and downloads the checkpoint from Hugging Face on first use.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model

REPO = "google/timesfm-2.5-200m-pytorch"


def _load(context: int, horizon: int, batch_size: int):
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
    ))
    return model


@register_model("timesfm")
class TimesFMZeroShot(Forecaster):
    """Median forecast of TimesFM 2.5 from the last `context` periods of units.

    Dates before a series' first record are left out of its history, so a new
    item is not read as a long run of zeros. Dates after its first record with
    no row count as zero sales, as in the other models. A series with no
    record in the context window is forecast at zero.
    """

    def __init__(self, context: int = 512, batch_size: int = 256, loader=None):
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
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        ahead = ((future[DATE].to_numpy() - self._end) / self._step).astype(int)
        horizon = int(ahead.max())
        rows = {s: i for i, s in enumerate(self._series)}
        inputs, has = [], np.zeros(len(self._series), dtype=bool)
        for i, h in enumerate(self._history):
            h = h[np.isfinite(h)]
            has[i] = len(h) > 0
            inputs.append(h if has[i] else np.zeros(3))
        model = self._loader(self.context, max(horizon, 1), self.batch_size)
        point, _ = model.forecast(horizon=horizon, inputs=inputs)
        point = np.clip(np.asarray(point, dtype="float64")[:, :horizon], 0.0, None)
        point[~has] = 0.0
        idx = future[SERIES].map(rows)
        out = np.zeros(len(future))
        known = idx.notna().to_numpy()
        out[known] = point[idx[known].astype(int).to_numpy(), ahead[known] - 1]
        return out
