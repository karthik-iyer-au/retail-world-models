"""TimesFM 3.0, Google's pretrained time-series foundation model (released
August 2026), used zero-shot: it is not trained or tuned on our data.

Unlike TimesFM 2.5 (`timesfm`), which takes the plan only through a separate
linear model fitted in context, TimesFM 3.0 reads past-and-future covariates
itself. With `plan` (and `price_plan`) each series is given its plan inputs
over its history and the horizon, built exactly as for TimesFM 2.5, so the two
versions differ only in the model.

The forecast is the mean of its nine quantiles (0.1 to 0.9), which tracks
sales totals better than the median for items that often sell nothing, as for
Chronos and Timestral; the quantiles are kept in `last_quantiles` for interval
scoring.

Needs `timesfm>=3.0` (the `foundation` dependencies) and downloads the
checkpoint `google/timesfm-3.0-pytorch` from Hugging Face on first use. The
checkpoint's licence is non-commercial, as is this repository's.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES
from rwm.forecaster import register_model
from rwm.prior_work.timesfm_zero_shot import TimesFMZeroShot

REPO = "google/timesfm-3.0-pytorch"
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _load(batch_size: int):
    import timesfm

    return timesfm.TimesFM3Forecaster.from_pretrained(REPO, per_core_batch_size=batch_size)


@register_model("timesfm3")
class TimesFM3ZeroShot(TimesFMZeroShot):
    def __init__(self, context: int = 512, batch_size: int = 256, point: str = "mean",
                 plan: list[str] | None = None, price_plan: bool = False, loader=None):
        if point not in ("median", "mean"):
            raise ValueError("point must be 'median' (its 0.5 quantile) or 'mean' (of its quantiles)")
        super().__init__(context=context, batch_size=batch_size, plan=plan, price_plan=price_plan)
        self.point = point
        self._tfm3_loader = loader or _load
        self.last_quantiles = {}

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        ahead = ((future[DATE].to_numpy() - self._end) / self._step).astype(int)
        horizon = int(ahead.max())
        has = np.isfinite(self._history).any(axis=1)
        keep = [np.isfinite(h) for h in self._history]
        contexts = [h[k] if k.any() else np.zeros(3) for h, k in zip(self._history, keep)]
        covariates = None
        if self._plan_past:
            cov = self._covariates(future, horizon, keep)
            covariates = [np.stack([np.asarray(cov[c][i], dtype=np.float32) for c in cov])
                          for i in range(len(contexts))]
        model = self._tfm3_loader(self.batch_size)
        q = np.zeros((len(contexts), horizon, len(LEVELS)))
        for lo in range(0, len(contexts), self.chunk):
            part = slice(lo, lo + self.chunk)
            outs = model.predict_batch(
                contexts[part], horizon,
                past_future_covariates=covariates[part] if covariates is not None else None,
                return_quantiles=True, make_positive=True,
            )
            for i, o in enumerate(outs, start=lo):
                q[i] = np.asarray(o.quantiles)[:horizon]
        q = np.clip(q, 0.0, None)
        q[~has] = 0.0
        point = q.mean(axis=2) if self.point == "mean" else q[:, :, LEVELS.index(0.5)]
        rows = {s: i for i, s in enumerate(self._series)}
        idx = future[SERIES].map(rows)
        known = idx.notna().to_numpy()
        pick = lambda m: np.where(known, m[idx.fillna(0).astype(int).to_numpy(), np.clip(ahead, 1, horizon) - 1], 0.0)
        self.last_quantiles = {lv: pick(q[:, :, k]) for k, lv in enumerate(LEVELS)}
        return pick(point)
