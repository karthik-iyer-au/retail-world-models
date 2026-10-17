"""Timestral 1.0, Mistral's pretrained time-series foundation model (released
15 October 2026), used zero-shot: it is not trained or tuned on our data.

By default it sees each series' unit history only. With `plan` (and
`price_plan`) it also gets the plan inputs as known-future covariates, read by
the model itself (no separate linear model, unlike TimesFM's covariate method).
Plan inputs that are only 0 or 1, such as promo, are treated by the model as
indicators.

The forecast is the mean of its 21 quantiles (0.01 to 0.99, each weighted by
the share of probability nearest it), which tracks
sales totals better than the median for items that often sell nothing, as for
Chronos; the quantiles are kept in `last_quantiles` for interval scoring.
Series history is built as for TimesFM, except that dates before a series'
first record are passed as missing rather than dropped, so every series keeps
the same calendar as its covariates.

Needs `pip install git+https://github.com/mistralai/timestral.git` and
downloads the checkpoint from Hugging Face on first use.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, SERIES
from rwm.forecaster import register_model
from rwm.prior_work.timesfm_zero_shot import TimesFMZeroShot

REPO = "mistralai/Timestral-209M"


def _load(repo: str, device: str):
    import torch
    from timestral import TimestralForecaster

    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    return TimestralForecaster.from_pretrained(repo, device=dev)


def _numpy(x) -> np.ndarray:
    """The model's output as a NumPy array; on a GPU it comes back in GPU memory."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _free_gpu() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


@register_model("timestral")
class TimestralZeroShot(TimesFMZeroShot):
    def __init__(self, context: int = 512, batch_size: int = 256, repo: str = REPO,
                 plan: list[str] | None = None, price_plan: bool = False,
                 device: str = "auto", loader=None):
        super().__init__(context=context, batch_size=batch_size, point="mean", plan=plan, price_plan=price_plan)
        self.repo = repo
        self.device = device
        self._ts_loader = loader or _load
        self.last_quantiles = {}

    def _known(self, future: pd.DataFrame, horizon: int) -> np.ndarray | None:
        """Plan inputs over the context window and the horizon: series x inputs x (context + horizon)."""
        if not self._plan_past:
            return None
        fut_dates = self._end + np.arange(1, horizon + 1) * self._step
        rows = []
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
            rows.append(both)
        return np.stack(rows, axis=1)

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        ahead = ((future[DATE].to_numpy() - self._end) / self._step).astype(int)
        horizon = int(ahead.max())
        model = self._ts_loader(self.repo, self.device)
        has = np.isfinite(self._history).any(axis=1)
        context = np.where(has[:, None], self._history, 0.0)  # a series with no record reads as zeros
        known = self._known(future, horizon)
        qs, levels = [], None
        lo, size = 0, self.batch_size
        while lo < len(context):
            part = slice(lo, lo + size)
            try:
                f = model.forecast(
                    context[part][:, None, :], prediction_length=horizon,
                    known_covariates=None if known is None else known[part],
                )
            except Exception as e:  # out of GPU memory: halve the batch and try again (results do not depend on it)
                if "out of memory" not in str(e).lower() or size == 1:
                    raise
                _free_gpu()
                size //= 2
                continue
            qs.append(_numpy(f.quantiles)[:, 0])  # series x horizon x levels
            levels = tuple(float(x) for x in f.quantile_levels)
            lo += size
        q = np.clip(np.concatenate(qs), 0.0, None)
        q[~has] = 0.0
        lv = np.asarray(levels)
        edges = np.concatenate([[0.0], (lv[1:] + lv[:-1]) / 2, [1.0]])
        point = q @ np.diff(edges)  # each quantile weighted by the share of probability nearest it
        rows = {s: i for i, s in enumerate(self._series)}
        idx = future[SERIES].map(rows)
        found = idx.notna().to_numpy()
        pick = lambda m: np.where(found, m[idx.fillna(0).astype(int).to_numpy(), np.clip(ahead, 1, horizon) - 1], 0.0)
        self.last_quantiles = {lv: pick(q[:, :, k]) for k, lv in enumerate(levels)}
        return pick(point)
