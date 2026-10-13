"""Chronos-Bolt, a pretrained time-series foundation model (Ansari et al., 2024),
used zero-shot on each series' unit history: no prices, promotions or calendar.

The forecast is the mean of its quantiles (0.1 to 0.9), which tracks sales
totals better than its median for items that often sell nothing; the
quantiles are kept in `last_quantiles` for interval scoring. Series history is
built exactly as for TimesFM. Needs `pip install chronos-forecasting`.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES
from rwm.forecaster import register_model
from rwm.prior_work.timesfm_zero_shot import TimesFMZeroShot

REPO = "amazon/chronos-bolt-base"
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _load(device: str):
    import torch
    from chronos import BaseChronosPipeline

    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    return BaseChronosPipeline.from_pretrained(REPO, device_map=dev, torch_dtype=torch.float32)


@register_model("chronos")
class ChronosZeroShot(TimesFMZeroShot):
    def __init__(self, context: int = 512, batch_size: int = 1024, device: str = "auto", loader=None):
        super().__init__(context=context, batch_size=batch_size, point="mean")
        self.device = device
        self._chronos_loader = loader or _load
        self.last_quantiles = {}

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        import torch

        ahead = ((future[DATE].to_numpy() - self._end) / self._step).astype(int)
        horizon = int(ahead.max())
        pipe = self._chronos_loader(self.device)
        has = np.isfinite(self._history).any(axis=1)
        qs = []
        for lo in range(0, len(self._history), self.batch_size):
            ctx = [torch.tensor(h[np.isfinite(h)] if np.isfinite(h).any() else np.zeros(3), dtype=torch.float32)
                   for h in self._history[lo:lo + self.batch_size]]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(LEVELS))
            qs.append(q.cpu().numpy())
        q = np.clip(np.concatenate(qs), 0.0, None)  # series x horizon x levels
        q[~has] = 0.0
        point = q.mean(axis=2)
        rows = {s: i for i, s in enumerate(self._series)}
        idx = future[SERIES].map(rows)
        known = idx.notna().to_numpy()
        pick = lambda m: np.where(known, m[idx.fillna(0).astype(int).to_numpy(), np.clip(ahead, 1, horizon) - 1], 0.0)
        self.last_quantiles = {lv: pick(q[:, :, k]) for k, lv in enumerate(LEVELS)}
        return pick(point)
