"""Chronos-2, Amazon's pretrained time-series foundation model (Ansari et
al., October 2025), used zero-shot: it is not trained or tuned on our data.

Unlike Chronos-Bolt (`chronos`), which sees each series' unit history only,
Chronos-2 reads known-future covariates. With `plan` (and `price_plan`) each
series is given its plan inputs over its history and the horizon, built
exactly as for TimesFM, so every foundation model in the comparison is given
the same plan.

The forecast is the mean of its nine quantiles (0.1 to 0.9), as for
Chronos-Bolt; the quantiles are kept in `last_quantiles` for interval scoring.
Needs `chronos-forecasting>=2.0` and downloads `amazon/chronos-2` from Hugging
Face on first use.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES
from rwm.forecaster import register_model
from rwm.prior_work.timesfm_zero_shot import TimesFMZeroShot

REPO = "amazon/chronos-2"
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _load(device: str):
    import torch
    from chronos import Chronos2Pipeline

    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    return Chronos2Pipeline.from_pretrained(REPO, device_map=dev, dtype=torch.float32)


@register_model("chronos2")
class Chronos2ZeroShot(TimesFMZeroShot):
    def __init__(self, context: int = 512, batch_size: int = 256, plan: list[str] | None = None,
                 price_plan: bool = False, device: str = "auto", loader=None):
        super().__init__(context=context, batch_size=batch_size, point="mean", plan=plan, price_plan=price_plan)
        self.device = device
        self._chronos_loader = loader or _load
        self.last_quantiles = {}

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        ahead = ((future[DATE].to_numpy() - self._end) / self._step).astype(int)
        horizon = int(ahead.max())
        has = np.isfinite(self._history).any(axis=1)
        keep = [np.isfinite(h) for h in self._history]
        targets = [h[k] if k.any() else np.zeros(3) for h, k in zip(self._history, keep)]
        if self._plan_past:
            cov = self._covariates(future, horizon, keep)
            inputs = []
            for i, y in enumerate(targets):
                both = {c: np.asarray(cov[c][i], dtype=np.float32) for c in cov}
                inputs.append({
                    "target": y.astype(np.float32),
                    "past_covariates": {c: v[: len(y)] for c, v in both.items()},
                    "future_covariates": {c: v[len(y):] for c, v in both.items()},
                })
        else:
            inputs = [y.astype(np.float32) for y in targets]
        pipe = self._chronos_loader(self.device)
        q = np.zeros((len(inputs), horizon, len(LEVELS)))
        for lo in range(0, len(inputs), self.chunk):
            got, _ = pipe.predict_quantiles(inputs[lo:lo + self.chunk], prediction_length=horizon,
                                            quantile_levels=list(LEVELS), batch_size=self.batch_size)
            for i, t in enumerate(got, start=lo):
                q[i] = t.cpu().numpy()[0]
        q = np.clip(q, 0.0, None)
        q[~has] = 0.0
        point = q.mean(axis=2)
        rows = {s: i for i, s in enumerate(self._series)}
        idx = future[SERIES].map(rows)
        known = idx.notna().to_numpy()
        pick = lambda m: np.where(known, m[idx.fillna(0).astype(int).to_numpy(), np.clip(ahead, 1, horizon) - 1], 0.0)
        self.last_quantiles = {lv: pick(q[:, :, k]) for k, lv in enumerate(LEVELS)}
        return pick(point)
