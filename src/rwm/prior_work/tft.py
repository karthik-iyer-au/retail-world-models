"""The Temporal Fusion Transformer (Lim et al., 2021), through the
neuralforecast implementation, as a covariate-aware neural benchmark.

One global model over every series, trained on the last `train_periods`
periods. It sees each series' units, and for every period, past and future,
the plan: the price against the series' highest price over the last
`price_window` periods, and each lever column. Its loss is the negative
binomial likelihood, which suits counts; the forecast is the predictive mean
and its quantiles are kept in `last_quantiles`. Needs `pip install neuralforecast`.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model

LEVELS = (50, 80, 95)


@register_model("tft")
class TFTForecaster(Forecaster):
    def __init__(self, horizon: int = 8, input_size: int = 26, train_periods: int = 156, extra=None,
                 price_window: int = 12, hidden_size: int = 64, max_steps: int = 1500, batch_size: int = 64,
                 windows_batch_size: int = 512, learning_rate: float = 1e-3, seed: int = 1, accelerator: str = "auto"):
        self.horizon, self.input_size, self.train_periods = horizon, input_size, train_periods
        self.extra = list(extra or [])
        self.price_window = price_window
        self.hidden_size, self.max_steps, self.batch_size = hidden_size, max_steps, batch_size
        self.windows_batch_size, self.lr, self.seed, self.accelerator = windows_batch_size, learning_rate, seed, accelerator
        self.last_quantiles = {}

    def _plan(self, frame: pd.DataFrame, top: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame({"unique_id": frame[SERIES].astype(str).to_numpy(), "ds": frame[DATE].to_numpy()})
        t = frame[SERIES].map(top).to_numpy(dtype=float)
        p = frame[PRICE].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            r = p / t
        out["price_ratio"] = np.where(np.isfinite(r) & (r > 0), r, 1.0)
        for c in self.extra:
            out[c] = frame[c].fillna(0).to_numpy(dtype=float)
        return out

    def fit(self, train: pd.DataFrame) -> "TFTForecaster":
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import TFT

        dates = np.sort(train[DATE].unique())
        self._step = np.diff(dates).min()
        self._freq = f"{int(self._step / np.timedelta64(1, 'D'))}D"  # weeks here start on any weekday
        recent = train[train[DATE] >= dates[max(0, len(dates) - self.train_periods)]]
        top_dates = dates[-self.price_window:]
        self._top = train[train[DATE].isin(top_dates)].groupby(SERIES)[PRICE].max()
        df = self._plan(recent, self._top)
        df["y"] = recent[UNITS].to_numpy(dtype=float)
        # a regular grid per series from its first record (missing periods sold nothing)
        df = df.sort_values(["unique_id", "ds"])
        self._futr = ["price_ratio"] + self.extra
        model = TFT(h=self.horizon, input_size=self.input_size, hidden_size=self.hidden_size,
                    futr_exog_list=self._futr, loss=DistributionLoss("NegativeBinomial", level=list(LEVELS)),
                    max_steps=self.max_steps, batch_size=self.batch_size,
                    windows_batch_size=self.windows_batch_size, learning_rate=self.lr, random_seed=self.seed,
                    scaler_type="robust", start_padding_enabled=True, accelerator=self.accelerator,
                    enable_progress_bar=False, logger=False, enable_model_summary=False)
        self._nf = NeuralForecast(models=[model], freq=self._freq)
        self._df = self._regular(df)
        self._nf.fit(df=self._df)
        return self

    def _regular(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill each series' gaps from its first record to the last training date with zero sales."""
        end = df["ds"].max()
        parts = []
        for uid, g in df.groupby("unique_id", sort=False):
            idx = pd.date_range(g["ds"].min(), end, freq=pd.Timedelta(self._step))
            g = g.set_index("ds").reindex(idx)
            g["unique_id"] = uid
            g["y"] = g["y"].fillna(0.0)
            g["price_ratio"] = g["price_ratio"].fillna(1.0)
            for c in self.extra:
                g[c] = g[c].fillna(0.0)
            parts.append(g.rename_axis("ds").reset_index())
        return pd.concat(parts, ignore_index=True)

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        fut = self._plan(future, self._top)
        grid = self._nf.make_future_dataframe()
        grid = grid.merge(fut, on=["unique_id", "ds"], how="left")
        grid["price_ratio"] = grid["price_ratio"].fillna(1.0)
        for c in self.extra:
            grid[c] = grid[c].fillna(0.0)
        pred = self._nf.predict(futr_df=grid).reset_index()
        key = pd.MultiIndex.from_arrays([future[SERIES].astype(str).to_numpy(), future[DATE].to_numpy()])
        pred = pred.set_index(["unique_id", "ds"]).reindex(key)
        point = np.clip(pred["TFT"].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        self.last_quantiles = {}
        for lv in LEVELS:
            lo, hi = (1 - lv / 100) / 2, 1 - (1 - lv / 100) / 2
            self.last_quantiles[round(lo, 3)] = np.clip(pred[f"TFT-lo-{lv}"].fillna(0).to_numpy(float), 0, None)
            self.last_quantiles[round(hi, 3)] = np.clip(pred[f"TFT-hi-{lv}"].fillna(0).to_numpy(float), 0, None)
        return point
