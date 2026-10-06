"""Gradient boosted trees, one model per store, forecasting all days directly.

Method: LightGBM (Ke et al., 2017) with a Tweedie objective, trained on
sales lagged by at least the forecast horizon, so one model forecasts every
day of the horizon without feeding its own forecasts back in. This is the
store-level, non-recursive design used by the top M5 entries (Makridakis,
Spiliotis and Assimakopoulos, 2022). It is one model per store with fixed
settings, not the winning entry's ensemble of 220 models.

Code: written here, on the `lightgbm` package (MIT licence).

Inputs per row: sales at set lags no shorter than the horizon, averages and
spread of sales ending one horizon earlier, price and how it compares with
the item's highest recent price and its price a little earlier, calendar
position, and any extra columns named in the config. Lags and window lengths
count periods and are set in the config; the defaults suit daily data with a
28-day horizon.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model
from rwm.utils.frames import frame_to_matrix

def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if k < x.shape[1]:
        out[:, k:] = x[:, : x.shape[1] - k]
    return out


def _rolling(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the `w` columns ending at each column."""
    z = np.nan_to_num(x).astype(np.float64)
    c1 = np.cumsum(z, axis=1)
    c2 = np.cumsum(z * z, axis=1)
    s1, s2 = c1.copy(), c2.copy()
    s1[:, w:] -= c1[:, :-w]
    s2[:, w:] -= c2[:, :-w]
    mean = s1 / w
    std = np.sqrt(np.maximum(s2 / w - mean**2, 0))
    known = np.cumsum(~np.isnan(x), axis=1)
    full = np.zeros(x.shape, dtype=bool)
    full[:, w - 1 :] = (known[:, w - 1 :] - np.c_[np.zeros((len(x), 1)), known[:, :-w]]) == w
    return (
        np.where(full, mean, np.nan).astype(np.float32),
        np.where(full, std, np.nan).astype(np.float32),
    )


def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    """Highest known value in the `w` columns ending at each column."""
    padded = np.c_[np.full((len(x), w - 1), np.nan, dtype=np.float32), x]
    windows = np.lib.stride_tricks.sliding_window_view(padded, w, axis=1)
    return np.fmax.reduce(windows, axis=2)


@register_model("lightgbm_direct")
class LightGBMDirect(Forecaster):
    def __init__(
        self,
        horizon: int = 28,
        train_periods: int = 730,
        group_by: str | None = None,
        categorical: list[str] | None = None,
        extra: list[str] | None = None,
        lags: tuple = (0, 7, 14, 21, 28),
        windows: tuple = (7, 14, 28, 56),
        spread_window: int = 28,
        price_window: int = 84,
        price_lag: int = 7,
        rounds: int = 800,
        seed: int = 0,
        threads: int = 2,
        params: dict | None = None,
        cold_start: bool = False,
    ):
        # forecast a series with no history from its labels, price and plan
        # alone (sales-history inputs are left missing); otherwise it gets zero
        self.cold_start = cold_start
        self.horizon = horizon
        self.train_periods = train_periods
        self.group_by = group_by
        self.categorical = categorical or []
        self.extra = extra or []
        # All of these count periods (days for daily data, weeks for weekly data).
        self.lags = tuple(lags)  # added to the horizon: sales this many periods before the earliest usable one
        self.windows = tuple(windows)  # lengths of the trailing sales averages
        self.spread_window = spread_window  # length of the trailing window for the spread of sales
        self.price_window = price_window  # look-back for the item's highest recent price
        self.price_lag = price_lag  # price is compared with the price this many periods earlier
        self.rounds = rounds
        # periods of history needed to build the inputs for a forecast
        self.history = max(horizon + max(self.lags) + max(self.windows), price_window + price_lag)
        self.params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "max_bin": 127,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": threads,
            "verbose": -1,
            **(params or {}),
        }

    # one table of model inputs for every series and every column of `dates`
    def _features(self, units, price, extra, dates, cats) -> dict[str, np.ndarray]:
        h = self.horizon
        f = {}
        for lag in self.lags:
            f[f"sales_{h + lag}_periods_ago"] = _shift(units, h + lag)
        base = _shift(units, h)
        for w in self.windows:
            mean, std = _rolling(base, w)
            f[f"mean_sales_{w}_periods"] = mean
            if w == self.spread_window:
                f[f"spread_sales_{w}_periods"] = std
        with np.errstate(invalid="ignore", divide="ignore"):
            f["price"] = price
            f[f"price_vs_highest_in_{self.price_window}_periods"] = price / _rolling_max(price, self.price_window)
            f[f"price_vs_{self.price_lag}_periods_ago"] = price / _shift(price, self.price_lag)
        d = pd.DatetimeIndex(dates)
        shape = units.shape
        for name, values in {
            "day_of_week": d.dayofweek,
            "day_of_month": d.day,
            "week_of_year": d.isocalendar().week.to_numpy(),
            "month": d.month,
            "year": d.year,
        }.items():
            f[name] = np.broadcast_to(np.asarray(values, dtype=np.float32), shape)
        for name, m in extra.items():
            f[name] = m
        for name, codes in cats.items():
            f[name] = np.broadcast_to(codes.astype(np.float32)[:, None], shape)
        return f

    def _prepare(self, frame: pd.DataFrame, names: pd.Index, dates: np.ndarray):
        units = frame_to_matrix(frame, names, dates, UNITS) if UNITS in frame else None
        price = frame_to_matrix(frame, names, dates, PRICE)
        extra = {c: frame_to_matrix(frame, names, dates, c) for c in self.extra}
        return units, price, extra

    def _fit_table(self, part: pd.DataFrame, names: pd.Index, dates: np.ndarray):
        """Inputs and targets for one group, plus the history kept for forecasting."""
        units, price, extra = self._prepare(part, names, dates)
        units = np.nan_to_num(units)  # a missing row means nothing sold
        first = part.drop_duplicates(SERIES).set_index(SERIES).loc[names]
        cats = {c: self._cat_levels[c].get_indexer(first[c].astype(str)) for c in self.categorical}
        f = self._features(units, price, extra, dates, cats)
        cols = slice(len(dates) - min(self.train_periods, len(dates)), len(dates))
        keep = ~np.isnan(price[:, cols]).ravel()  # only days the item was on sale
        x = np.column_stack([v[:, cols].ravel()[keep] for v in f.values()])
        tail = slice(len(dates) - self.history, len(dates))
        state = {
            "cats": cats,
            "dates": dates[tail],
            "units": units[:, tail],
            "price": price[:, tail],
            "extra": {c: m[:, tail] for c, m in extra.items()},
        }
        return x, units[:, cols].ravel()[keep], list(f), state

    def _predict_table(self, g: dict, part: pd.DataFrame, fut_dates: np.ndarray) -> np.ndarray:
        """Inputs for the future days of one group, built by the same code as training."""
        _, price, extra = self._prepare(part, g["names"], fut_dates)
        n, h = len(g["names"]), len(fut_dates)
        dates = np.r_[g["dates"], fut_dates]
        units = np.c_[g["units"], np.full((n, h), np.nan, dtype=np.float32)]
        price = np.c_[g["price"], price]
        extra = {c: np.c_[g["extra"][c], m] for c, m in extra.items()}
        f = self._features(units, price, extra, dates, g["cats"])
        return np.column_stack([v[:, -h:].ravel() for v in f.values()])

    def fit(self, train: pd.DataFrame) -> "LightGBMDirect":
        import lightgbm as lgb

        all_dates = np.sort(train[DATE].unique())
        dates = all_dates[-(self.train_periods + self.history) :]
        recent = (train[DATE] >= dates[0]).to_numpy()
        # distinct values first, then text: converting every row to text would not fit in memory
        self._cat_levels = {
            c: pd.Index(train[c].unique().astype(str)).sort_values() for c in self.categorical
        }
        self._groups = {}
        if self.group_by:
            group_codes, group_keys = pd.factorize(train[self.group_by], sort=True)
            group_codes = group_codes.astype(np.int32)
        else:
            group_codes, group_keys = np.zeros(len(train), dtype=np.int8), ["all"]
        for code, key in enumerate(group_keys):
            part = train[recent & (group_codes == code)]  # one store at a time
            part = part.assign(**{SERIES: part[SERIES].astype(str)})
            names = pd.Index(part[SERIES].unique()).sort_values()
            x, y, names_f, state = self._fit_table(part, names, dates)
            data = lgb.Dataset(
                x,
                label=y,
                feature_name=names_f,
                categorical_feature=list(self.categorical),
                free_raw_data=True,
            )
            booster = lgb.train(self.params, data, num_boost_round=self.rounds)
            booster.free_dataset()  # the fitted model does not need its training table
            self.feature_names = names_f
            del x, y, data, part
            self._groups[key] = {"booster": booster, "names": names, **state}
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(future), dtype=np.float64)
        fut_dates = np.sort(future[DATE].unique())
        if len(fut_dates) > self.horizon:
            raise ValueError("asked to forecast further ahead than the model was built for")
        series = future[SERIES].astype(str).to_numpy()
        col = pd.Index(fut_dates).get_indexer(future[DATE])
        keys = future[self.group_by].to_numpy() if self.group_by else np.full(len(future), "all")
        for key, g in self._groups.items():
            rows = np.flatnonzero(keys == key)
            if len(rows) == 0:
                continue
            part = future.iloc[rows].assign(**{SERIES: series[rows]})
            n, h = len(g["names"]), len(fut_dates)
            x = self._predict_table(g, part, fut_dates)
            pred = g["booster"].predict(x).reshape(n, h)
            r = g["names"].get_indexer(series[rows])
            ok = r >= 0
            out[rows[ok]] = pred[r[ok], col[rows[ok]]]
            if self.cold_start and (~ok).any():
                new = part.iloc[np.flatnonzero(~ok)]
                names = pd.Index(new[SERIES].unique()).sort_values()
                first = new.drop_duplicates(SERIES).set_index(SERIES).loc[names]
                m, back = len(names), len(g["dates"])
                blank = lambda: np.full((m, back), np.nan, dtype=np.float32)
                fresh = {
                    "names": names,
                    "dates": g["dates"],
                    "units": blank(),
                    "price": blank(),
                    "extra": {c: blank() for c in self.extra},
                    # a label not seen in training is coded -1, which LightGBM reads as missing
                    "cats": {c: self._cat_levels[c].get_indexer(first[c].astype(str)) for c in self.categorical},
                }
                got = g["booster"].predict(self._predict_table(fresh, new, fut_dates)).reshape(m, h)
                out[rows[~ok]] = got[names.get_indexer(series[rows[~ok]]), col[rows[~ok]]]
        return np.clip(out, 0, None)
