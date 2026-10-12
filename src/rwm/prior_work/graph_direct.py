"""A graph neural network over product attributes, forecasting all days directly.

Method: each item-store pair is a node. Nodes are joined when they share a
set of attributes named in the config (for M5: the same item in other
stores, and the same department in the same store). One round of
message passing in the style of GraphSAGE (Hamilton, Ying and Leskovec,
2017): each node's own inputs pass through a small network, the mean of
those over each relation's neighbours is added, and a second network reads
the result. Trained with the same Tweedie objective as LightGBM.

Code: written here, on PyTorch. The inputs per node and day are exactly the
LightGBM baseline's (`lightgbm_direct._features`): sales lagged by at least
the horizon, trailing averages, price against its recent highest, calendar,
extra columns and labels. So the comparison with LightGBM is the graph and
the network, not different inputs.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model
from rwm.prior_work.lightgbm_direct import LightGBMDirect
from rwm.utils.frames import frame_to_matrix


def _groups(attrs: pd.DataFrame, relations: list[list[str]]) -> list[np.ndarray]:
    """For each relation, a group number per node: nodes sharing those attributes share it."""
    out = []
    for cols in relations:
        key = attrs[cols].astype(str).agg("|".join, axis=1)
        out.append(pd.factorize(key, sort=True)[0].astype(np.int64))
    return out


@register_model("graph_direct")
class GraphDirect(Forecaster):
    def __init__(
        self,
        horizon: int = 28,
        train_periods: int = 730,
        relations: list[list[str]] | None = None,
        categorical: list[str] | None = None,
        extra: list[str] | None = None,
        hidden: int = 128,
        steps: int = 3000,
        lr: float = 1e-3,
        tweedie_power: float = 1.1,
        seed: int = 0,
        device: str | None = None,
        inputs: dict | None = None,
    ):
        # `inputs`: settings for the inputs, as for lightgbm_direct (lags, windows, price_window, ...)
        self.horizon = horizon
        self.train_periods = train_periods
        self.relations = relations or []
        self.categorical = categorical or []
        self.extra = extra or []
        self.hidden = hidden
        self.steps = steps
        self.lr = lr
        self.rho = tweedie_power
        self.seed = seed
        self.device = device
        # the LightGBM baseline builds the inputs, so both see the same ones
        self._inputs = LightGBMDirect(horizon=horizon, train_periods=train_periods,
                                      categorical=self.categorical, extra=self.extra, **(inputs or {}))

    def _matrices(self, frame, dates):
        units = frame_to_matrix(frame, self._names, dates, UNITS) if UNITS in frame else None
        price = frame_to_matrix(frame, self._names, dates, PRICE)
        extra = {c: frame_to_matrix(frame, self._names, dates, c) for c in self.extra}
        return units, price, extra

    def _stack(self, f: dict, cols, scale=None) -> np.ndarray:
        """Inputs as nodes by days by inputs (each input, then a flag for "missing"),
        scaled by `scale` (mean, spread) when given. Kept in half precision, filled
        one input at a time, so the whole training table fits in memory."""
        n, k = next(iter(f.values())).shape[0], len(f)
        days = len(range(*cols.indices(next(iter(f.values())).shape[1])))
        out = np.empty((n, days, 2 * k), dtype=np.float16)
        for i, v in enumerate(f.values()):
            v = v[:, cols].astype(np.float32)
            miss = np.isnan(v)
            v = np.nan_to_num(v)
            out[:, :, k + i] = miss
            if scale is not None:
                v = (v - scale[0][i]) / scale[1][i]
            out[:, :, i] = np.clip(v, -60000, 60000)
        return out

    def _net(self, n_in: int):
        import torch
        from torch import nn

        h, k = self.hidden, len(self.relations)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.own = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
                self.msg = nn.ModuleList(nn.Linear(h, h) for _ in range(k))
                self.read = nn.Sequential(nn.Linear(h * (1 + k), h), nn.GELU(), nn.Linear(h, 1))

            def forward(self, x, groups, sizes):
                # x: nodes by inputs for one day (or nodes by days by inputs)
                z = self.own(x)
                parts = [z]
                for g, n, m in zip(groups, sizes, self.msg):
                    total = torch.zeros((n,) + z.shape[1:], device=z.device, dtype=z.dtype).index_add_(0, g, z)
                    count = torch.zeros(n, device=z.device, dtype=z.dtype).index_add_(0, g, torch.ones_like(g, dtype=z.dtype))
                    shape = (-1,) + (1,) * (z.dim() - 1)
                    # the mean over the group without the node itself
                    others = (total[g] - z) / (count[g] - 1).clamp(min=1).view(shape)
                    parts.append(m(others))
                return self.read(torch.cat(parts, dim=-1)).squeeze(-1)

        return Net()

    def fit(self, train: pd.DataFrame) -> "GraphDirect":
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        inp = self._inputs
        all_dates = np.sort(train[DATE].unique())
        dates = all_dates[-(self.train_periods + inp.history):]
        part = train[train[DATE] >= dates[0]]
        part = part.assign(**{SERIES: part[SERIES].astype(str)})
        self._names = pd.Index(part[SERIES].unique()).sort_values()
        attrs = part.drop_duplicates(SERIES).set_index(SERIES).loc[self._names]
        inp._cat_levels = {c: pd.Index(train[c].unique().astype(str)).sort_values() for c in self.categorical}
        cats = {c: inp._cat_levels[c].get_indexer(attrs[c].astype(str)) for c in self.categorical}
        self._cats = cats
        self._group_np = _groups(attrs.reset_index(), self.relations)
        units, price, extra = self._matrices(part, dates)
        units = np.nan_to_num(units)
        f = inp._features(units, price, extra, dates, cats)
        cols = slice(len(dates) - min(self.train_periods, len(dates)), len(dates))
        # inputs are scaled by their mean and spread over training
        self._scale = (
            np.array([np.nanmean(v[:, cols]) if np.isfinite(v[:, cols]).any() else 0.0 for v in f.values()]),
            np.array([np.nanstd(v[:, cols]) + 1e-6 if np.isfinite(v[:, cols]).any() else 1.0 for v in f.values()]),
        )
        x = self._stack(f, cols, self._scale)
        del f
        y = units[:, cols].astype(np.float32)
        on = ~np.isnan(price[:, cols])  # only days the item was on sale count
        tail = slice(len(dates) - inp.history, len(dates))
        self._state = {"dates": dates[tail], "units": units[:, tail], "price": price[:, tail],
                       "extra": {c: m[:, tail] for c, m in extra.items()}}

        dev = self._dev
        groups = [torch.as_tensor(g, device=dev) for g in self._group_np]
        sizes = [int(g.max()) + 1 for g in self._group_np]
        self._model = self._net(x.shape[2]).to(dev)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        X, Y, ON = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(on)
        days = x.shape[1]
        self._model.train()
        for step in range(self.steps):
            d = rng.choice(days, size=min(4, days), replace=False)  # a few whole days: every node, so every neighbour
            xb = X[:, d].to(dev, non_blocking=True).float()
            yb, ob = Y[:, d].to(dev), ON[:, d].to(dev)
            out = self._model(xb, groups, sizes).clamp(-10, 10)  # log expected sales
            p = self.rho
            loss = -yb * torch.exp(out * (1 - p)) / (1 - p) + torch.exp(out * (2 - p)) / (2 - p)
            loss = (loss * ob).sum() / ob.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            opt.step()
            sched.step()
        self._groups_t, self._sizes = groups, sizes
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        import torch

        fut_dates = np.sort(future[DATE].unique())
        if len(fut_dates) > self.horizon:
            raise ValueError("asked to forecast further ahead than the model was built for")
        part = future.assign(**{SERIES: future[SERIES].astype(str)})
        _, price, extra = self._matrices(part, fut_dates)
        s, n, h = self._state, len(self._names), len(fut_dates)
        dates = np.r_[s["dates"], fut_dates]
        units = np.c_[s["units"], np.full((n, h), np.nan, dtype=np.float32)]
        price = np.c_[s["price"], price]
        extra = {c: np.c_[s["extra"][c], m] for c, m in extra.items()}
        f = self._inputs._features(units, price, extra, dates, self._cats)
        x = self._stack(f, slice(len(dates) - h, len(dates)), self._scale)
        self._model.eval()
        with torch.no_grad():
            out = torch.exp(self._model(torch.as_tensor(x, device=self._dev).float(), self._groups_t, self._sizes).clamp(-10, 10))
        pred = out.cpu().numpy()
        r = self._names.get_indexer(part[SERIES])
        c = pd.Index(fut_dates).get_indexer(part[DATE])
        return np.clip(np.where(r >= 0, pred[np.clip(r, 0, None), c], 0.0), 0, None)
