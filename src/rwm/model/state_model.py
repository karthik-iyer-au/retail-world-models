"""Retail state model: item encoder and sales readout.

For each item in a store the model reads the last `history` periods (sales,
price, any extra columns, calendar position) and turns them into a state
vector. A readout then gives expected sales for each of the next `horizon`
periods from that state plus what is planned for the period: its price, the
extra columns, and its calendar position.

Choices recorded in docs/DECISIONS.md:
- one state per item and store
- trained on forecast error, with a Tweedie loss because most days are zero
- a lever the dataset does not record is passed as "not known", not as zero
- sales and prices are scaled by each item's own recent level, so one model
  serves items of very different size

Training and forecasting build their inputs with the same function
(`_window`), so they cannot drift apart.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.forecaster import Forecaster, register_model
from rwm.utils.frames import frame_to_matrix, series_rows


def _calendar(dates: np.ndarray) -> np.ndarray:
    """Position in the week and in the year, as smooth cycles."""
    d = pd.DatetimeIndex(dates)
    week = 2 * np.pi * d.dayofweek.to_numpy() / 7
    year = 2 * np.pi * (d.dayofyear.to_numpy() - 1) / 365.25
    return np.stack([np.sin(week), np.cos(week), np.sin(year), np.cos(year)], axis=1).astype(np.float32)


def _build_net(n_hist: int, n_fut: int, cat_sizes: list[int], history: int, horizon: int,
               d_model: int, layers: int, heads: int, dropout: float):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_hist, d_model)
            self.pos = nn.Parameter(torch.zeros(history, d_model))
            block = nn.TransformerEncoderLayer(
                d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.emb = nn.ModuleList([nn.Embedding(n + 1, 16) for n in cat_sizes])  # 0 = not seen in training
            self.step = nn.Embedding(horizon, 16)
            self.head = nn.Sequential(
                nn.Linear(d_model + 16 * len(cat_sizes) + 16 + n_fut, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        def state(self, hist):
            """The item's state: one vector summarising its recent history."""
            return self.norm(self.encoder(self.inp(hist) + self.pos)).mean(dim=1)

        def forward(self, hist, fut, cats):
            h = fut.shape[1]
            parts = [self.state(hist)] + [e(cats[:, i]) for i, e in enumerate(self.emb)]
            fixed = torch.cat(parts, dim=1)[:, None, :].expand(-1, h, -1)
            steps = self.step(torch.arange(h, device=fut.device))[None].expand(len(fut), -1, -1)
            return self.head(torch.cat([fixed, steps, fut], dim=2)).squeeze(-1).clamp(-10, 10)

    return Net()


@register_model("state_model")
class StateModel(Forecaster):
    def __init__(
        self,
        horizon: int = 28,
        history: int = 112,
        train_periods: int = 730,
        categorical: list[str] | None = None,
        extra: list[str] | None = None,
        d_model: int = 96,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        steps: int = 20000,
        batch: int = 1024,
        lr: float = 1e-3,
        tweedie_power: float = 1.5,
        seed: int = 0,
        device: str | None = None,
        mixed_precision: bool = False,
    ):
        self._settings = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.horizon, self.history, self.train_periods = horizon, history, train_periods
        self.categorical, self.extra = categorical or [], extra or []
        self.net_args = dict(d_model=d_model, layers=layers, heads=heads, dropout=dropout)
        self.steps, self.batch, self.lr = steps, batch, lr
        self.power, self.seed, self.device = tweedie_power, seed, device
        self.mixed_precision = mixed_precision  # faster on a GPU; has no effect on a CPU

    def _tensors(self, frame: pd.DataFrame, dates: np.ndarray, with_units: bool):
        """Series-by-date arrays for one stretch of dates."""
        import torch

        to = lambda a: torch.as_tensor(a, device=self._dev)
        out = {"calendar": to(_calendar(dates))}
        if with_units:
            out["units"] = to(np.nan_to_num(frame_to_matrix(frame, self._names, dates, UNITS)))
        if self._has_price:
            out["price"] = to(frame_to_matrix(frame, self._names, dates, PRICE))
        else:
            out["price"] = torch.full((len(self._names), len(dates)), float("nan"), device=self._dev)
        out["extra"] = [to(frame_to_matrix(frame, self._names, dates, c)) for c in self.extra]
        return out

    def _window(self, past: dict, future: dict, rows, start):
        """Model inputs for the items in `rows`: the `history` columns of `past`
        ending just before column `start`, and `horizon` columns of `future`
        beginning at `future["start"]`. Also returns each item's scale, which
        future periods the item is on sale in, and the future column positions."""
        import torch

        L = self.history
        back = start[:, None] + torch.arange(-L, 0, device=self._dev)[None]
        r = rows[:, None]
        units, price = past["units"][r, back], past["price"][r, back]
        known = ~torch.isnan(price)
        n_known = known.sum(1, keepdim=True).clamp(min=1)
        # each item's own recent level, over the periods it was on sale
        scale = (units * known).sum(1, keepdim=True) / n_known
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        level = torch.nan_to_num(price).sum(1, keepdim=True) / n_known
        level = torch.where(level > 0, level, torch.ones_like(level))

        def lever_inputs(src, cols):
            p = src["price"][r, cols]
            k = ~torch.isnan(p)
            feats = [torch.nan_to_num(p) / level, k.float()]
            for e in src["extra"]:
                v = e[r, cols]
                feats += [torch.nan_to_num(v), (~torch.isnan(v)).float()]
            cal = src["calendar"][cols]
            return torch.cat([torch.stack(feats, dim=2), cal], dim=2), k

        hist_levers, _ = lever_inputs(past, back)
        hist = torch.cat([(units / scale)[..., None], torch.log1p(units)[..., None], hist_levers], dim=2)
        ahead = future["start"][:, None] + torch.arange(self.horizon, device=self._dev)[None]
        fut, on_sale = lever_inputs(future, ahead)
        return hist, fut, scale, on_sale, ahead

    def fit(self, train: pd.DataFrame) -> "StateModel":
        import torch

        torch.manual_seed(self.seed)
        self._dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        L, H = self.history, self.horizon
        dates = np.sort(train[DATE].unique())[-(self.train_periods + L) :]
        if len(dates) < L + H:
            raise ValueError("not enough history to train on")
        train = train[train[DATE] >= dates[0]]
        self._names = pd.Index(train[SERIES].unique().astype(str)).sort_values()
        self._has_price = PRICE in train
        data = self._tensors(train, dates, with_units=True)

        first = train.drop_duplicates(SERIES)
        first = first.set_index(first[SERIES].astype(str)).loc[self._names]
        self._levels = {c: pd.Index(first[c].astype(str).unique()).sort_values() for c in self.categorical}
        cats = np.stack(
            [self._levels[c].get_indexer(first[c].astype(str)) + 1 for c in self.categorical], axis=1
        ) if self.categorical else np.zeros((len(self._names), 0), dtype=np.int64)
        self._cats = torch.as_tensor(cats, device=self._dev, dtype=torch.long)

        n_lever = 2 + 2 * len(self.extra) + 4
        self._net = _build_net(
            2 + n_lever, n_lever, [len(v) for v in self._levels.values()], L, H, **self.net_args
        ).to(self._dev)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        warm = max(1, self.steps // 20)  # learning rate rises for the first 5% of steps, then falls away
        shape = lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + np.cos(np.pi * (k - warm) / max(1, self.steps - warm)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, shape)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        amp = self.mixed_precision and self._dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        n, t, p = len(self._names), len(dates), self.power
        self._net.train()
        self.loss_log = []
        for step in range(self.steps):
            rows = torch.randint(0, n, (self.batch,), generator=gen).to(self._dev)
            start = torch.randint(L, t - H + 1, (self.batch,), generator=gen).to(self._dev)
            hist, fut, scale, on_sale, ahead = self._window(data, {**data, "start": start}, rows, start)
            target = data["units"][rows[:, None], ahead] / scale
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                log_mu = self._net(hist, fut, self._cats[rows])
            log_mu = log_mu.float()
            # Tweedie loss on sales relative to the item's own level
            loss = -target * torch.exp((1 - p) * log_mu) / (1 - p) + torch.exp((2 - p) * log_mu) / (2 - p)
            loss = (loss * on_sale).sum() / on_sale.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 50 == 0 or step == self.steps - 1:
                self.loss_log.append((step, loss.item()))
            if (step + 1) % max(1, self.steps // 20) == 0:
                print(f"training step {step + 1} of {self.steps}, loss {loss.item():.4f}", flush=True)

        tail = slice(t - L, t)
        self._past = {
            "units": data["units"][:, tail],
            "price": data["price"][:, tail],
            "extra": [e[:, tail] for e in data["extra"]],
            "calendar": data["calendar"][tail],
        }
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        import torch

        fut_dates = np.sort(future[DATE].unique())
        h = len(fut_dates)
        if h > self.horizon:
            raise ValueError("asked to forecast further ahead than the model was built for")
        pad = fut_dates[-1] + (np.arange(1, self.horizon - h + 1) * (fut_dates[-1] - fut_dates[-2] if h > 1 else np.timedelta64(1, "D")))
        fut = self._tensors(future, np.r_[fut_dates, pad], with_units=False)
        n = len(self._names)
        out = np.zeros((n, self.horizon), dtype=np.float64)
        self._net.eval()
        with torch.no_grad():
            for lo in range(0, n, 4096):
                rows = torch.arange(lo, min(lo + 4096, n), device=self._dev)
                start = torch.full_like(rows, self.history)
                zero = torch.zeros_like(rows)
                hist, f, scale, _, _ = self._window(self._past, {**fut, "start": zero}, rows, start)
                out[lo : lo + len(rows)] = (torch.exp(self._net(hist, f, self._cats[rows])) * scale).cpu().numpy()
        r = series_rows(future[SERIES], self._names)
        c = pd.Index(fut_dates).get_indexer(future[DATE])
        pred = np.where(r >= 0, out[np.clip(r, 0, None), c], 0.0)
        return np.clip(pred, 0, None)

    def save(self, path) -> None:
        """Write the fitted model to one file: settings, weights, and the
        recent history it needs to forecast."""
        import torch

        cpu = lambda t: t.detach().cpu()
        torch.save(
            {
                "settings": self._settings,
                "weights": {k: cpu(v) for k, v in self._net.state_dict().items()},
                "names": list(self._names),
                "levels": {c: list(v) for c, v in self._levels.items()},
                "cats": cpu(self._cats),
                "has_price": self._has_price,
                "past": {
                    k: [cpu(e) for e in v] if isinstance(v, list) else cpu(v)
                    for k, v in self._past.items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str | None = None) -> "StateModel":
        import torch

        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**{**saved["settings"], "device": device})
        model._dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model._names = pd.Index(saved["names"])
        model._levels = {c: pd.Index(v) for c, v in saved["levels"].items()}
        model._cats = saved["cats"].to(model._dev)
        model._has_price = saved["has_price"]
        model._past = {
            k: [e.to(model._dev) for e in v] if isinstance(v, list) else v.to(model._dev)
            for k, v in saved["past"].items()
        }
        n_lever = 2 + 2 * len(model.extra) + 4
        model._net = _build_net(
            2 + n_lever, n_lever, [len(v) for v in model._levels.values()],
            model.history, model.horizon, **model.net_args,
        ).to(model._dev)
        model._net.load_state_dict(saved["weights"])
        return model
