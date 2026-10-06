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

Item-to-item part (optional, `neighbours`). Items are put in groups
(`group_by`, for example all items in one store). For each future period an
item receives a weighted sum of messages from the other items in its group;
each message is built from the sender's state, its labels (the `categorical`
columns) and its plan for that period.
The weights come from one of:
- "similarity": a learned position for each product, compared under several
  learned conditions (a conditional similarity network). The weights depend
  only on which two products they are, so they can be read off as a table.
- "attention": the receiver's and sender's state, labels and plan for the period.
- "both": the two scores added. The attention score starts at zero, so the
  model starts as "similarity" and attention learns only what that misses.
An item can always give weight to "no neighbour", so weights sum to at most 1.
`neighbour_weights` returns the weights behind a forecast.

Lever step (optional). "Levers off" means the item at its regular price (its
highest price in the last `regular_window` periods) with every column named
in `levers` set to zero.
- `regular_price`: the model also sees each price as a share of the regular
  price, so it can tell a deep cut from a shallow one.
- `lift_readout`: the forecast is built as baseline times lift. The baseline
  sees only the levers-off plan. The lift is a second readout evaluated on
  the real plan minus the same readout on the levers-off plan, so it is
  exactly zero when nothing is planned. `breakdown` returns both parts.
- `latent_weight`: a supporting loss. From the state now and the plan for
  the horizon, the model predicts the state the encoder will give once those
  periods are history. The target comes from a slowly updated copy of the
  encoder. The forecast loss stays on throughout.
- `pretrain_steps`: train the encoder on the latent loss alone first, then
  train for the forecast. `finetune` says what happens to the encoder in
  that second stretch: `full` (trained as usual), `low_lr` (trained at
  `encoder_lr_scale` of the usual rate) or `frozen` (left as pretraining
  made it, so only the readout learns). `latent_ema` is how slowly the
  target copy follows.
- `validation_periods`: the last periods of the training data are kept out
  of training targets, scored during training at three levels (single
  series, each `categorical` grouping, the total), and the weights from the
  best point are kept.
- `total_weight`: adds to the loss the squared gap between summed forecasts
  and summed sales over the items drawn for each date, so a small bias in
  the same direction on many items is penalised.
- `readout_dropout`, `weight_decay`: the usual two.
- `regular_hold`: where no regular price is recorded, a price held for this
  many periods in a row becomes the regular price (the latest such price),
  in place of "highest in the last `regular_window` periods".
- `regular_column`: a column holding the regular price the retailer recorded
  (for example `base_price`), used in place of the 12-period rule wherever
  it is filled in.
- `plan_split` (with `rollout`): the rolled state is the sum of two parts.
  One is rolled under the "nothing planned" plan and never sees the plan. The
  other is what the plan adds: exactly zero until something is planned, and
  able to linger afterwards. `split_weight` keeps the two parts pointing in
  different directions. After the DWM paper, "Separating world effects
  from actions in latent world models" (arXiv 2607.18715), adapted: here the no-plan part
  is separate by construction.
- `state_spread_weight`: a penalty that keeps the state from collapsing: each
  direction keeps some spread across items and directions do not copy each
  other (variance and covariance terms, after VICReg, Bardes et al. 2022).
- `rollout_discount`: below 1, the latent check counts near periods for more
  than far ones (after TD-MPC2, Hansen et al. 2024).
- `cold_start`: a series with no history is forecast by borrowing the state
  and scale of the `cold_neighbours` known series most like it in the same
  store (most `categorical` labels in common), run with its own labels and
  its own plan, and averaged. `unknown_label_rate` hides the item label in
  that share of training examples so the model learns to do without it.
- `likelihood`: `tweedie` (the default, on sales relative to the item's own
  level) or `negative_binomial` (on units, as a count with a learned spread).
- `rollout`: the world-model form. A transition step takes the state and one
  period's plan and gives the state one period later. It is applied period
  by period across the horizon, and each period's sales are read from the
  state as it stands after the plan so far, so an earlier promotion can
  change a later period through the state. With `lift_readout` the
  baseline comes from a second rollout under the "nothing planned" plan.
  With the latent loss on, each rolled state is checked against the state
  the encoder gives once those periods are history (at the last step and at
  two steps on the way): the transition is the JEPA predictor.
- `plan_lags`: each forecast period also sees what was planned for the
  periods just before it, so a promotion last week can lower this week
  (pantry loading). With `lift_readout` that dip is part of the lift.
- `year_ago`: each forecast period also sees the item's sales and levers the
  same period a year ago (`year_ago: 52` for weekly data) and
  `year_ago_window` periods either side.

Training and forecasting build their inputs with the same function
(`_window`), so they cannot drift apart.
"""
import numpy as np
import pandas as pd

from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.forecaster import Forecaster, register_model
from rwm.utils.frames import frame_to_matrix, series_rows


TOTAL_DATES = 8  # dates per batch when the totals term is on


def _calendar(dates: np.ndarray) -> np.ndarray:
    """Position in the week and in the year, as smooth cycles."""
    d = pd.DatetimeIndex(dates)
    week = 2 * np.pi * d.dayofweek.to_numpy() / 7
    year = 2 * np.pi * (d.dayofyear.to_numpy() - 1) / 365.25
    return np.stack([np.sin(week), np.cos(week), np.sin(year), np.cos(year)], axis=1).astype(np.float32)


def _build_net(n_hist: int, n_fut: int, cat_sizes: list[int], history: int, horizon: int,
               d_model: int, layers: int, heads: int, dropout: float,
               neighbours: str | None = None, n_products: int = 0, conditions: int = 4,
               lift_readout: bool = False, latent: bool = False, readout_dropout: float = 0.0,
               rollout: bool = False, spread: bool = False, plan_split: bool = False):
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
            self.neighbours, self.k = neighbours, conditions
            if neighbours:
                n_item = d_model + 16 * len(cat_sizes) + n_fut  # an item's state, labels and plan
                self.msg = nn.Linear(n_item, d_model)
                if neighbours in ("similarity", "both"):
                    self.product = nn.Embedding(n_products, 32)
                    self.masks = nn.Parameter(torch.rand(conditions, 32))
                    self.bias = nn.Parameter(torch.zeros(conditions))
                    self.sharp = nn.Parameter(torch.ones(conditions))
                if neighbours in ("attention", "both"):
                    self.query = nn.Linear(n_item, d_model)
                    self.key = nn.Linear(n_item, d_model)
                    if neighbours == "both":  # start as similarity alone
                        nn.init.zeros_(self.query.weight)
                        nn.init.zeros_(self.query.bias)
            readout = lambda: nn.Sequential(
                nn.Linear(d_model * (2 if neighbours else 1) + 16 * len(cat_sizes) + 16 + n_fut, 2 * d_model),
                nn.GELU(),
                nn.Dropout(readout_dropout),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(readout_dropout),
                nn.Linear(d_model, 1),
            )
            self.rollout = rollout
            # for the negative binomial: how much more spread out sales are than a Poisson count
            self.spread = nn.Parameter(torch.zeros(())) if spread else None
            if rollout:  # the transition: this period's plan moves the state on by one period
                self.cell = nn.GRUCell(n_fut, d_model)
            self.plan_split = plan_split
            if plan_split:
                # what the plan adds to the state, carried forward on its own. With no
                # bias terms it is exactly zero until something is planned, and it can
                # linger afterwards (a heavy promotion still matters weeks later).
                self.effect_in = nn.Linear(d_model, d_model, bias=False)
                self.effect = nn.GRUCell(n_fut + d_model, d_model, bias=False)
            self.head = readout()
            self.lift = readout() if lift_readout else None
            self.next = nn.Sequential(
                nn.Linear(d_model + horizon * n_fut, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
            ) if latent and not rollout else None

        def state(self, hist):
            """The item's state: one vector summarising its recent history."""
            return self.norm(self.encoder(self.inp(hist) + self.pos)).mean(dim=1)

        def context(self, fixed, fut, group):
            """What each item receives from the others in its group, for each
            future period, and the weights used. `group` holds the batch
            layout: `shape` (groups, slots), `product` per row, and `open`
            (rows by periods): whether that row is a real item on sale then."""
            (G, M), H, K = group["shape"], fut.shape[1], self.k
            x = torch.cat([fixed, fut], dim=2)
            split = lambda v: v.view(G, M, H, K, -1).permute(0, 2, 3, 1, 4)  # groups, periods, conditions, items, size
            msg = split(self.msg(x))
            score = torch.zeros(G, H, K, M, M, device=fut.device)
            if self.neighbours in ("similarity", "both"):
                e = self.product(group["product"]).view(G, M, -1)
                # each condition looks at its own part of the product position;
                # positions are put on a common length so distances stay bounded
                seen = nn.functional.normalize(e[:, None] * torch.relu(self.masks)[None, :, None, :], dim=-1)
                dist = ((seen[:, :, :, None, :] - seen[:, :, None, :, :]) ** 2).sum(-1)
                sharp = nn.functional.softplus(self.sharp)[None, :, None, None]
                score = score + (self.bias[None, :, None, None] - sharp * dist)[:, None]
            if self.neighbours in ("attention", "both"):
                q, k = split(self.query(x)), split(self.key(x))
                score = score + (q @ k.transpose(-1, -2)).float() / q.shape[-1] ** 0.5
            open_ = group["open"].view(G, M, H).permute(0, 2, 1)
            shut = ~open_[:, :, None, None, :] | torch.eye(M, dtype=torch.bool, device=fut.device)
            score = score.masked_fill(shut, float("-inf"))
            score = torch.cat([score, torch.zeros(G, H, K, M, 1, device=fut.device)], dim=-1)  # "no neighbour"
            w = torch.softmax(score, dim=-1)[..., :M]
            got = (w.to(msg.dtype) @ msg).permute(0, 3, 1, 2, 4).reshape(G * M, H, -1)
            return got, w

        def roll(self, state, plan):
            """The state carried forward one period at a time under a plan:
            entry h is the state after living through periods 0..h."""
            out = []
            for h in range(plan.shape[1]):
                state = self.cell(plan[:, h, :], state)
                out.append(state)
            return torch.stack(out, dim=1)

        def roll_split(self, state, plan, plan_off):
            """The state under a plan as two parts: the state had nothing been
            planned (it never sees the plan), plus what the plan added."""
            world = self.roll(state, plan_off)
            change = plan - plan_off
            planned = (change.abs().sum(dim=2, keepdim=True) > 0).to(change.dtype)
            e, out = torch.zeros_like(state), []
            for h in range(plan.shape[1]):
                e = self.effect(torch.cat([change[:, h], planned[:, h] * self.effect_in(world[:, h])], dim=1), e)
                out.append(e)
            return world, torch.stack(out, dim=1)

        def forward(self, hist, fut, cats, group=None, fut_off=None, full=False):
            """Log of expected sales relative to the item's scale. With `full`,
            also the baseline part, the state and the item-to-item weights."""
            h = fut.shape[1]
            state = self.state(hist)
            labels = [e(cats[:, i]) for i, e in enumerate(self.emb)]
            fixed = torch.cat([state] + labels, dim=1)[:, None, :].expand(-1, h, -1)
            steps = self.step(torch.arange(h, device=fut.device))[None].expand(len(fut), -1, -1)
            rest, w = [torch.cat(labels, dim=1)[:, None, :].expand(-1, h, -1)] if labels else [], None
            rest.append(steps)
            if self.neighbours:
                got, w = self.context(fixed, fut, group)
                rest.append(got)
            still = state[:, None, :].expand(-1, h, -1)
            # with rollout each period is read from the state as it stands after the plan so far
            states = lambda plan: self.roll(state, plan) if self.rollout else still
            run = lambda net, plan, st: net(torch.cat([st] + rest + [plan], dim=2)).squeeze(-1)
            world = effect = None
            if self.plan_split:
                world, effect = self.roll_split(state, fut, fut_off)
                on = world + effect
            else:
                on = states(fut)
            if self.lift is None:
                base = out = run(self.head, fut, on).clamp(-10, 10)
            else:
                off = world if self.plan_split else states(fut_off)
                base = run(self.head, fut_off, off).clamp(-10, 10)
                out = (base + run(self.lift, fut, on) - run(self.lift, fut_off, off)).clamp(-10, 10)
            if not full:
                return out
            return {"log_mu": out, "base": base, "state": state, "weights": w, "rolled": on if self.rollout else None,
                    "world": world, "effect": effect}

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
        group_by: list[str] | None = None,
        neighbours: str | None = None,
        conditions: int = 4,
        regular_price: bool = False,
        regular_window: int = 12,
        levers: list[str] | None = None,
        lift_readout: bool = False,
        latent_weight: float = 0.0,
        plan_lags: int = 0,
        year_ago: int = 0,
        year_ago_window: int = 1,
        regular_column: str | None = None,
        pretrain_steps: int = 0,
        finetune: str = "full",
        encoder_lr_scale: float = 0.1,
        latent_ema: float = 0.99,
        regular_hold: int = 0,
        validation_periods: int = 0,
        total_weight: float = 0.0,
        readout_dropout: float = 0.0,
        weight_decay: float = 1e-4,
        rollout: bool = False,
        likelihood: str = "tweedie",
        plan_split: bool = False,
        split_weight: float = 0.1,
        state_spread_weight: float = 0.0,
        rollout_discount: float = 1.0,
        cold_start: bool = False,
        cold_neighbours: int = 10,
        unknown_label_rate: float = 0.0,
    ):
        self._settings = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.horizon, self.history, self.train_periods = horizon, history, train_periods
        self.categorical, self.extra = categorical or [], extra or []
        if neighbours not in (None, "similarity", "attention", "both"):
            raise ValueError(f"unknown neighbours setting: {neighbours}")
        if neighbours and not group_by:
            raise ValueError("neighbours needs group_by: which items can affect each other")
        if neighbours and d_model % conditions:
            raise ValueError("d_model must divide evenly by conditions")
        if lift_readout and neighbours:
            raise ValueError("lift_readout with neighbours is not supported yet")
        self.group_by = group_by or []
        self.regular_price, self.regular_window = regular_price, regular_window
        self.levers = list(self.extra) if levers is None else levers
        if set(self.levers) - set(self.extra):
            raise ValueError("every lever must also be listed in extra")
        self.latent_weight = latent_weight
        self.plan_lags = plan_lags
        self.regular_column = regular_column
        self.regular_hold = regular_hold
        if validation_periods and validation_periods < horizon:
            raise ValueError("validation_periods must be at least the horizon")
        self.validation_periods, self.total_weight = validation_periods, total_weight
        if likelihood not in ("tweedie", "negative_binomial"):
            raise ValueError("likelihood must be tweedie or negative_binomial")
        self.likelihood = likelihood
        if plan_split and not rollout:
            raise ValueError("plan_split needs rollout")
        self.plan_split, self.split_weight = plan_split, split_weight
        self.state_spread_weight, self.rollout_discount = state_spread_weight, rollout_discount
        if cold_start and neighbours:
            raise ValueError("cold_start with neighbours is not supported yet")
        self.cold_start, self.cold_neighbours = cold_start, cold_neighbours
        self.unknown_label_rate = unknown_label_rate
        self.weight_decay = weight_decay
        if finetune not in ("full", "low_lr", "frozen"):
            raise ValueError("finetune must be full, low_lr or frozen")
        self.pretrain_steps, self.finetune = pretrain_steps, finetune
        self.encoder_lr_scale, self.latent_ema = encoder_lr_scale, latent_ema
        # periods back to look for "this time last year", e.g. 51, 52, 53
        self._year_offsets = (
            [year_ago + d for d in range(-year_ago_window, year_ago_window + 1)] if year_ago else []
        )
        if self._year_offsets and min(self._year_offsets) < horizon:
            raise ValueError("year_ago minus its window must be at least the horizon")
        # how many past periods a forecast needs to keep
        self._keep = max([history] + self._year_offsets)
        self.net_args = dict(d_model=d_model, layers=layers, heads=heads, dropout=dropout,
                             neighbours=neighbours, conditions=conditions,
                             lift_readout=lift_readout, latent=latent_weight > 0 or pretrain_steps > 0,
                             readout_dropout=readout_dropout, rollout=rollout, plan_split=plan_split,
                             spread=likelihood == "negative_binomial")
        self.steps, self.batch, self.lr = steps, batch, lr
        self.power, self.seed, self.device = tweedie_power, seed, device
        self.mixed_precision = mixed_precision  # faster on a GPU; has no effect on a CPU

    def _tensors(self, frame: pd.DataFrame, dates: np.ndarray, with_units: bool, names=None):
        """Series-by-date arrays for one stretch of dates."""
        import torch

        names = self._names if names is None else names
        to = lambda a: torch.as_tensor(a, device=self._dev)
        out = {"calendar": to(_calendar(dates))}
        if with_units:
            out["units"] = to(np.nan_to_num(frame_to_matrix(frame, names, dates, UNITS)))
        if self._has_price:
            out["price"] = to(frame_to_matrix(frame, names, dates, PRICE))
        else:
            out["price"] = torch.full((len(names), len(dates)), float("nan"), device=self._dev)
        out["extra"] = [to(frame_to_matrix(frame, names, dates, c)) for c in self.extra]
        if self.regular_column:
            out["regular"] = to(frame_to_matrix(frame, names, dates, self.regular_column))
        return out

    def _window(self, past: dict, future: dict, rows, start, fut_rows=None):
        """Model inputs for the items in `rows`: the `history` columns of `past`
        ending just before column `start`, and `horizon` columns of `future`
        beginning at `future["start"]`. Also returns each item's scale, which
        future periods the item is on sale in, and the future column positions and the
        levers-off version of the future inputs."""
        import torch

        L = self.history
        back = start[:, None] + torch.arange(-L, 0, device=self._dev)[None]
        r = rows[:, None]
        fr = r if fut_rows is None else fut_rows[:, None]  # the plan may belong to other series (new items)
        units, price = past["units"][r, back], past["price"][r, back]
        known = ~torch.isnan(price)
        n_known = known.sum(1, keepdim=True).clamp(min=1)
        # each item's own recent level, over the periods it was on sale
        scale = (units * known).sum(1, keepdim=True) / n_known
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        level = torch.nan_to_num(price).sum(1, keepdim=True) / n_known
        level = torch.where(level > 0, level, torch.ones_like(level))

        # regular price: the highest price in the last `regular_window` periods
        recent = torch.nan_to_num(price[:, -self.regular_window :], nan=float("-inf")).amax(1, keepdim=True)
        regular = torch.where(torch.isfinite(recent) & (recent > 0), recent, level)
        if self.regular_hold:
            # a price held for `regular_hold` periods in a row is the regular
            # price from then on, so a lasting price change is not read as a
            # promotion; the latest such price wins
            run = torch.ones_like(price)
            for j in range(1, L):
                same = (price[:, j] - price[:, j - 1]).abs() <= 0.01 * price[:, j - 1]  # False where either is missing
                run[:, j] = torch.where(same, run[:, j - 1] + 1, run[:, j])
            held = (run >= self.regular_hold) & known
            last = (held * torch.arange(1, L + 1, device=self._dev)[None]).amax(1, keepdim=True)
            regular = torch.where(last > 0, torch.nan_to_num(price).gather(1, (last - 1).clamp(min=0)), regular)

        def lever_inputs(src, cols, off=False, r=r):
            p = src["price"][r, cols]
            k = ~torch.isnan(p)
            reg = regular.expand_as(p)
            if self.regular_column:  # the recorded regular price, where there is one
                given = src["regular"][r, cols]
                reg = torch.where(torch.isnan(given) | (given <= 0), reg, given)
            if off:
                p = torch.where(k, reg, p)
            feats = [torch.nan_to_num(p) / level, k.float()]
            for name, e in zip(self.extra, src["extra"]):
                v = e[r, cols]
                known = ~torch.isnan(v)
                if off and name in self.levers:
                    v = torch.zeros_like(v)
                feats += [torch.nan_to_num(v), known.float()]
            if self.regular_price:
                feats.append(torch.nan_to_num(p) / reg)
            cal = src["calendar"][cols]
            return torch.cat([torch.stack(feats, dim=2), cal], dim=2), k

        hist_levers, _ = lever_inputs(past, back)
        hist = torch.cat([(units / scale)[..., None], torch.log1p(units)[..., None], hist_levers], dim=2)
        steps = torch.arange(self.horizon, device=self._dev)[None]
        ahead = future["start"][:, None] + steps
        fut, on_sale = lever_inputs(future, ahead, r=fr)
        fut_off, _ = lever_inputs(future, ahead, off=True, r=fr)

        def earlier(offset, off=False, with_units=False):
            """Inputs for the period `offset` before each forecast period.
            That period is in the plan when it falls inside the horizon and
            in history otherwise; before the start of the data it is "not
            known". Calendar columns are left out."""
            rel = steps - offset  # position relative to the first forecast period
            in_plan = (rel >= 0).expand(len(rows), -1)
            f_cols = (future["start"][:, None] + rel).clamp(min=0)
            p_cols = start[:, None] + rel
            seen = (p_cols >= 0) | in_plan
            p_cols = p_cols.clamp(0, past["price"].shape[1] - 1)
            from_plan = lever_inputs(future, f_cols, off, r=fr)[0][..., :-4]
            from_past = lever_inputs(past, p_cols)[0][..., :-4]
            out = torch.where(in_plan[..., None], from_plan, from_past) * seen[..., None]
            if with_units:  # only ever asked for periods that are history
                u = past["units"][r, p_cols] * seen
                out = torch.cat([out, (u / scale)[..., None], seen.float()[..., None]], dim=2)
            return out

        extra, extra_off = [], []
        for i in range(1, self.plan_lags + 1):  # what was planned just before, for pantry loading
            extra.append(earlier(i))
            extra_off.append(earlier(i, off=True))
        for lag in self._year_offsets:  # the same period a year ago, and its neighbours
            both = earlier(lag, with_units=True)
            extra.append(both)
            extra_off.append(both)
        if extra:
            fut = torch.cat([fut] + extra, dim=2)
            fut_off = torch.cat([fut_off] + extra_off, dim=2)
        return hist, fut, scale, on_sale, ahead, fut_off

    def _new_net(self):
        q = 2 + 2 * len(self.extra) + int(self.regular_price)  # inputs describing one period's plan
        n_fut = q + 4 + self.plan_lags * q + len(self._year_offsets) * (q + 2)
        return _build_net(
            2 + q + 4, n_fut, [len(v) for v in self._levels.values()], self.history, self.horizon,
            n_products=len(self._products), **self.net_args,
        ).to(self._dev)

    def _forward(self, past, future, rows, start, picked=None, full=False, fut_rows=None, cats=None):
        """Run the network for `rows`. `picked` is the group layout (groups by
        slots, -1 for an empty slot) when rows were drawn as whole groups.
        Returns log expected sales (relative to scale), scale, which periods
        count, and the future column positions."""
        hist, fut, scale, on_sale, ahead, fut_off = self._window(past, future, rows, start, fut_rows)
        group = None
        if picked is not None:
            on_sale = on_sale & (picked.reshape(-1) >= 0)[:, None]
            group = {"shape": tuple(picked.shape), "product": self._product[rows], "open": on_sale}
        cats = self._cats[rows] if cats is None else cats
        out = self._net(hist, fut, cats, group if self._net.neighbours else None, fut_off, full)
        if full:
            out["plan"] = fut
        return out, scale, on_sale, ahead

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

        self._stores = first[STORE].astype(str).to_numpy()
        self._products = pd.Index(first[ITEM].astype(str).unique()).sort_values()
        self._product = torch.as_tensor(self._products.get_indexer(first[ITEM].astype(str)), device=self._dev)
        self._members = None
        if self.group_by:
            key = first[self.group_by].astype(str).agg("|".join, axis=1).to_numpy()
            lists = [np.flatnonzero(key == k) for k in np.unique(key)]
            members = np.full((len(lists), max(map(len, lists))), -1, dtype=np.int64)
            for i, m in enumerate(lists):
                members[i, : len(m)] = m
            self._members = torch.as_tensor(members, device=self._dev)

        self._net = self._new_net()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        amp = self.mixed_precision and self._dev.type == "cuda"
        n, t, p = len(self._names), len(dates), self.power
        uses_latent = self.latent_weight > 0 or self.pretrain_steps > 0
        slow = None
        if uses_latent:  # slowly updated copy of the network, used only for target states
            import copy

            slow = copy.deepcopy(self._net).requires_grad_(False)
        encoder = [q for part in (self._net.inp, self._net.encoder, self._net.norm) for q in part.parameters()]
        encoder.append(self._net.pos)
        self.loss_log, self.validation_log = [], []
        V = self.validation_periods
        last = t - H - V  # latest start whose forecast periods are all open to training
        if last < L:
            raise ValueError("not enough history left to train on after the validation periods")
        best = {"score": float("inf"), "step": 0, "weights": None}

        def train(steps, forecast, latent_weight, encoder_lr, label):
            """One stretch of training. `forecast` switches the forecast loss
            on; `encoder_lr` is the encoder's learning rate as a share of
            the rest (0 freezes it)."""
            if steps == 0:
                return
            ids = {id(q) for q in encoder}
            rest = [q for q in self._net.parameters() if id(q) not in ids]
            for q in encoder:
                q.requires_grad_(encoder_lr > 0)
            groups = [{"params": rest, "lr": self.lr}]
            if encoder_lr > 0:
                groups.append({"params": encoder, "lr": self.lr * encoder_lr})
            opt = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
            warm = max(1, steps // 20)  # learning rate rises for the first 5% of steps, then falls away
            shape = lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + np.cos(np.pi * (k - warm) / max(1, steps - warm)))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, shape)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            with_latent = latent_weight > 0
            self._net.train()
            for step in range(steps):
                if self._members is None:
                    picked = None
                    rows = torch.randint(0, n, (self.batch,), generator=gen).to(self._dev)
                    if self.total_weight > 0:  # a few shared dates, so totals within the batch mean something
                        dates_in_batch = torch.randint(L, last + 1, (TOTAL_DATES,), generator=gen)
                        start = dates_in_batch.repeat_interleave(-(-self.batch // TOTAL_DATES))[: self.batch].to(self._dev)
                    else:
                        start = torch.randint(L, last + 1, (self.batch,), generator=gen).to(self._dev)
                else:  # whole groups, each at one point in time
                    n_groups, slots = self._members.shape
                    g = max(1, self.batch // slots)
                    picked = self._members[torch.randint(0, n_groups, (g,), generator=gen).to(self._dev)]
                    start = torch.randint(L, last + 1, (g,), generator=gen).to(self._dev).repeat_interleave(slots)
                    rows = picked.reshape(-1).clamp(min=0)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    cats = None
                    if self.unknown_label_rate > 0 and ITEM in self.categorical:
                        # now and then hide which item it is, so the model learns to
                        # forecast from the other labels alone (needed for new items)
                        cats = self._cats[rows].clone()
                        hide = torch.rand(len(rows), generator=gen).to(self._dev) < self.unknown_label_rate
                        cats[hide, self.categorical.index(ITEM)] = 0
                    out, scale, on_sale, ahead = self._forward(
                        data, {**data, "start": start}, rows, start, picked, full=True, cats=cats
                    )
                    latent = 0.0
                    if with_latent:
                        # the state once the horizon has become history, from the slow copy
                        later = start + H
                        with torch.no_grad():
                            hist_later = self._window(data, {**data, "start": start}, rows, later)[0]
                            want = slow.state(hist_later)
                        far = lambda a, b: (1 - torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)).mean()
                        if self._net.rollout:
                            # each rolled state must match the state the encoder gives once
                            # those periods are history; checked at the end and at two steps on the way
                            # with a discount below 1, near periods count for more than far ones
                            rho = self.rollout_discount
                            latent, weight = rho ** (H - 1) * far(out["rolled"][:, H - 1], want), rho ** (H - 1)
                            for k in torch.randint(1, H, (2,), generator=gen).tolist() if H > 1 else []:
                                with torch.no_grad():
                                    seen = slow.state(self._window(data, {**data, "start": start}, rows, start + k)[0])
                                latent = latent + rho ** (k - 1) * far(out["rolled"][:, k - 1], seen)
                                weight = weight + rho ** (k - 1)
                            latent = latent / weight
                        else:
                            guess = self._net.next(torch.cat([out["state"], out["plan"].flatten(1)], dim=1))
                            latent = far(guess, want)
                loss = latent_weight * latent
                if self.plan_split and self.split_weight > 0:
                    # keep what the plan adds apart from what would have happened anyway
                    w_, e_ = out["world"].float().flatten(0, 1), out["effect"].float().flatten(0, 1)
                    live = e_.abs().sum(1) > 0
                    if live.any():
                        cos = torch.nn.functional.cosine_similarity(w_[live], e_[live], dim=1)
                        loss = loss + self.split_weight * cos.abs().mean()
                if self.state_spread_weight > 0:
                    # stop the state shrinking onto a few directions: every direction
                    # keeps some spread across items, and directions do not copy each other
                    z = out["state"].float()
                    z = z - z.mean(0)
                    spread = torch.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()
                    cov = (z.T @ z) / max(len(z) - 1, 1)
                    copy = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / z.shape[1]
                    loss = loss + self.state_spread_weight * (spread + 0.04 * copy)
                if forecast:
                    log_mu = out["log_mu"].float()
                    target = data["units"][rows[:, None], ahead] / scale
                    if self.likelihood == "tweedie":
                        # Tweedie loss on sales relative to the item's own level
                        each = -target * torch.exp((1 - p) * log_mu) / (1 - p) + torch.exp((2 - p) * log_mu) / (2 - p)
                    else:
                        # negative binomial on units: a count with mean m and a learned spread
                        y = data["units"][rows[:, None], ahead].round()
                        m = (torch.exp(log_mu) * scale).clamp(min=1e-6)
                        r = 1 / torch.nn.functional.softplus(self._net.spread).clamp(min=1e-4)
                        each = -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
                                 + r * torch.log(r / (r + m)) + y * torch.log(m / (r + m)))
                    loss = loss + (each * on_sale).sum() / on_sale.sum().clamp(min=1)
                    if self.total_weight > 0:
                        # the same forecast summed over the items drawn for each date, against
                        # their summed sales: a bias shared by many items shows up here
                        units_f = torch.exp(log_mu) * scale * on_sale
                        units_y = data["units"][rows[:, None], ahead] * on_sale
                        same = (start[:, None] == start.unique()[None, :]).float()  # rows by dates
                        gap = (same.T @ units_f - same.T @ units_y) / (same.T @ units_y).clamp(min=1.0)
                        loss = loss + self.total_weight * (gap**2).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if slow is not None:
                    with torch.no_grad():
                        for a, b in zip(slow.parameters(), self._net.parameters()):
                            a.lerp_(b, 1 - self.latent_ema)
                if step % 50 == 0 or step == steps - 1:
                    self.loss_log.append((label, step, loss.item()))
                if forecast and V and ((step + 1) % max(1, steps // 20) == 0 or step == steps - 1):
                    score = self._validate(data, t - V)
                    self.validation_log.append((step + 1, score))
                    if score < best["score"]:
                        best.update(score=score, step=step + 1,
                                    weights={k: v.detach().clone() for k, v in self._net.state_dict().items()})
                    print(f"validation after step {step + 1}: {score:.4f} (best {best['score']:.4f} at {best['step']})", flush=True)
                    self._net.train()
                if (step + 1) % max(1, steps // 20) == 0:
                    print(f"{label} step {step + 1} of {steps}, loss {loss.item():.4f}", flush=True)

        # optional first stretch: learn the state from the latent loss alone
        train(self.pretrain_steps, False, 1.0, 1.0, "pretraining")
        encoder_lr = {"full": 1.0, "low_lr": self.encoder_lr_scale, "frozen": 0.0}[self.finetune]
        train(self.steps, True, self.latent_weight, encoder_lr if self.pretrain_steps else 1.0, "training")
        for q in encoder:
            q.requires_grad_(True)
        if best["weights"] is not None:  # go back to the point that did best on the validation periods
            self._net.load_state_dict(best["weights"])
        self.best_step = best["step"]

        tail = slice(max(0, t - self._keep), t)
        self._past = {
            "units": data["units"][:, tail],
            "price": data["price"][:, tail],
            "extra": [e[:, tail] for e in data["extra"]],
            **({"regular": data["regular"][:, tail]} if self.regular_column else {}),
            "calendar": data["calendar"][tail],
        }
        return self

    def _validate(self, data, origin: int) -> float:
        """Error on the `horizon` periods from column `origin`, which training
        never used as targets. Averaged over three levels (single series,
        each `categorical` grouping, and the total) so that a bias which adds
        up counts as much as single-series error."""
        import torch

        n, H = len(self._names), self.horizon
        f, y = torch.zeros(n, H, device=self._dev), torch.zeros(n, H, device=self._dev)
        self._net.eval()
        with torch.no_grad():
            for rows, picked in self._batches():
                start = torch.full_like(rows, origin)
                log_mu, scale, on_sale, ahead = self._forward(data, {**data, "start": start}, rows, start, picked)
                real = on_sale if picked is None else on_sale & (picked.reshape(-1) >= 0)[:, None]
                f[rows] += torch.exp(log_mu.float()) * scale * real
                y[rows] += data["units"][rows[:, None], ahead] * real
        level = lambda a, b: (((a - b) ** 2).mean().sqrt() / b.abs().mean().clamp(min=1e-9)).item()
        scores = [level(f, y), level(f.sum(0), y.sum(0))]
        for i in range(self._cats.shape[1]):
            g = self._cats[:, i]
            sums = lambda a: torch.zeros(int(g.max()) + 1, H, device=self._dev).index_add_(0, g, a)
            scores.append(level(sums(f), sums(y)))
        return float(np.mean(scores))

    def _future(self, future: pd.DataFrame):
        fut_dates = np.sort(future[DATE].unique())
        h = len(fut_dates)
        if h > self.horizon:
            raise ValueError("asked to forecast further ahead than the model was built for")
        pad = fut_dates[-1] + (np.arange(1, self.horizon - h + 1) * (fut_dates[-1] - fut_dates[-2] if h > 1 else np.timedelta64(1, "D")))
        return fut_dates, self._tensors(future, np.r_[fut_dates, pad], with_units=False)

    def _batches(self):
        """Rows to forecast together, with their group layout if there is one."""
        import torch

        if self._members is None:
            n = len(self._names)
            for lo in range(0, n, 4096):
                yield torch.arange(lo, min(lo + 4096, n), device=self._dev), None
        else:
            per = max(1, 4096 // self._members.shape[1])
            for lo in range(0, len(self._members), per):
                picked = self._members[lo : lo + per]
                yield picked.reshape(-1).clamp(min=0), picked

    def _predict(self, future: pd.DataFrame, key: str) -> np.ndarray:
        import torch

        fut_dates, fut = self._future(future)
        out = np.zeros((len(self._names), self.horizon), dtype=np.float64)
        self._net.eval()
        with torch.no_grad():
            for rows, picked in self._batches():
                start = torch.full_like(rows, self._past["price"].shape[1])
                got, scale, _, _ = self._forward(
                    self._past, {**fut, "start": torch.zeros_like(rows)}, rows, start, picked, full=True
                )
                real = torch.ones_like(rows, dtype=torch.bool) if picked is None else picked.reshape(-1) >= 0
                out[rows[real].cpu().numpy()] = (torch.exp(got[key]) * scale)[real].cpu().numpy()
        r = series_rows(future[SERIES], self._names)
        c = pd.Index(fut_dates).get_indexer(future[DATE])
        pred = np.where(r >= 0, out[np.clip(r, 0, None), c], 0.0)
        if self.cold_start and (r < 0).any():
            new = future[r < 0]
            pred[r < 0] = self._new_series(new, fut_dates, key)
        return np.clip(pred, 0, None)

    def _new_series(self, new: pd.DataFrame, fut_dates: np.ndarray, key: str) -> np.ndarray:
        """Forecasts for series the model has no history for. Each borrows the
        state and scale of the known series most like it in the same store
        (most labels in common), is run with its own labels and its own plan,
        and the results are averaged."""
        import torch

        names = pd.Index(new[SERIES].astype(str).unique())
        first = new.drop_duplicates(SERIES)
        first = first.set_index(first[SERIES].astype(str)).loc[names]
        cats = np.stack(
            [self._levels[c].get_indexer(first[c].astype(str)) + 1 for c in self.categorical], axis=1
        ) if self.categorical else np.zeros((len(names), 0), dtype=np.int64)
        known, stores, K = self._cats.cpu().numpy(), first[STORE].astype(str).to_numpy(), self.cold_neighbours
        picks = np.zeros((len(names), K), dtype=np.int64)
        for i in range(len(names)):
            pool = np.flatnonzero(self._stores == stores[i])
            if len(pool) == 0:
                pool = np.arange(len(self._names))
            alike = ((known[pool] == cats[i]) & (cats[i] > 0)).sum(1)
            best = pool[np.argsort(-alike, kind="stable")[:K]]
            picks[i] = np.resize(best, K)
        pad = fut_dates[-1] + (np.arange(1, self.horizon - len(fut_dates) + 1) * (fut_dates[-1] - fut_dates[-2] if len(fut_dates) > 1 else np.timedelta64(1, "D")))
        plan = self._tensors(new, np.r_[fut_dates, pad], with_units=False, names=names)
        out = np.zeros((len(names), self.horizon))
        self._net.eval()
        with torch.no_grad():
            for lo in range(0, len(names), max(1, 4096 // K)):
                hi = min(lo + max(1, 4096 // K), len(names))
                rows = torch.as_tensor(picks[lo:hi].reshape(-1), device=self._dev)
                own = torch.arange(lo, hi, device=self._dev).repeat_interleave(K)
                labels = torch.as_tensor(cats[lo:hi], device=self._dev, dtype=torch.long).repeat_interleave(K, dim=0)
                start = torch.full_like(rows, self._past["price"].shape[1])
                got, scale, _, _ = self._forward(
                    self._past, {**plan, "start": torch.zeros_like(rows)}, rows, start, None, full=True,
                    fut_rows=own, cats=labels,
                )
                units = (torch.exp(got[key]) * scale).view(hi - lo, K, -1).mean(1)
                out[lo:hi] = units.cpu().numpy()
        return out[names.get_indexer(new[SERIES].astype(str)), pd.Index(fut_dates).get_indexer(new[DATE])]

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return self._predict(future, "log_mu")

    def breakdown(self, future: pd.DataFrame) -> pd.DataFrame:
        """Each forecast split in two: `baseline`, what the item would sell at
        its regular price with no levers, and `lift`, what the plan adds or
        takes away. The two sum to the forecast."""
        if self._net.lift is None:
            raise ValueError("this model was not built with lift_readout")
        forecast, baseline = self.predict(future), self._predict(future, "base")
        return pd.DataFrame({"forecast": forecast, "baseline": baseline, "lift": forecast - baseline}, index=future.index)

    def neighbour_weights(self, future: pd.DataFrame) -> pd.DataFrame:
        """The weights behind a forecast: one row per receiving series, sending
        series, condition and date. Pairs with no weight are left out."""
        import torch

        if not self._net.neighbours:
            raise ValueError("this model has no item-to-item part")
        fut_dates, fut = self._future(future)
        names, out = self._names.to_numpy(), []
        self._net.eval()
        with torch.no_grad():
            for rows, picked in self._batches():
                start = torch.full_like(rows, self._past["price"].shape[1])
                got, _, _, _ = self._forward(
                    self._past, {**fut, "start": torch.zeros_like(rows)}, rows, start, picked, full=True
                )
                w = got["weights"][:, : len(fut_dates)].cpu().numpy()  # groups, dates, conditions, receiver, sender
                g, d, k, i, j = np.nonzero(w > 0)
                p = picked.cpu().numpy()
                out.append(pd.DataFrame({
                    "series": names[p[g, i]], "neighbour": names[p[g, j]], "condition": k,
                    DATE: fut_dates[d], "weight": w[g, d, k, i, j],
                }))
        return pd.concat(out, ignore_index=True)

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
                "products": list(self._products),
                "stores": list(self._stores),
                "product": cpu(self._product),
                "members": None if self._members is None else cpu(self._members),
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
        model._products = pd.Index(saved["products"])
        model._stores = np.array(saved.get("stores", []), dtype=object)
        model._product = saved["product"].to(model._dev)
        model._members = None if saved["members"] is None else saved["members"].to(model._dev)
        model._net = model._new_net()
        model._net.load_state_dict(saved["weights"])
        return model
