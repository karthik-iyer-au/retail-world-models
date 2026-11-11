"""A model fitted on the target data set together with other data sets (study U part b,
docs/PAPER1_REVISION_PLAN.md, entry of 11 November 2026): the pooled counterpart for LightGBM
of the world model's pooled pretraining, so both models see the same extra sales.

  model:
    name: pooled
    params:
      model: lightgbm_direct     # the model fitted on the pooled data
      params: {...}              # its own settings; `source` is added to its labels
      pool_with: [...]           # rwm.data.pool sources

Each source keeps its own sales history but its dates are moved, by whole weeks, so that its
last period falls on the target's last training period: a model that keeps only the latest
periods for training then keeps the latest periods of every data set. The calendar columns were
worked out before the move, from the source's own dates. Where the model fits one model per
store (`group_by`), the pooled fit is one model for all stores with the store as a label, as in
the transfer test, since per-store models would never share anything across data sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rwm.data.schema import DATE, PRICE, STORE, UNITS
from rwm.forecaster import Forecaster, build_model, register_model

TARGET = "target"


@register_model("pooled")
class Pooled(Forecaster):
    def __init__(self, model: str, pool_with: list[dict], params: dict | None = None, seed: int | None = None):
        p = dict(params or {})
        if seed is not None:  # a five-copy average gives each copy its own seed (rwm.seed_average)
            p["seed"] = seed
        if p.get("group_by") == STORE:
            p["group_by"] = None
            p["categorical"] = list(p.get("categorical", [])) + [STORE]
        p["categorical"] = list(p.get("categorical", [])) + ["source"]
        self.inner = build_model(model, **p)
        self.pool_with = list(pool_with)
        self._columns = [UNITS, PRICE] + list(p.get("extra", [])) + [c for c in p["categorical"] if c != "source"]

    def _sources(self, train: pd.DataFrame) -> list[pd.DataFrame]:
        from rwm.data.pool import load_source

        end = pd.Timestamp(train[DATE].max())
        step = int(np.median(np.diff(np.sort(train[DATE].unique())).astype("timedelta64[D]").astype(int)))  # days between periods
        out = []
        for spec in self.pool_with:
            frame = load_source(spec, self._columns, until=end)
            # the source's dates must fall on the target's own grid of dates: weekly data sets start their
            # weeks on different weekdays (Dominick's on Thursday), so a weekly source is moved by the exact
            # gap, which puts its last week on the target's last week; daily data move by whole weeks, which
            # keeps each day on its own weekday (11 November 2026: a shift by whole weeks alone left Frat's
            # weeks between Dominick's, so the pooled LightGBM read lags on a mixed date axis)
            shift = (end - pd.Timestamp(frame[DATE].max())).days
            if step < 7:
                shift = 7 * int(round(shift / 7))
            frame[DATE] = frame[DATE] + pd.Timedelta(days=shift)
            frame["source"] = spec.get("tag") or spec["name"]
            out.append(frame)
        return out

    def fit(self, train: pd.DataFrame) -> "Pooled":
        train = train.assign(source=TARGET)
        frames = [train] + self._sources(train)
        cols = sorted(set.intersection(*[set(f.columns) for f in frames]))
        self.inner.fit(pd.concat([f[cols] for f in frames], ignore_index=True))
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return self.inner.predict(future.assign(source=TARGET))

    def refresh(self, history: pd.DataFrame) -> "Pooled":
        self.inner.refresh(history.assign(source=TARGET))
        return self
