import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.forecaster import build_model
from rwm.model.state_model import _calendar


def test_weekday_flags_put_sunday_no_closer_to_monday_than_to_thursday():
    days = pd.date_range("2016-05-23", periods=7).to_numpy()  # Monday to Sunday
    smooth, flags = _calendar(days), _calendar(days, weekday_flags=True)
    d = lambda c, a, b: np.linalg.norm(c[a] - c[b])
    assert d(smooth, 6, 0) < d(smooth, 6, 3)  # on the cycle Sunday sits next to Monday
    assert np.isclose(d(flags[:, :7], 6, 0), d(flags[:, :7], 6, 3))  # as flags every pair of days is equally apart
    assert flags.shape == (7, 9) and (flags[:, :7].sum(1) == 1).all()


def test_world_model_trains_and_forecasts_with_weekday_flags():
    p = load_dataset("synthetic", n_stores=2, n_items=3, n_periods=120, seed=1).panel
    cut = np.sort(p[DATE].unique())[-9]
    train, test = p[p[DATE] <= cut], p[p[DATE] > cut]
    m = build_model("state_model", horizon=8, history=16, train_periods=90, categorical=["item_id", "store_id"],
                    extra=["promo"], d_model=16, layers=1, heads=2, steps=20, batch=32, device="cpu",
                    rollout=True, roll_norm=True, weekday_flags=True).fit(train)
    f = m.predict(test.drop(columns=[UNITS]))
    assert len(f) == len(test) and np.isfinite(f).all()


def test_world_model_trains_with_the_longer_run_level():
    p = load_dataset("synthetic", n_stores=2, n_items=3, n_periods=160, seed=1).panel
    cut = np.sort(p[DATE].unique())[-9]
    train, test = p[p[DATE] <= cut], p[p[DATE] > cut]
    m = build_model("state_model", horizon=8, history=16, train_periods=140, categorical=["item_id", "store_id"],
                    extra=["promo"], d_model=16, layers=1, heads=2, steps=20, batch=32, device="cpu",
                    rollout=True, roll_norm=True, weekday_flags=True, long_level=56).fit(train)
    f = m.predict(test.drop(columns=[UNITS]))
    assert len(f) == len(test) and np.isfinite(f).all()
