import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.forecaster import build_model
from rwm.prior_work.lightgbm_direct import _rolling, _shift

SETTINGS = dict(
    horizon=28, train_periods=250, group_by="store_id", categorical=["item_id"],
    extra=["promo"], rounds=150, params={"min_data_in_leaf": 20},
)


@pytest.fixture(scope="module")
def split():
    p = load_dataset("synthetic", n_stores=2, n_items=6, n_periods=400, seed=1).panel
    cut = np.sort(p[DATE].unique())[-29]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


def test_rolling_and_shift():
    x = np.arange(10, dtype=np.float32)[None, :]
    assert np.isnan(_shift(x, 3)[0, :3]).all() and _shift(x, 3)[0, 3:].tolist() == list(range(7))
    mean, std = _rolling(x, 3)
    assert np.isnan(mean[0, :2]).all()
    np.testing.assert_allclose(mean[0, 2:], np.arange(1, 9))
    np.testing.assert_allclose(std[0, 2:], np.sqrt(2 / 3), rtol=1e-6)
    late = x.copy()
    late[0, 0] = np.nan
    assert np.isnan(_rolling(late, 3)[0][0, :3]).all()  # no average until 3 known values


def test_same_forecast_whatever_the_row_order_and_on_refit(split):
    train, test = split
    future = test.drop(columns=[UNITS])
    model = build_model("lightgbm_direct", **SETTINGS).fit(train)
    a = model.predict(future)
    shuffled = future.sample(frac=1, random_state=1)
    b = pd.Series(model.predict(shuffled), index=shuffled.index).sort_index().to_numpy()
    np.testing.assert_allclose(a, b)
    again = build_model("lightgbm_direct", **SETTINGS).fit(train).predict(future)
    np.testing.assert_array_equal(a, again)


def test_beats_seasonal_naive_on_generated_data(split):
    train, test = split
    future, y = test.drop(columns=[UNITS]), test[UNITS].to_numpy()
    rmse = lambda f: float(np.sqrt(((f - y) ** 2).mean()))
    lgb = rmse(build_model("lightgbm_direct", **SETTINGS).fit(train).predict(future))
    naive = rmse(build_model("seasonal_naive").fit(train).predict(future))
    assert lgb < naive


def test_refuses_to_forecast_beyond_its_horizon(split):
    train, test = split
    model = build_model("lightgbm_direct", **{**SETTINGS, "horizon": 28}).fit(train)
    longer = pd.concat([test, test.assign(**{DATE: test[DATE] + pd.Timedelta(days=28)})])
    with pytest.raises(ValueError):
        model.predict(longer.drop(columns=[UNITS]))
