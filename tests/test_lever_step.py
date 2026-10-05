"""The lever step: regular price, baseline and lift, and the latent loss."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.forecaster import build_model

SETTINGS = dict(
    horizon=7, history=28, train_periods=330, categorical=["item_id", "store_id"], extra=["promo"],
    d_model=32, layers=1, heads=4, steps=500, batch=128, lr=3e-3, device="cpu",
    regular_price=True, lift_readout=True,
)


@pytest.fixture(scope="module")
def split():
    # on promotion the price is 20% lower and sales are 80% higher
    p = load_dataset("synthetic", n_stores=3, n_items=6, n_periods=400, seed=3).panel
    cut = np.sort(p[DATE].unique())[-8]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted(split):
    return build_model("state_model", **SETTINGS).fit(split[0])


def _plans(train, test):
    """The same future with nothing planned, and with everything on promotion."""
    regular = train.groupby("series_id")["price"].max()
    off = test.drop(columns=[UNITS]).assign(promo=0)
    off["price"] = off["series_id"].map(regular).to_numpy()
    on = off.assign(promo=1, price=(off["price"] * 0.8).round(2))
    return off, on


def test_lift_is_zero_when_nothing_is_planned(split, fitted):
    off, _ = _plans(*split)
    parts = fitted.breakdown(off)
    np.testing.assert_allclose(parts["lift"], 0, atol=1e-4 * parts["forecast"].max())


def test_baseline_does_not_move_with_the_plan_and_lift_is_learned(split, fitted):
    off, on = _plans(*split)
    a, b = fitted.breakdown(off), fitted.breakdown(on)
    np.testing.assert_allclose(a["baseline"], b["baseline"], rtol=1e-5)
    np.testing.assert_allclose(b["baseline"] + b["lift"], b["forecast"], rtol=1e-6)
    ratio = b["forecast"].sum() / a["forecast"].sum()
    assert 1.5 < ratio < 2.1  # the true lift is 1.8


def test_forecast_inputs_equal_training_inputs_for_the_same_days(split):
    import pandas as pd

    from rwm.data.schema import SERIES

    train, test = split
    model = build_model("state_model", **{**SETTINGS, "steps": 1}).fit(train)
    future = test.drop(columns=[UNITS])
    fut_dates = np.sort(future[DATE].unique())
    rows = torch.arange(len(model._names))
    zero, start = torch.zeros_like(rows), torch.full_like(rows, model.history)
    fut = model._tensors(future, fut_dates, with_units=False)
    got = model._window(model._past, {**fut, "start": zero}, rows, start)
    full = pd.concat([train, test]).sort_values([SERIES, DATE]).reset_index(drop=True)
    dates = np.sort(full[DATE].unique())[-(model.history + 7):]
    data = model._tensors(full[full[DATE] >= dates[0]], dates, with_units=True)
    want = model._window(data, {**data, "start": start}, rows, start)
    for i in (0, 1, 2, 3, 5):  # history, plan, scale, on-sale flags, levers-off plan
        np.testing.assert_allclose(got[i].numpy(), want[i].numpy(), rtol=1e-6)


def test_latent_loss_trains_and_repeats(split):
    train, test = split
    quick = {**SETTINGS, "steps": 150, "latent_weight": 0.5}
    future = test.drop(columns=[UNITS])
    a = build_model("state_model", **quick).fit(train).predict(future)
    b = build_model("state_model", **quick).fit(train).predict(future)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_saved_model_gives_the_same_breakdown_when_loaded(split, fitted, tmp_path):
    from rwm.model.state_model import StateModel

    future = split[1].drop(columns=[UNITS])
    fitted.save(tmp_path / "model.pt")
    loaded = StateModel.load(tmp_path / "model.pt", device="cpu")
    np.testing.assert_array_equal(fitted.breakdown(future).to_numpy(), loaded.breakdown(future).to_numpy())


def test_a_column_not_named_as_a_lever_is_left_alone(split):
    train, test = split
    extra = lambda p: p.assign(shelf=1.0)
    model = build_model("state_model", **{**SETTINGS, "steps": 5, "extra": ["promo", "shelf"], "levers": ["promo"]})
    model.fit(extra(train))
    off, _ = _plans(extra(train), extra(test))
    np.testing.assert_allclose(model.breakdown(off)["lift"], 0, atol=1e-4)


@pytest.fixture(scope="module")
def dip_split():
    # the period after its own promotion an item sells 50% less
    p = load_dataset("synthetic", n_stores=3, n_items=6, n_periods=400, seed=4, dip=0.5).panel
    cut = np.sort(p[DATE].unique())[-8]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


def test_a_promotion_lowers_the_following_period(dip_split):
    train, test = dip_split
    model = build_model("state_model", **SETTINGS, plan_lags=1).fit(train)
    off, _ = _plans(train, test)
    first = off[DATE] == off[DATE].min()
    on = off.assign(promo=first.astype(int), price=np.where(first, (off["price"] * 0.8).round(2), off["price"]))
    second = (off[DATE] == np.sort(off[DATE].unique())[1]).to_numpy()
    a, b = model.breakdown(off), model.breakdown(on)
    assert b["forecast"][second].sum() / a["forecast"][second].sum() < 0.8
    np.testing.assert_allclose(a["baseline"], b["baseline"], rtol=1e-5)
    np.testing.assert_allclose(a["lift"], 0, atol=1e-4 * a["forecast"].max())


@pytest.mark.parametrize("more", [dict(plan_lags=2), dict(year_ago=52), dict(plan_lags=1, year_ago=52, history=80)])
def test_earlier_period_inputs_match_between_training_and_forecasting(split, more, tmp_path):
    import pandas as pd

    from rwm.data.schema import SERIES
    from rwm.model.state_model import StateModel

    train, test = split
    model = build_model("state_model", **{**SETTINGS, "steps": 1, **more}).fit(train)
    future = test.drop(columns=[UNITS])
    fut_dates = np.sort(future[DATE].unique())
    rows = torch.arange(len(model._names))
    keep = model._past["price"].shape[1]
    fut = model._tensors(future, fut_dates, with_units=False)
    got = model._window(model._past, {**fut, "start": torch.zeros_like(rows)}, rows, torch.full_like(rows, keep))
    full = pd.concat([train, test]).sort_values([SERIES, DATE]).reset_index(drop=True)
    dates = np.sort(full[DATE].unique())[-(keep + 7):]
    data = model._tensors(full[full[DATE] >= dates[0]], dates, with_units=True)
    start = torch.full_like(rows, keep)
    want = model._window(data, {**data, "start": start}, rows, start)
    for i in (0, 1, 2, 3, 5):
        np.testing.assert_allclose(got[i].numpy(), want[i].numpy(), rtol=1e-6)
    model.save(tmp_path / "m.pt")
    np.testing.assert_array_equal(model.predict(future), StateModel.load(tmp_path / "m.pt", device="cpu").predict(future))
