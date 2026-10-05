"""The world-model form: the state rolled forward period by period under a plan."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.forecaster import build_model

SETTINGS = dict(
    horizon=7, history=28, train_periods=330, categorical=["item_id", "store_id"], extra=["promo"],
    d_model=32, layers=1, heads=4, steps=500, batch=128, lr=3e-3, device="cpu",
    regular_price=True, lift_readout=True, rollout=True,
)


@pytest.fixture(scope="module")
def split():
    # the period after its own promotion an item sells 50% less
    p = load_dataset("synthetic", n_stores=3, n_items=6, n_periods=400, seed=4, dip=0.5).panel
    cut = np.sort(p[DATE].unique())[-8]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted(split):
    return build_model("state_model", **SETTINGS).fit(split[0])


def _plans(train, test):
    """Nothing planned, and a promotion in the first period only."""
    regular = train.groupby("series_id")["price"].max()
    off = test.drop(columns=[UNITS]).assign(promo=0)
    off["price"] = off["series_id"].map(regular).to_numpy()
    first = (off[DATE] == off[DATE].min()).to_numpy()
    on = off.assign(promo=first.astype(int), price=np.where(first, (off["price"] * 0.8).round(2), off["price"]))
    return off, on, first


def test_lift_is_zero_when_nothing_is_planned_and_the_baseline_ignores_the_plan(split, fitted):
    off, on, _ = _plans(*split)
    a, b = fitted.breakdown(off), fitted.breakdown(on)
    np.testing.assert_allclose(a["lift"], 0, atol=1e-4 * a["forecast"].max())
    np.testing.assert_allclose(a["baseline"], b["baseline"], rtol=1e-5)


def test_a_promotion_raises_its_own_period_and_lowers_the_next_through_the_state(split, fitted):
    """No earlier-period inputs are switched on: the only route from the
    first period's plan to the second period's forecast is the rolled state."""
    off, on, first = _plans(*split)
    second = (off[DATE] == np.sort(off[DATE].unique())[1]).to_numpy()
    a, b = fitted.predict(off), fitted.predict(on)
    assert b[first].sum() / a[first].sum() > 1.4
    assert b[second].sum() / a[second].sum() < 0.8


def test_without_rollout_a_plan_cannot_reach_a_later_period(split):
    still = build_model("state_model", **{**SETTINGS, "rollout": False, "steps": 5}).fit(split[0])
    off, on, first = _plans(*split)
    np.testing.assert_allclose(still.predict(on)[~first], still.predict(off)[~first], rtol=1e-5)


@pytest.mark.parametrize("more", [dict(latent_weight=0.5), dict(pretrain_steps=40, finetune="low_lr")])
def test_latent_check_on_the_rolled_states_trains_and_repeats(split, more):
    train, test = split
    quick = {**SETTINGS, "steps": 80, **more}
    future = test.drop(columns=[UNITS])
    a = build_model("state_model", **quick).fit(train).predict(future)
    b = build_model("state_model", **quick).fit(train).predict(future)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_saved_world_model_gives_the_same_forecast_when_loaded(split, fitted, tmp_path):
    from rwm.model.state_model import StateModel

    future = split[1].drop(columns=[UNITS])
    fitted.save(tmp_path / "model.pt")
    np.testing.assert_array_equal(
        fitted.predict(future), StateModel.load(tmp_path / "model.pt", device="cpu").predict(future)
    )
