"""The item-to-item part of our model, in each of its three versions."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import build_model

SETTINGS = dict(
    horizon=7, history=28, train_periods=330, categorical=["item_id"], extra=["promo"],
    d_model=32, layers=1, heads=4, steps=600, batch=128, lr=3e-3, device="cpu", group_by=["store_id"],
)
KINDS = ["similarity", "attention", "both"]


@pytest.fixture(scope="module")
def split():
    # odd-numbered items lose 60% of sales when the item before them is on promotion
    p = load_dataset("synthetic", n_stores=6, n_items=4, n_periods=400, seed=2, steal=0.6).panel
    cut = np.sort(p[DATE].unique())[-8]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


@pytest.fixture(scope="module", params=KINDS)
def fitted(request, split):
    return build_model("state_model", **SETTINGS, neighbours=request.param).fit(split[0])


def _plans(test, promoted_item):
    """The same future twice: nothing on promotion, then one item on promotion."""
    off = test.drop(columns=[UNITS]).assign(promo=0)
    on = off.assign(promo=(off["item_id"] == promoted_item).astype(int))
    return off, on


def test_a_neighbours_promotion_lowers_the_forecast_of_the_item_it_steals_from(split, fitted):
    off, on = _plans(split[1], "I0")
    change = fitted.predict(on) / np.maximum(fitted.predict(off), 1e-9)
    victim, bystander = (off["item_id"] == "I1").to_numpy(), (off["item_id"] == "I3").to_numpy()
    assert change[victim].mean() < 0.8
    assert abs(change[bystander].mean() - 1) < abs(change[victim].mean() - 1) / 2


def test_without_the_item_to_item_part_a_neighbours_plan_changes_nothing(split):
    model = build_model("state_model", **{**SETTINGS, "steps": 5}).fit(split[0])
    off, on = _plans(split[1], "I0")
    other = (off["item_id"] != "I0").to_numpy()
    np.testing.assert_allclose(model.predict(on)[other], model.predict(off)[other], rtol=1e-5)


def test_weights_are_within_one_group_never_to_itself_and_sum_to_at_most_one(split, fitted):
    future = split[1].drop(columns=[UNITS])
    w = fitted.neighbour_weights(future)
    store = lambda s: s.str.split("_").str[0]
    assert (store(w["series"]) == store(w["neighbour"])).all()
    assert (w["series"] != w["neighbour"]).all()
    total = w.groupby(["series", "condition", DATE])["weight"].sum()
    assert total.max() <= 1 + 1e-5 and (w["weight"] > 0).all()


def test_same_forecast_whatever_the_row_order(split, fitted):
    future = split[1].drop(columns=[UNITS])
    shuffled = future.sample(frac=1, random_state=1)
    back = np.empty(len(future))
    back[shuffled.index.to_numpy()] = fitted.predict(shuffled)
    np.testing.assert_allclose(fitted.predict(future), back, rtol=1e-5)


def test_saved_model_gives_the_same_forecast_when_loaded(split, fitted, tmp_path):
    from rwm.model.state_model import StateModel

    future = split[1].drop(columns=[UNITS])
    fitted.save(tmp_path / "model.pt")
    np.testing.assert_array_equal(
        fitted.predict(future), StateModel.load(tmp_path / "model.pt", device="cpu").predict(future)
    )


def test_both_starts_out_identical_to_similarity(split):
    quick = {**SETTINGS, "steps": 0}
    future = split[1].drop(columns=[UNITS])
    a = build_model("state_model", **quick, neighbours="both").fit(split[0])
    b = build_model("state_model", **quick, neighbours="similarity").fit(split[0])
    b._net.load_state_dict(a._net.state_dict(), strict=False)
    np.testing.assert_allclose(a.predict(future), b.predict(future), rtol=1e-5)


def test_groups_of_different_sizes(split):
    train, test = split
    drop = lambda p: p[~((p["store_id"] == "S0") & (p["item_id"] == "I3"))]
    model = build_model("state_model", **{**SETTINGS, "steps": 20}, neighbours="both").fit(drop(train))
    pred = model.predict(drop(test).drop(columns=[UNITS]))
    assert np.isfinite(pred).all()
