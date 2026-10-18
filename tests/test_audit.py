"""Paper 2 audit checks on generated data, where the true lift is known."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwm.audit import (construction_checks, lever_contributions, lift_vs_actual, price_monotonicity,
                       randomised_copies, rank_agreement, regular_prices)
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


@pytest.fixture(scope="module")
def regular(split):
    return regular_prices(split[0], 12)


def test_construction_holds(split, fitted, regular):
    c = construction_checks(fitted, split[1], regular, ["promo"])
    assert c["zero_lift_when_nothing_planned"] < 1e-3
    assert c["baseline_moves_with_plan"] < 1e-4
    assert c["parts_do_not_add_up"] < 1e-6


def test_contributions_find_the_promotion(split, fitted, regular):
    parts = lever_contributions(fitted, split[1], regular, ["promo"])
    promo = split[1]["promo"].to_numpy() == 1
    assert (parts.loc[promo, "total_lift"] > 0).mean() > 0.9
    np.testing.assert_allclose(parts.loc[~promo, "total_lift"], 0, atol=1e-3 * parts["forecast"].max())
    # promotion and its price cut together: total lift is the sum of the parts plus their interaction
    total = parts["lift_promo"] + parts["lift_price"] + parts["interaction"]
    np.testing.assert_allclose(total, parts["total_lift"], rtol=1e-9, atol=1e-9)
    ratio = parts.loc[promo, "forecast"].sum() / parts.loc[promo, "nothing_planned"].sum()
    assert 1.4 < ratio < 2.2  # the true lift is 1.8


def test_randomising_the_model_changes_the_lifts(split, fitted, regular):
    before = lever_contributions(fitted, split[1], regular, ["promo"])["total_lift"]
    names, agreement = [], []
    for name, m in randomised_copies(fitted, seed=1):
        names.append(name)
        agreement.append(rank_agreement(before, lever_contributions(m, split[1], regular, ["promo"])["total_lift"]))
    assert names[0] == "lift readout" and "encoder" in names
    assert np.nanmax(np.abs(agreement)) < 0.6
    # the original model is untouched
    after = lever_contributions(fitted, split[1], regular, ["promo"])["total_lift"]
    np.testing.assert_allclose(before, after)


def test_price_monotonicity_and_lift_against_sales(split, fitted, regular):
    share = price_monotonicity(fitted, split[1])["share_lower_after_deeper_cut"]
    assert 0.0 <= share <= 1.0
    parts = fitted.breakdown(split[1].drop(columns=[UNITS]))
    promo = split[1]["promo"].to_numpy() == 1
    r = lift_vs_actual(parts["forecast"], parts["baseline"], split[1][UNITS], promo)
    assert r["n"] == promo.sum() and 0.6 < r["bias"] < 1.5


def test_lightgbm_shap_adds_up_to_the_forecast(split):
    pytest.importorskip("lightgbm")
    m = build_model("lightgbm_direct", horizon=7, train_periods=330, group_by="store_id",
                    categorical=["item_id"], extra=["promo"], rounds=50, lags=[0, 7], windows=[7, 28]).fit(split[0])
    fut = split[1].drop(columns=[UNITS])
    shap = m.shap(fut)
    raw = shap.sum(axis=1).to_numpy()
    np.testing.assert_allclose(np.exp(raw), m.predict(fut), rtol=1e-6, atol=1e-6)
    assert "promo" in shap.columns


def test_nothing_planned_is_the_models_own(split):
    """Given the model's own "nothing planned" plan the lift is zero, also for a
    series with no price in its recent history, where the audit's plan keeps the
    planned price and so leaves a lift."""
    from rwm.audit.lifts import off_plan
    from rwm.data.schema import PRICE, SERIES

    train, test = split
    gap = train[SERIES].iloc[0]
    recent = np.sort(train[DATE].unique())[-30:]
    train = train.copy()
    train.loc[(train[SERIES] == gap) & train[DATE].isin(recent), PRICE] = np.nan  # no price in the history window
    m = build_model("state_model", **{**SETTINGS, "steps": 20}).fit(train)
    plan = test.drop(columns=[UNITS])
    own = m.nothing_planned(plan)
    assert (own["promo"].fillna(0) == 0).all()
    top = m.predict(plan).max()
    assert np.abs(m.breakdown(own)["lift"]).max() < 1e-5 * top
    audit = m.breakdown(off_plan(plan, regular_prices(train, 12), ["promo"]))
    assert np.abs(audit.loc[plan[SERIES] == gap, "lift"]).max() > 1e-5 * top
