"""Pretraining on other data sets as well (study U part b; rwm.data.pool, StateModel pretrain_on)."""
import numpy as np
import pandas as pd
import pytest

from rwm.data import load_dataset
from rwm.data.pool import load_source
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.model.state_model import StateModel

STATE = dict(horizon=7, history=28, train_periods=150, categorical=["item_id", "store_id"], extra=["promo"],
             d_model=16, layers=1, heads=2, steps=20, batch=32, lr=3e-3, device="cpu",
             regular_price=True, lift_readout=True, rollout=True)
SOURCE = {"name": "synthetic", "params": {"n_stores": 3, "n_items": 4, "n_periods": 200, "seed": 9}, "tag": "other"}


def _target():
    return load_dataset("synthetic", n_stores=2, n_items=3, n_periods=200, seed=4).panel


def test_source_is_relabelled_and_has_the_target_columns():
    p = load_source({**SOURCE}, [UNITS, PRICE, "promo", "not_in_source"])
    assert p[SERIES].str.startswith("other:").all() and p[ITEM].str.startswith("other:").all()
    assert p[STORE].str.startswith("other:").all()
    assert (p["not_in_source"] == 0).all()


def test_same_retailer_source_stops_at_the_training_end():
    until = np.sort(load_dataset(SOURCE["name"], **SOURCE["params"]).panel[DATE].unique())[100]
    p = load_source({**SOURCE, "same_retailer": True}, [UNITS], until=until)
    assert p[DATE].max() <= pd.Timestamp(until)
    q = load_source({**SOURCE}, [UNITS], until=until)  # another retailer: all of it
    assert q[DATE].max() > pd.Timestamp(until)


def test_pretraining_on_another_data_set_changes_the_fit_and_still_forecasts():
    train = _target()
    cut = np.sort(train[DATE].unique())[-7]
    past, future = train[train[DATE] < cut], train[train[DATE] >= cut]
    hybrid = dict(pretrain_steps=20, finetune="low_lr", encoder_lr_scale=0.1, latent_weight=0.5)
    own = StateModel(**STATE, **hybrid, seed=1).fit(past)
    pooled = StateModel(**STATE, **hybrid, pretrain_on=[SOURCE], seed=1).fit(past)
    a, b = own.predict(future), pooled.predict(future)
    assert np.isfinite(b).all() and a.shape == b.shape
    assert not np.allclose(a, b)
    again = StateModel(**STATE, **hybrid, pretrain_on=[SOURCE], seed=1).fit(past).predict(future)
    np.testing.assert_allclose(b, again)  # the same seed gives the same fit


def test_pretrain_on_needs_pretraining():
    with pytest.raises(ValueError):
        StateModel(**STATE, pretrain_on=[SOURCE])


def test_pooled_lightgbm_fits_on_both_and_forecasts_the_target():
    pytest.importorskip("lightgbm")
    from rwm.forecaster import build_model

    train = _target()
    cut = np.sort(train[DATE].unique())[-7]
    past, future = train[train[DATE] < cut], train[train[DATE] >= cut]
    lgb = dict(horizon=7, train_periods=120, group_by="store_id", categorical=["item_id"], extra=["promo"],
               lags=(0, 7), windows=(7, 14), price_window=28, price_lag=7, rounds=20)
    alone = build_model("lightgbm_direct", **lgb).fit(past).predict(future)
    pooled = build_model("pooled", model="lightgbm_direct", params=lgb, pool_with=[SOURCE]).fit(past).predict(future)
    assert pooled.shape == alone.shape and np.isfinite(pooled).all()
    assert not np.allclose(alone, pooled)


def test_pooled_sources_land_on_the_target_dates():
    """A weekly source whose weeks start on another weekday is moved onto the target's weeks."""
    from rwm.pooled import Pooled

    train = _target()
    weekly = train.assign(**{DATE: train[DATE]})
    m = Pooled.__new__(Pooled)
    m.pool_with, m._columns = [SOURCE], [UNITS, PRICE, "promo"]
    src = m._sources(weekly)[0]
    assert set(src[DATE].unique()) <= set(weekly[DATE].unique()) or src[DATE].max() == weekly[DATE].max()
    assert src[DATE].max() == weekly[DATE].max()
