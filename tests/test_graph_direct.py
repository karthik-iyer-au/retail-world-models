"""The graph baseline: builds, trains on a small panel, and beats repeating last week."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from rwm.data import load_dataset
from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import build_model
from rwm.prior_work.graph_direct import _groups


def test_groups_join_nodes_that_share_attributes():
    attrs = pd.DataFrame({"item_id": ["a", "a", "b"], "store_id": ["1", "2", "1"]})
    by_item, by_store = _groups(attrs, [["item_id"], ["store_id"]])
    assert by_item[0] == by_item[1] != by_item[2]
    assert by_store[0] == by_store[2] != by_store[1]


def test_trains_and_beats_last_week():
    p = load_dataset("synthetic", n_stores=3, n_items=8, n_periods=300, seed=2).panel
    dates = np.sort(p[DATE].unique())
    train, test = p[p[DATE] <= dates[-8]], p[p[DATE] > dates[-8]].reset_index(drop=True)
    settings = dict(horizon=7, train_periods=200, relations=[["item_id"], ["store_id"]],
                    categorical=["item_id", "store_id"], extra=["promo"], hidden=32, steps=300, lr=3e-3, device="cpu",
                    inputs={"lags": [0, 7], "windows": [7, 28]})
    model = build_model("graph_direct", **settings)
    pred = model.fit(train).predict(test.drop(columns=[UNITS]))
    assert pred.shape == (len(test),) and np.isfinite(pred).all() and (pred >= 0).all()
    naive = build_model("seasonal_naive", season=7).fit(train).predict(test.drop(columns=[UNITS]))
    err = lambda f: np.mean((f - test[UNITS].to_numpy()) ** 2)
    assert err(pred) < err(naive)
