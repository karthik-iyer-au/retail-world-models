"""Forecasting products the model never saw in training."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from rwm.data.schema import DATE, ITEM, SERIES, UNITS
from rwm.data.synthetic import load_synthetic
from rwm.experiments.run import run
from rwm.forecaster import build_model

PARAMS = dict(horizon=7, history=28, steps=60, d_model=16, layers=1, extra=["promo"], train_periods=90, batch=64,
              categorical=["store_id", "item_id"], device="cpu")


def _split():
    panel = load_synthetic(n_stores=2, n_items=6, n_periods=120).panel
    cut = np.sort(panel[DATE].unique())[-7]
    return panel[panel[DATE] < cut], panel[panel[DATE] >= cut]


def test_new_item_gets_a_forecast_from_similar_items():
    train, test = _split()
    new = train[ITEM] == "I5"
    model = build_model("state_model", cold_start=True, cold_neighbours=3, unknown_label_rate=0.2, **PARAMS)
    model.fit(train[~new])
    pred = model.predict(test.drop(columns=[UNITS]))
    mine = pred[(test[ITEM] == "I5").to_numpy()]
    assert np.isfinite(pred).all() and (mine > 0).all()


def test_without_cold_start_a_new_item_gets_zero():
    train, test = _split()
    model = build_model("state_model", **PARAMS)
    model.fit(train[train[ITEM] != "I5"])
    pred = model.predict(test.drop(columns=[UNITS]))
    assert (pred[(test[ITEM] == "I5").to_numpy()] == 0).all()


def test_known_items_are_not_changed_by_cold_start():
    train, test = _split()
    a = build_model("state_model", **PARAMS); a.fit(train)
    b = build_model("state_model", cold_start=True, **PARAMS); b.fit(train)
    plan = test.drop(columns=[UNITS])
    assert np.allclose(a.predict(plan), b.predict(plan))


def test_run_scores_new_items_apart(tmp_path):
    config = {
        "name": "new_items", "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_items": 6, "n_periods": 120}},
        "model": {"name": "state_model", "params": dict(PARAMS, cold_start=True, cold_neighbours=3)},
        "evaluation": {"horizon": 7, "n_origins": 1, "levels": [["store_id", "item_id"]],
                       "weight_window": 28, "new_items": {"share": 0.34, "seed": 0}},
    }
    m = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())
    s = m["splits"][0]
    assert np.isfinite(s["new_items"]) and np.isfinite(s["known_items"])


def test_lightgbm_cold_start_forecasts_a_new_item():
    pytest.importorskip("lightgbm")
    train, test = _split()
    kw = dict(horizon=7, train_periods=90, categorical=["store_id", "item_id"], extra=["promo"],
              lags=(0, 7), windows=(7, 14), spread_window=7, price_window=14, rounds=20,
              params={"min_data_in_leaf": 5})
    plan = test.drop(columns=[UNITS])
    new = (test[ITEM] == "I5").to_numpy()
    off = build_model("lightgbm_direct", **kw).fit(train[train[ITEM] != "I5"]).predict(plan)
    on = build_model("lightgbm_direct", cold_start=True, **kw).fit(train[train[ITEM] != "I5"]).predict(plan)
    assert (off[new] == 0).all() and (on[new] > 0).all()
    assert np.allclose(off[~new], on[~new])
