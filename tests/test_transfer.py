"""The transfer test: train without some stores or items, then forecast them from their own history."""
import numpy as np
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.experiments.run import run
from rwm.forecaster import build_model

EVAL = {"horizon": 7, "n_origins": 1}


def _config(model, params, by="store_id", trained_on_all=False):
    return {"name": f"transfer_{model}", "seed": 0,
            "dataset": {"name": "synthetic", "params": {"n_stores": 4, "n_items": 5, "n_periods": 200, "seed": 2}},
            "model": {"name": model, "params": params},
            "evaluation": {**EVAL, "transfer": {"by": by, "share": 0.25, "seed": 0, "trained_on_all": trained_on_all}}}


def test_lightgbm_forecasts_stores_it_was_not_trained_on(tmp_path):
    pytest.importorskip("lightgbm")
    params = {"horizon": 7, "train_periods": 120, "categorical": ["item_id"], "extra": ["promo"],
              "lags": [0, 7], "windows": [7, 14], "spread_window": 7, "price_window": 14, "price_lag": 7, "rounds": 30}
    out = run(_config("lightgbm_direct", params), out_root=tmp_path)
    import json

    s = json.loads((out / "metrics.json").read_text())["splits"][0]
    assert np.isfinite(s["kept_out"]) and np.isfinite(s["trained_on"])
    assert s["kept_out_forecast_to_actual"] > 0.3  # forecast from the held-out stores' own history, not zero


def test_one_model_per_store_cannot_serve_a_store_it_never_saw():
    pytest.importorskip("lightgbm")
    p = load_dataset("synthetic", n_stores=2, n_items=3, n_periods=120, seed=1).panel
    cut = np.sort(p[DATE].unique())[-7]
    train, test = p[p[DATE] <= cut], p[p[DATE] > cut]
    model = build_model("lightgbm_direct", horizon=7, train_periods=60, group_by="store_id", lags=[0], windows=[7],
                        spread_window=7, price_window=14, price_lag=7, rounds=10)
    seen = train[train["store_id"] != train["store_id"].iloc[0]]
    model.fit(seen).refresh(train)
    assert model.unserved == [train["store_id"].iloc[0]]


def test_world_model_reads_held_out_items_from_their_own_history():
    pytest.importorskip("torch")
    p = load_dataset("synthetic", n_stores=2, n_items=6, n_periods=200, seed=3).panel
    cut = np.sort(p[DATE].unique())[-7]
    train, test = p[p[DATE] <= cut], p[p[DATE] > cut]
    out_item = train["item_id"].astype(str).unique()[0]
    params = dict(horizon=7, history=28, train_periods=120, categorical=["item_id", "store_id"], extra=["promo"],
                  d_model=16, layers=1, heads=2, steps=60, batch=64, device="cpu", rollout=True, lift_readout=True)
    model = build_model("state_model", **params).fit(train[train["item_id"].astype(str) != out_item])
    model.refresh(train)
    f = model.predict(test.drop(columns=[UNITS]))
    held = (test["item_id"].astype(str) == out_item).to_numpy()
    assert held.any() and np.all(np.isfinite(f)) and f[held].sum() > 0
    # nothing about the network changed: refreshing again with the training data alone gives the same forecasts
    again = build_model("state_model", **params).fit(train[train["item_id"].astype(str) != out_item])
    again.refresh(train)
    assert np.allclose(again.predict(test.drop(columns=[UNITS])), f)


def test_the_comparison_run_scores_the_same_share(tmp_path):
    import json

    a = run(_config("seasonal_naive", {"season": 7}), out_root=tmp_path / "a")
    b = run(_config("seasonal_naive", {"season": 7}, trained_on_all=True), out_root=tmp_path / "b")
    sa = json.loads((a / "metrics.json").read_text())["splits"][0]
    sb = json.loads((b / "metrics.json").read_text())["splits"][0]
    # a model that learns nothing from history gives the same score either way
    assert sa["kept_out"] == pytest.approx(sb["kept_out"])
