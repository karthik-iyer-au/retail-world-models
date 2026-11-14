"""Forecasting far ahead in blocks (rwm.blocks), for the 52-week test."""
import json

import numpy as np
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.experiments.run import run
from rwm.forecaster import build_model

LGB = dict(horizon=7, train_periods=120, categorical=["item_id"], extra=["promo"], lags=[0, 7], windows=[7, 14],
           spread_window=7, price_window=14, price_lag=7, rounds=30)
WM = dict(horizon=7, history=28, train_periods=120, categorical=["item_id", "store_id"], extra=["promo"],
          d_model=16, layers=1, heads=2, steps=60, batch=64, device="cpu", rollout=True, lift_readout=True)


def _split(n_ahead):
    p = load_dataset("synthetic", n_stores=2, n_items=4, n_periods=220, seed=4).panel
    cut = np.sort(p[DATE].unique())[-n_ahead]
    return p[p[DATE] <= cut], p[p[DATE] > cut]


@pytest.mark.parametrize("name,params,lib", [("lightgbm_direct", LGB, "lightgbm"), ("state_model", WM, "torch")])
def test_one_block_is_the_wrapped_model(name, params, lib):
    pytest.importorskip(lib)
    train, test = _split(7)
    plain = build_model(name, **params).fit(train).predict(test.drop(columns=[UNITS]))
    blocked = build_model("blocks", model=name, block=7, params=params).fit(train).predict(test.drop(columns=[UNITS]))
    assert np.allclose(plain, blocked)


@pytest.mark.parametrize("name,params,lib", [("lightgbm_direct", LGB, "lightgbm"), ("state_model", WM, "torch")])
def test_later_blocks_start_from_the_earlier_forecasts(name, params, lib):
    pytest.importorskip(lib)
    train, test = _split(21)
    m = build_model("blocks", model=name, block=7, params=params).fit(train)
    future = test.drop(columns=[UNITS])
    f = m.predict(future)
    assert np.all(np.isfinite(f)) and f.sum() > 0
    dates = np.sort(test[DATE].unique())
    first = future[DATE].isin(dates[:7]).to_numpy()
    # the first block is the wrapped model's own 7-period forecast
    own = build_model(name, **params).fit(train).predict(future[first])
    assert np.allclose(f[first], own)
    # by hand: take in the first block's forecast as sales, then forecast the second block
    inner = build_model(name, **params).fit(train)
    import pandas as pd

    hist = pd.concat([train, test[first].assign(**{UNITS: own})[train.columns]], ignore_index=True)
    inner.refresh(hist)
    second = future[DATE].isin(dates[7:14]).to_numpy()
    assert np.allclose(f[second], inner.predict(future[second]))
    # a second call gives the same forecasts
    assert np.allclose(m.predict(future), f)


def test_runs_through_the_experiment_runner(tmp_path):
    pytest.importorskip("lightgbm")
    cfg = {"name": "blocks_lgb", "seed": 0,
           "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 4, "n_periods": 220, "seed": 4}},
           "model": {"name": "blocks", "params": {"model": "lightgbm_direct", "block": 7, "params": LGB}},
           "evaluation": {"horizon": 21, "n_origins": 2, "step": 7}}
    out = run(cfg, out_root=tmp_path)
    s = json.loads((out / "metrics.json").read_text())["splits"]
    assert len(s) == 2


def test_roll_on_matches_the_model_within_its_horizon_and_runs_past_it():
    pytest.importorskip("torch")
    train, test = _split(21)
    future = test.drop(columns=[UNITS])
    dates = np.sort(test[DATE].unique())
    first = future[DATE].isin(dates[:7]).to_numpy()
    own = build_model("state_model", **WM).fit(train).predict(future[first])
    m = build_model("roll_on", model="state_model", params=WM).fit(train)
    f = m.predict(future)
    assert np.all(np.isfinite(f)) and f.sum() > 0
    # the first seven periods: rolling on for 21 changes nothing in them
    assert np.allclose(f[first], own, rtol=1e-4, atol=1e-6)
    # and back to the trained horizon again
    assert np.allclose(m.predict(future[first]), own, rtol=1e-4, atol=1e-6)


def test_by_block_uses_model_k_for_block_k():
    pytest.importorskip("lightgbm")
    train, test = _split(21)
    future = test.drop(columns=[UNITS])
    dates = np.sort(test[DATE].unique())
    f = build_model("by_block", model="lightgbm_direct", block=7, blocks=3, params=LGB).fit(train).predict(future)
    assert np.all(np.isfinite(f))
    for k in range(3):
        reach = future[DATE].isin(dates[:7 * (k + 1)]).to_numpy()
        own = build_model("lightgbm_direct", **{**LGB, "horizon": 7 * (k + 1)}).fit(train).predict(future[reach])
        mine = future[reach][DATE].isin(dates[7 * k:7 * (k + 1)]).to_numpy()
        assert np.allclose(f[np.flatnonzero(reach)[mine]], own[mine])


def test_overlapping_windows_score_every_period(tmp_path):
    """Windows whose step is shorter than the horizon: every window keeps all its test rows (bug of 14 Nov 2026,
    when an earlier window lost the periods after a later window's start)."""
    pytest.importorskip("lightgbm")
    cfg = {"name": "overlap", "seed": 0,
           "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 4, "n_periods": 220, "seed": 4}},
           "model": {"name": "lightgbm_direct", "params": {**LGB, "horizon": 21}},
           "evaluation": {"horizon": 21, "n_origins": 3, "step": 7}}
    out = run(cfg, out_root=tmp_path)
    m = json.loads((out / "metrics.json").read_text())
    for s in m["splits"]:
        assert len(s["by_horizon"]) == 21 and np.all(np.isfinite(s["by_horizon"]))
