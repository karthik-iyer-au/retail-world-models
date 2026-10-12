"""The plan read-back probe on a small world model and a model without the roll-forward."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE
from rwm.forecaster import build_model

spec = importlib.util.spec_from_file_location("plan_readback", Path(__file__).parents[1] / "scripts" / "plan_readback.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

SETTINGS = dict(
    horizon=7, history=28, train_periods=330, categorical=["item_id", "store_id"], extra=["promo"],
    d_model=32, layers=1, heads=4, steps=300, batch=128, lr=3e-3, device="cpu",
    regular_price=True, lift_readout=True, rollout=True,
)


@pytest.fixture(scope="module")
def split():
    p = load_dataset("synthetic", n_stores=4, n_items=10, n_periods=400, seed=4, dip=0.5).panel
    cut = np.sort(p[DATE].unique())[-8]
    return p[p[DATE] <= cut], p[(p[DATE] > cut) & (p[DATE] <= np.sort(p[DATE].unique())[-1])].reset_index(drop=True)


def test_ridge_r2_recovers_a_linear_target_and_not_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = x @ np.array([1.0, -2.0, 0.0, 0.5, 0.0])
    assert probe.ridge_r2(x[:200], y[:200], x[200:], y[200:]) > 0.99
    assert probe.ridge_r2(x[:200], rng.normal(size=200), x[200:], rng.normal(size=200)) < 0.1


def test_rolled_state_reads_back_the_promotion(split):
    train, test = split
    model = build_model("state_model", **SETTINGS).fit(train)
    state, roll, plan = probe.collect(model, test, max_items=1000, seed=0)
    assert roll is not None and roll.shape[:2] == plan.shape[:2] == (len(state), 7)
    rows = probe.readback(state, roll, plan, probe.plan_columns(model), seed=0)
    promo_now = next(r for r in rows if r["plan input"] == "promo" and r["periods back"] == 0)
    # the state before the window cannot know this window's promotions; the rolled state can
    assert promo_now["from rolled state"] > promo_now["from state before window"]


def test_model_without_roll_forward_reports_the_control_only(split):
    train, test = split
    model = build_model("state_model", **{**SETTINGS, "rollout": False, "steps": 5}).fit(train)
    state, roll, plan = probe.collect(model, test, max_items=1000, seed=0)
    assert roll is None
    rows = probe.readback(state, roll, plan, probe.plan_columns(model), seed=0)
    assert rows and all("from rolled state" not in r for r in rows)
