"""The audit rebuilds each run's frames and models exactly."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rwm.audit.runs import load_checkpoint, plan_levers, windows
from rwm.data.schema import UNITS
from rwm.experiments.run import run

STATE = dict(horizon=7, history=28, train_periods=150, categorical=["item_id", "store_id"], extra=["promo"],
             d_model=16, layers=1, heads=2, steps=60, batch=64, lr=3e-3, device="cpu",
             regular_price=True, lift_readout=True, rollout=True, roll_norm=True)


def _config(model):
    return {"name": "audit_check", "seed": 0,
            "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 3, "n_periods": 120, "seed": 1}},
            "model": model,
            "evaluation": {"horizon": 7, "n_origins": 2, "scale_lag": 1, "weight_window": 14}}


@pytest.mark.parametrize("model", [
    {"name": "state_model", "params": STATE},
    {"name": "seed_average", "params": {"model": "state_model", "members": 2, "seed": 0, "params": STATE}},
])
def test_rebuilt_frames_and_checkpoints_give_the_recorded_forecasts(tmp_path, model):
    cfg = _config(model)
    run_dir = run(cfg, out_root=tmp_path)
    recorded = pd.read_csv(run_dir / "forecasts.csv", parse_dates=["date"])
    for origin, train, test in windows(cfg):
        m = load_checkpoint(run_dir, origin, device="cpu")
        got = m.predict(test.drop(columns=[UNITS]))
        want = recorded[recorded["origin"] == origin]["forecast"].to_numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
        parts = m.breakdown(test.drop(columns=[UNITS]))
        np.testing.assert_allclose(parts["baseline"] + parts["lift"], parts["forecast"], rtol=1e-6)
    assert plan_levers(cfg) == ["promo"]
