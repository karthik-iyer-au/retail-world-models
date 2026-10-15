"""Study B of the paper 1 amendment of 15 October 2026: scripts/input_perturbation.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import input_perturbation as IP  # noqa: E402

from rwm.experiments.run import run  # noqa: E402

STATE = dict(horizon=7, history=28, train_periods=100, categorical=["item_id", "store_id"], extra=["promo"],
             d_model=16, layers=1, heads=2, steps=40, batch=64, lr=3e-3, device="cpu",
             regular_price=True, lift_readout=True, roll_norm=True)


def _config(name, model):
    return {"name": name, "seed": 0,
            "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 3, "n_periods": 120, "seed": 1}},
            "model": model,
            "evaluation": {"horizon": 7, "n_origins": 2, "scale_lag": 1, "weight_window": 14}}


def test_permutation_keeps_each_series_values():
    plan = pd.DataFrame({"series_id": list("aaabbbb"), "promo": [1, 0, 0, 5, 6, 7, 8]})
    got = IP.permute_within_series(plan, "promo", seed=3)
    for s, g in plan.groupby("series_id"):
        assert sorted(got.loc[g.index, "promo"]) == sorted(g["promo"])


def test_occlusion_switches_the_input_off():
    plan = pd.DataFrame({"series_id": ["a", "a", "b"], "price": [8.0, 10.0, 4.0], "promo": [1, 0, 1]})
    regular = pd.Series({"a": 10.0, "b": 5.0})
    assert IP.occlude(plan, "price", regular)["price"].tolist() == [10.0, 10.0, 5.0]
    assert IP.occlude(plan, "promo", regular)["promo"].tolist() == [0, 0, 0]


def test_runs_on_saved_and_refitted_models(tmp_path):
    for model, params in [("world_model", {**STATE, "rollout": True}), ("lift", {**STATE, "rollout": False})]:
        run(_config(f"rev_frat_rolling_{model}_seed1", {"name": "state_model", "params": params}), out_root=tmp_path)
    lgb = dict(horizon=7, train_periods=100, group_by="store_id", categorical=["item_id"], extra=["promo"],
               rounds=30, params={"min_data_in_leaf": 5})
    run(_config("rev_frat_rolling_lightgbm_seed1", {"name": "lightgbm_direct", "params": lgb}), out_root=tmp_path)
    out = tmp_path / "perturbation.csv"
    IP.main(["--results", str(tmp_path), "--out", str(out), "--lightgbm-keys", "frat", "--lightgbm-last", "1"])
    df = pd.read_csv(out)
    assert set(df["model"]) == {"world_model", "lift", "lightgbm"}
    assert len(df[df["model"] == "lightgbm"]["origin"].unique()) == 1
    assert len(df[df["model"] == "world_model"]["origin"].unique()) == 2
    # saved checkpoints give the recorded forecasts; the LightGBM refit matches its run
    assert (df["refit_gap"] < 1e-3).all()
    assert {"price", "promo", "whole_plan"} <= set(df["input"])
    assert set(df[df["model"] == "lightgbm"]["input"]).isdisjoint({"history_last_4"})
    assert {"history_last_4", "history_last_13"} <= set(df[df["model"] == "world_model"]["input"])
    wm = df[df["model"] == "world_model"]
    hist = wm[wm["input"] == "history_last_4"]
    assert not np.allclose(hist["bottom"], hist["bottom_true"])  # masking history changes the forecast
    promo = df[(df["input"] == "promo") & (df["operation"] == "occlude")]
    assert (promo["units_change_on"].dropna() != 0).any()
    n = len(df)
    IP.main(["--results", str(tmp_path), "--out", str(out), "--lightgbm-keys", "frat", "--lightgbm-last", "1"])
    assert len(pd.read_csv(out)) == n  # windows already done are skipped


def test_table_means_over_windows():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import perturbation_tables as PT
    df = pd.DataFrame({"key": "frat", "model": "world_model", "input": "price", "operation": "occlude",
                       "origin": [0, 1], "bottom": [1.1, 1.3], "bottom_true": [1.0, 1.0],
                       "total": [0.5, 0.5], "total_true": [0.5, 0.5], "units_change_on": [-0.2, -0.4]})
    t = PT.table(df)
    assert np.isclose(t["bottom_change"].iloc[0], 0.2) and np.isclose(t["units_change_on"].iloc[0], -0.3)
    assert t["windows"].iloc[0] == 2
