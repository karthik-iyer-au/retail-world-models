"""A run split into pieces, each (window, member) fitted on its own and a last run that
loads them, gives exactly the forecasts and scores of the same run in one go
(rwm.experiments.run.piece_of; plan entry of 9 November 2026)."""
import json

import pandas as pd
import pytest

from rwm.experiments.run import run

STATE = dict(
    horizon=7, history=28, train_periods=200, categorical=["item_id", "store_id"], extra=["promo"],
    d_model=16, layers=1, heads=2, steps=40, batch=64, lr=3e-3, device="cpu",
    regular_price=True, lift_readout=True, rollout=True,
)


def _config(more: dict, average: bool = True) -> dict:
    model = {"name": "seed_average", "params": {"model": "state_model", "members": 2, "seed": 0,
                                                "params": {**STATE, **more}}} if average else \
        {"name": "state_model", "params": {**STATE, **more}}
    return {
        "name": "split",
        "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 3, "n_periods": 260, "seed": 4}},
        "model": model,
        "evaluation": {"horizon": 7, "n_origins": 2},
    }


@pytest.mark.parametrize("more,average", [({}, True),
                                          (dict(pretrain_steps=20, finetune="low_lr", encoder_lr_scale=0.1, latent_weight=0.5), True),
                                          (dict(latent_weight=0.1, latent_checks=3, weight_average=0.9), True),
                                          (dict(pretrain_steps=20, finetune="low_lr", encoder_lr_scale=0.1, latent_weight=0.5), False)])
def test_split_run_matches_one_run(tmp_path, monkeypatch, more, average):
    config = _config(more, average)
    members = 2 if average else 1
    whole = run(config, out_root=tmp_path / "a", partial=tmp_path / "a" / "_partial")
    for origin in range(2):
        for member in range(members):
            monkeypatch.setenv("RWM_PIECE", f"{origin},{member}")
            piece = run(config, out_root=tmp_path / "b", partial=tmp_path / "b" / "_partial")
            assert json.loads((piece / "metrics.json").read_text())["piece"] == {
                "origin": origin, "member": member, "fit_seconds": pytest.approx(json.loads((piece / "metrics.json").read_text())["piece"]["fit_seconds"])}
    monkeypatch.delenv("RWM_PIECE")
    last = run(config, out_root=tmp_path / "b", partial=tmp_path / "b" / "_partial")
    pd.testing.assert_frame_equal(pd.read_csv(whole / "forecasts.csv"), pd.read_csv(last / "forecasts.csv"), check_exact=True)
    a, b = (json.loads((d / "metrics.json").read_text()) for d in (whole, last))
    assert a["wrmsse"] == b["wrmsse"] and a["splits"] == b["splits"]
