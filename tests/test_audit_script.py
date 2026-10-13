"""The audit script end to end on generated data."""
import json
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_runs  # noqa: E402
from rwm.experiments.run import run  # noqa: E402

STATE = dict(horizon=7, history=28, train_periods=150, categorical=["item_id", "store_id"], extra=["promo"],
             d_model=16, layers=1, heads=2, steps=60, batch=64, lr=3e-3, device="cpu",
             regular_price=True, lift_readout=True, rollout=True, roll_norm=True)
DATA = {"name": "synthetic", "params": {"n_stores": 2, "n_items": 3, "n_periods": 120, "seed": 1}}
EV = {"horizon": 7, "n_origins": 1, "scale_lag": 1, "weight_window": 14}


def test_checkpoint_and_lightgbm_audits(tmp_path):
    run({"name": "w", "seed": 0, "dataset": DATA, "model": {"name": "state_model", "params": STATE},
         "evaluation": EV}, out_root=tmp_path)
    audit_runs.main(["checkpoints", "--results", str(tmp_path)])
    rec = json.loads(next(tmp_path.glob("*/audit/audit_origin0.json")).read_text())
    assert rec["construction"]["zero_lift_when_nothing_planned"] < 1e-3
    assert set(rec["randomisation"]) >= {"lift readout", "encoder"}
    lcfg = {"name": "l", "seed": 0, "dataset": DATA, "evaluation": EV,
            "model": {"name": "lightgbm_direct", "params": {"horizon": 7, "train_periods": 100, "group_by": "store_id",
                                                             "categorical": ["item_id"], "extra": ["promo"], "rounds": 20,
                                                             "lags": [0, 7], "windows": [7, 14]}}}
    cfg = tmp_path / "l.yaml"
    cfg.write_text(yaml.safe_dump(lcfg))
    audit_runs.main(["lightgbm", "--config", str(cfg), "--results", str(tmp_path)])
    out = tmp_path / "paper2_lightgbm_l"
    assert (out / "audit_origin0.json").exists() and (out / "lifts_origin0.csv.gz").exists()


def test_summary_reads_the_records(tmp_path):
    import audit_summary
    for seed in (1, 2):
        cfg = {"name": f"w_seed{seed}", "seed": seed, "dataset": DATA,
               "model": {"name": "state_model", "params": {**STATE, "seed": seed}}, "evaluation": EV}
        run(cfg, out_root=tmp_path)
    audit_runs.main(["checkpoints", "--results", str(tmp_path)])
    audit_summary.main(["--results", str(tmp_path), "--out", str(tmp_path / "s")])
    import pandas as pd
    v = pd.read_csv(tmp_path / "s" / "versions.csv")
    st = pd.read_csv(tmp_path / "s" / "stability.csv")
    assert len(v) == 2 and "wrmsse" in v and len(st) == 1 and st["seeds"].iloc[0] == 2
