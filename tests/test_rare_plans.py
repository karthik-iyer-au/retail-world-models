"""Accuracy on lever mixes that were rare in training."""
import json

import numpy as np

from rwm.experiments.run import run


def test_run_scores_rare_and_usual_plans_apart(tmp_path):
    config = {
        "name": "rare", "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_items": 6, "n_periods": 150}},
        "model": {"name": "seasonal_naive"},
        "evaluation": {"horizon": 14, "n_origins": 1,
                       # every promoted row counts as rare with the bar this high
                       "rare_plans": {"levers": ["promo"], "below": 0.5}},
    }
    s = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())["splits"][0]
    assert np.isfinite(s["rare_plans"])
    config["evaluation"]["rare_plans"]["below"] = 0.0
    config["name"] = "usual"
    s = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())["splits"][0]
    assert np.isfinite(s["usual_plans"])
