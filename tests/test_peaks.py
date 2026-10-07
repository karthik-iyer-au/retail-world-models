"""Forecast over actual in promoted periods, just after a promotion, and otherwise."""
import json

import numpy as np

from rwm.experiments.run import run


def test_run_reports_promoted_after_promo_and_ordinary_periods(tmp_path):
    config = {
        "name": "peaks", "seed": 0,
        # every item sells 40% less in the period after its own promotion
        "dataset": {"name": "synthetic", "params": {"n_items": 6, "n_periods": 200, "dip": 0.4}},
        "model": {"name": "recent_average"},
        "evaluation": {"horizon": 14, "n_origins": 1, "after_promo": {"levers": ["promo"], "periods": 1}},
    }
    s = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())["splits"][0]["peaks"]
    assert abs(sum(g["share_of_units"] for g in s.values()) - 1) < 1e-6
    # a forecast that ignores the plan is low when promoted and high just after
    assert s["promoted"]["forecast_to_actual"] < s["ordinary"]["forecast_to_actual"] < s["after_promo"]["forecast_to_actual"]
    assert all(np.isfinite(g["rmsse"]) for g in s.values())
