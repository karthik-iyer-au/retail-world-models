"""Plan A against plan B: two weeks of the same item with different plans."""
import json

from rwm.experiments.run import run


def _score(tmp_path, model, name):
    config = {
        "name": name, "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_items": 6, "n_periods": 200}},
        "model": model,
        "evaluation": {"horizon": 14, "n_origins": 1, "plan_pairs": {"levers": ["promo"]}},
    }
    return json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())["splits"][0]


def test_a_model_that_sees_the_plan_orders_the_weeks_better(tmp_path):
    blind = _score(tmp_path, {"name": "recent_average"}, "blind")
    sees = _score(tmp_path, {"name": "lightgbm_direct", "params": dict(
        horizon=14, train_periods=150, extra=["promo"], lags=(0, 7), windows=(7, 14), spread_window=7,
        price_window=14, rounds=60, params={"min_data_in_leaf": 5})}, "sees")
    assert blind["plan_pairs"] == sees["plan_pairs"] > 0
    assert sees["plan_pair_order"] > blind["plan_pair_order"]
    assert sees["plan_pair_change_error"] < blind["plan_pair_change_error"]
