"""Write the study D configs and jobs (docs/PAPER1_REVISION_PLAN.md, amendment of
15 October 2026): each model with the settings of its M5 rolling config, only
the data set's columns changed, on four rolling 28-day windows ending on the
last day of the Favorita data.

  python scripts/make_favorita_configs.py
"""
import copy

import yaml

from make_revision_configs import EXP, JOBS, load

EVAL = {"horizon": 28, "n_origins": 4, "scale_lag": 1, "weight_window": 28, "holdout_periods": 0, "step": 28}
STATE_CATS = ["item_id", "class", "family", "store_id", "city"]  # M5: item, dept, cat, store, state
TREE_CATS = ["item_id", "class", "family"]  # M5: item, dept, cat (stores are LightGBM's groups)
PLAN = ["promo", "event", "payday"]  # M5: event, snap; promo is the lever, as on Dominick's


def adapt(model: str, m5: dict) -> dict:
    cfg = copy.deepcopy(m5)
    cfg["name"] = f"rev_favorita_rolling_{model}"
    cfg["dataset"] = {"name": "favorita"}
    cfg["evaluation"] = dict(EVAL)
    p = cfg["model"]["params"]
    if cfg["model"]["name"] == "state_model":
        p.update(categorical=STATE_CATS, extra=PLAN, levers=["promo"])
    elif cfg["model"]["name"] == "lightgbm_direct":
        p.update(categorical=TREE_CATS, extra=PLAN)
    elif cfg["model"]["name"] == "timesfm":
        p.update(plan=PLAN, price_plan=False)  # Favorita records no prices
    return cfg


def main():
    sources = {"seasonal_naive": "rev_m5_rolling_seasonal_naive", "lightgbm": "rev_m5_rolling_lightgbm",
               "timesfm_plan": "rev_m5_rolling_timesfm_plan", "world_model": "rev_m5_rolling_world_model",
               "lift": "rev_m5_rolling_lift"}
    names = {}
    for model, src in sources.items():
        cfg = adapt(model, load(f"{src}.yaml"))
        path = EXP / f"{cfg['name']}.yaml"
        path.write_text(f"# Study D (plan amendment of 15 October 2026): {model} on Favorita, settings of {src}; "
                        "written by scripts/make_favorita_configs.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
        names[model] = f"configs/experiments/{path.name}"
    stock = adapt("world_model_stock", load("rev_m5_rolling_world_model.yaml"))
    stock["model"]["params"]["stock"] = True
    (EXP / f"{stock['name']}.yaml").write_text(
        "# Study D: the world model with the stock term; run only if it passes stage 1 of study C.\n"
        + yaml.safe_dump(stock, sort_keys=False))
    jobs = {"favorita_a": ["seasonal_naive", "lightgbm", "timesfm_plan"], "favorita_b": ["world_model", "lift"]}
    for job, models in jobs.items():
        body = {"datasets": [{"name": "favorita"}], "runs": [{"config": names[m], "seeds": [1]} for m in models]}
        (JOBS / f"{job}.yaml").write_text(f"# Study D, Favorita, part {job[-1]}.\n" + yaml.safe_dump(body, sort_keys=False))
    body = {"datasets": [{"name": "favorita"}],
            "runs": [{"config": f"configs/experiments/{stock['name']}.yaml", "seeds": [1]}]}
    (JOBS / "favorita_stock.yaml").write_text("# Study D, the stock variant (only after study C stage 1).\n"
                                              + yaml.safe_dump(body, sort_keys=False))


if __name__ == "__main__":
    main()
