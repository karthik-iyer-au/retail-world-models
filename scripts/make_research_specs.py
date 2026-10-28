"""Write the research specs: study M (docs/PAPER1_REVISION_PLAN.md, entry of
28 October 2026), the research loop as the companion work ran it on the world
model, on M5 and the six Dominick's categories, with every change screened at
full training length and AdaJEPA in its published form; and study N (entry of
29 October 2026), the same loop on LightGBM, so both models get the same search.

  python scripts/make_research_specs.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "research"
CATS = ["che", "sna", "sdr", "bjc", "ana", "fre"]

WEEKLY = {"horizon": 8, "n_origins": 4, "holdout_periods": 72, "step": 8, "scale_lag": 1, "weight_window": 13,
          "after_promo": {"levers": ["promo"], "periods": 2}}
DAILY = {"horizon": 28, "n_origins": 4, "holdout_periods": 196, "step": 28, "scale_lag": 1, "weight_window": 28}


def menu(base: dict, weekly: bool) -> list[dict]:
    p = base["model"]["params"]
    h = base["evaluation"]["horizon"]
    views = [4, 8] if weekly else [7, 28]
    adapt_periods = 4 if weekly else 28
    m = [
        ("volume_weight_0.25", {"volume_weight": 0.25}),
        ("volume_weight_0.5", {"volume_weight": 0.5}),
        ("peak_weight_0.5", {"peak_weight": 0.5}),
        ("after_promo_weight_0.5", {"after_promo_weight": 0.5}),
        ("total_weight_0.1", {"total_weight": 0.1}),
        ("plan_split", {"plan_split": True}),
        ("feedback", {"feedback": True}),
        ("level_views", {"level_views": views}),
        ("latent_check", {"latent_weight": 0.1}),
        ("latent_overshooting", {"latent_weight": 0.1, "latent_checks": min(h - 1, 7)}),
        ("state_spread", {"state_spread_weight": 0.01}),
        ("lr_half", {"lr": p["lr"] / 2}),
        ("lr_double", {"lr": p["lr"] * 2}),
        ("batch_half", {"batch": p["batch"] // 2}),
        ("steps_1.5x", {"steps": int(p["steps"] * 1.5)}),
        ("weight_average_0.999", {"weight_average": 0.999}),
        ("width_1.5x", {"d_model": int(p["d_model"] * 1.5)}),
        ("adajepa_1", {"adapt_steps": 1, "adapt_optimizer": "adam", "adapt_lr": 1e-4, "adapt_periods": adapt_periods}),
        ("adajepa_3", {"adapt_steps": 3, "adapt_optimizer": "adam", "adapt_lr": 1e-4, "adapt_periods": adapt_periods}),
    ]
    if p.get("scale_window"):
        m.append(("level_from_all_history", {"scale_window": None}))
    return [{"name": n, "change": c} for n, c in m]


def lightgbm_menu(base: dict, weekly: bool) -> list[dict]:
    """Settings of the trees, their inputs and their training: the counterpart
    of the world model's menu, 20 changes."""
    p = base["model"]["params"]
    rounds = p.get("rounds", 800)
    lags = list(p.get("lags", [0, 7, 14, 21, 28]))
    windows = list(p.get("windows", [7, 14, 28, 56]))
    step = 1 if weekly else 7
    m = [
        ("rounds_half", {"rounds": rounds // 2}),
        ("rounds_1.5x", {"rounds": int(rounds * 1.5)}),
        ("rounds_double", {"rounds": rounds * 2}),
        ("learning_rate_half", {"params": {"learning_rate": 0.025}}),
        ("learning_rate_double", {"params": {"learning_rate": 0.1}}),
        ("num_leaves_quarter", {"params": {"num_leaves": max(15, p.get("params", {}).get("num_leaves", 255) // 4)}}),
        ("num_leaves_4x", {"params": {"num_leaves": min(2047, p.get("params", {}).get("num_leaves", 255) * 4)}}),
        ("min_data_in_leaf_quarter", {"params": {"min_data_in_leaf": max(5, p.get("params", {}).get("min_data_in_leaf", 200) // 4)}}),
        ("min_data_in_leaf_4x", {"params": {"min_data_in_leaf": p.get("params", {}).get("min_data_in_leaf", 200) * 4}}),
        ("feature_fraction_0.8", {"params": {"feature_fraction": 0.8}}),
        ("bagging_fraction_0.8", {"params": {"bagging_fraction": 0.8}}),
        ("tweedie_power_1.3", {"params": {"tweedie_variance_power": 1.3}}),
        ("tweedie_power_1.5", {"params": {"tweedie_variance_power": 1.5}}),
        ("l2_1", {"params": {"lambda_l2": 1.0}}),
        ("max_bin_255", {"params": {"max_bin": 255}}),
        ("more_lags", {"lags": sorted(set(lags + [lags[-1] + step, lags[-1] + 2 * step]))}),
        ("longer_windows", {"windows": sorted(set(windows + [windows[-1] * 2]))}),
        ("train_periods_half", {"train_periods": p["train_periods"] // 2}),
        ("objective_poisson", {"params": {"objective": "poisson"}}),
        ("seed_bagging_off", {"params": {"bagging_fraction": 1.0, "bagging_freq": 0}}),
    ]
    return [{"name": n, "change": c} for n, c in m]


def spec(name: str, base_path: str, evaluation: dict, weekly: bool) -> dict:
    base = yaml.safe_load((ROOT / base_path).read_text())
    return {"name": name, "base": base_path, "evaluation": evaluation, "seeds": [1, 2, 3],
            "rung_fraction": 1.0, "promote": "beats_best", "menu": menu(base, weekly)}


HEADER = """# Study M (docs/PAPER1_REVISION_PLAN.md, entry of 28 October 2026): the research loop
# (src/rwm/research/loop.py) as the companion work ran it. Fixed before the first try and
# never changed by the loop. Every change is tried alone on top of the best settings so far,
# at full training length on seed 1; a change that beats the best on seed 1 is run on seeds
# 2 and 3, and the best such change is kept if its mean over the three seeds beats the best's
# mean by at least the seed spread of the starting settings. The windows lie before every
# window the paper reports; the latest of the four is held out, scored but never used to decide.
# Written by scripts/make_research_specs.py.
"""


HEADER_N = HEADER.replace("Study M (docs/PAPER1_REVISION_PLAN.md, entry of 28 October 2026)",
                          "Study N (docs/PAPER1_REVISION_PLAN.md, entry of 29 October 2026), on LightGBM")


def main():
    specs = {"pure_m5": spec("pure_m5", "configs/experiments/rev_m5_rolling_world_model.yaml", DAILY, False)}
    for c in CATS:
        specs[f"pure_dominicks_{c}"] = spec(f"pure_dominicks_{c}", f"configs/experiments/rev_dominicks_{c}_rolling_world_model.yaml",
                                            WEEKLY, True)
    for name, s in specs.items():
        (OUT / f"{name}.yaml").write_text(HEADER + yaml.safe_dump(s, sort_keys=False))
    # study N: the same loop on LightGBM, from each data set's revision config
    keys = [("m5", DAILY, False)] + [(f"dominicks_{c}", WEEKLY, True) for c in CATS]
    for key, evaluation, weekly in keys:
        base_path = f"configs/experiments/rev_{key}_rolling_lightgbm.yaml"
        base = yaml.safe_load((ROOT / base_path).read_text())
        s = {"name": f"pure_lightgbm_{key}", "base": base_path, "evaluation": evaluation, "seeds": [1, 2, 3],
             "rung_fraction": 1.0, "promote": "beats_best",
             # every core of the machine; the threads change how fast a run is, not what it models
             "base_overrides": {"threads": 16},
             "menu": lightgbm_menu(base, weekly)}
        (OUT / f"pure_lightgbm_{key}.yaml").write_text(HEADER_N + yaml.safe_dump(s, sort_keys=False))


if __name__ == "__main__":
    main()
