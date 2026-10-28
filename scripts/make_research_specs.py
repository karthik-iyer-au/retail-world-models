"""Write the study M research specs (docs/PAPER1_REVISION_PLAN.md, entry of
28 October 2026): the research loop as the companion work ran it, on M5 and
the six Dominick's categories, with every change screened at full training
length and AdaJEPA in its published form.

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


def main():
    specs = {"pure_m5": spec("pure_m5", "configs/experiments/rev_m5_rolling_world_model.yaml", DAILY, False)}
    for c in CATS:
        specs[f"pure_dominicks_{c}"] = spec(f"pure_dominicks_{c}", f"configs/experiments/rev_dominicks_{c}_rolling_world_model.yaml",
                                            WEEKLY, True)
    for name, s in specs.items():
        (OUT / f"{name}.yaml").write_text(HEADER + yaml.safe_dump(s, sort_keys=False))


if __name__ == "__main__":
    main()
