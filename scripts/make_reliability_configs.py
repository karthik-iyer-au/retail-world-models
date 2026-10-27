"""Write the study K configs and job (docs/PAPER1_REVISION_PLAN.md, entry of
27 October 2026, exploratory): the world model's failures in the rolling
windows, with and without the changes study J pointed to.

  python scripts/make_reliability_configs.py
"""
import copy

import yaml

from make_revision_configs import JOBS, load, write

CATS = ["che", "sna"]
HEADER = "Study K (plan entry of 27 October 2026, exploratory): {variant}, on the settings and windows of {src}; written by scripts/make_reliability_configs.py."
FIX = {"volume_weight": 0.5, "plan_split": True}
VARIANTS = {
    "as_is": {},
    "fix": FIX,
    "fix_weight_average": {**FIX, "weight_average": 0.999},
    "fix_adajepa": {**FIX, "adapt_steps": 1},
    "weight_average": {"weight_average": 0.999},
    "adajepa": {"adapt_steps": 1},
}


def main():
    runs = []
    for cat in CATS:
        src = f"rev_dominicks_{cat}_rolling_world_model"
        base = load(f"{src}.yaml")
        for variant, change in VARIANTS.items():
            cfg = copy.deepcopy(base)
            cfg["model"]["params"].update(change)
            path = write(f"reliability_dominicks_{cat}_{variant}", HEADER.format(variant=variant, src=src), cfg)
            runs.append({"config": path, "seeds": [1, 2, 3]})
    job = {"datasets": [{"name": "dominicks", "codes": CATS}], "runs": runs}
    (JOBS / "study_k_reliability.yaml").write_text("# Study K, the world model's failures with and without the study J changes.\n"
                                                   + yaml.safe_dump(job, sort_keys=False))


if __name__ == "__main__":
    main()
