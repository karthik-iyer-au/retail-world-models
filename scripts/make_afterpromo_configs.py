"""Write the after-promotion loss-weight study (docs/PAPER1_REVISION_PLAN.md,
amendment of 16 October 2026, study E).

  python scripts/make_afterpromo_configs.py

Each config is the development-window world model of the pantry study
(dominicks_{c}_dev_pantry_world_model) with only `after_promo_weight` added.
The world model itself is run again in the same job, so every comparison is
between runs on the same runtime.
"""
import copy

import yaml

from make_revision_configs import CATS, EXP, JOBS, load

WEIGHTS = [2, 5]


def main():
    runs = []
    for c in CATS:
        base = load(f"dominicks_{c}_dev_pantry_world_model.yaml")
        runs.append({"config": f"configs/experiments/dominicks_{c}_dev_pantry_world_model.yaml", "seeds": [1]})
        for w in WEIGHTS:
            cfg = copy.deepcopy(base)
            cfg["name"] = f"dominicks_{c}_dev_afterpromo_w{w}"
            cfg["model"]["params"]["after_promo_weight"] = w
            path = EXP / f"{cfg['name']}.yaml"
            path.write_text(f"# After-promotion loss weight {w} (plan amendment of 16 October 2026, study E); "
                            "written by scripts/make_afterpromo_configs.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
            runs.append({"config": f"configs/experiments/{path.name}", "seeds": [1]})
    (JOBS / "afterpromo_weight.yaml").write_text(
        "# Study E: after-promotion loss weight, development windows, six Dominick's categories, with the world model "
        "rerun on the same runtime.\n" + yaml.safe_dump({"datasets": [{"name": "dominicks", "codes": CATS}], "runs": runs},
                                                         sort_keys=False))
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
