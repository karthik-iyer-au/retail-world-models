"""Write the pantry-loading study configs and job (docs/PAPER1_REVISION_PLAN.md,
amendment of 15 October 2026, study C).

  python scripts/make_pantry_configs.py

Each config is the development config of the final Dominick's world model
(one fit in place of an average of three, as in the rolling windows), with
only the change under test:
  world_model        unchanged
  world_model_stock  + the stock term
  lift               roll-forward off (the transition ablation)
  lift_lags          + the plan of the two periods before each forecast period
  lift_stock         + the stock term, without the roll-forward
"""
import copy

import yaml

from make_revision_configs import CATS, EXP, JOBS, lift, load, single

VARIANTS = {
    "world_model": lambda m: m,
    "world_model_stock": lambda m: {**m, "params": {**m["params"], "stock": True}},
    "lift": lift,
    "lift_lags": lambda m: {**lift(m), "params": {**lift(m)["params"], "plan_lags": 2}},
    "lift_stock": lambda m: {**lift(m), "params": {**lift(m)["params"], "stock": True}},
}


def main():
    runs = []
    for c in CATS:
        base = load(f"dominicks_{c}_dev_peaks_world_model_switching_avg3_roll_norm.yaml")
        for v, change in VARIANTS.items():
            cfg = copy.deepcopy(base)
            cfg["name"] = f"dominicks_{c}_dev_pantry_{v}"
            cfg["model"] = change(single(base["model"]))
            path = EXP / f"{cfg['name']}.yaml"
            header = (f"# Pantry-loading study (plan amendment of 15 October 2026): {v} on the development "
                      f"windows of dominicks_{c}; written by scripts/make_pantry_configs.py.\n")
            path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
            runs.append({"config": f"configs/experiments/{path.name}", "seeds": [1]})
    job = {"datasets": [{"name": "dominicks", "codes": CATS}], "runs": runs}
    (JOBS / "pantry_loading.yaml").write_text(
        "# Pantry-loading study, development windows, six Dominick's categories, seed 1 (stage 1).\n"
        + yaml.safe_dump(job, sort_keys=False))
    print(len(runs), "configs")
    # stage 2 (plan, after stage 1): seeds 2 and 3 on the development windows, and the rolling windows
    stage2 = [{"config": f"configs/experiments/dominicks_{c}_dev_pantry_{v}.yaml", "seeds": [2, 3]}
              for c in CATS for v in ("world_model", "lift_stock")]
    for c in CATS:
        cfg = load(f"rev_dominicks_{c}_rolling_lift.yaml")
        cfg["name"] = f"rev_dominicks_{c}_rolling_lift_stock"
        cfg["model"]["params"]["stock"] = True
        (EXP / f"{cfg['name']}.yaml").write_text(
            "# Study C stage 2: the lift model with the stock term on the rolling windows; "
            "written by scripts/make_pantry_configs.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
    rolling = [{"config": f"configs/experiments/rev_dominicks_{c}_rolling_lift_stock.yaml", "seeds": [1]} for c in CATS]
    for name, rs in (("pantry_stage2_seeds", stage2), ("pantry_stage2_rolling", rolling)):
        (JOBS / f"{name}.yaml").write_text(f"# Study C stage 2: {name.split('_')[-1]}.\n" + yaml.safe_dump(
            {"datasets": [{"name": "dominicks", "codes": CATS}], "runs": rs}, sort_keys=False))


if __name__ == "__main__":
    main()
