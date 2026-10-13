"""Write the paper 1 revision configs and jobs (docs/PAPER1_REVISION_PLAN.md).

  python scripts/make_revision_configs.py

Every config is derived from a final config of the companion paper, so its
settings cannot drift: only the evaluation windows change, plus, for the
rolling windows, one fit (seed set by the job) in place of an average of three.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "configs" / "experiments"
JOBS = ROOT / "configs" / "jobs"
CATS = ["che", "sna", "sdr", "bjc", "ana", "fre"]
LIFT_DROP = ["rollout", "roll_norm", "plan_split", "feedback", "rollout_discount"]


def load(name):
    return yaml.safe_load((EXP / name).read_text())


def single(model):
    """An average of fits replaced by one fit with the same settings."""
    if model["name"] == "seed_average":
        p = copy.deepcopy(model["params"])
        return {"name": p["model"], "params": p["params"]}
    return copy.deepcopy(model)


def lift(model):
    """The transition ablation: the same network, read from the state before the window."""
    m = single(model)
    for k in LIFT_DROP:
        m["params"].pop(k, None)
    return m


def write(name, header, cfg):
    cfg = {"name": name, **{k: v for k, v in cfg.items() if k != "name"}}
    (EXP / f"{name}.yaml").write_text(f"# {header}\n" + yaml.safe_dump(cfg, sort_keys=False))
    return f"configs/experiments/{name}.yaml"


def models_for(dataset, cat=None):
    """Each model's config body, taken from the companion paper's final configs."""
    if dataset == "dominicks":
        pre = f"dominicks_{cat}_final_"
        wm = load(pre + "world_model_switching_avg3_roll_norm.yaml")
        lgb, tf_mean, tf_plan = load(pre + "lightgbm.yaml"), load(pre + "timesfm_mean.yaml"), load(pre + "timesfm_plan.yaml")
        levers, train_periods, input_size, season = ["promo"], 156, 26, 52
    elif dataset == "frat":
        wm, lgb = load("frat_final_world_model.yaml"), load("frat_final_lightgbm.yaml")
        tf_mean, tf_plan = load("frat_final_timesfm_mean.yaml"), load("frat_final_timesfm_plan.yaml")
        levers, train_periods, input_size, season = ["promo", "display", "feature", "tpr_only"], 90, 26, 52
    else:
        wm, lgb = load("m5_test_world_model_roll_norm.yaml"), load("m5_test_lightgbm.yaml")
        tf_mean, tf_plan = load("m5_test_timesfm_mean.yaml"), load("m5_test_timesfm_plan.yaml")
        levers, train_periods, input_size, season = ["event", "snap"], None, None, 7
    out = {
        "world_model": single(wm["model"]),
        "lift": (lift(wm["model"]) if dataset == "dominicks"
                 else single(load("frat_final_lift.yaml" if dataset == "frat" else "m5_test_lift.yaml")["model"])),
        "lightgbm": lgb["model"],
        "seasonal_naive": {"name": "seasonal_naive", "params": {"season": season}},
        "timesfm_mean": tf_mean["model"],
        "timesfm_plan": tf_plan["model"],
        "chronos": {"name": "chronos", "params": {"context": 512}},
    }
    if dataset != "m5":
        out["tft"] = {"name": "tft", "params": {"horizon": 8, "input_size": input_size, "train_periods": train_periods,
                                                "extra": levers, "max_steps": 1500}}
    return out, wm


def main():
    weekly_runs, m5_runs, final_runs = [], [], []
    for dataset, cats in (("dominicks", CATS), ("frat", [None]), ("m5", [None])):
        for cat in cats:
            body, wm = models_for(dataset, cat)
            tag = f"dominicks_{cat}" if cat else dataset
            ds = copy.deepcopy(wm["dataset"])
            ev = copy.deepcopy(wm["evaluation"])
            if dataset == "m5":
                ds = {"name": "m5"}  # data to 22 May 2016; the three months already reported are held out
                ev.update({"n_origins": 4, "holdout_periods": 84, "step": 28})
            else:
                ev.update({"n_origins": 6 if dataset == "dominicks" else 4, "holdout_periods": 24, "step": 8})
            for mname, model in body.items():
                header = (f"Paper 1 revision, rolling windows before the development windows; settings from the "
                          f"companion paper's final config for {tag} (docs/PAPER1_REVISION_PLAN.md).")
                path = write(f"rev_{tag}_rolling_{mname}", header,
                             {"seed": 0, "dataset": ds, "model": model, "evaluation": ev})
                neural = mname in ("world_model", "lift", "tft")
                (m5_runs if dataset == "m5" else weekly_runs).append({"config": path, **({"seeds": [1]} if neural else {})})
                # the final window for the models the companion paper did not run there
                new_here = mname in ("chronos", "tft") or (mname == "lift" and dataset == "dominicks")
                if new_here:
                    fev = copy.deepcopy(wm["evaluation"])
                    fds = copy.deepcopy(wm["dataset"])
                    path = write(f"rev_{tag}_finalwindow_{mname}",
                                 f"Paper 1 revision, the final window, a model not run there before; settings as {tag}'s rolling config.",
                                 {"seed": 0, "dataset": fds, "model": model, "evaluation": fev})
                    final_runs.append({"config": path, **({"seeds": [1]} if neural else {})})
    weekly = {"datasets": [{"name": "breakfast_at_the_frat", "from": "/content/drive/MyDrive/retail-world-model/data"},
                           {"name": "dominicks", "codes": CATS}],
              "runs": [r for r in final_runs if "m5" not in r["config"]] + weekly_runs}
    m5 = {"datasets": [{"name": "m5"}], "runs": [r for r in final_runs if "m5" in r["config"]] + m5_runs}
    (JOBS / "paper1_revision_weekly.yaml").write_text(
        "# Paper 1 revision: Dominick's and Frat, final window for new models, then rolling windows.\n"
        + yaml.safe_dump(weekly, sort_keys=False))
    (JOBS / "paper1_revision_m5.yaml").write_text(
        "# Paper 1 revision: M5, final window for new models, then four earlier months.\n"
        + yaml.safe_dump(m5, sort_keys=False))
    print(len(weekly["runs"]), "weekly runs,", len(m5["runs"]), "M5 runs")


if __name__ == "__main__":
    main()
