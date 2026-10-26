"""Write the study H configs and jobs (docs/PAPER1_REVISION_PLAN.md, entry of
27 October 2026): TimesFM 3.0 and Chronos-2, history only and with the plan,
and Timestral-209M with the plan, on the rolling windows of every data set,
each with the windows, context and plan inputs of the matching TimesFM config.

  python scripts/make_latest_foundation_configs.py
"""
import copy

import yaml

from make_revision_configs import CATS, JOBS, load, write

KEYS = [f"dominicks_{c}" for c in CATS] + ["frat", "m5", "favorita"]
HEADER = "Study H (plan entry of 27 October 2026): {model}, windows and inputs of {src}; written by scripts/make_latest_foundation_configs.py."
FRAT = {"name": "breakfast_at_the_frat", "from": "/content/drive/MyDrive/retail-world-model/data"}


def latest(src: dict, model: str, plan: bool) -> dict:
    cfg = copy.deepcopy(src)
    p = src["model"]["params"]
    params = {"context": p.get("context", 512), "batch_size": 256}
    if plan:
        params.update(plan=p.get("plan", []), price_plan=p.get("price_plan", False))
    if model == "timestral":
        params["repo"] = "mistralai/Timestral-209M"
    cfg["model"] = {"name": model, "params": params}
    return cfg


def main():
    paths = {}
    for key in KEYS:
        src = f"rev_{key}_rolling_timesfm_plan"
        base = load(f"{src}.yaml")
        for model, form, plan in (("timesfm3", "mean", False), ("timesfm3", "plan", True),
                                  ("chronos2", "mean", False), ("chronos2", "plan", True),
                                  ("timestral", "plan", True)):
            name = f"rev_{key}_rolling_{model}{'209' if model == 'timestral' else ''}_{form}"
            if name == "rev_frat_rolling_timestral209_plan":  # already run in study F's size check
                paths.setdefault(key, []).append(f"configs/experiments/{name}.yaml")
                continue
            label = {"timesfm3": "TimesFM 3.0", "chronos2": "Chronos-2", "timestral": "Timestral-209M"}[model]
            paths.setdefault(key, []).append(write(name, HEADER.format(model=label, src=src), latest(base, model, plan)))
    jobs = {
        "latest_foundation_weekly": {
            "datasets": [FRAT, {"name": "dominicks", "codes": CATS}],
            "runs": [{"config": c} for k in KEYS if k.startswith(("dominicks", "frat")) for c in paths[k]]},
        "latest_foundation_daily": {
            "datasets": [{"name": "m5"}, {"name": "favorita"}],
            "runs": [{"config": c} for k in ("m5", "favorita") for c in paths[k]]},
    }
    for name, job in jobs.items():
        (JOBS / f"{name}.yaml").write_text(f"# Study H, the latest foundation models ({name.split('_')[-1]} data).\n"
                                           + yaml.safe_dump(job, sort_keys=False))


if __name__ == "__main__":
    main()
