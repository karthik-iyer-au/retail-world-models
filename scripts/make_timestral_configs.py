"""Write the study F configs and jobs (docs/PAPER1_REVISION_PLAN.md, amendment of
17 October 2026): Timestral-209M on the rolling windows of every data set, in a
history-only form and a plan form, each with the windows, context and plan
inputs of the matching TimesFM config.

  python scripts/make_timestral_configs.py
"""
import copy

import yaml

from make_revision_configs import CATS, JOBS, load, write

KEYS = [f"dominicks_{c}" for c in CATS] + ["frat", "m5", "favorita"]
HEADER = "Study F (plan amendment of 17 October 2026): Timestral-209M, windows and inputs of {src}; written by scripts/make_timestral_configs.py."


def timestral(src: dict, plan: bool) -> dict:
    cfg = copy.deepcopy(src)
    p = src["model"]["params"]
    params = {"context": p.get("context", 512), "batch_size": 256, "repo": "mistralai/Timestral-209M"}
    if plan:
        params.update(plan=p.get("plan", []), price_plan=p.get("price_plan", False))
    cfg["model"] = {"name": "timestral", "params": params}
    return cfg


def main():
    paths = {}
    for key in KEYS:
        src = f"rev_{key}_rolling_timesfm_plan"
        base = load(f"{src}.yaml")
        for form, plan in (("plan", True), ("mean", False)):
            name = f"rev_{key}_rolling_timestral_{form}"
            paths.setdefault(key, []).append(write(name, HEADER.format(src=src), timestral(base, plan)))
    weekly = {"datasets": [{"name": "breakfast_at_the_frat", "from": "/content/drive/MyDrive/retail-world-model/data"},
                           {"name": "dominicks", "codes": CATS}],
              "runs": [{"config": c} for k in KEYS if k.startswith(("dominicks", "frat")) for c in paths[k]]}
    daily = {"datasets": [{"name": "favorita", "from": "/content/drive/MyDrive/retail-world-model/data/favorita"},
                          {"name": "m5", "from": "/content/drive/MyDrive/retail-world-model/data/m5"}],
             "runs": [{"config": c} for k in ("favorita", "m5") for c in paths[k]]}
    for name, job in (("timestral_weekly", weekly), ("timestral_daily", daily)):
        (JOBS / f"{name}.yaml").write_text(f"# Study F, Timestral ({name.split('_')[1]} data).\n" + yaml.safe_dump(job, sort_keys=False))


if __name__ == "__main__":
    main()
