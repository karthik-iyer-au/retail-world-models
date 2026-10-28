"""Write the study L configs and jobs (docs/PAPER1_REVISION_PLAN.md, entry of
28 October 2026): the world-model fix from study K, confirmed on data it was
not chosen on, and the transfer test rerun with it.

  python scripts/make_confirm_configs.py
"""
import copy

import yaml

from make_revision_configs import JOBS, load, write

FIX = {"volume_weight": 0.5, "plan_split": True, "weight_average": 0.999}
KEYS = ["m5", "favorita", "dominicks_ana", "dominicks_bjc", "dominicks_fre", "dominicks_sdr"]
HEADER = "Study L (plan entry of 28 October 2026): {what}, on the settings and windows of {src}; written by scripts/make_confirm_configs.py."
SEEDS = [1, 2, 3]


def fixed(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model"]["params"].update(FIX)
    return cfg


def main():
    confirm = []
    for key in KEYS:
        src = f"rev_{key}_rolling_world_model"
        base = load(f"{src}.yaml")
        for variant, cfg in (("as_is", copy.deepcopy(base)), ("fix", fixed(base))):
            path = write(f"confirm_{key}_{variant}", HEADER.format(what=f"world model {variant}", src=src), cfg)
            confirm.append({"config": path, "seeds": SEEDS})
    transfer = []
    for cat in ("che", "sna"):
        for kind in ("store", "item"):
            for form in ("kept_out", "trained_on_all"):
                src = f"transfer_dominicks_{cat}_{kind}_world_model_{form}"
                path = write(f"{src}_fix", HEADER.format(what="world model with the fix", src=src), fixed(load(f"{src}.yaml")))
                transfer.append({"config": path, "seeds": SEEDS})
    dom = [k.split("_")[1] for k in KEYS if k.startswith("dominicks")]
    jobs = {
        "study_l_confirm": {"datasets": [{"name": "m5"}, {"name": "favorita"}, {"name": "dominicks", "codes": dom}],
                            "runs": confirm},
        "study_l_transfer": {"datasets": [{"name": "dominicks", "codes": ["che", "sna"]}], "runs": transfer},
    }
    for name, job in jobs.items():
        (JOBS / f"{name}.yaml").write_text(f"# Study L ({name.split('_')[-1]}).\n" + yaml.safe_dump(job, sort_keys=False))


if __name__ == "__main__":
    main()
