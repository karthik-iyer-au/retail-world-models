"""Write the study I configs and job (docs/PAPER1_REVISION_PLAN.md, entry of
27 October 2026): the transfer test. A fifth of stores, or of items, is kept
out of training in every rolling window; the model then reads their own
history without learning from it and forecasts them. Each kept-out run has a
comparison run trained on everything and scored on the same fifth.

  python scripts/make_transfer_configs.py
"""
import copy

import yaml

from make_revision_configs import JOBS, load, write

KEYS = ["m5", "dominicks_che", "dominicks_sna"]
HEADER = "Study I (plan entry of 27 October 2026): transfer to {by}s not trained on, from {src}; written by scripts/make_transfer_configs.py."


def transfer(src: dict, by: str, trained_on_all: bool, pooled: bool) -> dict:
    cfg = copy.deepcopy(src)
    cfg["evaluation"]["transfer"] = {"by": by, "share": 0.2, "seed": 0, "trained_on_all": trained_on_all}
    p = cfg["model"]["params"]
    if pooled and p.get("group_by"):
        # one model per store cannot forecast a store it never saw: one model for all
        # stores, with the store as an input (missing for a store kept out)
        p["group_by"] = None
        p["categorical"] = list(p.get("categorical", [])) + ["store_id"]
    return cfg


def main():
    runs = []
    for key in KEYS:
        for by in ("store_id", "item_id"):
            short = by.split("_")[0]
            for model in ("world_model", "lightgbm", "seasonal_naive"):
                src = f"rev_{key}_rolling_{model}"
                base = load(f"{src}.yaml")
                pooled = model == "lightgbm" and by == "store_id"
                for form, all_ in (("kept_out", False), ("trained_on_all", True)):
                    if model == "seasonal_naive" and all_:
                        continue  # learns nothing from history: the same either way
                    name = f"transfer_{key}_{short}_{model}{'_pooled' if pooled else ''}_{form}"
                    path = write(name, HEADER.format(by=short, src=src), transfer(base, by, all_, pooled))
                    runs.append({"config": path, **({"seeds": [1, 2, 3]} if model == "world_model" else {})})
    job = {"datasets": [{"name": "m5"}, {"name": "dominicks", "codes": ["che", "sna"]}], "runs": runs}
    (JOBS / "study_i_transfer.yaml").write_text("# Study I, transfer to stores and items not trained on.\n"
                                                 + yaml.safe_dump(job, sort_keys=False))


if __name__ == "__main__":
    main()
