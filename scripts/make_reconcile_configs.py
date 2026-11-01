"""Study P (docs/PAPER1_REVISION_PLAN.md, entry of 1 November 2026): the store
totals forecast directly, by each model with the settings of its study O
starting runs (the five-copy average), on the same search windows, so they can
be reconciled with those runs' item-by-store forecasts.

  python scripts/make_reconcile_configs.py
"""
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwm.research import loop  # noqa: E402

CATS = ["che", "sna", "sdr", "bjc", "ana", "fre"]
OUT = ROOT / "configs" / "experiments"


def main() -> None:
    runs = []
    for cat in CATS:
        for model in ("world_model", "lightgbm"):
            spec = loop.load_spec(f"configs/research/avg5_{model}_dominicks_{cat}.yaml")
            base = yaml.safe_load((ROOT / spec["base"]).read_text())
            name = f"agg_store_{model}_dominicks_{cat}"
            config = {
                "name": name,
                "seed": 0,
                "dataset": {**base["dataset"], "aggregate": {"by": ["store_id"]}},
                "model": {"name": base["model"]["name"], "params": loop.base_params(spec)},
                "evaluation": spec["evaluation"],
            }
            path = OUT / f"{name}.yaml"
            head = (f"# Study P (docs/PAPER1_REVISION_PLAN.md, entry of 1 November 2026): store totals forecast directly,\n"
                    f"# settings of {spec['base']} as its study O loop starts it; written by scripts/make_reconcile_configs.py.\n")
            path.write_text(head + yaml.safe_dump(config, sort_keys=False))
            runs.append({"config": str(path.relative_to(ROOT)), "seeds": spec["seeds"]})
    job = ROOT / "configs" / "jobs" / "study_p_store_totals.yaml"
    job.write_text("# Study P: store totals for the six Dominick's categories, both models, seeds 1 to 3.\n"
                   + yaml.safe_dump({"runs": runs}, sort_keys=False))
    print(f"{len(runs)} configs, job {job.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
