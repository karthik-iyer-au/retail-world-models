"""Study U, part (a) (docs/PAPER1_REVISION_PLAN.md, entry of 9 November 2026): one more round for a data
set whose world-model search and study T round have ended, trying JEPA pretraining followed by hybrid
fine-tuning (the forecast loss with the latent loss kept on, the encoder at a tenth of the rate), on top of
its final settings.

  python scripts/make_hybrid_round.py configs/research/<study T spec>.yaml results/research/<study T name>/log.jsonl

The input is the study T round (its base already carries the world-model loop's kept changes; its log adds
study T's kept change, if any). Writes configs/experiments/hybridU_<name>.yaml and
configs/research/hybridU_<name>.yaml with the same windows, seeds and keep rule, the keep bar measured again
on these settings, and a menu of two recipes fixed in advance from the early Frat JEPA study
(pretrain_low_lr_joint, and the same with three times the pretraining). Run it as a loop limited to one round.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rwm.research import loop  # noqa: E402


def menu(steps: int) -> list:
    """Pretraining length is set from the final settings' training steps, as in the Frat study
    (pretraining equal to training, and three times it)."""
    hybrid = {"finetune": "low_lr", "encoder_lr_scale": 0.1, "latent_weight": 0.5}
    return [("jepa_hybrid", {**hybrid, "pretrain_steps": steps}),
            ("jepa_hybrid_long", {**hybrid, "pretrain_steps": 3 * steps})]


def kept_changes(rows: list) -> list:
    kept = []
    for r in rows:
        if r["kind"] == "kept" and r["change"] not in kept:
            kept.append(r["change"])
    return kept


def main(spec_path: str, log_path: str) -> None:
    import json

    spec = loop.load_spec(spec_path)
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    if not any(r["kind"] == "end" for r in rows):
        raise SystemExit("the study T round has not ended")
    kept = kept_changes(rows)
    changes = {c["name"]: c["change"] for c in spec["menu"]}
    params = loop.base_params(spec)
    for name in kept:
        params = loop.merge(params, changes[name])
    base = yaml.safe_load((ROOT / spec["base"]).read_text())
    wrapped = base["model"]["name"] == "seed_average"
    inner = params.get("params", params) if wrapped else params
    steps = int(inner["steps"])
    stem = spec["name"].removeprefix("lossT_r1_").removeprefix("lossT_")
    name = f"hybridU_{stem}"
    cfg = {**base, "name": name, "model": {**base["model"], "params": params}}
    exp = ROOT / "configs" / "experiments" / f"{name}.yaml"
    exp.write_text(f"# Study U part (a) (plan entry of 9 November 2026): {spec['base']} with study T's kept changes "
                   f"({', '.join(kept) or 'none'}); written by scripts/make_hybrid_round.py.\n"
                   + yaml.safe_dump(cfg, sort_keys=False))
    out = {"name": name, "base": str(exp.relative_to(ROOT)), "evaluation": spec["evaluation"], "seeds": spec["seeds"],
           "rung_fraction": spec.get("rung_fraction", 1.0), "promote": spec.get("promote", "beats_best"),
           "menu": [{"name": n, "change": {"params": c} if wrapped else c} for n, c in menu(steps)]}
    for k in ("base_overrides",):
        if k in spec:
            out[k] = spec[k]
    path = ROOT / "configs" / "research" / f"{name}.yaml"
    path.write_text("# Study U part (a) (docs/PAPER1_REVISION_PLAN.md, entry of 9 November 2026): JEPA pretraining and\n"
                    f"# hybrid fine-tuning on top of the final settings of {stem}; written by scripts/make_hybrid_round.py.\n"
                    "# Run with one round.\n" + yaml.safe_dump(out, sort_keys=False))
    print(path.relative_to(ROOT), "kept:", kept, "steps:", steps)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
