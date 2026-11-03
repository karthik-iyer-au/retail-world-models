"""Study T (docs/PAPER1_REVISION_PLAN.md, entry of 4 November 2026): one more round for a
finished world-model loop, trying count likelihoods that treat periods with no sales
on their own, on top of the loop's final settings.

  python scripts/make_loss_round.py configs/research/<world-model spec>.yaml results/research/<name>/log.jsonl

Writes configs/experiments/lossT_<name>.yaml (the loop's base config with its kept
changes applied, in the order the loop kept them) and configs/research/lossT_<name>.yaml
(the same windows, seeds and keep rule, the loop's keep bar measured again on these
settings, and a menu of the four likelihood changes). Run it as a loop limited to one round.
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwm.research import loop  # noqa: E402

# the four changes, each on top of the final settings
MENU = [
    ("negative_binomial", {"likelihood": "negative_binomial"}),
    ("negative_binomial_item_spread", {"likelihood": "negative_binomial", "spread_by_item": True}),
    ("zinb_item_spread", {"likelihood": "zinb", "spread_by_item": True}),
    ("hurdle_nb_item_spread", {"likelihood": "hurdle_nb", "spread_by_item": True}),
]


def main(spec_path: str, log_path: str) -> None:
    spec = loop.load_spec(spec_path)
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    if not any(r["kind"] == "end" for r in rows):
        raise SystemExit("the loop has not ended")
    kept = []
    for r in rows:
        if r["kind"] == "kept" and r["change"] not in kept:
            kept.append(r["change"])
    changes = {c["name"]: c["change"] for c in spec["menu"]}
    params = loop.base_params(spec)
    for name in kept:
        params = loop.merge(params, changes[name])
    base = yaml.safe_load((ROOT / spec["base"]).read_text())
    wrapped = base["model"]["name"] == "seed_average"  # a five-copy average: changes apply to the copies
    name = f"lossT_{spec['name']}"
    cfg = {**base, "name": name, "model": {**base["model"], "params": params}}
    exp = ROOT / "configs" / "experiments" / f"{name}.yaml"
    exp.write_text(f"# Study T (plan entry of 4 November 2026): {spec['base']} with the kept changes of {spec['name']} "
                   f"({', '.join(kept) or 'none'}); written by scripts/make_loss_round.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
    out = {"name": name, "base": str(exp.relative_to(ROOT)), "evaluation": spec["evaluation"], "seeds": spec["seeds"],
           "rung_fraction": spec.get("rung_fraction", 1.0), "promote": spec.get("promote", "beats_best"),
           "menu": [{"name": n, "change": {"params": c} if wrapped else c} for n, c in MENU]}
    for k in ("base_overrides",):
        if k in spec:
            out[k] = spec[k]
    path = ROOT / "configs" / "research" / f"{name}.yaml"
    path.write_text("# Study T (docs/PAPER1_REVISION_PLAN.md, entry of 4 November 2026): one round of count likelihoods on top of\n"
                    f"# the final settings of {spec['name']}; written by scripts/make_loss_round.py. Run with one round.\n"
                    + yaml.safe_dump(out, sort_keys=False))
    print(path.relative_to(ROOT), "kept:", kept)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
