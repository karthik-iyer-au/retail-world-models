"""Final runs (docs/PAPER1_REVISION_PLAN.md, entry of 10 November 2026): each data set's tuned
world model and tuned LightGBM on the paper's reported rolling windows.

For each model the searches ran as a chain of stages (the main loop, then for the world
model the loss round of study T and the hybrid round of study U). Each stage's base config
already carries the changes kept before it, so the final settings are the last finished
stage's base settings with that stage's kept changes applied, in the order it kept them.
The final config is that stage's base config (the data set's rolling config, with the
paper's windows) under a new name; nothing else changes.

  python scripts/make_final_configs.py            # every data set whose chain has ended
  python scripts/make_final_configs.py --check    # print the chain and settings, write nothing

Writes configs/experiments/final_<data>_{world_model,lightgbm}.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwm.research import loop  # noqa: E402

# data set -> (world-model chain, LightGBM chain), each the research specs in the order they ran
CHAINS = {
    "m5": (["pure_m5", "lossT_pure_m5", "hybridU_pure_m5"], ["pure_lightgbm_m5"]),
    "frat": (["avg5_world_model_frat", "lossT_avg5_world_model_frat", "hybridU_avg5_world_model_frat"],
             ["avg5_lightgbm_frat"]),
    "favorita": (["avg5_world_model_favorita", "lossT_r1_avg5_world_model_favorita"], ["avg5_lightgbm_favorita"]),  # study U rerun stopped early (plan entry of 11 November 2026)
    **{f"dominicks_{c}": ([f"clean_avg5_world_model_dominicks_{c}", f"lossT_clean_avg5_world_model_dominicks_{c}",
                           f"hybridU_clean_avg5_world_model_dominicks_{c}"],
                          [f"clean_avg5_lightgbm_dominicks_{c}"])
       for c in ("ana", "bjc", "che", "fre", "sdr", "sna")},
}


def _rows(name: str) -> list[dict] | None:
    p = ROOT / "results" / "research" / name / "log.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def final_settings(chain: list[str]) -> tuple[dict, dict, str, list[str]]:
    """(base config, final model params, last stage, kept changes of every stage).
    Raises if any stage has not ended."""
    kept_all = []
    for name in chain:
        rows = _rows(name)
        if rows is None or not any(r["kind"] == "end" for r in rows):
            raise LookupError(f"{name} has not ended")
    last = chain[-1]
    spec = loop.load_spec(ROOT / "configs" / "research" / f"{last}.yaml")
    params = loop.base_params(spec)
    changes = {c["name"]: c["change"] for c in spec["menu"]}
    for name in chain:
        for r in _rows(name):
            if r["kind"] == "kept" and f"{name}:{r['change']}" not in kept_all:
                kept_all.append(f"{name}:{r['change']}")
    for r in _rows(last):
        if r["kind"] == "kept":
            params = loop.merge(params, changes[r["change"]])
    base = yaml.safe_load((ROOT / spec["base"]).read_text())
    return base, params, last, kept_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for data, chains in CHAINS.items():
        for kind, chain in zip(("world_model", "lightgbm"), chains):
            try:
                base, params, last, kept = final_settings(chain)
            except LookupError as e:
                print(f"{data} {kind}: waits ({e})")
                continue
            name = f"final_{data}_{kind}"
            cfg = {**base, "name": name, "model": {**base["model"], "params": params}}
            print(f"{data} {kind}: from {last}; kept {', '.join(kept) or 'nothing'}")
            if args.check:
                continue
            out = ROOT / "configs" / "experiments" / f"{name}.yaml"
            out.write_text(f"# Final run (plan entry of 10 November 2026): the tuned settings of {data}'s {kind.replace('_', ' ')}, "
                           f"from {last} (kept: {', '.join(kept) or 'nothing'}), on the paper's rolling windows; "
                           "written by scripts/make_final_configs.py.\n" + yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
