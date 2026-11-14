"""Configs for the 52-week test with each model's settings tuned for 52 weeks (plan entry of 14 November 2026).

  python scripts/make_tuned_long_horizon_configs.py --logs <folder with h52s2_<model>_<key>/log.jsonl>

For each Dominick's category and model, reads the search's 'end' row (the changes it kept, in order), applies them to
the search's starting settings (configs/research/h52s2_<model>_<key>.yaml: base config plus menu changes, as
src/rwm/research/loop.py does), and writes configs/experiments/h52t_<key>_<model>_<setup>_w<window>.yaml from the
untuned 52-week configs (h52_<key>_<model>_<setup>_w<window>.yaml) with only the copy's settings replaced: five copies,
as there. The world model forecasting all 52 weeks at once; LightGBM directly and by block.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwm.research.loop import base_params, merge  # noqa: E402

CATS = ["ana", "bjc", "che", "fre", "sdr", "sna"]
SETUPS = {"world_model": ["direct"], "lightgbm": ["direct", "byblock"]}


def tuned_inner(spec: dict, kept: list[str]) -> dict:
    params = base_params(spec)
    menu = {m["name"]: m["change"] for m in spec["menu"]}
    for name in kept:
        params = merge(params, menu[name])
    return params["params"]  # the copy's settings inside the seed average


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    a = p.parse_args(argv)
    summary = {}
    for c in CATS:
        key = f"dominicks_{c}"
        for which, setups in SETUPS.items():
            name = f"h52s2_{which}_{key}"
            spec = yaml.safe_load((ROOT / f"configs/research/{name}.yaml").read_text())
            rows = [json.loads(x) for x in (Path(a.logs) / name / "log.jsonl").read_text().splitlines() if x.strip()]
            end = [r for r in rows if r.get("kind") == "end"]
            if not end:
                raise SystemExit(f"{name} has not finished")
            kept = end[-1]["kept"]
            inner = tuned_inner(spec, kept)
            summary[name] = {"kept": kept, "score": end[-1]["score"], "held_out": end[-1]["held_out"]}
            for setup in setups:
                for w in (0, 1):
                    src = ROOT / f"configs/experiments/h52_{key}_{which}_{setup}_w{w}.yaml"
                    cfg = yaml.safe_load(src.read_text())
                    holder = cfg["model"]["params"] if setup == "direct" else cfg["model"]["params"]["params"]
                    old = holder["params"]
                    new = copy.deepcopy(inner)
                    if setup == "byblock":
                        new.pop("horizon", None)  # set per block by rwm.by_block
                    elif "horizon" in old:
                        new["horizon"] = old["horizon"]
                    holder["params"] = new
                    cfg["name"] = f"h52t_{key}_{which}_{setup}_w{w}"
                    head = (f"# 52-week test with settings tuned for 52 weeks (plan entry of 14 November 2026): {src.name} "
                            f"with the copy's settings from the search {name} (kept: {', '.join(kept) or 'none'}); "
                            "written by scripts/make_tuned_long_horizon_configs.py.\n")
                    (ROOT / f"configs/experiments/{cfg['name']}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    (ROOT / "results" / "long_horizon").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "long_horizon" / "search_kept.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
