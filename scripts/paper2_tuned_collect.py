"""Gather the paper 2 audit of the tuned models into small tables (plan amendment of 13 November 2026).

  python scripts/paper2_tuned_collect.py --results <rwm-results root> --out <folder>

Builds two folders of links and runs scripts/audit_summary.py on each:

  final/    the nine final tuned world models (their audit/ folders), the final LightGBM audits (C6) and the
            zone-price episodes (C7)  -> <out>/final/{versions,stability,agreement,zones}.csv
  search/   every world-model version the settings searches saved, with its audit  -> <out>/search/...

and copies the plan read-backs (C8) to <out>/readback/. Reads; writes nothing beside <out>.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SEARCHES = ([f"{p}_world_model_dominicks_{k}" for k in ("ana", "bjc", "che", "fre", "sdr", "sna")
             for p in ("clean_avg5", "lossT_clean_avg5", "hybridU_clean_avg5")]
            + ["avg5_world_model_frat", "lossT_avg5_world_model_frat", "hybridU_avg5_world_model_frat",
               "pure_m5", "lossT_pure_m5", "hybridU_pure_m5", "avg5_world_model_favorita",
               "lossT_r1_avg5_world_model_favorita", "hybridU_r1_avg5_world_model_favorita",
               "hybridU_avg5_world_model_favorita"])


def link(src: Path, dst: Path):
    if not dst.exists():
        dst.symlink_to(src)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    tmp = Path("/tmp/paper2_collect")
    shutil.rmtree(tmp, ignore_errors=True)

    fin = tmp / "final"
    fin.mkdir(parents=True)
    for d in (res / "final").iterdir():
        if "_world_model_" in d.name and "_piece_" not in d.name and (d / "audit").exists():
            link(d, fin / d.name)
    tuned = res / "paper2_tuned"
    for d in tuned.glob("paper2_lightgbm_*"):
        link(d, fin / d.name)
    (fin / "paper2_zones").mkdir()
    for f in tuned.glob("zones_*/paper2_zones/*.json"):
        link(f, fin / "paper2_zones" / f.name)

    sea = tmp / "search"
    sea.mkdir()
    for s in SEARCHES:
        for d in (res / "research" / s).glob("*"):
            if (d / "audit").exists():
                link(d, sea / d.name)

    for name in ("final", "search"):
        subprocess.run([sys.executable, "scripts/audit_summary.py", "--results", str(tmp / name),
                        "--out", str(out / name)], check=True)
    rb = out / "readback"
    rb.mkdir(parents=True, exist_ok=True)
    for f in tuned.glob("readback_*.json"):
        shutil.copy(f, rb / f.name)
    print(f"readback files: {len(list(rb.glob('*.json')))}")


if __name__ == "__main__":
    main()
