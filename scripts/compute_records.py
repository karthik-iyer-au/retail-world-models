"""Wall-clock time of every run on the results volume, from each run's manifest (plan, compute records of
7 November 2026; results/compute/README.md).

  python scripts/compute_records.py --root /vol/results --out <csv>

One row per run folder (split runs' pieces included): top folder, run folder, config name, wall seconds, device,
start and finish times.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = []
    def manifests(folder: Path, depth: int):
        # run folders sit one to three levels down; partial saves and other "_" folders are skipped
        for e in os.scandir(folder):
            if not e.is_dir() or e.name.startswith("_"):
                continue
            m = Path(e.path) / "manifest.json"
            if m.exists():
                yield m
            elif depth > 1:
                yield from manifests(Path(e.path), depth - 1)

    for m in manifests(Path(a.root), 3):
        rel = m.parent.relative_to(a.root)
        try:
            d = json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"top": rel.parts[0], "run": str(rel), "config": d.get("config", {}).get("name"),
                     "wall_seconds": d.get("wall_seconds"), "device": d.get("device"),
                     "created_utc": d.get("created_utc"), "finished_utc": d.get("finished_utc")})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(rows), "runs")


if __name__ == "__main__":
    main()
