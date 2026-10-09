"""Table of every saved run: what it ran on and how long it took.

  python scripts/run_times.py <results folder> [--out run_times.csv]

Runs saved before run times were recorded get their finish time from when
metrics.json was written; their start time is in the manifest either way.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def row(folder: Path) -> dict | None:
    man_p, met_p = folder / "manifest.json", folder / "metrics.json"
    if not (man_p.exists() and met_p.exists()):
        return None
    man = json.loads(man_p.read_text())
    met = json.loads(met_p.read_text())
    start = datetime.fromisoformat(man["created_utc"])
    if "finished_utc" in man:
        end, how = datetime.fromisoformat(man["finished_utc"]), "recorded"
    else:
        end, how = datetime.fromtimestamp(met_p.stat().st_mtime, timezone.utc), "file time"
    hw = man.get("hardware", {})
    return {
        "run": folder.name,
        "config": man["config"]["name"],
        "seed": man.get("seed"),
        "commit": (man["git"].get("commit") or "")[:8],
        "device": man.get("device"),
        "gpu_memory_gb": hw.get("gpu_memory_gb"),
        "cpu": hw.get("cpu"),
        "logical_cores": hw.get("logical_cores"),
        "memory_gb": hw.get("memory_gb"),
        "started_utc": start.isoformat(timespec="seconds"),
        "finished_utc": end.isoformat(timespec="seconds"),
        "wall_minutes": round(man.get("wall_seconds", (end - start).total_seconds()) / 60, 1),
        "wall_source": how,
        "wrmsse": round(met["wrmsse"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="run_times.csv")
    args = ap.parse_args()
    rows = [r for f in sorted(Path(args.results).iterdir()) if f.is_dir() and (r := row(f))]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["wall_minutes"] for r in rows) / 60
    print(f"{len(rows)} runs, {total:.1f} GPU-hours in total, written to {args.out}")


if __name__ == "__main__":
    main()
