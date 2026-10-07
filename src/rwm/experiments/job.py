"""Run a job: a named list of experiment runs, kept as a file in the repository.

  python -m rwm.experiments.job --job configs/jobs/frat_seeds.yaml --out <folder>

A job file lists the datasets to fetch and the runs to make:

  datasets:
    - name: m5
    - name: breakfast_at_the_frat
      from: /folder/that/holds/the/file     # for data with no public download
  runs:
    - config: configs/experiments/frat_dev_lightgbm.yaml
      seeds: [1, 2, 3]

Every run is strict: committed code and data that matches its checksums.
A run already saved in the output folder for exactly the same config is kept
and not repeated, so a job that was cut short can be started again. The job
ends by printing one summary line per run.
"""
import argparse
import json
import traceback
from pathlib import Path

import yaml

from rwm.data import fetch
from rwm.experiments.run import run, with_seed
from rwm.utils.hashing import sha256_obj
from rwm.utils.paths import REPO_ROOT, RESULTS


def finished_run(out: Path, config: dict) -> Path | None:
    """The folder of an earlier strict run of exactly this config in `out`, if any."""
    tag = f"_{config['name']}_{sha256_obj(config)[:8]}"
    for folder in sorted(out.glob(f"*{tag}")):
        if (folder / "metrics.json").exists() and (folder / "manifest.json").exists():
            return folder
    return None


def summary_row(folder: Path) -> str:
    m = json.loads((folder / "metrics.json").read_text())
    row = f"{folder.name.split('_', 1)[1]:<48} overall {m['wrmsse']:.4f}"
    mean = lambda v: sum(v) / len(v)
    levels = m["splits"][0].get("by_level", [])
    if len(levels) > 1:  # the top and bottom of the hierarchy, to show where the error sits
        for i in (0, -1):
            row += f" | {levels[i]['level']} {mean([s['by_level'][i]['wrmsse'] for s in m['splits']]):.4f}"
    if "by_horizon" in m["splits"][0]:  # first and last period ahead
        first = mean([s["by_horizon"][0] for s in m["splits"]])
        final = mean([s["by_horizon"][-1] for s in m["splits"]])
        row += f" | ahead 1 {first:.4f} last {final:.4f}"
    if "peaks" in m["splits"][0]:
        for g in ("promoted", "after_promo", "ordinary"):
            fa = mean([s["peaks"][g]["forecast_to_actual"] for s in m["splits"]])
            er = mean([s["peaks"][g]["rmsse"] for s in m["splits"]])
            row += f" | {g} f/a {fa:.3f} err {er:.3f}"
    if "plan_pair_order" in m["splits"][0]:
        row += f" | plan pairs right {mean([s['plan_pair_order'] for s in m['splits']]):.3f} change error {mean([s['plan_pair_change_error'] for s in m['splits']]):.3f}"
    if "rare_plans" in m["splits"][0]:
        row += f" | rare plans {mean([s['rare_plans'] for s in m['splits']]):.3f} usual {mean([s['usual_plans'] for s in m['splits']]):.3f}"
    if "new_items" in m["splits"][0]:
        row += f" | new items {mean([s['new_items'] for s in m['splits']]):.3f} known {mean([s['known_items'] for s in m['splits']]):.3f}"
    if "forecast_to_actual" in m["splits"][0]:
        row += " | forecast/actual " + " ".join(f"{s['forecast_to_actual']:.3f}" for s in m["splits"])
    conditions = m["splits"][0].get("by_condition", {})
    for c in conditions:
        on = [s["by_condition"][c]["on"]["wrmsse"] for s in m["splits"]]
        off = [s["by_condition"][c]["off"]["wrmsse"] for s in m["splits"]]
        row += f" | {c} on {sum(on) / len(on):.4f} off {sum(off) / len(off):.4f}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, nargs="+", help="one or more job files, run in order")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else RESULTS
    for job_file in args.job:
        job = yaml.safe_load(Path(job_file).read_text())
        for d in job.get("datasets", []):
            # one part at a time, so a download that fails does not stop the others;
            # runs that needed the missing part fail on their own and the queue goes on
            for codes in [[c] for c in d.get("codes", [])] or [[]]:
                try:
                    fetch.main([d["name"], *codes, *(["--from", d["from"]] if d.get("from") else [])])
                except BaseException:  # includes a download tool exiting
                    print(f"COULD NOT FETCH {d['name']} {' '.join(codes)}", flush=True)
                    traceback.print_exc()
        folders, failed = [], []
        for r in job["runs"]:
            base = yaml.safe_load((REPO_ROOT / r["config"]).read_text())
            for seed in r.get("seeds", [None]):
                config = with_seed(base, seed)
                # A run already saved for this exact config is kept, not repeated.
                folder = finished_run(out, config)
                if folder is None:
                    try:
                        folder = run(config, strict=True, out_root=out)
                    except Exception:  # one broken run must not stop the rest of the queue
                        failed.append(config["name"])
                        print(f"FAILED {config['name']}", flush=True)
                        traceback.print_exc()
                        continue
                    print(f"finished {folder.name}", flush=True)
                else:
                    print(f"already saved {folder.name}", flush=True)
                folders.append(folder)
        print(f"\nSummary of {job_file} (average over each run's windows; lower is better)")
        for folder in folders:
            print(summary_row(folder), flush=True)
        for name in failed:
            print(f"{name:<48} FAILED, see the error above", flush=True)


if __name__ == "__main__":
    main()
