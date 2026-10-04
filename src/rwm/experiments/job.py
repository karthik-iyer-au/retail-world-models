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
"""
import argparse
import json
from pathlib import Path

import yaml

from rwm.data import fetch
from rwm.experiments.run import run, with_seed
from rwm.utils.paths import REPO_ROOT, RESULTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    job = yaml.safe_load(Path(args.job).read_text())
    for d in job.get("datasets", []):
        fetch.main([d["name"], *d.get("codes", []), *(["--from", d["from"]] if d.get("from") else [])])
    out = Path(args.out) if args.out else RESULTS
    for r in job["runs"]:
        config = yaml.safe_load((REPO_ROOT / r["config"]).read_text())
        for seed in r.get("seeds", [None]):
            folder = run(with_seed(config, seed), strict=True, out_root=out)
            score = json.loads((folder / "metrics.json").read_text())["wrmsse"]
            print(f"finished {folder.name}: WRMSSE {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
