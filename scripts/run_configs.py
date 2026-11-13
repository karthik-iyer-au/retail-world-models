"""Run named experiment configs one after another (used for the 52-week test on Modal).

  python scripts/run_configs.py --configs h52_dominicks_che_lightgbm_blocks_w0 ... --out <folder>
"""
import argparse
from pathlib import Path

import yaml

from rwm.experiments.run import run

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfgs = {n: yaml.safe_load((ROOT / f"configs/experiments/{n}.yaml").read_text()) for n in a.configs}
    if any(c["model"]["name"] == "tft" for c in cfgs.values()):  # the Temporal Fusion Transformer's library
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.3.0"], check=True)
    for name in a.configs:
        cfg = cfgs[name]
        print(f"wrote {run(cfg, out_root=Path(a.out))}", flush=True)


if __name__ == "__main__":
    main()
