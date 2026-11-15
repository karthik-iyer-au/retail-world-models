"""Configs for scoring the held-out final periods (plan entry of 15 November 2026; Karthik, 23:01, "850 is okay").

  python scripts/make_heldout_configs.py

Each data set's final world model and final LightGBM (configs/experiments/final_<key>_{world_model,lightgbm}.yaml) with
only the windows changed: the periods every final run held out (24 weeks on Dominick's and Frat, 84 days on M5) are
scored as rolling windows of the paper's length (three of 8 weeks; three of 28 days), and on M5 the competition's final
28 days, from the organisers' test file, are added as a fourth window. Each window trains on the history before it,
as in the paper. Favorita held nothing out (its final windows end on its last day) and is not here.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KEYS = [f"dominicks_{c}" for c in ("ana", "bjc", "che", "fre", "sdr", "sna")] + ["frat", "m5"]


def main():
    n = 0
    for key in KEYS:
        for which in ("world_model", "lightgbm"):
            src = ROOT / f"configs/experiments/final_{key}_{which}.yaml"
            cfg = yaml.safe_load(src.read_text())
            out = copy.deepcopy(cfg)
            ev = out["evaluation"]
            held, h = ev["holdout_periods"], ev["horizon"]
            ev["holdout_periods"] = 0
            ev["n_origins"] = held // h + (1 if key == "m5" else 0)
            ev["step"] = h
            if key == "m5":
                out["dataset"].setdefault("params", {})["include_test"] = True
            out["name"] = f"heldout_{key}_{which}"
            head = (f"# Held-out periods scored (plan entry of 15 November 2026): {src.name} with only the windows changed; "
                    "written by scripts/make_heldout_configs.py.\n")
            (ROOT / f"configs/experiments/{out['name']}.yaml").write_text(head + yaml.safe_dump(out, sort_keys=False))
            n += 1
    print(n, "configs")


if __name__ == "__main__":
    main()
