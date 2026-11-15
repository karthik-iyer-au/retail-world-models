"""Configs for the history-length test (plan entry of 15 November 2026; Karthik, 14:26, "okay lets do it").

  python scripts/make_history_configs.py

Each data set's final world model and final LightGBM (configs/experiments/final_<key>_{world_model,lightgbm}.yaml),
with only the training length changed (train_periods) and, on Favorita, the data loaded from 1 January 2013 instead
of 2015 so that four years fit before the windows. Windows, settings and copies as in the final runs. The final runs
already give Dominick's at three years and M5 at two; Favorita is run at every length on the 2013 load, so that all
its lengths are scored on the same history.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATS = ["ana", "bjc", "che", "fre", "sdr", "sna"]
LENGTHS = {**{f"dominicks_{c}": {"1y": 52, "2y": 104, "5y": 260} for c in CATS},
           "m5": {"1y": 365, "4y": 1460},
           "favorita": {"1y": 365, "2y": 730, "4y": 1460}}


def set_train_periods(model: dict, n: int) -> int:
    hits = 0
    if isinstance(model, dict):
        for k, v in model.items():
            if k == "train_periods":
                model[k] = n
                hits += 1
            else:
                hits += set_train_periods(v, n)
    return hits


def main():
    n = 0
    for key, lengths in LENGTHS.items():
        for which in ("world_model", "lightgbm"):
            src = ROOT / f"configs/experiments/final_{key}_{which}.yaml"
            base = yaml.safe_load(src.read_text())
            for tag, periods in lengths.items():
                cfg = copy.deepcopy(base)
                assert set_train_periods(cfg["model"], periods) == 1, (key, which)
                if key == "favorita":
                    cfg["dataset"].setdefault("params", {})["start"] = "2013-01-01"
                cfg["name"] = f"hist_{key}_{which}_{tag}"
                head = (f"# History-length test (plan entry of 15 November 2026): {src.name} with only the training "
                        f"length changed to {periods} periods" + (", data from 1 January 2013" if key == "favorita" else "")
                        + "; written by scripts/make_history_configs.py.\n")
                (ROOT / f"configs/experiments/{cfg['name']}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
                n += 1
    print(n, "configs")


if __name__ == "__main__":
    main()
