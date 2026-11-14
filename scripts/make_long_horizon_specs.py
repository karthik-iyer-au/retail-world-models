"""Research specs for tuning both models for 52 weeks (docs/PAPER1_REVISION_PLAN.md, entry of 14 November 2026).

  python scripts/make_long_horizon_specs.py

For each Dominick's category, from the final tuned settings (configs/experiments/final_<key>_<model>.yaml):
configs/experiments/h52s_<key>_<model>.yaml, the model fitted to forecast all 52 weeks at once, one copy; and
configs/research/h52s_<model>_<key>.yaml, the research loop of the paper's searches with the same menus
(scripts/make_research_specs.py) on windows of 52 weeks that end where the 52-week test's windows begin.
"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_research_specs import lightgbm_menu, menu  # noqa: E402

CATS = ["ana", "bjc", "che", "fre", "sdr", "sna"]
# three windows of 52 weeks, 13 weeks apart; the latest ends 102 weeks before the data's end, where the 52-week
# test's earlier window begins (24 held-out weeks, 26 between its two windows, 52 for the window)
EVALUATION = {"horizon": 52, "n_origins": 3, "holdout_periods": 102, "step": 13, "scale_lag": 1, "weight_window": 13,
              "after_promo": {"levers": ["promo"], "periods": 2}}
HEADER = """# Tuning for 52 weeks (docs/PAPER1_REVISION_PLAN.md, entry of 14 November 2026): the research loop
# (src/rwm/research/loop.py) as the paper's searches ran it, with the same menu, from the final tuned settings,
# the model fitted to forecast all 52 weeks at once, one copy. Fixed before the first try and never changed by the
# loop. Every change is tried alone on top of the best settings so far on seed 1; a change that beats the best on
# seed 1 is run on seeds 2 and 3, and the best such change is kept if its mean over the three seeds beats the best's
# mean by at least the seed spread of the starting settings. Three windows of 52 weeks, all before the 52-week
# test's windows; the latest is held out, scored but never used to decide. At most three rounds.
# Written by scripts/make_long_horizon_specs.py.
"""


def main():
    for c in CATS:
        key = f"dominicks_{c}"
        for which in ("world_model", "lightgbm"):
            cfg = yaml.safe_load((ROOT / f"configs/experiments/final_{key}_{which}.yaml").read_text())
            inner = cfg["model"]["params"]["params"]
            inner["horizon"] = 52
            if inner.get("validation_periods") and inner["validation_periods"] < 52:
                inner["validation_periods"] = 52
            cfg["model"]["params"]["members"] = 1
            cfg["evaluation"] = copy.deepcopy(EVALUATION)
            cfg["name"] = f"h52s_{key}_{which}"
            base_path = f"configs/experiments/h52s_{key}_{which}.yaml"
            (ROOT / base_path).write_text("# Base of the 52-week search (written by scripts/make_long_horizon_specs.py)\n"
                                          + yaml.safe_dump(cfg, sort_keys=False))
            pseudo = {"model": {"params": inner}, "evaluation": {"horizon": 52}}
            items = menu(pseudo, True) if which == "world_model" else lightgbm_menu(pseudo, True)
            # the changes apply to the copy's settings, inside the average
            items = [{"name": m["name"], "change": {"params": m["change"]}} for m in items]
            spec = {"name": f"h52s_{which}_{key}", "base": base_path, "evaluation": copy.deepcopy(EVALUATION),
                    "seeds": [1, 2, 3], "rung_fraction": 1.0, "promote": "beats_best", "menu": items}
            if which == "lightgbm":
                spec["base_overrides"] = {"params": {"threads": 16}}
            (ROOT / f"configs/research/h52s_{which}_{key}.yaml").write_text(HEADER + yaml.safe_dump(spec, sort_keys=False))
    print("written")


if __name__ == "__main__":
    main()
