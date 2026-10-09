"""Every final-test config must be its development config unchanged, apart from
the name and the window it is scored on, so nothing is tuned on the final window."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "configs" / "experiments"
FINALS = sorted(p for p in ROOT.glob("*.yaml") if re.search(r"_final_|^m5_test_(world_model|lift)", p.stem))


def test_there_are_final_configs():
    assert len(FINALS) >= 17


def test_final_configs_match_their_development_configs():
    for p in FINALS:
        src = re.search(r"development config (\S+?);", p.read_text()).group(1)
        final, dev = yaml.safe_load(p.read_text()), yaml.safe_load((ROOT / src).read_text())
        assert final["model"] == dev["model"], p.name
        assert final["seed"] == dev["seed"], p.name
        assert final["dataset"]["name"] == dev["dataset"]["name"], p.name
        fe, de = dict(final["evaluation"]), dict(dev["evaluation"])
        if p.stem.startswith("m5_test"):
            assert final["dataset"]["params"] == {"include_test": True} and fe == de, p.name
        else:
            assert final["dataset"] == dev["dataset"], p.name
            assert fe.pop("holdout_periods") == 0 and fe.pop("n_origins") == 1, p.name
            de.pop("holdout_periods"), de.pop("n_origins")
            assert fe == de, p.name
