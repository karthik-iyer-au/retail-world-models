"""Every experiment config's model can be built from its settings (catches misplaced keys)."""
from pathlib import Path

import pytest
import yaml

from rwm.forecaster import build_model

CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_model_builds(path):
    model = yaml.safe_load(path.read_text())["model"]
    build_model(model["name"], **model.get("params", {}))
