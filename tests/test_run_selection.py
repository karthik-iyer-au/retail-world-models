"""revision_analysis.index keeps the lowest seed and, among repeats of one configuration, the latest run."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("revision_analysis", ROOT / "scripts" / "revision_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(root: Path, folder: str, name: str):
    d = root / folder
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"config": {"name": name}}))
    (d / "forecasts.csv").write_text("x\n")


def test_latest_repeat_and_lowest_seed(tmp_path, monkeypatch):
    ra = _load()
    monkeypatch.setattr(ra, "TUNED", True)
    _run(tmp_path, "20261110T000000_final_frat_lightgbm_aaaaaaaa", "final_frat_lightgbm")
    _run(tmp_path, "20261111T000000_final_frat_lightgbm_bbbbbbbb", "final_frat_lightgbm")
    _run(tmp_path, "20261109T000000_final_frat_world_model_seed2_cccccccc", "final_frat_world_model_seed2")
    _run(tmp_path, "20261110T000000_final_frat_world_model_seed1_dddddddd", "final_frat_world_model_seed1")
    idx = ra.index(tmp_path).set_index("model")
    assert idx.loc["lightgbm", "run"].startswith("20261111")
    assert idx.loc["world_model", "seed"] == 1
