"""scripts/records_snapshot.py: the recorded numbers of paper 1, built from run folders.

Two kinds of test:

1. Once results/records/snapshot.json exists (built on Colab from the Drive run
   folders), it must agree with the numbers transcribed from results/README.md
   (results/records/snapshot_transcribed.json) to the decimals README shows, so a
   transcription error, or a wrong run behind a number, surfaces. Skipped until then.
2. The extraction itself, on small synthetic run folders made here.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
RECORDS = ROOT / "results" / "records"
SNAPSHOT = RECORDS / "snapshot.json"
TRANSCRIBED = RECORDS / "snapshot_transcribed.json"

spec = importlib.util.spec_from_file_location("records_snapshot", ROOT / "scripts" / "records_snapshot.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


# --------------------------------------------- decimals shown in README ---
def decimals(block: str, path: tuple) -> int:
    """How many decimals results/README.md shows for a value."""
    last = path[-1] if path else None
    if block in ("PRICE_CUT", "PRICE_CUT_FOODS"):
        return 1 if path[0] in ("lgb", "tfm") else 0
    if block == "LIFT_APR_SEED2":
        return 2 if path[0] == "top5_share" else 0
    if block in ("FRAT_ACTUAL", "BAND_UNITS"):
        return 0
    if block in ("FRAT_BIAS", "BAND_BIAS", "BAND_CLOSER"):
        return 1
    if block in ("READBACK", "SWITCH_SHARE"):
        return 2
    if block == "PLAN_PAIRS" and last == 1:
        return 1
    return 3


def walk(x, path=()):
    """(path, leaf) for every leaf of a nested value."""
    if isinstance(x, dict):
        for k in x:
            yield from walk(x[k], path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, path + (i,))
    else:
        yield path, x


def same_shape(a, b, exact_text=True) -> list[str]:
    """Differences in structure between two block values (keys, lengths, kinds of leaf)."""
    out = []
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return [f"keys {sorted(a) if isinstance(a, dict) else a} != {sorted(b) if isinstance(b, dict) else b}"]
        for k in a:
            out += [f"{k}: {d}" for d in same_shape(a[k], b[k], exact_text)]
    elif isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return [f"lists differ: {a} != {b}"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += [f"[{i}]: {d}" for d in same_shape(x, y, exact_text)]
    elif a is None or b is None:
        if not (a is None and b is None):
            out.append(f"{a} != {b} (None)")
    elif isinstance(a, str) or isinstance(b, str):
        if not (isinstance(a, str) and isinstance(b, str)) or (exact_text and a != b):
            out.append(f"{a!r} != {b!r}")
    elif not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append(f"{a!r} / {b!r} not numbers")
    return out


def disagreements(block: str, got, want) -> list[str]:
    bad = same_shape(got, want)
    if bad:
        return [f"{block}: {d}" for d in bad]
    for (path, g), (_, w) in zip(walk(got), walk(want)):
        if isinstance(g, (int, float)) and isinstance(w, (int, float)):
            d = decimals(block, path)
            if abs(g - w) > 0.5 * 10 ** -d + 1e-9:
                out = f"{block}{list(path)}: records {g:.{d + 2}f}, README {w:.{d}f}"
                bad.append(out)
    return bad


# ---------------------------------------- the real snapshot against README ---
@pytest.mark.skipif(not SNAPSHOT.exists(), reason="results/records/snapshot.json not built yet (run on Colab)")
def test_snapshot_agrees_with_readme_transcription():
    snap = json.loads(SNAPSHOT.read_text())
    want = json.loads(TRANSCRIBED.read_text())["blocks"]
    assert snap.get("kind") == "records", "snapshot.json must be a full records build"
    assert set(snap["blocks"]) == set(want), "snapshot and transcription hold different blocks"
    bad = []
    for name, block in want.items():
        bad += disagreements(name, snap["blocks"][name]["value"], block["value"])
    assert not bad, "records and README disagree:\n" + "\n".join(bad)


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="results/records/snapshot.json not built yet (run on Colab)")
def test_snapshot_names_a_run_folder_for_every_value():
    snap = json.loads(SNAPSHOT.read_text())
    folder = re.compile(r"^(\d{8}T\d{6}_)?.+_[0-9a-f]{8}$")
    for name, block in snap["blocks"].items():
        assert block["sources"], f"{name} names no run"
        for key, folders in block["sources"].items():
            assert folders and all(folder.match(f) for f in folders), (name, key, folders)


def test_transcription_is_what_readme_shows():
    """Guard on the comparison file itself: every number in it appears in results/README.md as shown there."""
    readme = (ROOT / "results" / "README.md").read_text()
    doc = json.loads(TRANSCRIBED.read_text())
    assert doc["kind"] == "transcribed"
    assert set(doc["blocks"]) == set(rs.BLOCKS)
    skip = {("LIFT_APR_SEED2", ("top5_share",))}  # README states it as a percentage (101%)
    missing = []
    for name, block in doc["blocks"].items():
        if name == "LEDGER":
            continue
        for path, v in walk(block["value"]):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or (name, path) in skip:
                continue
            shown = f"{abs(v):,}" if isinstance(v, int) else f"{abs(v):,.{decimals(name, path)}f}"
            if shown not in readme:
                missing.append(f"{name}{list(path)} = {shown}")
    assert not missing, missing


def test_rwdata_refuses_to_use_transcribed_values_unless_asked():
    if SNAPSHOT.exists():
        pytest.skip("snapshot.json exists; rwdata reads it")
    env = {k: v for k, v in os.environ.items() if k != "RWDATA_ALLOW_TRANSCRIBED"}
    code = "import rwdata as D; print(D.RECORDS_KIND, D.READBACK['mar'][7], D.DOM28['ana'])"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "paper" / "figs", env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "snapshot.json is missing" in r.stderr
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "paper" / "figs",
                       env={**env, "RWDATA_ALLOW_TRANSCRIBED": "1"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stderr and r.stdout.split()[0] == "transcribed"
    assert "(0.0, 0.59)" in r.stdout and "(0.702, 0.718, 0.666, 0.679)" in r.stdout


# ------------------------------------------------- synthetic run folders ---
M5_IDS = list(rs._script("after_confirm").FAILED) + ["FOODS_1_001_CA_1", "HOBBIES_1_002_TX_2"]
FRAT_IDS = ["1111_2277", "2222_2277", "1111_389"]


def make_run(root: Path, name: str, stamp="20261012T000000", finished="2026-10-12T01:00:00+00:00", *,
             overall=0.5, splits=None, forecasts=None, files=(), hexid=None):
    hexid = hexid or f"{abs(hash((name, stamp))) % 16**8:08x}"
    d = root / f"{stamp}_{name}_{hexid}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"config": {"name": name}, "finished_utc": finished}))
    splits = splits or [split(overall)]
    (d / "metrics.json").write_text(json.dumps({"wrmsse": float(np.mean([s["wrmsse"] for s in splits])),
                                                "splits": splits}))
    if forecasts is not None:
        forecasts.to_csv(d / "forecasts.csv", index=False)
    for f in files:
        (d / f).write_bytes(b"")
    return d


def split(w=0.5, ixs=0.8, fa=1.0, **extra):
    s = {"wrmsse": w, "forecast_to_actual": fa,
         "by_level": [{"level": "total", "wrmsse": w / 2}, {"level": rs.ITEM_STORE, "wrmsse": ixs}],
         "by_condition": {"promo": {"on": {"wrmsse": 1.2}, "off": {"wrmsse": 0.4}}},
         "peaks": {g: {"forecast_to_actual": v, "rmsse": 0.5, "share_of_units": 1 / 3}
                   for g, v in (("promoted", 0.9), ("after_promo", 1.1), ("ordinary", 1.0))},
         "plan_pair_order": 0.75, "plan_pair_change_error": 0.45, "rare_plans": 0.7, "usual_plans": 0.69,
         "new_items": 1.3, "known_items": 0.5}
    s.update(extra)
    return s


def m5_forecasts(scale=1.0, units=1.0):
    dates = pd.date_range("2016-03-28", periods=28).strftime("%Y-%m-%d")
    df = pd.DataFrame([(s, d) for s in M5_IDS for d in dates], columns=["series_id", "date"])
    df["units"] = units
    df["forecast"] = scale * units
    df["origin"] = 0
    return df


MARCH_RATE = [0.0, 0.3, 0.7, 2.0, 5.0, 20.0, 20.0]  # one series in each sales band, two in the top one


def m5_march():
    f = m5_forecasts()
    f["units"] = f["series_id"].map(dict(zip(M5_IDS, MARCH_RATE)))
    return f


def test_latest_finished_run_is_used_and_recorded(tmp_path):
    make_run(tmp_path, "m5_test_timesfm_plan_seed1", "20261012T000000", "2026-10-12T02:00:00+00:00", overall=0.9)
    late = make_run(tmp_path, "m5_test_timesfm_plan_seed1", "20261011T000000", "2026-10-13T02:00:00+00:00",
                    overall=0.7)
    make_run(tmp_path, "m5_test_timesfm_plan_extra_seed1", overall=5.0)  # a longer name is another config
    rec = rs.Records([tmp_path])
    assert rec.folder("m5_test_timesfm_plan", 1) == late
    b = rs.Builder(rec)
    assert b.one("x", "m5_test_timesfm_plan") == 0.7
    assert b.sources == {"x": [late.name]}
    with pytest.raises(rs.MissingRun):
        rec.folder("m5_test_timesfm_plan", 2)
    with pytest.raises(rs.MissingRun):
        rec.folder("m5_test_timesfm_plan", 1, need="forecasts.csv")


def test_measures_average_windows_then_seeds(tmp_path):
    for s, (a, b_) in zip((1, 2, 3), ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0))):
        make_run(tmp_path, f"frat_dev_x_seed{s}", splits=[split(a, ixs=a, new_items=a), split(b_, ixs=b_, new_items=b_)])
    b = rs.Builder(rs.Records([tmp_path]))
    r = b.seeds("x", "frat_dev_x", (1, 2, 3))
    assert np.isclose(r["mean"], 0.7) and np.isclose(r["lo"], 0.5) and np.isclose(r["hi"], 0.9)
    assert np.isclose(b.one("x", "frat_dev_x", (1, 2, 3), rs.ixs), 0.7)
    assert np.isclose(b.one("x", "frat_dev_x", (1, 2, 3), rs.key("new_items")), 0.7)
    assert np.isclose(b.one("x", "frat_dev_x", (1,), rs.key("plan_pair_order", 100.0)), 75.0)
    assert np.isclose(b.one("x", "frat_dev_x", (1,), rs.peak_fa("after_promo")), 1.1)
    assert np.isclose(b.one("x", "frat_dev_x", (1,), rs.condition_on("promo")), 1.2)
    assert len(b.sources["x"]) == 3


def test_unseeded_run_is_found_by_its_plain_name(tmp_path):
    plain = make_run(tmp_path, "dominicks_ana_dev_lightgbm", overall=0.715)
    make_run(tmp_path, "dominicks_ana_dev_lightgbm_seed1", overall=0.7)
    rec = rs.Records([tmp_path])
    assert rec.folder("dominicks_ana_dev_lightgbm", None) == plain


def test_price_cut_and_seed_two_excess_from_forecasts(tmp_path):
    hh = [s for s in M5_IDS if s.startswith("HOUSEHOLD_1_146")]
    for cfg, seeds in rs.APR_MODELS.values():
        for s in seeds:
            f = m5_forecasts(scale=s)  # each seed forecasts s times the sales
            if cfg == "m5_confirm_apr_lift" and s == 2:
                f.loc[f["series_id"] == hh[0], "forecast"] = 100.0  # the blow-up
            make_run(tmp_path, f"{cfg}_seed{s}", forecasts=f)
    blocks, errors = rs.build(rs.Records([tmp_path]), only=["PRICE_CUT", "PRICE_CUT_FOODS", "LIFT_APR_SEED2"])
    assert not errors
    pc = blocks["PRICE_CUT"]["value"]
    assert pc["sold"] == 4 * 28 and pc["world"] == [112.0, 224.0, 336.0] and pc["tfm"] == [112.0]
    assert pc["lift"][1] == 100.0 * 28 + 3 * 2 * 28
    assert blocks["PRICE_CUT_FOODS"]["value"]["graph"] == [28.0, 56.0, 84.0]
    assert len(blocks["PRICE_CUT"]["sources"]["world"]) == 3
    lift = blocks["LIFT_APR_SEED2"]["value"]
    n = len(M5_IDS) * 28
    assert lift["sold"] == n and lift["forecast"] == 2 * n + (100 - 2) * 28
    # the excess is n (seed 2 doubles every series) plus 98 a day on one series, all within the top five
    top5 = (98 * 28 + 5 * 28) / (n + 98 * 28)
    assert np.isclose(lift["top5_share"], top5)


def test_frat_weekly_bias_and_bands(tmp_path):
    weeks = pd.date_range("2011-11-16", periods=8, freq="7D").strftime("%Y-%m-%d")
    for cfg, scale in (("frat_final_world_model", 1.1), ("frat_final_lightgbm", 0.9)):
        for s in (1, 2, 3):
            f = pd.DataFrame([(i, w, 10.0, 10.0 * scale, 0) for i in FRAT_IDS for w in weeks],
                             columns=["series_id", "date", "units", "forecast", "origin"])
            make_run(tmp_path, f"{cfg}_seed{s}", forecasts=f)
    for cfg, scale in (("m5_confirm_apr_world_model_v1", 1.2), ("m5_confirm_apr_lightgbm", 0.8)):
        for s in (1, 2, 3):
            make_run(tmp_path, f"{cfg}_seed{s}", forecasts=m5_forecasts(scale=scale, units=2.0))
    make_run(tmp_path, "m5_confirm_mar_world_model_v1_seed1", forecasts=m5_march())
    blocks, errors = rs.build(rs.Records([tmp_path]), only=["FRAT_WEEKS", "FRAT_ACTUAL", "FRAT_BIAS", "BAND_UNITS",
                                                              "BAND_BIAS", "BAND_CLOSER"])
    assert not errors
    assert blocks["FRAT_WEEKS"]["value"][:2] == ["16 Nov", "23 Nov"]
    assert blocks["FRAT_ACTUAL"]["value"] == [30] * 8
    assert np.allclose(blocks["FRAT_BIAS"]["value"]["world"], 10.0)
    assert np.allclose(blocks["FRAT_BIAS"]["value"]["lgb"], -10.0)
    assert blocks["BAND_UNITS"]["value"] == [56] * 5 + [112]  # banded by March sales, scored on April's
    assert blocks["BAND_BIAS"]["value"]["world"] == [20.0] * 6
    assert blocks["BAND_BIAS"]["value"]["lgb"] == [-20.0] * 6
    assert blocks["BAND_CLOSER"]["value"] == [0.0] * 6  # equally far off: the world model is never closer
    assert "band_from" in blocks["BAND_UNITS"]["sources"]


def test_readback_uses_rows_saved_by_plan_readback(tmp_path):
    for w, cfg in rs.READBACK_RUNS.items():
        d = make_run(tmp_path, f"{cfg}_seed1", files=["checkpoint_origin0.pt"])
        rows = [{"plan input": rs.DISCOUNT, "periods back": lag, "from state before window": 0.01 * lag,
                 "from rolled state": 0.9 - 0.05 * lag} for lag in (0, 1, 7)]
        rows.append({"plan input": "snap", "periods back": 0, "from state before window": 0.0, "from rolled state": 0.99})
        (d / "plan_readback.json").write_text(json.dumps({"run": d.name, "rows": rows}))
    blocks, errors = rs.build(rs.Records([tmp_path]), only=["READBACK"])
    assert not errors
    assert blocks["READBACK"]["value"]["apr"]["7"] == [0.07, 0.9 - 0.35]


def test_switch_share_from_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    for c in rs.CATS:
        for s in (1, 2, 3):
            d = make_run(tmp_path, f"dominicks_{c}_dev_peaks_world_model_switching_seed{s}")
            for o in (0, 1):
                torch.save({"new_share": 0.5 + 0.1 * s + 0.01 * o}, d / f"checkpoint_origin{o}.pt")
    blocks, errors = rs.build(rs.Records([tmp_path]), only=["SWITCH_SHARE"])
    assert not errors
    assert np.allclose(blocks["SWITCH_SHARE"]["value"]["sna"], [0.6, 0.81])


def test_missing_runs_stop_the_snapshot(tmp_path):
    (tmp_path / "runs").mkdir()
    out = tmp_path / "snapshot.json"
    assert rs.main(["--results", str(tmp_path / "runs"), "--out", str(out)]) == 1
    assert not out.exists()


def _fill(root: Path, name: str, need: str | None, hexid: str):
    """A synthetic run good enough for every block that asks for it."""
    if "timesfm" in name or "final" in name or "confirm" in name or "m5_" in name:
        stamp = "20261012T000000"
    else:
        stamp = "20261008T000000"
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    w = float(rng.uniform(0.4, 1.2))
    if name.startswith("frat_final"):
        weeks = pd.date_range("2011-11-16", periods=8, freq="7D").strftime("%Y-%m-%d")
        fc = pd.DataFrame([(i, d, 10.0, 10.0 * w, 0) for i in FRAT_IDS for d in weeks],
                          columns=["series_id", "date", "units", "forecast", "origin"])
    elif name.startswith("m5_confirm_mar_world_model_v1"):
        fc = m5_march()
    elif name.startswith("m5_"):
        fc = m5_forecasts(scale=w)
    else:
        fc = None
    d = make_run(root, name, stamp, splits=[split(w), split(w * 1.01)], forecasts=fc, hexid=hexid)
    if need and need.startswith("checkpoint"):
        if "switching" in name:
            import torch

            for o in (0, 1):
                torch.save({"new_share": float(rng.uniform(0, 1))}, d / f"checkpoint_origin{o}.pt")
        else:
            (d / "checkpoint_origin0.pt").write_bytes(b"")
            rows = [{"plan input": rs.DISCOUNT, "periods back": lag, "from state before window": 0.0,
                     "from rolled state": 0.8} for lag in (0, 1, 7)]
            (d / "plan_readback.json").write_text(json.dumps({"run": d.name, "rows": rows}))


def test_every_block_builds_with_the_shape_rwdata_expects(tmp_path):
    """Make every run the snapshot asks for, then check each block against the transcribed one's shape."""
    pytest.importorskip("torch")
    runs = tmp_path / "runs"
    runs.mkdir()
    for _ in range(400):
        blocks, errors = rs.build(rs.Records([runs]))
        missing = [re.search(r"MissingRun: (\S+)( \(with (\S+)\))?", e) for e in errors]
        missing = [m for m in missing if m]
        if not missing:
            break
        for i, (name, need) in enumerate(sorted({(m.group(1), m.group(3) or "") for m in missing})):
            _fill(runs, name, need or None, f"{_:04d}{i:04d}")
    assert not errors, errors
    want = json.loads(TRANSCRIBED.read_text())["blocks"]
    assert list(blocks) == rs.BLOCKS
    for name, block in blocks.items():
        diff = same_shape(block["value"], want[name]["value"], exact_text=name not in ("FRAT_WEEKS", "LEDGER"))
        assert not diff, (name, diff[:3])
        assert block["sources"], name
    out = tmp_path / "snapshot.json"
    assert rs.main(["--results", str(runs), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["kind"] == "records"
