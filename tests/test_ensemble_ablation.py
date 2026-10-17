"""Study A of the paper 1 amendment of 15 October 2026: scripts/ensemble_ablation.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ensemble_ablation as EA  # noqa: E402

from rwm.evaluation import compare as C  # noqa: E402
from rwm.evaluation.splits import rolling_origins  # noqa: E402


def test_grid_covers_the_simplex():
    g = EA.grid()
    assert len(g) == 232 and np.allclose(g.sum(axis=1), 1) and (g >= 0).all()
    assert np.allclose(g[-1], 1 / 3)
    assert {0.0, 0.33, 0.5, 1.0} <= set(np.round(g[np.isclose(g[:, 1], g[:, 2]), 0], 2))


def test_choose_uses_the_earlier_windows_only():
    pts = EA.grid()
    rows = [{"origin": 0, "point": i, "level": lv, "wrmsse": 1.0 + (0 if i == 7 else 0.1)}
            for i in range(len(pts)) for lv in ("total", "item_idxstore_id")]
    prior = pd.DataFrame(rows)
    assert np.allclose(EA.choose(prior, "total", "item_idxstore_id", pts), pts[7])


@pytest.fixture(scope="module")
def fr():
    rng = np.random.default_rng(0)
    items, stores, T = 6, 3, 80
    attrs = pd.DataFrame([(f"i{i}_s{s}", f"i{i}", f"s{s}") for i in range(items) for s in range(stores)],
                         columns=["series_id", "item_id", "store_id"]).sort_values("series_id").reset_index(drop=True)
    units = rng.poisson(rng.uniform(2, 20, (len(attrs), 1)), (len(attrs), T)).astype(np.float32)
    dates = pd.date_range("2020-01-01", periods=T, freq="7D").to_numpy()
    levels = [[], ["store_id"], ["item_id", "store_id"]]
    return C.Frame(attrs, pd.Index(attrs["series_id"]), units, np.ones_like(units), dates, levels,
                   rolling_origins(dates, 8, 3), 13, 1)


def test_main_on_a_small_hierarchy(fr, tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    fcs = {}
    noise = {"world_model": 0.3, "lightgbm": 0.3, "timesfm_plan": 0.3, "lift": 0.6, "lift_matched": 0.5}
    for m, s in noise.items():
        recs = []
        for sp in fr.splits:
            t0 = int(np.searchsorted(fr.dates, np.datetime64(sp.train_end))) + 1
            act = fr.units[:, t0:t0 + 8]
            f = act * np.exp(rng.normal(0, s, act.shape))
            for i, n in enumerate(fr.names):
                for h, d in enumerate(sp.test_dates):
                    recs.append({"origin": sp.origin, "series_id": n, "date": d, "forecast": f[i, h]})
        fcs[m] = pd.DataFrame(recs)
    idx = pd.DataFrame({"key": "toy", "set": "rolling", "model": list(fcs), "seed": 1, "run": list(fcs), "config": ""})
    monkeypatch.setattr(EA, "index", lambda results: idx)
    monkeypatch.setattr(EA, "frames_and_forecasts", lambda results, rows: (fr, fcs))
    out = tmp_path / "abl"
    EA.main(["--results", str(tmp_path), "--out", str(out), "--boot", "50"])
    sc = pd.read_csv(out / "scores.csv")
    assert set(EA.VARIANTS) <= set(sc["model"])
    wt = pd.read_csv(out / "weights.csv")
    assert sorted(wt["origin"].unique()) == [1, 2]  # the first window has no earlier window
    pr = pd.read_csv(out / "paired.csv")
    assert (pr["b"] == "ensemble").all()
    # the ensemble's score equals the grid point at equal weights only up to the grid step; the weights sum to 1
    assert np.allclose(wt[["w_world_model", "w_lightgbm", "w_timesfm_plan"]].sum(axis=1), 1)
    # starting again skips the windows already done
    EA.main(["--results", str(tmp_path), "--out", str(out), "--boot", "50"])
    assert len(pd.read_csv(out / "paired.csv")) == len(pr)
    EA.main(["--tables", str(out), "--out", str(tmp_path / "tables")])
    sweep = pd.read_csv(tmp_path / "tables" / "ablation_sweep.csv")
    assert sweep["w_world_model"].min() == 0 and sweep["w_world_model"].max() == 1


def test_grid_scores_equal_level_scores(fr):
    rng = np.random.default_rng(5)
    t0, t1, _ = fr.cut(2)
    act = fr.units[:, t0:t1]
    members = [act * np.exp(rng.normal(0, 0.3, act.shape)) for _ in range(3)]
    pts = EA.grid()[[0, 50, 120, 231]]
    names, sc = EA.grid_scores(fr, 2, members, pts, chunk=3)
    for i, w in enumerate(pts):
        want = C.level_scores(fr, 2, sum(wi * m for wi, m in zip(w, members)))
        assert np.allclose(sc[i], [want[n] for n in names])
