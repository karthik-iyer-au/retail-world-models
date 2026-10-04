import json

import numpy as np
import pandas as pd
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, SERIES, UNITS, validate
from rwm.evaluation.hierarchy import rmsse_where, to_matrix, wrmsse
from rwm.evaluation.metrics import rmsse
from rwm.evaluation.splits import rolling_origins
from rwm.experiments.run import run
from rwm.forecaster import build_model
from rwm.utils.hashing import sha256_obj

CONFIG = {
    "name": "t",
    "seed": 0,
    "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 3, "n_periods": 120}},
    "model": {"name": "seasonal_naive", "params": {"season": 7}},
    "evaluation": {"horizon": 14, "n_origins": 2},
}


def test_synthetic_is_deterministic():
    a = load_dataset("synthetic", seed=1).panel
    b = load_dataset("synthetic", seed=1).panel
    pd.testing.assert_frame_equal(a, b)


def test_schema_rejects_duplicates():
    p = load_dataset("synthetic").panel
    with pytest.raises(ValueError):
        validate(pd.concat([p, p.head(1)]), ["price"])


def test_splits_never_overlap_training():
    dates = pd.date_range("2021-01-01", periods=100)
    splits = rolling_origins(dates, horizon=10, n_origins=3)
    assert [s.origin for s in splits] == [0, 1, 2]
    for s in splits:
        assert len(s.test_dates) == 10
        assert s.train_end < s.test_dates[0]
    assert splits[-1].test_dates[-1] == dates[-1]


def test_rmsse_known_value():
    # naive one-step error on [1,2,3,4] is 1, forecast error is 2 -> rmsse 2
    assert rmsse([1, 2, 3, 4], [5, 5], [3, 7]) == pytest.approx(2.0)
    assert np.isnan(rmsse([0, 0, 0], [1], [1]))


def test_seasonal_naive_repeats_last_week_in_row_order():
    p = load_dataset("synthetic", n_periods=60).panel
    cut = sorted(p[DATE].unique())[45]
    train, test = p[p[DATE] <= cut], p[p[DATE] > cut].sample(frac=1, random_state=0)
    pred = build_model("seasonal_naive", season=7).fit(train).predict(test.drop(columns=[UNITS]))
    one = test.assign(pred=pred)
    one = one[one[SERIES] == "S0_I0"].sort_values(DATE)
    last_week = train[train[SERIES] == "S0_I0"].sort_values(DATE)[UNITS].to_numpy()[-7:]
    np.testing.assert_array_equal(one["pred"].to_numpy()[:14], np.tile(last_week, 2))


def test_run_is_repeatable_and_recorded(tmp_path):
    a = run(CONFIG, out_root=tmp_path / "a")
    b = run(CONFIG, out_root=tmp_path / "b")
    ma, mb = (json.loads((d / "metrics.json").read_text()) for d in (a, b))
    assert ma == mb
    man = json.loads((a / "manifest.json").read_text())
    assert man["config_sha256"] == sha256_obj(CONFIG)
    assert {"git", "seed", "packages", "data_files", "python"} <= set(man)
    assert np.isfinite(ma["wrmsse"])


def test_m5_loader_on_files_in_m5_layout(tmp_path):
    days = [f"d_{i}" for i in range(1, 15)]
    cal = pd.DataFrame(
        {
            "d": days,
            "date": pd.date_range("2011-01-29", periods=14).astype(str),
            "wm_yr_wk": [11101] * 7 + [11102] * 7,
            "event_name_1": [None] * 13 + ["X"],
            "snap_CA": [1] * 14, "snap_TX": [0] * 14, "snap_WI": [0] * 14,
        }
    )
    sales = pd.DataFrame(
        [["A_1_CA_1", "A_1", "A", "CAT", "CA_1", "CA", *range(14)],
         ["A_1_TX_1", "A_1", "A", "CAT", "TX_1", "TX", *[0] * 14]],
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *days],
    )
    prices = pd.DataFrame(
        {"store_id": ["CA_1", "CA_1"], "item_id": ["A_1", "A_1"],
         "wm_yr_wk": [11101, 11102], "sell_price": [2.0, 1.5]}
    )
    cal.to_csv(tmp_path / "calendar.csv", index=False)
    sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    prices.to_csv(tmp_path / "sell_prices.csv", index=False)

    ds = load_dataset("m5", root=str(tmp_path))
    p = ds.panel
    assert len(p) == 28 and ds.levers == ["price"] and len(ds.files) == 3
    ca = p[p[SERIES] == "A_1_CA_1"]
    assert ca[UNITS].tolist() == list(map(float, range(14)))
    assert ca["price"].tolist() == [2.0] * 7 + [1.5] * 7
    assert ca["snap"].eq(1).all() and ca["event"].sum() == 1
    tx = p[p[SERIES] == "A_1_TX_1"]
    assert tx["price"].isna().all() and tx["snap"].eq(0).all()


def test_wrmsse_matches_series_by_series_calculation():
    p = load_dataset("synthetic", n_stores=2, n_items=3, n_periods=90, seed=3).panel
    p.loc[p[SERIES] == "S0_I0", UNITS] = 0.0  # a series that never sells
    late = (p[SERIES] == "S1_I1") & (p[DATE] < p[DATE].min() + pd.Timedelta(days=20))
    p.loc[late, UNITS] = 0.0  # a series that starts late
    dates = np.sort(p[DATE].unique())
    y = to_matrix(p, SERIES, DATE, UNITS, dates)
    hist, act = y[:, :76], y[:, 76:]
    fc = np.full(act.shape, 3.0)
    attrs = p.drop_duplicates(SERIES).sort_values(SERIES).reset_index(drop=True)
    rev = np.arange(1.0, len(attrs) + 1)
    levels = [[], ["store_id"], ["item_id"], [SERIES]]
    got = wrmsse(hist, act, fc, rev, attrs, levels)

    def by_hand(cols):
        keys = attrs[cols].astype(str).agg("|".join, axis=1) if cols else pd.Series("all", index=attrs.index)
        total = 0.0
        for k in keys.unique():
            m = (keys == k).to_numpy()
            r = rmsse(hist[m].sum(0), act[m].sum(0), fc[m].sum(0))
            if np.isfinite(r):
                total += r * rev[m].sum() / rev.sum()
        return total

    for lv, cols in zip(got["by_level"], levels):
        assert lv["wrmsse"] == pytest.approx(by_hand(cols))
    assert got["wrmsse"] == pytest.approx(np.mean([by_hand(c) for c in levels]))


def test_naive_forecasts_follow_dates_when_a_series_has_gaps():
    days = pd.date_range("2021-01-01", periods=21)
    full = pd.DataFrame({SERIES: "a", "store_id": "s", "item_id": "i", DATE: days, UNITS: np.arange(21.0)})
    gappy = full.drop(index=[16, 18])  # two days missing in the last week
    future = pd.DataFrame({SERIES: "a", DATE: pd.date_range("2021-01-22", periods=10)})
    got = build_model("seasonal_naive", season=7).fit(gappy).predict(future)
    assert got.tolist() == [14, 15, 0, 17, 0, 19, 20, 14, 15, 0]
    flat = build_model("recent_average", window=7).fit(gappy).predict(future)
    assert flat.tolist() == [(14 + 15 + 17 + 19 + 20) / 7] * 10


def test_run_scores_series_that_are_absent_from_a_test_window(tmp_path):
    """A series with no rows in the forecast window must not shift the others."""
    from rwm.data.registry import register_dataset
    from rwm.data.synthetic import load_synthetic

    @register_dataset("synthetic_with_gap")
    def _load(**kw):
        ds = load_synthetic(**kw)
        last = ds.panel[DATE] > ds.panel[DATE].max() - pd.Timedelta(days=14)
        ds.panel = ds.panel[~(last & (ds.panel[SERIES] == "S0_I1"))].reset_index(drop=True)
        return ds

    base = {**CONFIG, "evaluation": {"horizon": 14, "n_origins": 1}}
    gap = {**base, "dataset": {"name": "synthetic_with_gap", "params": CONFIG["dataset"]["params"]}}
    m = json.loads((run(gap, out_root=tmp_path) / "metrics.json").read_text())
    assert np.isfinite(m["wrmsse"])


def test_rmsse_where_scores_only_the_chosen_periods():
    hist = np.array([[1.0, 2, 3, 4], [2.0, 2, 4, 4]])  # one-step scale: 1 and 4/3
    act = np.array([[5.0, 5], [5.0, 5]])
    fc = np.array([[3.0, 5], [5.0, 1]])
    where = np.array([[True, False], [False, True]])
    got = rmsse_where(hist, act, fc, where, np.array([1.0, 3.0]))
    want = 0.25 * rmsse([1, 2, 3, 4], [5], [3]) + 0.75 * rmsse([2, 2, 4, 4], [5], [1])
    assert got["periods"] == 2 and got["series"] == 2 and got["wrmsse"] == pytest.approx(want)
    none = rmsse_where(hist, act, fc, np.zeros_like(where), np.array([1.0, 3.0]))
    assert none["series"] == 0 and np.isnan(none["wrmsse"])


def test_run_reports_accuracy_with_a_lever_on_and_off(tmp_path):
    config = {**CONFIG, "evaluation": {"horizon": 14, "n_origins": 1, "conditions": ["promo"]}}
    m = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())
    c = m["splits"][0]["by_condition"]["promo"]
    assert c["on"]["periods"] + c["off"]["periods"] == 2 * 3 * 14
    assert c["on"]["periods"] > 0 and np.isfinite(c["off"]["wrmsse"])


def test_holdout_periods_are_never_scored(tmp_path):
    held = {**CONFIG, "evaluation": {"horizon": 14, "n_origins": 2, "holdout_periods": 14}}
    full = {**CONFIG, "evaluation": {"horizon": 14, "n_origins": 3}}
    a = json.loads((run(held, out_root=tmp_path / "a") / "metrics.json").read_text())["splits"]
    b = json.loads((run(full, out_root=tmp_path / "b") / "metrics.json").read_text())["splits"]
    assert [s["test_end"] for s in a] == [s["test_end"] for s in b[:2]]
    assert [s["wrmsse"] for s in a] == pytest.approx([s["wrmsse"] for s in b[:2]])


def test_with_seed_renames_the_run_and_sets_every_seed():
    from rwm.experiments.run import with_seed

    base = {"name": "x", "seed": 0, "model": {"name": "m", "params": {"seed": 0, "k": 1}}}
    got = with_seed(base, 3)
    assert got["name"] == "x_seed3" and got["seed"] == 3 and got["model"]["params"] == {"seed": 3, "k": 1}
    assert with_seed(base, None) is base and base["seed"] == 0


def test_fetch_copies_registered_files_from_a_folder_and_checks_them(tmp_path, monkeypatch):
    from rwm.data import fetch, manifest
    from rwm.utils.hashing import sha256_file

    source, raw, manifests = tmp_path / "src", tmp_path / "raw", tmp_path / "manifests"
    for d in (source, raw, manifests):
        d.mkdir()
    (source / "a.zip").write_bytes(b"data")
    (manifests / "toy.json").write_text(json.dumps(
        {"dataset": "toy", "files": [{"file": "a.zip", "bytes": 4, "sha256": sha256_file(source / "a.zip")}]}
    ))
    monkeypatch.setattr(manifest, "DATA_RAW", raw)
    monkeypatch.setattr(manifest, "DATA_MANIFESTS", manifests)
    monkeypatch.setattr(fetch, "DATA_RAW", raw)
    fetch.main(["toy", "--from", str(source)])
    assert (raw / "toy" / "a.zip").read_bytes() == b"data"
    (raw / "toy" / "a.zip").write_bytes(b"changed")
    with pytest.raises(ValueError):
        fetch.main(["toy", "--from", str(source)])


def test_verify_checks_only_the_files_a_run_reads(tmp_path, monkeypatch):
    import pytest

    from rwm.data import manifest

    monkeypatch.setattr(manifest, "DATA_RAW", tmp_path / "raw")
    monkeypatch.setattr(manifest, "DATA_MANIFESTS", tmp_path)
    root = tmp_path / "raw" / "d"
    root.mkdir(parents=True)
    (root / "a.csv").write_text("a")
    (root / "b.csv").write_text("b")
    manifest.register("d")
    (root / "b.csv").unlink()
    assert [e["file"] for e in manifest.verify("d", files=[root / "a.csv"])] == ["a.csv"]
    with pytest.raises(FileNotFoundError):
        manifest.verify("d")
    (root / "c.csv").write_text("c")
    with pytest.raises(ValueError):
        manifest.verify("d", files=[root / "c.csv"])
