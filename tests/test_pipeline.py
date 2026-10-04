import json

import numpy as np
import pandas as pd
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, SERIES, UNITS, validate
from rwm.evaluation.metrics import rmsse, rmsse_by_series, weighted_rmsse
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
    assert weighted_rmsse([1.0, 3.0, np.nan], [1.0, 3.0, 5.0]) == pytest.approx(2.5)


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
    assert np.isfinite(ma["weighted_rmsse"])


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


def test_rmsse_by_series_matches_per_series_definition():
    p = load_dataset("synthetic", n_periods=90, seed=3).panel
    p.loc[p[SERIES] == "S0_I0", UNITS] = 0.0  # a series that never sells
    first = p[SERIES] == "S1_I1"
    p.loc[first & (p[DATE] < p[DATE].min() + pd.Timedelta(days=20)), UNITS] = 0.0  # late start
    cut = sorted(p[DATE].unique())[75]
    train, test = p[p[DATE] <= cut], p[p[DATE] > cut].copy()
    test["forecast"] = 3.0
    got = rmsse_by_series(train, test, SERIES, UNITS)
    for k, g in test.groupby(SERIES):
        want = rmsse(train.loc[train[SERIES] == k, UNITS].to_numpy(), g[UNITS].to_numpy(), g["forecast"].to_numpy())
        assert (np.isnan(want) and np.isnan(got[k])) or got[k] == pytest.approx(want)


def test_last_k_rows_takes_the_end_of_each_series():
    from rwm.utils.panel import last_k_rows

    s = pd.Series(["a"] * 5 + ["b"] * 2 + ["c"] * 4)
    assert last_k_rows(s, 3).tolist() == [2, 3, 4, 5, 6, 8, 9, 10]
