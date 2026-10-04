"""Checks against the M5 organisers' published files.

Needs data/raw/m5/ (with sales_test_evaluation.csv) and data/raw/m5_reference/.
Skipped when those are not present. See data/README.md.
"""
import numpy as np
import pandas as pd
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.evaluation.hierarchy import to_matrix, wrmsse
from rwm.utils.paths import DATA_RAW

REF = DATA_RAW / "m5_reference"
needs_files = pytest.mark.skipif(
    not (REF / "Accuracy_benchmarks_overall.csv").exists()
    or not (DATA_RAW / "m5" / "sales_test_evaluation.csv").exists(),
    reason="M5 reference files not present",
)


@pytest.fixture(scope="module")
def m5():
    ds = load_dataset("m5", include_test=True)
    p = ds.panel
    dates = np.sort(p[DATE].unique())
    y = to_matrix(p, SERIES, DATE, UNITS, dates)
    price = to_matrix(p, SERIES, DATE, PRICE, dates)
    attrs = p.iloc[:: len(dates)].reset_index(drop=True)
    names = attrs[SERIES].astype(str).to_numpy()
    history, actual = y[:, :-28], y[:, -28:]
    revenue = (history[:, -28:] * price[:, -56:-28]).sum(axis=1, dtype=np.float64)
    ds.panel = None  # the tests below only need the arrays
    return ds, names, history, actual, revenue, attrs


def _published_forecast(file: str, names: np.ndarray) -> np.ndarray:
    f = pd.read_csv(REF / file)
    f = f[f["id"].str.endswith("_evaluation")]
    f.index = f["id"].str.replace("_evaluation", "", regex=False)
    return f.loc[names, [f"F{i}" for i in range(1, 29)]].to_numpy(dtype=np.float64)


@needs_files
@pytest.mark.parametrize(
    "file, table, column",
    [
        ("benchmark_sNaive.csv", "Accuracy_benchmarks_overall.csv", "sNaive"),
        ("benchmark_ES_bu.csv", "Accuracy_benchmarks_overall.csv", "ES_bu"),
        ("submission_0001_YJ_STU.csv", "Accuracy_top50_overall.csv", "YJ_STU"),
    ],
)
def test_wrmsse_reproduces_published_scores(m5, file, table, column):
    ds, names, history, actual, revenue, attrs = m5
    got = wrmsse(history, actual, _published_forecast(file, names), revenue, attrs, ds.hierarchy)
    # Published table: one row per level 1 to 12, then a final row of averages.
    published = pd.read_csv(REF / table)[column].to_numpy(dtype=float)
    assert len(published) == 13
    np.testing.assert_allclose([lv["wrmsse"] for lv in got["by_level"]], published[:12], rtol=1e-4)
    assert got["wrmsse"] == pytest.approx(published[12], rel=1e-4)


@needs_files
def test_revenue_weights_match_published_weights(m5):
    _, names, _, _, revenue, _ = m5
    w = pd.read_csv(REF / "weights_evaluation.csv")
    w = w[w["Level_id"] == "Level12"]
    w.index = w["Agg_Level_1"] + "_" + w["Agg_Level_2"]
    np.testing.assert_allclose(revenue, w.loc[names, "Dollar_Sales"].to_numpy(), rtol=1e-4, atol=0.011)
