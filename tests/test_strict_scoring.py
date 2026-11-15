"""Scoring refuses broken forecasts and undefined levels (outside code review, 15 November 2026)."""
import numpy as np
import pandas as pd
import pytest

from rwm.evaluation.hierarchy import to_matrix, wrmsse

NAMES = pd.Index(["a", "b"])
DATES = pd.to_datetime(["2020-01-01", "2020-01-02"])


def frame(rows):
    return pd.DataFrame(rows, columns=["s", "d", "f"]).assign(d=lambda x: pd.to_datetime(x["d"]))


FULL = [("a", "2020-01-01", 1.0), ("a", "2020-01-02", 2.0), ("b", "2020-01-01", 3.0), ("b", "2020-01-02", 4.0)]


def test_complete_forecasts_pass():
    m = to_matrix(frame(FULL), "s", "d", "f", DATES, NAMES, complete=True)
    assert m.tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("rows, word", [
    (FULL[:3], "missing"),
    (FULL + [FULL[0]], "duplicate"),
    ([*FULL[:3], ("b", "2020-01-02", float("nan"))], "non-finite"),
    (FULL + [("c", "2020-01-01", 1.0)], "outside"),
])
def test_broken_forecasts_raise(rows, word):
    with pytest.raises(ValueError, match=word):
        to_matrix(frame(rows), "s", "d", "f", DATES, NAMES, complete=True)


def test_a_row_the_data_lacks_is_not_required():
    need = np.array([[True, True], [True, False]])  # b was not stocked on the second day
    m = to_matrix(frame(FULL[:3]), "s", "d", "f", DATES, NAMES, complete=True, required=need)
    assert m[1, 1] == 0.0


def test_lenient_default_still_reads_gaps_as_zero():
    assert to_matrix(frame(FULL[:3]), "s", "d", "f", DATES, NAMES)[1, 1] == 0.0


def test_a_level_with_nothing_to_score_is_undefined_not_zero():
    hist = np.ones((2, 5))  # constant history: no scale anywhere
    out = wrmsse(hist, np.ones((2, 2)), np.full((2, 2), 100.0), np.array([1.0, 1.0]),
                 pd.DataFrame({"s": ["a", "b"]}), [["s"], []])
    assert all(np.isnan(lv["wrmsse"]) for lv in out["by_level"])
    assert all(lv["excluded_weight"] == pytest.approx(1.0) for lv in out["by_level"])
