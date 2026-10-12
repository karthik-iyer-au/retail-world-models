import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import build_model


class FakeModel:
    """Stands in for TimesFM: forecasts the mean of each history, flat."""

    def __init__(self, seen):
        self.seen = seen

    def forecast(self, horizon, inputs):
        self.seen.extend(inputs)
        point = np.array([[np.mean(x)] * horizon for x in inputs])
        return point, None


def test_histories_and_alignment():
    dates = pd.date_range("2016-01-01", periods=10, freq="D")
    rows = []
    for d in dates:  # A sells 2 a day, with 6 January missing
        if d != dates[5]:
            rows.append(("A", d, 2.0))
    for d in dates[6:]:  # B starts on day 7, selling 4 a day
        rows.append(("B", d, 4.0))
    rows.append(("C", dates[0], 9.0))  # C sold only before the context window
    train = pd.DataFrame(rows, columns=[SERIES, DATE, UNITS])
    seen = []
    model = build_model("timesfm", context=8, loader=lambda c, h, b: FakeModel(seen)).fit(train)
    future = pd.DataFrame({
        SERIES: ["B", "A", "C", "Z"],
        DATE: [dates[-1] + pd.Timedelta(days=k) for k in (2, 1, 1, 1)],
    })
    out = model.predict(future)
    # A's missing day inside the window counts as zero sales.
    assert len(seen[0]) == 8 and sorted(seen[0])[:2] == [0.0, 2.0]
    # B's history starts at its first record, so it is not read as zeros before launch.
    assert len(seen[1]) == 4
    assert out.tolist() == [4.0, 1.75, 0.0, 0.0]


def test_missing_day_after_first_record_counts_as_zero():
    dates = pd.date_range("2016-01-01", periods=6, freq="D")
    train = pd.DataFrame({SERIES: "A", DATE: dates.delete(3), UNITS: 3.0})
    seen = []
    model = build_model("timesfm", context=6, loader=lambda c, h, b: FakeModel(seen)).fit(train)
    model.predict(pd.DataFrame({SERIES: ["A"], DATE: [dates[-1] + pd.Timedelta(days=1)]}))
    assert seen[0].tolist() == [3.0, 3.0, 3.0, 0.0, 3.0, 3.0]
