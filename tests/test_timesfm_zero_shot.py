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
        full = np.stack([point * 2] + [point] * 9, axis=-1)  # the mean output first, then nine quantiles
        return point, full


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


def test_mean_output_is_used_when_asked():
    dates = pd.date_range("2016-01-01", periods=6, freq="D")
    train = pd.DataFrame({SERIES: "A", DATE: dates, UNITS: 3.0})
    future = pd.DataFrame({SERIES: ["A"], DATE: [dates[-1] + pd.Timedelta(days=1)]})
    loader = lambda c, h, b: FakeModel([])
    assert build_model("timesfm", context=6, loader=loader).fit(train).predict(future).tolist() == [3.0]
    assert build_model("timesfm", context=6, point="mean", loader=loader).fit(train).predict(future).tolist() == [6.0]


class FakeCovariateModel:
    """Stands in for forecast_with_covariates: forecast = history mean + 10 x the plan column in the horizon."""

    def __init__(self, seen):
        self.seen = seen

    def forecast_with_covariates(self, inputs, dynamic_numerical_covariates, xreg_mode, ridge):
        self.seen.append(dynamic_numerical_covariates)
        out = []
        for i, x in enumerate(inputs):
            promo = np.array(dynamic_numerical_covariates["plan_promo"][i][len(x):])
            out.append(np.mean(x) + 10 * promo)
        full = [np.stack([o] * 10, axis=-1) for o in out]
        return out, full


def test_plan_reaches_the_covariate_forecast_aligned_with_each_history():
    dates = pd.date_range("2016-01-01", periods=6, freq="D")
    rows = [("A", d, 2.0, 0, 5.0) for d in dates] + [("B", d, 1.0, 0, 4.0) for d in dates[3:]]
    train = pd.DataFrame(rows, columns=[SERIES, DATE, UNITS, "promo", "price"])
    fut_dates = [dates[-1] + pd.Timedelta(days=k) for k in (1, 2)]
    future = pd.DataFrame({SERIES: ["A", "A", "B", "B"], DATE: fut_dates * 2,
                           "promo": [1, 0, 0, 1], "price": [4.0, 5.0, 4.0, 2.0]})
    seen = []
    model = build_model("timesfm", context=6, point="mean", plan=["promo"], price_plan=True,
                        loader=lambda c, h, b, back=False: FakeCovariateModel(seen)).fit(train)
    out = model.predict(future)
    assert out.tolist() == [12.0, 2.0, 1.0, 11.0]
    cov = seen[0]
    # B's covariates start where its history starts (3 days) and run 2 days into the horizon
    assert len(cov["plan_promo"][1]) == 5
    # price against the highest price in the window: A at 4.0 of 5.0, B at 2.0 of 4.0
    assert cov["plan_price"][0][-2:] == [0.8, 1.0] and cov["plan_price"][1][-1] == 0.5


def test_covariate_forecast_in_chunks_gives_the_same_result():
    dates = pd.date_range("2016-01-01", periods=6, freq="D")
    rows = [(f"S{i}", d, float(i), int(d == dates[2]), 5.0) for i in range(5) for d in dates]
    train = pd.DataFrame(rows, columns=[SERIES, DATE, UNITS, "promo", "price"])
    future = pd.DataFrame({SERIES: [f"S{i}" for i in range(5)], DATE: dates[-1] + pd.Timedelta(days=1),
                           "promo": [1, 0, 1, 0, 1], "price": 5.0})
    loader = lambda c, h, b, back=False: FakeCovariateModel([])
    whole = build_model("timesfm", context=6, point="mean", plan=["promo"], loader=loader).fit(train).predict(future)
    parts = build_model("timesfm", context=6, point="mean", plan=["promo"], chunk=2, loader=loader).fit(train).predict(future)
    assert whole.tolist() == parts.tolist()
