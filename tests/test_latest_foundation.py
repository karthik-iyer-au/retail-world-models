"""TimesFM 3.0 and Chronos-2 are given the same plan, built the same way, as TimesFM 2.5."""
import numpy as np
import pandas as pd
import pytest

from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.forecaster import build_model

LEVELS = np.linspace(0.1, 0.9, 9)


def _data():
    dates = pd.date_range("2016-01-01", periods=10, freq="D")
    rows = [("A", d, 2.0, int(i == 4), 1.0) for i, d in enumerate(dates)]
    rows += [("B", d, 4.0, 0, 2.0) for d in dates[6:]]  # B starts on day 7
    train = pd.DataFrame(rows, columns=[SERIES, DATE, UNITS, "promo", PRICE])
    future = pd.DataFrame({SERIES: ["A", "A", "B"], DATE: [dates[-1] + pd.Timedelta(days=k) for k in (1, 2, 1)],
                           "promo": [1, 0, 0], PRICE: [0.8, 1.0, 2.0]})
    return train, future


class FakeTimesFM3:
    """Quantiles are the history's mean times each level times 2, so the mean of them is the history's mean."""

    def __init__(self, seen):
        self.seen = seen

    def predict_batch(self, contexts, horizon, past_future_covariates=None, return_quantiles=False, make_positive=False):
        self.seen.append((contexts, past_future_covariates))
        for c in contexts:
            q = np.mean(c) * 2 * LEVELS[None, :].repeat(horizon, 0)
            yield type("Out", (), {"quantiles": q})()


class FakeChronos2:
    def __init__(self, seen):
        self.seen = seen

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, batch_size):
        import torch

        self.seen.append(inputs)
        out = []
        for x in inputs:
            y = x["target"] if isinstance(x, dict) else x
            out.append(torch.tensor(np.mean(y) * 2 * LEVELS[None, None, :].repeat(prediction_length, 1)))
        return out, None


def test_timesfm3_reads_the_plan_over_history_and_horizon():
    train, future = _data()
    seen = []
    model = build_model("timesfm3", context=8, plan=["promo"], price_plan=True,
                        loader=lambda b: FakeTimesFM3(seen)).fit(train)
    out = model.predict(future)
    contexts, cov = seen[0]
    assert len(contexts[1]) == 4  # B from its first record only
    assert cov[0].shape == (2, len(contexts[0]) + 2)  # promo and price, history then horizon
    assert cov[0][0, -2:].tolist() == [1.0, 0.0]  # the planned promotion
    assert np.allclose(cov[0][1, -2:], [0.8, 1.0])  # price against the highest in the window
    assert np.allclose(out, [2.0, 2.0, 4.0])  # the mean of its quantiles
    assert set(model.last_quantiles) == set(np.round(LEVELS, 1))


def test_chronos2_splits_the_plan_into_past_and_future():
    pytest.importorskip("torch")
    train, future = _data()
    seen = []
    model = build_model("chronos2", context=8, plan=["promo"], price_plan=True,
                        loader=lambda d: FakeChronos2(seen)).fit(train)
    out = model.predict(future)
    a = seen[0][0]
    assert len(a["past_covariates"]["plan_promo"]) == len(a["target"])
    assert a["future_covariates"]["plan_promo"].tolist() == [1.0, 0.0]
    assert np.allclose(out, [2.0, 2.0, 4.0])


def test_without_a_plan_both_see_history_only():
    pytest.importorskip("torch")
    train, future = _data()
    seen = []
    build_model("timesfm3", context=8, loader=lambda b: FakeTimesFM3(seen)).fit(train).predict(future)
    assert seen[0][1] is None
    seen = []
    build_model("chronos2", context=8, loader=lambda d: FakeChronos2(seen)).fit(train).predict(future)
    assert not isinstance(seen[0][0], dict)
