"""Chronos-Bolt and the Temporal Fusion Transformer, on generated data."""
import numpy as np
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.forecaster import build_model


@pytest.fixture(scope="module")
def split():
    p = load_dataset("synthetic_levers", n_zones=1, stores_per_zone=3, n_items=4, n_periods=120, seed=2).panel
    p = p.drop(columns=[c for c in p.columns if c.startswith("true_")])
    d = np.sort(p[DATE].unique())
    return p[p[DATE] <= d[-9]], p[p[DATE] > d[-9]].reset_index(drop=True)


def test_chronos_forecasts_every_row_with_ordered_quantiles(split):
    pytest.importorskip("chronos")
    m = build_model("chronos").fit(split[0])
    f = m.predict(split[1].drop(columns=[UNITS]))
    assert len(f) == len(split[1]) and (f >= 0).all()
    q = m.last_quantiles
    assert (q[0.1] <= q[0.5] + 1e-6).all() and (q[0.5] <= q[0.9] + 1e-6).all()


def test_tft_learns_and_gives_intervals(split):
    pytest.importorskip("neuralforecast")
    m = build_model("tft", horizon=8, extra=["promo", "display"], max_steps=40, accelerator="cpu").fit(split[0])
    f = m.predict(split[1].drop(columns=[UNITS]))
    assert len(f) == len(split[1]) and np.isfinite(f).all()
    assert 0.5 < f.sum() / split[1][UNITS].sum() < 2.0
    assert (m.last_quantiles[0.025] <= m.last_quantiles[0.975] + 1e-6).all()


@pytest.mark.parametrize("plan", [None, ["promo"]])
def test_timestral_forecasts_every_row_with_ordered_quantiles(split, plan):
    pytest.importorskip("timestral")
    m = build_model("timestral", repo="mistralai/Timestral-5M", plan=plan, price_plan=bool(plan)).fit(split[0])
    f = m.predict(split[1].drop(columns=[UNITS]))
    assert len(f) == len(split[1]) and np.isfinite(f).all() and (f >= 0).all()
    assert 0.3 < f.sum() / split[1][UNITS].sum() < 3.0
    q = m.last_quantiles
    assert len(q) == 21 and (q[0.1] <= q[0.5] + 1e-6).all() and (q[0.5] <= q[0.9] + 1e-6).all()


def test_timestral_reads_the_plan():
    """Given a promo that always doubles sales, the plan forecast rises on promoted days."""
    pytest.importorskip("timestral")
    import pandas as pd
    from rwm.data.schema import PRICE, SERIES
    rng = np.random.default_rng(0)
    n, dates = 200, pd.date_range("2020-01-01", periods=208, freq="D")
    promo = (rng.random((4, 208)) < 0.2).astype(float)
    units = rng.poisson(10 * (1 + promo))
    p = pd.DataFrame({SERIES: np.repeat([f"s{i}" for i in range(4)], 208), DATE: np.tile(dates, 4),
                      UNITS: units.ravel().astype(float), "promo": promo.ravel(), PRICE: 1.0})
    train, test = p[p[DATE] < dates[n]], p[p[DATE] >= dates[n]].reset_index(drop=True)
    f = build_model("timestral", repo="mistralai/Timestral-5M", plan=["promo"]).fit(train).predict(test.drop(columns=[UNITS]))
    on = test["promo"].to_numpy() > 0
    assert f[on].mean() > 1.3 * f[~on].mean()
