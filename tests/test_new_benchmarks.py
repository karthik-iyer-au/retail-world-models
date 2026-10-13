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
