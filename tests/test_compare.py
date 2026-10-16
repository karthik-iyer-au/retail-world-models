"""Paper 1 revision analyses (rwm.evaluation.compare, reconcile) on a small hand-made hierarchy."""
import numpy as np
import pandas as pd
import pytest

from rwm.evaluation import compare as C
from rwm.evaluation.splits import rolling_origins


@pytest.fixture(scope="module")
def fr():
    rng = np.random.default_rng(0)
    items, stores, T = 6, 3, 80
    attrs = pd.DataFrame([(f"i{i}_s{s}", f"i{i}", f"s{s}", "c") for i in range(items) for s in range(stores)],
                         columns=["series_id", "item_id", "store_id", "cat_id"]).sort_values("series_id").reset_index(drop=True)
    units = rng.poisson(rng.uniform(2, 20, (len(attrs), 1)), (len(attrs), T)).astype(np.float32)
    dates = pd.date_range("2020-01-01", periods=T, freq="7D").to_numpy()
    levels = [[], ["store_id"], ["item_id"], ["item_id", "store_id"]]
    return C.Frame(attrs, pd.Index(attrs["series_id"]), units, np.ones_like(units), dates, levels,
                   rolling_origins(dates, 8, 3), 13, 1)


def test_bootstrap_with_every_item_once_matches_the_score(fr):
    fc = fr.units[:, -8:] * 1.1
    point = C.level_scores(fr, 2, fc)
    bs = C.Bootstrap(fr, 2, n=3)
    bs.counts[:] = 1
    got = bs.scores({"m": fc})["m"][0]
    assert np.allclose(got, [point[k] for k in bs.level_names])


def test_paired_interval_excludes_zero_for_a_much_worse_model(fr):
    good, bad = fr.units[:, -8:] * 1.02, fr.units[:, -8:] * 1.6
    point = {"good": C.level_scores(fr, 2, good), "bad": C.level_scores(fr, 2, bad)}
    bs = C.Bootstrap(fr, 2, n=200)
    df = C.paired(bs.scores({"good": good, "bad": bad}), point, bs.level_names, [("bad", "good")])
    assert (df["lo"] > 0).all()


def test_conformal_intervals_cover_about_as_planned(fr):
    errs = []
    for o in (0, 1):
        t0, t1, _ = fr.cut(o)
        errs.append(C.scaled_errors(fr, o, fr.units[:, t0 - 8:t0], ["item_id", "store_id"]))  # last 8 weeks as forecast
    q = C.conformal_quantiles(errs)
    t0, t1, _ = fr.cut(2)
    s = C.interval_scores(fr, 2, fr.units[:, t0 - 8:t0], q, ["item_id", "store_id"])
    assert 0.6 < s["coverage_80"] <= 1.0 and np.isfinite(s["spl"])


def test_model_confidence_set_drops_a_clearly_worse_model():
    rng = np.random.default_rng(1)
    L = pd.DataFrame({"a": 1 + rng.normal(0, 0.01, 12), "b": 1.002 + rng.normal(0, 0.01, 12), "c": 1.5 + rng.normal(0, 0.01, 12)})
    kept = C.model_confidence_set(L)
    assert "c" not in kept and "a" in kept


def test_reconciliation_is_coherent_and_ols_keeps_a_coherent_forecast(fr):
    pytest.importorskip("statsforecast")
    from rwm.evaluation.reconcile import gls_bottom, summing_matrix
    S, _ = summing_matrix(fr)
    b = np.random.default_rng(2).uniform(1, 5, (S.shape[1], 4))
    y = S @ b  # already coherent: every method must give it back
    assert np.allclose(gls_bottom(S, y, np.ones(S.shape[0])), b, atol=1e-5)
    assert np.allclose(gls_bottom(S, y, np.asarray(S.sum(axis=1)).ravel()), b, atol=1e-5)


def test_reconcile_all_runs_and_returns_bottom_forecasts(fr):
    pytest.importorskip("statsforecast")
    from rwm.evaluation.reconcile import reconcile_all
    # make one item start late, so nodes have different history lengths
    fr.units[0, :30] = 0
    out = reconcile_all(fr, 2, season=1, bottoms={"m": fr.units[:, -8:]})
    for name, b in out.items():
        assert b.shape == (len(fr.attrs), 8), name
        assert np.isfinite(b).all() and (b >= 0).all(), name
    assert "ets_mint_shrink" in out and "m_with_ets_wls_var" in out


def test_fast_bootstrap_equals_the_direct_computation(fr):
    rng = np.random.default_rng(3)
    fcs = {"a": fr.units[:, -8:] * 1.1, "b": fr.units[:, -8:] * rng.uniform(0.5, 1.5, fr.units[:, -8:].shape)}
    bs = C.Bootstrap(fr, 2, n=40, seed=5)
    fast, slow = bs.scores(fcs, chunk=7), bs.scores_reference(fcs)
    for m in fcs:
        assert np.allclose(fast[m], slow[m], rtol=1e-6, atol=1e-9)


def test_fast_bootstrap_handles_an_item_missing_from_a_resample_and_a_coarse_level(fr):
    fc = fr.units[:, -8:] * 0.9
    bs = C.Bootstrap(fr, 2, n=4, seed=1)
    bs.counts[:] = 0
    bs.counts[:, 0] = 6  # only the first item, six times
    bs.counts[1] = [1, 1, 1, 1, 1, 1]
    fast, slow = bs.scores({"m": fc}), bs.scores_reference({"m": fc})
    assert np.allclose(fast["m"], slow["m"], rtol=1e-6)


def test_conformal_quantiles_pool_horizons_when_a_level_has_few_errors():
    import numpy as np

    from rwm.evaluation import compare as C

    rng = np.random.default_rng(0)
    few = [rng.normal(size=(1, 8)), rng.normal(size=(1, 8))]  # the total: one node, two windows
    per_h = C.conformal_quantiles(few)
    pooled = C.conformal_quantiles(few, min_per_horizon=30)
    assert np.allclose(pooled, pooled[0])  # the same quantiles at every horizon
    assert (pooled[:, -1] - pooled[:, 0]).mean() > (per_h[:, -1] - per_h[:, 0]).mean()
    many = [rng.normal(size=(500, 8))]
    np.testing.assert_allclose(C.conformal_quantiles(many, min_per_horizon=30), C.conformal_quantiles(many))


def test_whole_unit_coverage_is_never_lower(fr):
    errs = []
    for o in (0, 1):
        t0, t1, _ = fr.cut(o)
        errs.append(C.scaled_errors(fr, o, fr.units[:, t0 - 8:t0], ["item_id", "store_id"]))
    q = C.conformal_quantiles(errs)
    t0, t1, _ = fr.cut(2)
    a = C.interval_scores(fr, 2, fr.units[:, t0 - 8:t0], q, ["item_id", "store_id"])
    b = C.interval_scores(fr, 2, fr.units[:, t0 - 8:t0], q, ["item_id", "store_id"], whole_units=True)
    assert all(b[f"coverage_{k}"] >= a[f"coverage_{k}"] for k in (50, 80, 95)) and b["spl"] == a["spl"]
