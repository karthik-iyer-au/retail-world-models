import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE, SERIES, UNITS
from rwm.forecaster import build_model

SETTINGS = dict(
    horizon=28, history=56, train_periods=380, categorical=["item_id", "store_id"], extra=["promo"],
    d_model=32, layers=2, heads=4, steps=300, batch=128, lr=2e-3, device="cpu",
)


@pytest.fixture(scope="module")
def split():
    p = load_dataset("synthetic", n_stores=3, n_items=8, n_periods=500, seed=1).panel
    cut = np.sort(p[DATE].unique())[-29]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted(split):
    return build_model("state_model", **SETTINGS).fit(split[0])


def test_beats_the_simple_benchmarks_on_generated_data(split, fitted):
    train, test = split
    future, y = test.drop(columns=[UNITS]), test[UNITS].to_numpy()
    rmse = lambda f: float(np.sqrt(((f - y) ** 2).mean()))
    ours = rmse(fitted.predict(future))
    assert ours < rmse(build_model("seasonal_naive").fit(train).predict(future))
    assert ours < rmse(build_model("recent_average").fit(train).predict(future))


def test_same_forecast_whatever_the_row_order_or_horizon_asked(split, fitted):
    future = split[1].drop(columns=[UNITS])
    a = fitted.predict(future)
    shuffled = future.sample(frac=1, random_state=1)
    b = pd.Series(fitted.predict(shuffled), index=shuffled.index).sort_index().to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-5)
    first10 = (future[DATE] <= future[DATE].min() + pd.Timedelta(days=9)).to_numpy()
    np.testing.assert_allclose(fitted.predict(future[first10]), a[first10], rtol=1e-5)


def test_refit_with_the_same_seed_gives_the_same_forecast(split, fitted):
    train, test = split
    again = build_model("state_model", **SETTINGS).fit(train)
    future = test.drop(columns=[UNITS])
    np.testing.assert_allclose(fitted.predict(future), again.predict(future), rtol=1e-4)


def test_forecast_inputs_equal_training_inputs_for_the_same_days(split):
    """Inputs built when forecasting 28 future days must equal the inputs the
    training code builds for those days once they are history."""
    train, test = split
    quick = {**SETTINGS, "steps": 1}
    model = build_model("state_model", **quick).fit(train)
    future = test.drop(columns=[UNITS])
    fut_dates = np.sort(future[DATE].unique())
    rows = torch.arange(len(model._names))
    zero, start = torch.zeros_like(rows), torch.full_like(rows, model.history)
    fut = model._tensors(future, fut_dates, with_units=False)
    got = model._window(model._past, {**fut, "start": zero}, rows, start)

    full = pd.concat([train, test]).sort_values([SERIES, DATE]).reset_index(drop=True)
    dates = np.sort(full[DATE].unique())[-(model.history + 28):]
    data = model._tensors(full[full[DATE] >= dates[0]], dates, with_units=True)
    want = model._window(data, {**data, "start": start}, rows, start)
    for g, w in zip(got[:4], want[:4]):  # history inputs, future inputs, scale, on-sale flags
        np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=1e-6)


def test_missing_lever_is_passed_as_not_known(split):
    """A dataset with no price column still trains and forecasts."""
    train, test = split
    quick = {**SETTINGS, "steps": 20, "extra": []}
    model = build_model("state_model", **quick).fit(train.drop(columns=["price"]))
    pred = model.predict(test.drop(columns=[UNITS, "price"]))
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_saved_model_gives_the_same_forecast_when_loaded(split, fitted, tmp_path):
    from rwm.model.state_model import StateModel

    future = split[1].drop(columns=[UNITS])
    fitted.save(tmp_path / "model.pt")
    loaded = StateModel.load(tmp_path / "model.pt", device="cpu")
    np.testing.assert_array_equal(fitted.predict(future), loaded.predict(future))


def test_run_saves_a_checkpoint_with_its_checksum(tmp_path):
    import json

    from rwm.experiments.run import run
    from rwm.utils.hashing import sha256_file

    config = {
        "name": "t", "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_stores": 2, "n_items": 3, "n_periods": 200}},
        "model": {"name": "state_model", "params": {**SETTINGS, "train_periods": 100, "steps": 5}},
        "evaluation": {"horizon": 28, "n_origins": 1},
    }
    out = run(config, out_root=tmp_path)
    saved = json.loads((out / "metrics.json").read_text())["checkpoints"]
    assert len(saved) == 1 and sha256_file(out / saved[0]["file"]) == saved[0]["sha256"]


def test_negative_binomial_likelihood_also_beats_the_simple_benchmarks(split):
    train, test = split
    future, y = test.drop(columns=[UNITS]), test[UNITS].to_numpy()
    rmse = lambda f: float(np.sqrt(((f - y) ** 2).mean()))
    model = build_model("state_model", **SETTINGS, likelihood="negative_binomial").fit(train)
    ours = rmse(model.predict(future))
    assert ours < rmse(build_model("seasonal_naive").fit(train).predict(future))
    assert ours < rmse(build_model("recent_average").fit(train).predict(future))
    with pytest.raises(ValueError):
        build_model("state_model", **SETTINGS, likelihood="other")


def test_level_options_change_only_the_level(split):
    """A capped level and a latest-weeks level both run, and the latest-weeks
    level follows an item whose sales have fallen away."""
    train, test = split
    quick = {**SETTINGS, "steps": 5}
    for extra in ({"scale_cap": 0.9}, {"scale_window": 8}, {"scale_unpromoted": True, "regular_price": True}, {"volume_weight": 0.5}, {"volume_weight": 0.5, "volume_by": "dollars"}, {"level_views": [4, 8]}):
        model = build_model("state_model", **quick, **extra).fit(train)
        assert np.isfinite(model.predict(test.drop(columns=[UNITS]))).all()
    faded = train.copy()
    late = faded[DATE] > np.sort(faded[DATE].unique())[-9]
    one = faded[SERIES] == faded[SERIES].astype(str).min()  # the first series in the model's order
    faded.loc[late & one, UNITS] = 0.0
    future = test.drop(columns=[UNITS])
    fut_dates = np.sort(future[DATE].unique())
    rows = torch.zeros(1, dtype=torch.long)
    levels = []
    for extra in ({}, {"scale_window": 8}):
        m = build_model("state_model", **quick, **extra).fit(faded)
        start = torch.full_like(rows, m._past["price"].shape[1])
        fut = m._tensors(future, fut_dates, with_units=False)
        levels.append(float(m._window(m._past, {**fut, "start": torch.zeros_like(rows)}, rows, start)[2]))
    assert levels[1] < levels[0]


def test_switching_keeps_group_totals_to_the_learned_share():
    import numpy as np
    import pytest

    pytest.importorskip("torch")
    from rwm.data.synthetic import load_synthetic
    from rwm.model.state_model import StateModel

    panel = load_synthetic(n_items=6, n_periods=80).panel
    dates = np.sort(panel["date"].unique())
    train, future = panel[panel["date"] <= dates[-5]], panel[panel["date"] > dates[-5]]
    m = StateModel(horizon=4, history=12, categorical=["item_id", "store_id"], extra=["promo"], levers=["promo"],
                   d_model=16, layers=1, heads=2, steps=30, batch=32, regular_price=True, lift_readout=True,
                   validation_periods=4, switching=["store_id"]).fit(train)
    assert 0.0 <= m.new_share <= 1.0
    f, b = m.predict(future), m._predict(future, "base")
    m.new_share = 1.0  # all lift is new sales: forecasts unchanged
    with_one, group = m.predict(future), m._group
    m._group = None
    assert np.allclose(with_one, m.predict(future))
    m._group = group
    m.new_share = 0.0  # all lift is switching: each store's total equals its baseline total
    zero = future.assign(f=m.predict(future), b=b).groupby(["store_id", "date"])[["f", "b"]].sum()
    assert np.allclose(zero["f"], zero["b"], rtol=1e-6)
    with pytest.raises(ValueError):
        StateModel(horizon=4, switching=["store_id"])
