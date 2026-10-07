"""Item roles: measured from the past only, and usable as a label."""
import numpy as np
import pandas as pd

from rwm.data.roles import ROLE, add_roles, item_measures


def _panel():
    dates = pd.date_range("2020-01-06", periods=40, freq="W-MON")
    rows = []

    def series(item, store, units, promo, keep=None):
        for t, d in enumerate(dates):
            if keep is not None and not keep[t]:
                continue
            rows.append((f"{item}_{store}", store, item, d, float(units[t]), int(promo[t])))

    promo = np.zeros(40, dtype=int)
    promo[::4] = 1
    for store, size in (("s1", 1.0), ("s2", 10.0)):
        series("steady", store, size * np.full(40, 10.0), promo)
        series("responsive", store, size * np.where(promo == 1, 25.0, 10.0), promo)
        series("driven", store, size * np.where(promo == 1, 90.0, 5.0), promo)
        series("in_out", store, size * np.full(40, 10.0), promo, keep=np.arange(40) % 4 == 0)
        series("late", store, size * np.full(40, 10.0), promo, keep=np.arange(40) >= 36)
    return pd.DataFrame(rows, columns=["series_id", "store_id", "item_id", "date", "units", "promo"]), dates


def test_roles_follow_the_measures():
    panel, dates = _panel()
    m = add_roles(panel, until=dates[31])
    assert m.loc["steady", ROLE] == "steady"
    assert m.loc["responsive", ROLE] == "responsive"
    assert m.loc["driven", ROLE] == "promotion_driven"
    assert m.loc["in_out", ROLE] == "in_and_out"
    assert np.isclose(m.loc["responsive", "lift"], 2.5)
    # first seen after the cut-off: no measures, labelled new
    assert "late" not in m.index
    assert set(panel.loc[panel["item_id"] == "late", ROLE]) == {"new"}
    # one role per item, the same in every store
    assert (panel.groupby("item_id")[ROLE].nunique() == 1).all()


def test_roles_do_not_see_later_periods():
    panel, dates = _panel()
    before = item_measures(panel, until=dates[31])
    changed = panel.copy()
    late = changed["date"] > dates[31]
    changed.loc[late, "units"] = 1000.0
    changed.loc[late, "promo"] = 1
    pd.testing.assert_frame_equal(before, item_measures(changed, until=dates[31]))


def test_run_with_roles_as_a_label_and_as_the_level_picker(tmp_path):
    import json

    import pytest

    pytest.importorskip("torch")
    from rwm.experiments.run import run

    config = {
        "name": "roles", "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_items": 6, "n_periods": 120}, "roles": {"lever": "promo"}},
        "model": {"name": "state_model", "params": {
            "horizon": 4, "history": 12, "categorical": ["item_id", "item_role"], "extra": ["promo"],
            "levers": ["promo"], "d_model": 16, "layers": 1, "heads": 2, "steps": 20, "batch": 32,
            "scale_window_by": {"label": "item_role", "windows": {"steady": 4, "responsive": 8}},
        }},
        "evaluation": {"horizon": 4, "n_origins": 2},
    }
    m = json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())
    assert np.isfinite(m["wrmsse"])


def test_lightgbm_takes_the_role_as_a_label(tmp_path):
    import json

    import pytest

    pytest.importorskip("lightgbm")
    from rwm.experiments.run import run

    config = {
        "name": "roles_lgb", "seed": 0,
        "dataset": {"name": "synthetic", "params": {"n_items": 6, "n_periods": 120}, "roles": {"lever": "promo"}},
        "model": {"name": "lightgbm_direct", "params": {
            "horizon": 4, "categorical": ["item_id", "item_role"], "extra": ["promo"], "rounds": 20,
            "params": {"min_data_in_leaf": 5},
        }},
        "evaluation": {"horizon": 4, "n_origins": 2, "holdout_periods": 4},
    }
    assert np.isfinite(json.loads((run(config, out_root=tmp_path) / "metrics.json").read_text())["wrmsse"])
