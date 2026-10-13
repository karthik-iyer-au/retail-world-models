"""The generated data of the paper 2 known-answer study."""
import numpy as np
import pandas as pd

from rwm.audit.zones import Episode, event_study
from rwm.data import load_dataset
from rwm.data.schema import DATE, PRICE, SERIES
from rwm.data.synthetic_levers import Truth, expected_units, planned


def small(**kw):
    return load_dataset("synthetic_levers", n_zones=2, stores_per_zone=5, n_items=4, n_periods=120,
                        seed=1, test_origin=101, **kw).panel


def test_expected_units_match_the_effects():
    p = small()
    t = Truth(**p.attrs["truth"])
    one = p[p[SERIES] == p[SERIES].iloc[0]].sort_values(DATE).reset_index(drop=True)
    base = expected_units(one.assign(**{PRICE: one["true_regular"], "promo": 0, "display": 0}), t)
    assert np.allclose(base, one["true_level"])
    promo = expected_units(one.assign(**{PRICE: one["true_regular"], "promo": 1, "display": 0}), t)
    assert np.allclose(promo / base, 1 + t.promo_effect)
    cut = expected_units(one.assign(**{PRICE: one["true_regular"] * 0.8, "promo": 0, "display": 0}), t)
    assert np.allclose(cut / base, 0.8 ** t.cut_elasticity)


def test_dip_follows_a_planned_week():
    p = small()
    t = Truth(**p.attrs["truth"])
    one = p[p[SERIES] == p[SERIES].iloc[0]].sort_values(DATE).reset_index(drop=True)
    off = one.assign(**{PRICE: one["true_regular"], "promo": 0, "display": 0})
    on = off.copy()
    on.loc[10, "promo"] = 1
    ratio = expected_units(on, t) / expected_units(off, t)
    assert np.isclose(ratio[11], 1 - t.dip) and np.isclose(ratio[12], 1.0)


def test_origin_episodes_and_confounding():
    p = small()
    eps = p.attrs["episodes"]
    assert sum(e["at_origin"] for e in eps) == 4 and all(e["first_week"] == 101 for e in eps if e["at_origin"])
    clean, conf = small(), small(confounding=1.5)
    shock = lambda d: np.log(d["true_level"]) - np.log(d.groupby(SERIES)["true_level"].transform("median"))
    gap = lambda d: shock(d)[planned(d)].mean() - shock(d)[~planned(d)].mean()
    assert gap(conf) > gap(clean) + 0.05


def test_event_study_finds_a_step():
    rng = np.random.default_rng(0)
    rows = [{"store": s, "week": w, "units": 100 * (0.8 if (s == 1 and w >= 40) else 1.0) * np.exp(rng.normal(0, 0.01))}
            for s in range(1, 5) for w in range(1, 60)]
    es = event_study(pd.DataFrame(rows), Episode(1, 1, 40, 55, 0.1, [2, 3, 4]))
    assert abs(es.loc[-10]) < 0.05 and abs(es.loc[3] - np.log(0.8)) < 0.05
