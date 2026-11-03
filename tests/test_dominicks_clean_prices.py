"""The Dominick's price cleaning rule (study S, docs/PAPER1_REVISION_PLAN.md)."""
import numpy as np
import pandas as pd

from rwm.data.dominicks import _clean_prices


def frame(prices, upc=1, week=10):
    n = len(prices)
    return pd.DataFrame({"store": np.arange(n), "upc": upc, "week": week,
                         "price": np.array(prices, dtype=np.float32), "cost": np.array(prices, dtype=np.float32) * 0.7})


def test_far_prices_become_missing_both_ways():
    d = _clean_prices(frame([1.29, 1.29, 1.25, 84.72, 0.10]), 5)
    assert d["price"].isna().tolist() == [False, False, False, True, True]
    assert d["cost"].isna().tolist() == [False, False, False, True, True]


def test_ordinary_promotion_prices_are_kept():
    d = _clean_prices(frame([1.29, 1.29, 0.79, 0.50, 1.29]), 5)
    assert not d["price"].isna().any()


def test_needs_three_stores():
    d = _clean_prices(frame([1.29, 84.72]), 5)
    assert not d["price"].isna().any()


def test_compares_within_item_and_week():
    d = pd.concat([frame([1.0, 1.0, 1.0], upc=1), frame([20.0, 20.0, 20.0], upc=2)], ignore_index=True)
    assert not _clean_prices(d, 5)["price"].isna().any()
