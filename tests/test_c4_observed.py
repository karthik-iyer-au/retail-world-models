"""C4 within the observed price range (paper 2 plan amendment of 16 November 2026)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("c4o", Path(__file__).resolve().parents[1] / "scripts" / "c4_observed.py")
c4o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c4o)


class Odd:
    """Sales fall once the price drops below 0.85: a model that fails C4 only on deep cuts."""
    def predict(self, plan):
        return np.where(plan["price"].to_numpy() < 0.85, 1.0, 10.0 / plan["price"].to_numpy())


def test_counts_only_prices_seen_in_training():
    train = pd.DataFrame({"series_id": ["a"] * 3 + ["b"] * 3, "price": [1.0, 0.9, 0.8, 1.0, 1.0, 0.5], "units": 1.0})
    test = pd.DataFrame({"series_id": ["a", "b"], "price": [1.0, 1.0], "units": 1.0})
    r = c4o.c4_observed(Odd(), train, test)
    assert r["steps"] == 8
    # a: lowest seen 0.8, so cuts to 0.95 and 0.9 kept, 0.8 kept, 0.7 not; b: lowest 0.5, all four kept
    assert r["kept"] == 7
    # the failing step (0.9 -> 0.8 for both series) is inside the range for both
    assert r["c4_all"] == 2 / 8 and abs(r["c4_observed"] - 2 / 7) < 1e-12
