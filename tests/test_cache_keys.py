"""Partway saves are keyed by the data as loaded (outside review, 15 November 2026)."""
import pandas as pd

from rwm.experiments.run import data_fingerprint


def test_fingerprint_follows_the_data():
    a = pd.DataFrame({"series": ["x", "y"], "units": [1.0, 2.0]})
    assert data_fingerprint(a) == data_fingerprint(a.copy())
    assert data_fingerprint(a) != data_fingerprint(a.assign(units=[1.0, 3.0]))
    assert data_fingerprint(a) != data_fingerprint(a.rename(columns={"units": "sales"}))
