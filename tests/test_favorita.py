"""Study D: the Favorita loader, on small files in the competition's format."""
import numpy as np
import pandas as pd
import pytest

from rwm.data import load_dataset


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-12-01", "2017-01-31", freq="D")
    dates = dates[~((dates.month == 12) & (dates.day == 25))]  # stores closed: no rows
    rows = []
    for s in (1, 2, 3):
        for i in (10, 11, 12):
            if s == 3 and i == 12:
                continue  # item 12 not sold in store 3, so it is dropped with 3 stores
            for d in dates:
                if rng.random() < 0.7:
                    rows.append((d.date().isoformat(), s, i, float(rng.integers(-1, 9)) * (4 - s),
                                 "" if d.year == 2015 else rng.choice(["True", "False"])))
    pd.DataFrame(rows, columns=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]) \
        .assign(id=lambda f: range(len(f)))[["id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]] \
        .to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"item_nbr": [10, 11, 12], "family": ["DAIRY", "DAIRY", "BREAD"], "class": [1, 2, 3],
                  "perishable": [1, 1, 0]}).to_csv(tmp_path / "items.csv", index=False)
    pd.DataFrame({"store_nbr": [1, 2, 3], "city": ["Quito", "Quito", "Guayaquil"], "state": ["P", "P", "G"],
                  "type": ["A", "A", "B"], "cluster": [1, 1, 2]}).to_csv(tmp_path / "stores.csv", index=False)
    pd.DataFrame({"date": ["2016-01-01", "2016-05-24", "2016-08-10", "2016-03-02"],
                  "type": ["Holiday", "Holiday", "Holiday", "Holiday"],
                  "locale": ["National", "National", "National", "Local"], "locale_name": ["Ecuador"] * 3 + ["Quito"],
                  "description": ["a", "b", "c", "d"], "transferred": [False, False, True, False]}) \
        .to_csv(tmp_path / "holidays_events.csv", index=False)
    return tmp_path


def test_selection_zeros_and_flags(root):
    ds = load_dataset("favorita", root=str(root), start="2016-01-01", n_stores=3, select_year=2016, cache=False)
    p = ds.panel
    assert set(p["item_id"].astype(str)) == {"I10", "I11"}  # item 12 was not sold in every store
    assert p["units"].min() >= 0 and (p["price"] == 1).all()
    days = pd.date_range("2016-01-01", p["date"].max(), freq="D")
    assert p.groupby("series_id", observed=True).size().eq(len(days)).all()  # every day, 25 December included
    xmas = p[p["date"] == pd.Timestamp("2016-12-25")]
    assert (xmas["units"] == 0).all()
    ev = p.drop_duplicates("date").set_index("date")["event"]
    assert ev[pd.Timestamp("2016-01-01")] == 1 and ev[pd.Timestamp("2016-08-10")] == 0  # transferred
    assert ev[pd.Timestamp("2016-03-02")] == 0  # local
    pay = p.drop_duplicates("date").set_index("date")["payday"]
    assert pay[pd.Timestamp("2016-01-15")] == 1 and pay[pd.Timestamp("2016-01-31")] == 1 and pay[pd.Timestamp("2016-01-16")] == 0
    assert ds.hierarchy[-1] == ["item_id", "store_id"]


def test_top_stores_by_units_and_cache(root):
    ds = load_dataset("favorita", root=str(root), start="2016-01-01", n_stores=2, select_year=2016)
    assert set(ds.panel["store_id"].astype(str)) == {"S01", "S02"}  # the stores selling most
    assert set(ds.panel["item_id"].astype(str)) == {"I10", "I11", "I12"}
    again = load_dataset("favorita", root=str(root), start="2016-01-01", n_stores=2, select_year=2016)
    pd.testing.assert_frame_equal(ds.panel.reset_index(drop=True), again.panel.reset_index(drop=True), check_categorical=False)
