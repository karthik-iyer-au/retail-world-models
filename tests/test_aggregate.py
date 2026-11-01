"""Adding a panel up to store totals (study P)."""
import numpy as np
import pandas as pd

from rwm.data.aggregate import aggregate_panel


def _panel():
    dates = pd.date_range("2020-01-02", periods=3, freq="7D")
    rows = []
    for store in ("001", "002"):
        for item, price in (("a", [2.0, 1.0, 2.0]), ("b", [4.0, 4.0, 4.0])):
            for d, pr, u in zip(dates, price, [1.0, 3.0, 2.0]):
                rows.append({"series_id": f"{item}_{store}", "store_id": store, "item_id": item, "date": d,
                             "units": u, "price": pr, "promo": int(pr < 2.0 and item == "a"), "cost": pr / 2, "cat_id": "x"})
    return pd.DataFrame(rows)


def test_store_totals_add_units_and_turn_the_plan_into_shares():
    out = aggregate_panel(_panel(), by=["store_id"])
    assert sorted(out["series_id"].astype(str).unique()) == ["001", "002"]
    s = out[out["series_id"] == "001"].reset_index(drop=True)
    assert s["units"].tolist() == [2.0, 6.0, 4.0]
    # week 2: item a at half its regular price, item b at its regular price
    assert np.isclose(s.loc[1, "price"], 0.75) and np.isclose(s.loc[1, "promo"], 0.5)
    assert np.isclose(s.loc[0, "price"], 1.0) and s.loc[0, "promo"] == 0.0
    assert (out["item_id"].astype(str) == "all").all() and (out["cat_id"].astype(str) == "x").all()
