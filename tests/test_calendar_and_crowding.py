import numpy as np
import pandas as pd
import pytest

from rwm.data.calendar import add_calendar, events
from rwm.data.crowding import add_crowding
from rwm.data.schema import DATE, ITEM, SERIES, STORE


def test_us_event_dates_match_the_real_ones():
    got = events("US", [1996, 2011]).set_index("event")["date"].astype(str)
    real = {
        "1996-01-28": "super_bowl", "1996-04-07": "easter", "1996-11-28": "thanksgiving",
        "2011-02-06": "super_bowl", "2011-04-24": "easter", "2011-05-08": "mothers_day",
        "2011-05-30": "memorial_day", "2011-09-05": "labor_day", "2011-11-24": "thanksgiving",
    }
    for date, name in real.items():
        assert date in set(got[name]), (name, date)
    with pytest.raises(KeyError):
        events("XX", [2011])


def test_event_lands_in_the_week_that_holds_it_and_the_week_before_is_marked():
    weeks = pd.DataFrame({DATE: pd.date_range("2011-11-02", periods=6, freq="7D")})  # week-ending Wednesdays
    cols = add_calendar(weeks, "US", anchor="end")
    assert "ev_thanksgiving" in cols and "pre_thanksgiving" in cols
    by = weeks.set_index(weeks[DATE].astype(str))
    # Thanksgiving 2011 was Thursday 24 November: the week ending Wednesday 30 November
    assert by.loc["2011-11-30", "ev_thanksgiving"] == 1 and by["ev_thanksgiving"].sum() == 1
    assert by.loc["2011-11-23", "pre_thanksgiving"] == 1 and by["pre_thanksgiving"].sum() == 1
    starts = pd.DataFrame({DATE: pd.date_range("2011-11-03", periods=6, freq="7D")})  # week-starting Thursdays
    add_calendar(starts, "US", anchor="start")
    assert starts.set_index(starts[DATE].astype(str)).loc["2011-11-24", "ev_thanksgiving"] == 1


def test_crowding_counts_the_other_items_on_the_lever():
    rows = [("S0", f"I{i}", "A" if i < 2 else "B", d, on) for d in (1, 2) for i, on in enumerate([1, 1, 0, 0])]
    p = pd.DataFrame(rows, columns=[STORE, ITEM, "cat", "d", "display"])
    p[DATE] = pd.Timestamp("2020-01-01") + pd.to_timedelta(p["d"] * 7, unit="D")
    p[SERIES] = p[STORE] + "_" + p[ITEM]
    p.loc[(p["d"] == 2), "display"] = 0
    assert add_crowding(p, ["display"], within="cat") == ["display_share", "display_share_within"]
    first = p[p["d"] == 1].set_index(ITEM)
    np.testing.assert_allclose(first["display_share"], [1 / 3, 1 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(first["display_share_within"], [1, 1, 0, 0])
    assert (p.loc[p["d"] == 2, "display_share"] == 0).all()
