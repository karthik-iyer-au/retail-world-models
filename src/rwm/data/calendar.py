"""Seasonal calendar: the holidays and retail events of a country.

Every date here follows from a fixed rule, so the calendar is known years in
advance and a forecast may use it for future periods.

  events("US", [2011])            # date and name of each event
  add_calendar(panel, "US", ...)  # event columns on a panel

Add a country by writing one function and listing it in `_COUNTRIES`.
"""
import datetime as dt

import numpy as np
import pandas as pd

from rwm.data.schema import DATE


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th given weekday (Monday=0) of a month; n=-1 for the last."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> dt.date:
    """Western Easter Sunday (the anonymous Gregorian rule)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 19 * m) // 433
    month = (h + m - 7 * n + 90) // 25
    return dt.date(year, month, (h + m - 7 * n + 33 * month + 19) % 32)


def _super_bowl(year: int) -> dt.date:
    # last Sunday of January to 2001, first Sunday of February 2002 to 2021
    # (2003: last Sunday of January), second Sunday of February since
    if year >= 2022:
        return _nth_weekday(year, 2, 6, 2)
    if year >= 2004 or year == 2002:
        return _nth_weekday(year, 2, 6, 1)
    return _nth_weekday(year, 1, 6, -1)


def _us(year: int) -> dict[str, dt.date]:
    return {
        "new_year": dt.date(year, 1, 1),
        "super_bowl": _super_bowl(year),
        "valentines": dt.date(year, 2, 14),
        "easter": _easter(year),
        "mothers_day": _nth_weekday(year, 5, 6, 2),
        "memorial_day": _nth_weekday(year, 5, 0, -1),
        "july_4": dt.date(year, 7, 4),
        "labor_day": _nth_weekday(year, 9, 0, 1),
        "halloween": dt.date(year, 10, 31),
        "thanksgiving": _nth_weekday(year, 11, 3, 4),
        "christmas": dt.date(year, 12, 25),
    }


_COUNTRIES = {"US": _us}


def events(country: str, years) -> pd.DataFrame:
    if country not in _COUNTRIES:
        raise KeyError(f"no calendar for '{country}'. Known: {sorted(_COUNTRIES)}")
    rows = [(pd.Timestamp(d), name) for y in years for name, d in _COUNTRIES[country](y).items()]
    return pd.DataFrame(rows, columns=["date", "event"]).sort_values("date", ignore_index=True)


def add_calendar(panel: pd.DataFrame, country: str, anchor: str = "start", period_days: int | None = None) -> list[str]:
    """Adds two columns per event to `panel`, in place: `ev_<name>` is 1 in
    the period that holds the event, and `pre_<name>` is 1 in the period
    before it (when the shopping for it happens). Returns the new column
    names. `anchor` says whether a row's date is the first or last day of
    its period; `period_days` defaults to the usual gap between dates."""
    if anchor not in ("start", "end"):
        raise ValueError("anchor must be 'start' or 'end'")
    dates = pd.DatetimeIndex(np.sort(panel[DATE].unique()))
    n = period_days or int(pd.Series(dates).diff().dt.days.median())
    first = dates - pd.Timedelta(days=n - 1) if anchor == "end" else dates
    ev = events(country, range(first.min().year - 1, dates.max().year + 2))
    start, day = first.to_numpy(), np.timedelta64(1, "D")
    lookup, added = pd.Index(dates).get_indexer(panel[DATE]), []
    for name, when in ev.groupby("event")["date"]:
        when = when.to_numpy()[None, :]
        here = ((when >= start[:, None]) & (when < start[:, None] + n * day)).any(1)
        before = ((when >= start[:, None] + n * day) & (when < start[:, None] + 2 * n * day)).any(1)
        panel[f"ev_{name}"] = here[lookup].astype(np.int8)
        panel[f"pre_{name}"] = before[lookup].astype(np.int8)
        added += [f"ev_{name}", f"pre_{name}"]
    return added
