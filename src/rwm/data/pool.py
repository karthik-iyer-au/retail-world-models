"""Other data sets brought in to train alongside a target data set (study U part b,
docs/PAPER1_REVISION_PLAN.md, entry of 11 November 2026).

A source is given as a data set config, as an experiment's `dataset` block, plus two keys:

  name, params, calendar, season   as in an experiment config (the derived columns are added the
                                   same way, so they mean the same in both data sets)
  same_retailer: true              the source shares the target's stores and weeks (another Dominick's
                                   category): only its periods up to the target's training end are
                                   used, so no sales from the target's test weeks get in
  tag: <text>                      prefixed to the source's series, item and store labels, so they
                                   never match the target's

The source is returned in the target's columns: a column the target uses and the source lacks is
added as zeros (a lever that was never used), a column the source has and the target does not use
is dropped by the model that reads it.
"""
from __future__ import annotations

import pandas as pd

from rwm.data import load_dataset
from rwm.data.calendar import add_calendar, add_season
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE


def load_source(spec: dict, columns: list[str], until=None) -> pd.DataFrame:
    """The source's panel with `columns` present, labels prefixed, and, for a source from the
    target's own retailer, only periods before `until`."""
    ds = load_dataset(spec["name"], **spec.get("params", {}))
    panel = ds.panel
    if "calendar" in spec:
        add_calendar(panel, **spec["calendar"])
    if spec.get("season"):
        add_season(panel, **(spec["season"] if isinstance(spec["season"], dict) else {}))
    if spec.get("same_retailer") and until is not None:
        panel = panel[panel[DATE] <= pd.Timestamp(until)]
    tag = spec.get("tag") or spec["name"]
    panel = panel.copy()
    for c in (SERIES, ITEM, STORE):
        panel[c] = f"{tag}:" + panel[c].astype(str)
    for c in columns:
        if c not in panel:
            panel[c] = 1.0 if c == PRICE else 0.0  # no recorded price: one constant price, as run.py treats it
    return panel.reset_index(drop=True)
