"""C7 zone price deviations on a generated chain where the answer is known."""
import numpy as np
import pandas as pd

from rwm.audit.zones import did, find_episodes, regular_prices, store_ratios


def _chain(effect=-0.3, seed=0):
    """One zone of 6 stores, 15 items, 80 weeks. Store 1 raises regular prices
    by 15% in weeks 40 to 59, which lowers its units by `effect` in logs."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, 7):
        level = rng.uniform(50, 100)
        for u in range(15):
            p = 1.0 + 0.1 * u
            for w in range(1, 81):
                treated = s == 1 and 40 <= w <= 59
                price = round(p * (1.15 if treated else 1.0), 2)
                mu = level * np.exp(0.1 * np.sin(w / 5) + (effect if treated else 0.0))
                rows.append((s, 1000 + u, w, rng.poisson(mu), 1, price, np.nan, 1))
    return pd.DataFrame(rows, columns=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "OK"])


def test_finds_the_episode_and_its_effect():
    move = _chain()
    zone = pd.Series(1, index=range(1, 7))
    reg = regular_prices(move)
    eps = find_episodes(store_ratios(reg, zone))
    assert len(eps) == 1
    e = eps[0]
    assert e.store == 1 and e.first_week == 40 and e.last_week == 59
    assert abs(e.log_ratio - np.log(1.15)) < 0.01
    assert e.controls == [2, 3, 4, 5, 6]
    r = did(reg, e)
    assert r["lo"] < -0.3 < r["hi"] and abs(r["estimate"] + 0.3) < 0.05


def test_deal_weeks_and_short_spells_are_not_episodes():
    move = _chain(effect=0.0)
    move["SALE"] = move["SALE"].astype(object)
    treated = (move["STORE"] == 1) & move["WEEK"].between(40, 59)
    move.loc[treated, "SALE"] = "S"  # the higher prices were deal weeks, not regular prices
    short = (move["STORE"] == 2) & move["WEEK"].between(10, 17)  # an 8-week spell is too short
    move.loc[short, "PRICE"] = (move.loc[short, "PRICE"] * 1.2).round(2)
    zone = pd.Series(1, index=range(1, 7))
    assert find_episodes(store_ratios(regular_prices(move), zone)) == []
