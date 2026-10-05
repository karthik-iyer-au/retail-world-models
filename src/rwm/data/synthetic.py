"""Generated data for tests and the smoke run only.

Never used for any reported result. It exists so the full pipeline can be
checked end to end without downloading anything.
"""
import numpy as np
import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS


@register_dataset("synthetic")
def load_synthetic(
    n_stores: int = 2, n_items: int = 5, n_periods: int = 200, seed: int = 0,
    steal: float = 0.0,
    dip: float = 0.0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    weekly = 1.0 + 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
    rows = []
    for s in range(n_stores):
        previous = np.zeros(n_periods, dtype=bool)
        for i in range(n_items):
            base = rng.uniform(2, 20)
            list_price = rng.uniform(1, 10)
            promo = rng.random(n_periods) < 0.1
            # with `steal`, an item loses that share of its sales in periods
            # when the item before it in the same store is on promotion
            base = base * np.where(previous & (i % 2 == 1), 1 - steal, 1.0)
            previous = promo
            # with `dip`, an item loses that share of its sales in the period
            # after its own promotion (shoppers stocked up)
            base = base * np.where(np.r_[False, promo[:-1]] & ~promo, 1 - dip, 1.0)
            price = np.where(promo, list_price * 0.8, list_price)
            mean = base * weekly * np.where(promo, 1.8, 1.0)
            rows.append(
                pd.DataFrame(
                    {
                        SERIES: f"S{s}_I{i}",
                        STORE: f"S{s}",
                        ITEM: f"I{i}",
                        DATE: dates,
                        UNITS: rng.poisson(mean).astype(float),
                        PRICE: price.round(2),
                        "promo": promo.astype(int),
                    }
                )
            )
    return Dataset("synthetic", pd.concat(rows, ignore_index=True), ["price", "promo"])
