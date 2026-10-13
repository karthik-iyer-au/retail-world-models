"""Price path of each scored zone episode: the store's median price against its
zone peers by week relative to the episode start, from 52 weeks before to 8
after (data only). Shows whether prices had already moved before the start or
in the placebo windows.

  python scripts/zone_price_path.py --episodes results/paper2/zone_episodes.csv --out results/paper2/price_path.csv
"""
import argparse
import zipfile

import pandas as pd

import rwm.audit.zones as Z
from rwm.utils.paths import DATA_RAW

from zone_episodes import movement


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    eps = pd.read_csv(a.episodes)
    eps = eps[eps["scored"]]
    with zipfile.ZipFile(DATA_RAW / "dominicks" / "demo_stata.zip") as z:
        z.extract("demo.dta", "/tmp")
    zone = Z.zones("/tmp/demo.dta")
    rows = []
    for code, ce in eps.groupby("category"):
        ratios = Z.store_ratios(Z.regular_prices(movement(code)), zone)
        for _, e in ce.iterrows():
            d = ratios[(ratios["store"] == e["store"]) & ratios["week"].between(e["first_week"] - 52, e["first_week"] + 7)]
            for _, r in d.iterrows():
                rows.append({"category": code, "store": e["store"], "first_week": e["first_week"],
                             "price_log_ratio": e["price_log_ratio"], "rel_week": int(r["week"] - e["first_week"]),
                             "ratio": r["median"], "items": r["items"]})
        print(code, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
