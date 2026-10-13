"""List the zone price deviation episodes on Dominick's and their measured
effect on category units (check C7 of docs/PAPER2_PLAN.md). Reads no model.

  python scripts/zone_episodes.py --categories che sna sdr bjc ana fre --out results/paper2/zone_episodes.csv
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

from rwm.audit.zones import SCORED_WEEKS, did, scorable, find_episodes, regular_prices, store_ratios, zones
from rwm.utils.paths import DATA_RAW

COLS = ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "OK"]


def movement(code: str) -> pd.DataFrame:
    root = DATA_RAW / "dominicks"
    for name in (f"w{code}.zip", f"w{code}_csv.zip"):
        if (root / name).exists():
            return pd.read_csv(root / name, usecols=COLS, dtype={"SALE": "object"})
    raise FileNotFoundError(code)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    with zipfile.ZipFile(DATA_RAW / "dominicks" / "demo_stata.zip") as z:
        z.extract("demo.dta", "/tmp")
    zone = zones("/tmp/demo.dta")
    rows = []
    for code in a.categories:
        move = movement(code)
        reg = regular_prices(move)
        allunits = move[move["OK"] == 1].rename(columns={"STORE": "store", "UPC": "upc", "WEEK": "week",
                                                         "MOVE": "units"})[["store", "upc", "week", "units"]]
        eps = find_episodes(store_ratios(reg, zone))
        print(f"{code}: {len(eps)} episodes", flush=True)
        for e in eps:
            r = did(allunits, e)
            r8 = did(allunits, e, last_week=e.first_week + SCORED_WEEKS - 1)
            r.update({f"{k}_8wk": v for k, v in r8.items() if k in ("estimate", "se", "lo", "hi")})
            r["scored"] = scorable(e)
            rows.append({"category": code, "zone": e.zone, "store": e.store, "first_week": e.first_week,
                         "last_week": e.last_week, "weeks": e.last_week - e.first_week + 1,
                         "price_log_ratio": round(e.log_ratio, 4), "controls": " ".join(map(str, e.controls)),
                         "n_controls": len(e.controls), **{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in r.items()}})
        del move, reg, allunits
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
