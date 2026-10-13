"""C7 robustness on Dominick's, data only (docs/PAPER2_PLAN.md, amendment of
13 October 2026): event study, placebo start dates, and detector sensitivity.

  python scripts/zone_robustness.py --episodes results/paper2/zone_episodes.csv --out results/paper2

Writes event_study.csv, placebo.csv and sensitivity.csv next to the episode list.
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import rwm.audit.zones as Z
from rwm.utils.paths import DATA_RAW

from zone_episodes import movement


def slope(est, ratio):
    return float(np.polyfit(np.asarray(ratio, float), np.asarray(est, float), 1)[0]) if len(est) > 2 else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    eps = pd.read_csv(a.episodes)
    with zipfile.ZipFile(DATA_RAW / "dominicks" / "demo_stata.zip") as z:
        z.extract("demo.dta", "/tmp")
    zone = Z.zones("/tmp/demo.dta")
    ev_rows, pl_rows, sens_rows = [], [], []
    base_week, base_min = Z.WEEK_SIZE, Z.MIN_SIZE
    for code in eps["category"].unique():
        move = movement(code)
        units = move[move["OK"] == 1].rename(columns={"STORE": "store", "WEEK": "week", "MOVE": "units"})[
            ["store", "week", "units"]]
        for _, r in eps[(eps["category"] == code) & eps["scored"]].iterrows():
            e = Z.Episode(int(r["zone"]), int(r["store"]), int(r["first_week"]), int(r["last_week"]),
                          float(r["price_log_ratio"]), [int(x) for x in str(r["controls"]).split()])
            for rel, coef in Z.event_study(units, e).items():
                ev_rows.append({"category": code, "store": e.store, "first_week": e.first_week,
                                "price_log_ratio": e.log_ratio, "rel_week": int(rel), "coef": float(coef)})
            # placebo: the same 8-week estimator with the start moved PRE_WEEKS earlier
            fake = Z.Episode(e.zone, e.store, e.first_week - Z.PRE_WEEKS, e.first_week - Z.PRE_WEEKS + Z.SCORED_WEEKS - 1,
                             e.log_ratio, e.controls)
            if fake.first_week - Z.PRE_WEEKS >= 1:
                d = Z.did(units, fake, last_week=fake.last_week)
                pl_rows.append({"category": code, "store": e.store, "first_week": e.first_week,
                                "price_log_ratio": e.log_ratio, **{f"placebo_{k}": v for k, v in d.items()}})
        # detector sensitivity: episode count and pooled 8-week slope
        reg = Z.regular_prices(move)
        ratios = Z.store_ratios(reg, zone)
        for ws in (0.02, 0.03, 0.04):
            for ms in (0.04, 0.05, 0.07):
                Z.WEEK_SIZE, Z.MIN_SIZE = ws, ms
                found = Z.find_episodes(ratios)
                for e in found:
                    scored = Z.scorable(e)
                    est = Z.did(units, e, last_week=e.first_week + Z.SCORED_WEEKS - 1)["estimate"] if scored else np.nan
                    sens_rows.append({"category": code, "week_size": ws, "min_size": ms, "store": e.store,
                                      "first_week": e.first_week, "price_log_ratio": e.log_ratio,
                                      "scored": scored, "estimate_8wk": est})
                Z.WEEK_SIZE, Z.MIN_SIZE = base_week, base_min
        print(f"{code}: done", flush=True)
        del move, units, reg, ratios
    out = Path(a.out)
    pd.DataFrame(ev_rows).to_csv(out / "event_study.csv", index=False)
    pd.DataFrame(pl_rows).to_csv(out / "placebo.csv", index=False)
    s = pd.DataFrame(sens_rows)
    s.to_csv(out / "sensitivity_episodes.csv", index=False)
    summ = s.groupby(["week_size", "min_size"]).apply(lambda d: pd.Series({
        "episodes": len(d), "scored": int(d["scored"].sum()),
        "slope_8wk": slope(d.loc[d["scored"], "estimate_8wk"], d.loc[d["scored"], "price_log_ratio"])}),
        include_groups=False).reset_index()
    summ.to_csv(out / "sensitivity.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
