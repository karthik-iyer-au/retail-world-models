"""Paper 1 revision: promotion-decision regret where the true lift is known
(docs/PAPER1_REVISION_PLAN.md, "Decisions").

  python scripts/promo_regret.py --known results/paper2/known_answer --out results/paper1_revision/promo_regret.csv

In paper 2's generated worlds, for every item-week that had a promotion, a
merchant keeps the promotion when the model's promotion lift (units) is above
the promotion's cost in units of margin. The value of a kept promotion is its
true lift minus the cost. Regret is the value of deciding with the true lift
minus the value of deciding with the model's, as a share of the former. The
cost is set at the 25th, 50th and 75th percentile of the true lifts, so that
some promotions are worth keeping and some are not.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--known", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = []
    for f in sorted(Path(a.known).glob("*_lifts.csv.gz")):
        world, model = f.name.replace("_lifts.csv.gz", "").split("_", 1)
        d = pd.read_csv(f)
        on = d["true_lift_promo"].abs() > 1e-9
        d = d[on]
        true, pred = d["true_lift_promo"].to_numpy(), d["lift_promo"].to_numpy()
        for pct in (25, 50, 75):
            cost = np.percentile(true, pct)
            best = np.clip(true - cost, 0, None).sum()
            got = ((true - cost) * (pred > cost)).sum()
            rows.append({"world": world, "model": model, "cost_percentile": pct, "promotions": int(on.sum()),
                         "kept_by_model": float((pred > cost).mean()), "kept_by_truth": float((true > cost).mean()),
                         "regret_share": float((best - got) / best) if best > 0 else np.nan})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).pivot_table(index=["world", "model"], columns="cost_percentile", values="regret_share").round(3).to_string())


if __name__ == "__main__":
    main()
