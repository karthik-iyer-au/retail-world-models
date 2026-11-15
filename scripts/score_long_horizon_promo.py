"""52-week forecasts scored on price-cut weeks and regular weeks separately (plan entry of 15 November 2026).

  python scripts/score_long_horizon_promo.py --results <h52 folder> --out <folder>

Question (Karthik, 15 November): is the world model's gap to LightGBM on Dominick's in the promotion weeks? A week is a
price-cut week for an item in a store when its price is at most 95% of that item's most common price in the store over
the year before the window (52 weeks; 364 days on M5); the same rule on both data sets, since M5 records no
promotions. On Dominick's the deal code is scored as well (the manual warns it is not always set). The score is the
item-by-store RMSSE over the weeks of each kind, weighted by revenue over the year before the window (as WRMSSE's
bottom level), with the 95% interval of each model's difference to LightGBM from 1,000 resamples of items.
Fixed before it ran: the 95% rule, the year before the window, the weights and the resampling.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_long_horizon import index  # noqa: E402
from rwm.evaluation import compare as C  # noqa: E402

MODELS = ["world_model_direct", "world_model_tdirect", "world_model_rollon", "world_model_blocks", "lightgbm_direct",
          "lightgbm_tdirect", "lightgbm_blocks", "average_direct", "average_tdirect", "average_rollon"]
BASE = "lightgbm_direct"


def regular_price(price: np.ndarray) -> np.ndarray:
    out = np.full(len(price), np.nan)
    for i, row in enumerate(price):
        x = np.round(row[np.isfinite(row) & (row > 0)], 2)
        if len(x):
            v, c = np.unique(x, return_counts=True)
            out[i] = v[c.argmax()]
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = index(results)
    rows = []
    rng = np.random.default_rng(0)
    for (key, window), runs in idx.groupby(["key", "window"]):
        cfg, _ = C.load_run(results / runs["run"].iloc[0])
        fr = C.frame(cfg)
        t0, t1, rev = fr.cut(0)
        mats = {}
        for _, r in runs.iterrows():
            f = pd.read_csv(results / r["run"] / "forecasts.csv")
            mats[f"{r['model']}_{r['setup']}"] = C.forecast_matrix(f, fr, 0)
        for t in ("", "t"):
            if f"world_model_{t}direct" in mats and f"lightgbm_{t}direct" in mats:
                mats[f"average_{t}direct"] = (mats[f"world_model_{t}direct"] + mats[f"lightgbm_{t}direct"]) / 2
        if "world_model_rollon" in mats and "lightgbm_direct" in mats:
            mats["average_rollon"] = (mats["world_model_rollon"] + mats["lightgbm_direct"]) / 2
        if BASE not in mats:
            continue
        reg = regular_price(fr.price[:, t0 - fr.window:t0])
        pt = fr.price[:, t0:t1]
        cut = np.isfinite(pt) & (pt <= 0.95 * reg[:, None])
        kinds = {"price_cut": cut, "regular": ~cut}
        if fr.promo is not None and key.startswith("dominicks"):
            deal = fr.promo[:, t0:t1] > 0
            kinds |= {"deal": deal, "no_deal": ~deal}
        u_hist, u = fr.units[:, :t0], fr.units[:, t0:t1]
        lag = fr.lag
        d = u_hist[:, lag:] - u_hist[:, :-lag]
        started = np.cumsum(u_hist, axis=1)[:, lag:] > 0
        scale = np.array([np.mean(r[k] ** 2) if k.any() else np.nan for r, k in zip(d, started)])
        for kind, m in kinds.items():
            n = m.sum(axis=1)
            ok = (n > 0) & np.isfinite(scale) & (scale > 0) & (rev > 0)
            w = rev[ok] / rev[ok].sum()
            err = {}
            for name in MODELS:
                if name not in mats:
                    continue
                e = np.where(m[ok], (u[ok] - mats[name][ok]) ** 2, 0.0).sum(axis=1) / n[ok]
                err[name] = np.sqrt(e / scale[ok])
            draws = rng.multinomial(len(w), np.full(len(w), 1 / len(w)), size=a.boot)  # items resampled
            for name, e in err.items():
                point = float((w * e).sum())
                diff = (draws * (w * (e - err[BASE]))).sum(axis=1) / (draws * w).sum(axis=1)
                rows.append(dict(key=key, window=window, kind=kind, model=name, rmsse=point,
                                 share_weeks=float(m.mean()), items=int(ok.sum()),
                                 diff_to_lgb=float((w * (e - err[BASE])).sum()),
                                 lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5))))
        print(key, window, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "promo_scores.csv", index=False)


if __name__ == "__main__":
    main()
