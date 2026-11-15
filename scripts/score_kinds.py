"""When is the world model more accurate than LightGBM: the final runs scored by kind of item and kind of week
(plan entry of 15 November 2026; Karthik, 14:40, "yes i want to add that too").

  python scripts/score_kinds.py --results <final runs folder> --out <folder>

The final tuned world model, LightGBM and their average, on every data set's rolling windows (as paper 1's RQ results),
scored on item-by-store series of each kind, every kind fixed from the year before each window (52 weeks; 364 days):
  selling rate     fast lines (the items making 80% of units) and slow lines, as RQ2 (rwm.evaluation.compare)
  history          short (first sale under 26 weeks, 182 days, before the window) and the rest, as RQ2
  intermittency    intermittent (no sales in more than half the periods) and steady (no sales in at most a tenth)
  promotion        none (no price cut), light (a price cut in under a quarter of the periods), heavy (a quarter or more);
                   a price cut is a price at most 95% of the series' most common price that year; Favorita records no
                   prices, so its promotion flag stands in
and on kinds of period inside the window: price-cut periods and the others (Favorita: promotion flag on and off).
Score: item-by-store RMSSE over the series (and periods) of the kind, weighted by revenue over the window's weight
period as WRMSSE (units on Favorita); differences to LightGBM with 95% intervals from 1,000 resamples of series,
windows pooled. Reading, fixed now: a kind where the world model is clearly more accurate than LightGBM on at least
five of the nine data sets is one where it works; clearly less accurate on at least five, one where it does not.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_analysis as RA  # noqa: E402
from rwm.evaluation import compare as C  # noqa: E402

MODELS = ["world_model", "lightgbm", "average"]


def year_before(fr, t0):
    daily = np.median(np.diff(fr.dates[-10:]).astype("timedelta64[D]").astype(int)) < 7
    w = 364 if daily else 52
    return max(0, t0 - w), t0


def regular(price):
    out = np.full(len(price), np.nan)
    for i, row in enumerate(price):
        x = np.round(row[np.isfinite(row) & (row > 0)], 2)
        if len(x):
            v, c = np.unique(x, return_counts=True)
            out[i] = v[c.argmax()]
    return out


def kinds(fr, o, has_price):
    t0, t1, rev = fr.cut(o)
    a, b = year_before(fr, t0)
    u = fr.units[:, a:b]
    zero = (u <= 0).mean(axis=1)
    masks, _ = C.line_segments(fr, o)
    p = fr.price[:, a:b]
    if has_price:
        reg = regular(p)
        cut = np.isfinite(p) & (p <= 0.95 * reg[:, None])
    else:
        cut = (fr.promo[:, a:b] > 0) if fr.promo is not None else np.zeros_like(u, bool)
    share = cut.mean(axis=1)
    series = {
        "fast": masks.get("fast"), "slow": masks.get("slow"),
        "short_history": masks["short_history"], "long_history": ~masks["short_history"],
        "intermittent": zero > 0.5, "steady": zero <= 0.1,
        "no_promotion": share == 0, "light_promotion": (share > 0) & (share < 0.25), "heavy_promotion": share >= 0.25,
    }
    series = {k: v for k, v in series.items() if v is not None}
    if has_price:
        reg_now = regular(fr.price[:, a:b])
        pw = fr.price[:, t0:t1]
        cutw = np.isfinite(pw) & (pw <= 0.95 * reg_now[:, None])
    else:
        cutw = (fr.promo[:, t0:t1] > 0) if fr.promo is not None else np.zeros((len(fr.attrs), t1 - t0), bool)
    periods = {"price_cut_periods": cutw, "other_periods": ~cutw}
    return series, periods


def rows_for(fr, o, mats, series, periods):
    t0, t1, rev = fr.cut(o)
    u_hist, u = fr.units[:, :t0], fr.units[:, t0:t1]
    lag = fr.lag
    d = u_hist[:, lag:] - u_hist[:, :-lag]
    started = np.cumsum(u_hist, axis=1)[:, lag:] > 0
    scale = np.array([np.mean(r[k] ** 2) if k.any() else np.nan for r, k in zip(d, started)])
    out = []
    every = np.ones_like(u, bool)
    tasks = [(k, m[:, None] & every) for k, m in series.items()] + list(periods.items())
    for kind, mask in tasks:
        n = mask.sum(axis=1)
        ok = (n > 0) & np.isfinite(scale) & (scale > 0) & (rev > 0)
        if not ok.any():
            continue
        e = {}
        for name in MODELS:
            if name in mats:
                sq = np.where(mask[ok], (u[ok] - mats[name][ok]) ** 2, 0.0).sum(axis=1) / n[ok]
                e[name] = np.sqrt(sq / scale[ok])
        out.append(pd.DataFrame({"origin": o, "kind": kind, "w": rev[ok] / rev[ok].sum(),
                                 **{f"e_{m}": v for m, v in e.items()}}))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args(argv)
    RA.TUNED, RA.COMPARISON = True, False
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = RA.index(results)
    idx = idx[(idx["set"] == "rolling") & idx["model"].isin(["world_model", "lightgbm"])]
    idx.to_csv(out / "runs.csv", index=False)
    summary, shares = [], []
    rng = np.random.default_rng(0)
    for key, rows in idx.groupby("key"):
        if set(rows["model"]) != {"world_model", "lightgbm"}:
            continue
        fr, fcs = RA.frames_and_forecasts(results, rows)
        has_price = not key.startswith("favorita")
        parts = []
        for o in sorted(set(fcs["world_model"]["origin"]) & set(fcs["lightgbm"]["origin"])):
            mats = {m: C.forecast_matrix(f, fr, o) for m, f in fcs.items()}
            mats["average"] = (mats["world_model"] + mats["lightgbm"]) / 2
            series, periods = kinds(fr, o, has_price)
            for k, m in series.items():
                shares.append(dict(key=key, origin=o, kind=k, share=float(m.mean())))
            for k, m in periods.items():
                shares.append(dict(key=key, origin=o, kind=k, share=float(m.mean())))
            parts += rows_for(fr, o, mats, series, periods)
        df = pd.concat(parts)
        for kind, g in df.groupby("kind"):
            w = g["w"].to_numpy() / g["origin"].nunique()
            draws = rng.multinomial(len(w), np.full(len(w), 1 / len(w)), size=a.boot)
            base = g["e_lightgbm"].to_numpy()
            for m in MODELS:
                e = g[f"e_{m}"].to_numpy()
                diff = (draws * (w * (e - base))).sum(axis=1) / (draws * w).sum(axis=1)
                summary.append(dict(key=key, kind=kind, model=m, rmsse=float((w * e).sum() / w.sum()),
                                    diff_to_lgb=float((w * (e - base)).sum() / w.sum()),
                                    lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)),
                                    series_windows=len(g)))
        print(key, "done", flush=True)
        pd.DataFrame(summary).to_csv(out / "kinds.csv", index=False)
        pd.DataFrame(shares).to_csv(out / "kind_shares.csv", index=False)


if __name__ == "__main__":
    main()
