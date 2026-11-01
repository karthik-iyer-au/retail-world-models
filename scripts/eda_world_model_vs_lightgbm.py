"""Error breakdown, 1 November 2026: where does the averaged world model lose to LightGBM on Dominick's? Search windows (dev origins only)."""
import os, sys, json
import numpy as np, pandas as pd
_rc=pd.read_csv
def _r(*a,**k):
    k.setdefault("low_memory",False); return _rc(*a,**k)
pd.read_csv=_r
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwm.data.dominicks import load_dominicks
from rwm.evaluation.hierarchy import series_scale

import os
S = os.environ.get("EDA_DIR", "eda")  # holds fc/{world_model,lightgbm}_<cat>.csv (start_s1 forecasts of the avg5 loops) and raw/dominicks
LEVELS = [[], ["store"], ["item"], ["item", "store"]]  # one category: cat levels equal total/store
out = {}
for cat in sys.argv[1:]:
    ds = load_dominicks([cat], root=f"{S}/raw/dominicks")
    p = ds.panel
    p["series_id"] = p["series_id"].astype(str) if "series_id" in p else p.iloc[:, 0].astype(str)
    sid = [c for c in p.columns if "series" in c][0]
    p["sid"] = p[sid].astype(str); p["store"] = p["sid"].str[-3:]; p["item"] = p["sid"].str[:12]
    wm = pd.read_csv(f"{S}/fc/world_model_{cat}.csv", parse_dates=["date"])
    lg = pd.read_csv(f"{S}/fc/lightgbm_{cat}.csv", parse_dates=["date"])
    f = wm.merge(lg[["series_id", "date", "origin", "forecast"]], on=["series_id", "date", "origin"], suffixes=("_wm", "_lg"))
    dev = sorted(f["origin"].unique())[:-1]
    f = f[f["origin"].isin(dev)]
    p["date"] = pd.to_datetime(p["date"])
    rows = []
    for o in dev:
        fo = f[f["origin"] == o]
        t0 = fo["date"].min()
        names = pd.Index(sorted(fo["series_id"].unique()))
        hist = p[(p["date"] < t0) & p["sid"].isin(names)]
        dates = sorted(hist["date"].unique())
        H = np.zeros((len(names), len(dates))); ri = names.get_indexer(hist["sid"]); ci = pd.Index(dates).get_indexer(hist["date"])
        H[ri, ci] = hist["units"].to_numpy()
        last = hist[hist["date"] >= dates[-13]]
        rev = (last["units"] * last["price"].fillna(0)).groupby(last["sid"]).sum().reindex(names).fillna(0).to_numpy()
        tdates = sorted(fo["date"].unique())
        def mat(col):
            M = np.zeros((len(names), len(tdates))); M[names.get_indexer(fo["series_id"]), pd.Index(tdates).get_indexer(fo["date"])] = fo[col].to_numpy(); return M
        A, W, L = mat("units"), mat("forecast_wm"), mat("forecast_lg")
        promo = p[(p["date"] >= t0) & p["sid"].isin(names)]
        P = np.zeros_like(A); P[names.get_indexer(promo["sid"]), pd.Index(tdates).get_indexer(promo["date"])] = promo["promo"].to_numpy()
        attrs = pd.DataFrame({"store": names.str[-3:], "item": names.str[:12]})
        lv = {}
        for cols in LEVELS:
            codes = attrs.groupby(cols).ngroup().to_numpy() if cols else np.zeros(len(names), int)
            g = codes.max() + 1
            agg = lambda X: np.array([X[codes == k].sum(0) for k in range(g)])
            h, a, w_, l_ = agg(H), agg(A), agg(W), agg(L)
            wt = np.array([rev[codes == k].sum() for k in range(g)]); wt = wt / wt.sum()
            sc = series_scale(h)
            s_w = np.sqrt(((a - w_) ** 2).mean(1) / sc); s_l = np.sqrt(((a - l_) ** 2).mean(1) / sc)
            ok = np.isfinite(s_w) & np.isfinite(s_l) & (wt > 0)
            lv["x".join(cols) or "total"] = ((s_w[ok] * wt[ok]).sum(), (s_l[ok] * wt[ok]).sum(), a.sum(), w_.sum(), l_.sum())
        # bottom-level splits: weighted scaled squared error by segment
        sc = series_scale(H); ok = np.isfinite(sc) & (rev > 0)
        zero_share = (H[:, -52:] == 0).mean(1)
        vol = H[:, -13:].mean(1)
        segs = {
            "promo weeks": P > 0, "non-promo weeks": P == 0,
            "after promo (1-2 wk)": (np.roll(P, 1, 1) + np.roll(P, 2, 1) > 0) & (P == 0),
        }
        for name, m in segs.items():
            wse = lambda F: np.nansum(rev[ok, None] * ((A - F)[ok] ** 2) / sc[ok, None] * m[ok]) / rev[ok].sum()
            rows.append(dict(origin=o, seg=name, wm=wse(W), lg=wse(L), share=m[ok].mean()))
        for name, m in {"intermittent (>50% zero wks)": zero_share > .5, "steady": zero_share <= .5,
                        "top-volume fifth": vol >= np.quantile(vol, .8), "rest": vol < np.quantile(vol, .8)}.items():
            mm = ok & m
            wse = lambda F: np.nansum(rev[mm, None] * ((A - F)[mm] ** 2) / sc[mm, None]) / rev[ok].sum()
            rows.append(dict(origin=o, seg=name, wm=wse(W), lg=wse(L), share=mm.sum() / ok.sum()))
        for step in range(A.shape[1]):
            wse = lambda F: np.nansum(rev[ok] * ((A - F)[ok, step] ** 2) / sc[ok]) / rev[ok].sum()
            rows.append(dict(origin=o, seg=f"week {step+1}", wm=wse(W), lg=wse(L), share=1))
        rows.append(dict(origin=o, seg="_levels", **{k: v for k, v in lv.items()}))
        # data profile
        rows.append(dict(origin=o, seg="_profile", series=len(names), items=attrs.item.nunique(), stores=attrs.store.nunique(),
                         zero_share=float((H[:, -52:] == 0).mean()), promo_share=float(P.mean()),
                         mean_units=float(H[:, -13:].mean()), new_series=float((np.cumsum(H != 0, 1)[:, -52] == 0).mean()),
                         cv=float(np.nanmedian(H[:, -52:].std(1) / np.maximum(H[:, -52:].mean(1), 1e-9)))))
    out[cat] = rows
    print(cat, "done", flush=True)
json.dump(out, open(f"{S}/eda_{'_'.join(sys.argv[1:])}.json", "w"), default=float)
