"""Dominick's accuracy by merchant grain: brand-week, item-week (fast vs slow sellers), category event weeks."""
import os, sys, json, numpy as np, pandas as pd
_rc = pd.read_csv
pd.read_csv = lambda *a, **k: _rc(*a, **{**k, "low_memory": False})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); os.chdir(os.environ.get("EDA_DIR", "eda"))
from rwm.data.dominicks import load_dominicks
from rwm.evaluation.hierarchy import series_scale
out = {}
for cat in sys.argv[1:]:
    p = load_dominicks([cat], root="raw/dominicks").panel
    p["sid"] = p["series_id"].astype(str); p["date"] = pd.to_datetime(p["date"])
    upc = _rc(f"raw/dominicks/upc{cat}.csv", encoding="latin-1").drop_duplicates("UPC")
    brand = upc.set_index(upc["UPC"].map(lambda u: f"{int(u):012d}"))["DESCRIP"].astype(str).str.split().str[0]
    w = pd.read_csv(f"fc/world_model_{cat}.csv"); l = pd.read_csv(f"fc/lightgbm_{cat}.csv")
    w["lg"] = l["forecast"].to_numpy(); w["date"] = pd.to_datetime(w["date"])
    res = []
    for o in sorted(w.origin.unique())[:-1]:
        fo = w[w.origin == o].copy(); t0 = fo.date.min()
        fo["item"] = fo.series_id.str[:12]; fo["brand"] = fo["item"].map(brand).fillna("?")
        h = p[(p.date < t0) & (p.sid.isin(fo.series_id.unique()))].copy()
        h["item"] = h.sid.str[:12]; h["brand"] = h["item"].map(brand).fillna("?")
        h["rev"] = h.units * h.price.fillna(0)
        last = h[h.date >= h.date.max() - pd.Timedelta(weeks=12)]
        def grain(key):
            H = h.pivot_table(index=key, columns="date", values="units", aggfunc="sum", fill_value=0)
            F = fo.groupby([key, "date"])[["units", "forecast", "lg"]].sum()
            A, W, L = (F[c].unstack(fill_value=0).reindex(H.index, fill_value=0) for c in ("units", "forecast", "lg"))
            sc = pd.Series(series_scale(H.to_numpy().astype(float)), H.index)
            rev = last.groupby(key).rev.sum().reindex(H.index).fillna(0)
            sw = np.sqrt(((A - W) ** 2).mean(1) / sc); sl = np.sqrt(((A - L) ** 2).mean(1) / sc)
            return pd.DataFrame({"wm": sw, "lg": sl, "rev": rev}).dropna()
        b = grain("brand"); b = b[b.rev > 0]
        res.append(dict(origin=int(o), grain="brand-week", wm=float((b.wm * b.rev).sum() / b.rev.sum()), lg=float((b.lg * b.rev).sum() / b.rev.sum()), n=len(b)))
        it = grain("item"); it = it[it.rev > 0]
        rank = it.rev.rank(ascending=False, pct=True)
        for name, m in {"item-week, top 20% sellers": rank <= .2, "item-week, bottom 50%": rank > .5}.items():
            s = it[m]
            res.append(dict(origin=int(o), grain=name, wm=float((s.wm * s.rev).sum() / s.rev.sum()), lg=float((s.lg * s.rev).sum() / s.rev.sum()),
                            n=int(m.sum()), rev_share=float(s.rev.sum() / it.rev.sum())))
        tot = fo.groupby("date")[["units", "forecast", "lg"]].sum()
        med = h.groupby("date").units.sum().iloc[-52:].median()
        ev = tot[tot.units > 1.3 * med]
        for d, r in ev.iterrows():
            res.append(dict(origin=int(o), grain="category event week", date=str(d.date()), size=float(r.units / med),
                            wm=float(r.forecast / r.units - 1), lg=float(r.lg / r.units - 1)))
    out[cat] = res
    print(cat, "done", flush=True)
json.dump(out, open(f"grains_{'_'.join(sys.argv[1:])}.json", "w"))
