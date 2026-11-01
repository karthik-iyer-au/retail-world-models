"""Split each week's category-total error into the part that comes from single-item errors and the part
that comes from items being wrong in the same direction at once (covariance of errors across series)."""
import sys, json, numpy as np, pandas as pd
out = {}
for cat in sys.argv[1:]:
    w = pd.read_csv(f"fc/world_model_{cat}.csv"); l = pd.read_csv(f"fc/lightgbm_{cat}.csv")
    w["lg"] = l["forecast"].to_numpy()
    w = w[w.origin < w.origin.max()]
    w["e_wm"] = w.forecast - w.units; w["e_lg"] = w.lg - w.units
    g = w.groupby(["origin", "date"])
    res = {}
    for m in ("wm", "lg"):
        tot_sq = g[f"e_{m}"].sum() ** 2            # category-total squared error per week
        own = g[f"e_{m}"].apply(lambda e: (e ** 2).sum())  # what it would be if item errors were independent
        res[m] = dict(total_rmse=float(np.sqrt(tot_sq.mean())), independent_rmse=float(np.sqrt(own.mean())),
                      shared_share=float(1 - own.sum() / tot_sq.sum()),
                      item_rmse=float(np.sqrt((w[f"e_{m}"] ** 2).mean())),
                      mean_week_bias=float(g[f"e_{m}"].sum().abs().mean() / g["units"].sum().mean()))
    out[cat] = res
    print(cat, json.dumps(res), flush=True)
json.dump(out, open("corr.json", "w"))
