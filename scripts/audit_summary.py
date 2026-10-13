"""Gather the paper 2 audit records into small tables to commit.

  python scripts/audit_summary.py --results <folder> --out <folder>/paper2_summary

Writes:
  versions.csv    one row per audited run and window: accuracy from its
                  metrics.json and every check from its audit record
  stability.csv   C3: for each configuration and window with two or more seeds,
                  rank agreement of item-week lifts between seeds, and of
                  forecasts, averaged over seed pairs
  agreement.csv   C6: rank agreement of each lever's contribution between a
                  model's audit and LightGBM's (on/off and TreeSHAP), same window
  zones.csv       C7: one row per episode scored
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.audit import rank_agreement


def base_config(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def accuracy(run_dir: Path, origin: int) -> dict:
    m = json.loads((run_dir / "metrics.json").read_text())
    sp = next((s for s in m["splits"] if s["origin"] == origin), None)
    if sp is None:
        return {}
    lv = {b["level"]: b["wrmsse"] for b in sp["by_level"]}
    out = {"wrmsse": sp["wrmsse"], "total": lv.get("total"), "item_by_store": lv.get("item_idxstore_id"),
           "forecast_to_actual": sp.get("forecast_to_actual")}
    for c, v in sp.get("by_condition", {}).items():
        out[f"{c}_on"] = v["on"]["wrmsse"]
    if "peaks" in sp:
        out["promoted_fa"] = sp["peaks"]["promoted"]["forecast_to_actual"]
    return out


def flat(rec: dict) -> dict:
    out = {k: rec[k] for k in ("run", "config", "seed", "origin", "rows") if k in rec}
    out["dataset"] = rec.get("config", "").split("_")[0]
    out.update({f"c1_{k}": v for k, v in rec.get("construction", {}).items()})
    out.update({f"c2_{k.replace(' ', '_')}": v for k, v in rec.get("randomisation", {}).items()})
    out["c4_share_lower_after_deeper_cut"] = rec["monotonicity"]["share_lower_after_deeper_cut"]
    for k, v in rec["against_sales"].items():
        out[f"c5_{k}_bias"], out[f"c5_{k}_rank"], out[f"c5_{k}_n"] = v["bias"], v["rank"], v["n"]
    out["interaction_share"] = rec.get("share_of_lift_in_interaction")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, lifts = [], {}
    for f in sorted(res.glob("*/audit/audit_origin*.json")):
        rec = json.loads(f.read_text())
        run_dir = f.parent.parent
        rows.append({**flat(rec), **accuracy(run_dir, rec["origin"])})
        lifts[(base_config(rec["config"]), rec.get("seed"), rec["origin"])] = f.parent / f"lifts_origin{rec['origin']}.csv.gz"
    versions = pd.DataFrame(rows)
    versions.to_csv(out / "versions.csv", index=False)

    stab = []
    by_cfg = {}
    for (cfg, seed, origin), path in lifts.items():
        by_cfg.setdefault((cfg, origin), []).append((seed, path))
    for (cfg, origin), items in sorted(by_cfg.items()):
        if len(items) < 2:
            continue
        frames = {s: pd.read_csv(pth) for s, pth in items}
        lift_r, fc_r = [], []
        for (s1, d1), (s2, d2) in itertools.combinations(frames.items(), 2):
            j = d1.merge(d2, on=["series_id", "date"], suffixes=("_a", "_b"))
            lift_r.append(rank_agreement(j["total_lift_a"], j["total_lift_b"]))
            fc_r.append(rank_agreement(j["forecast_a"], j["forecast_b"]))
        stab.append({"config": cfg, "origin": origin, "seeds": len(items),
                     "lift_rank_agreement": float(np.nanmean(lift_r)), "forecast_rank_agreement": float(np.nanmean(fc_r))})
    pd.DataFrame(stab).to_csv(out / "stability.csv", index=False)

    agree = []
    for lg in sorted(res.glob("paper2_lightgbm_*/audit_origin*.json")):
        rec = json.loads(lg.read_text())
        lgb = pd.read_csv(lg.parent / f"lifts_origin{rec['origin']}.csv.gz")
        stem = rec["config"].replace("_lightgbm", "")  # for example dominicks_che_final or m5_test
        for (cfg, seed, origin), path in lifts.items():
            same_window = cfg.startswith(stem + "_") or cfg.startswith("paper2_refit_" + stem + "_")
            if origin != rec["origin"] or not same_window:
                continue
            mine = pd.read_csv(path)
            j = mine.merge(lgb, on=["series_id", "date"], suffixes=("", "_lgb"))
            row = {"model_config": cfg, "seed": seed, "lightgbm_config": rec["config"], "origin": origin, "rows": len(j)}
            for c in [c for c in mine.columns if c.startswith("lift_")]:
                lever = c[len("lift_"):]
                row[f"{lever}_onoff"] = rank_agreement(j[c], j[f"{c}_lgb"])
                if f"shap_{lever}" in j:
                    row[f"{lever}_shap"] = rank_agreement(j[c], j[f"shap_{lever}"])
            row["total_lift_onoff"] = rank_agreement(j["total_lift"], j["total_lift_lgb"])
            agree.append(row)
    pd.DataFrame(agree).to_csv(out / "agreement.csv", index=False)

    zones = []
    for f in sorted(res.glob("paper2_zones/*.json")):
        for r in json.loads(f.read_text()):
            d = r.pop("did")
            zones.append({**r, **{f"did_{k}": v for k, v in d.items()}})
    pd.DataFrame(zones).to_csv(out / "zones.csv", index=False)
    print(f"{len(versions)} audited windows, {len(stab)} stability rows, {len(agree)} agreement rows, "
          f"{len(zones)} zone episodes -> {out}")


if __name__ == "__main__":
    main()
