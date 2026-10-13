"""Paper 2 audit runs (docs/PAPER2_PLAN.md). Reads models; never tunes them.

  # C1 to C5 on every saved world-model or lift-model checkpoint in a results folder
  python scripts/audit_runs.py checkpoints --results <folder> [--pattern GLOB ...]

  # LightGBM refitted on a run config's windows, with on/off contributions and TreeSHAP (C6)
  python scripts/audit_runs.py lightgbm --config configs/experiments/<config>.yaml --results <folder>

  # C7: models fitted up to each zone-price episode and scored on it
  python scripts/audit_runs.py zones --episodes results/paper2/zone_episodes.csv \
      --world-config <yaml> --lightgbm-config <yaml> --results <folder>

Each writes `audit/` inside the run folder it reads (checkpoints) or a new
folder `paper2_<what>/` under --results, with one JSON record per window and
the per-row lifts as CSV, so seed stability (C3) and agreement (C6) can be
computed afterwards by `scripts/audit_summary.py`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rwm.audit import (construction_checks, lever_contributions, lift_vs_actual, price_monotonicity,
                       randomised_copies, rank_agreement, regular_prices)
from rwm.audit.runs import load_checkpoint, model_params, plan_levers, windows
from rwm.data.schema import DATE, PRICE, SERIES, STORE, UNITS
from rwm.forecaster import build_model

KEEP = ["state_model", "seed_average"]


def audit_model(model, train, test, cfg, randomise=True) -> tuple[dict, pd.DataFrame]:
    p = model_params(cfg)
    levers = [c for c in plan_levers(cfg) if c in test.columns]
    regular = regular_prices(train, p.get("regular_window", 12))
    t0 = time.time()
    parts = lever_contributions(model, test, regular, levers)
    rec = {"levers": levers, "rows": len(test)}
    if hasattr(model, "breakdown"):
        rec["construction"] = construction_checks(model, test, regular, levers)
    rec["monotonicity"] = price_monotonicity(model, test)
    on = np.zeros(len(test), dtype=bool)
    for c in levers:
        on |= test[c].fillna(0).to_numpy() > 0
    price_cut = (test[PRICE].to_numpy() < 0.95 * test[SERIES].map(regular).to_numpy())
    rec["against_sales"] = {
        "any_plan": lift_vs_actual(parts["forecast"], parts["nothing_planned"], test[UNITS], on | price_cut),
        "price_cut_only": lift_vs_actual(parts["forecast"], parts["nothing_planned"], test[UNITS], price_cut & ~on),
        **{c: lift_vs_actual(parts["forecast"], parts["nothing_planned"], test[UNITS],
                             test[c].fillna(0).to_numpy() > 0) for c in levers},
        "nothing_planned": lift_vs_actual(parts["forecast"], parts["nothing_planned"], test[UNITS], ~on & ~price_cut),
    }
    rec["share_of_lift_in_interaction"] = float(parts["interaction"].abs().sum() / max(parts["total_lift"].abs().sum(), 1e-9))
    if randomise and hasattr(model, "_net"):
        rec["randomisation"] = {}
        for name, m in randomised_copies(model, seed=0):
            r = lever_contributions(m, test, regular, levers)
            rec["randomisation"][name] = rank_agreement(parts["total_lift"], r["total_lift"])
    rec["seconds"] = round(time.time() - t0, 1)
    out = pd.concat([test[[SERIES, DATE, UNITS]].reset_index(drop=True), parts.reset_index(drop=True)], axis=1)
    return rec, out


def cmd_checkpoints(a):
    results = Path(a.results)
    folders = sorted({d for pat in a.pattern for d in results.glob(pat) if (d / "manifest.json").exists()})
    for d in folders:
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        if cfg["model"]["name"] not in KEEP or not list(d.glob("checkpoint_origin*.pt")):
            continue
        if not model_params(cfg).get("lift_readout"):
            continue
        out = d / "audit"
        if (out / "done").exists():
            continue
        out.mkdir(exist_ok=True)
        print(f"== {d.name}", flush=True)
        for origin, train, test in windows(cfg):
            ck = d / f"checkpoint_origin{origin}.pt"
            if not ck.exists():
                continue
            model = load_checkpoint(d, origin)
            if cfg["model"]["name"] == "seed_average":
                rec, lifts = audit_model(model, train, test, cfg, randomise=False)
                # randomisation on the first member: averaging would hide a member that ignores its weights
                regular = regular_prices(train, model_params(cfg).get("regular_window", 12))
                base = lever_contributions(model.models[0], test, regular, rec["levers"])["total_lift"]
                rec["randomisation"] = {
                    name: rank_agreement(base, lever_contributions(m, test, regular, rec["levers"])["total_lift"])
                    for name, m in randomised_copies(model.models[0], seed=0)
                }
            else:
                rec, lifts = audit_model(model, train, test, cfg)
            rec.update({"run": d.name, "config": cfg["name"], "seed": cfg.get("seed"), "origin": origin})
            (out / f"audit_origin{origin}.json").write_text(json.dumps(rec, indent=1, default=float))
            lifts.to_csv(out / f"lifts_origin{origin}.csv.gz", index=False)
            print(json.dumps({k: rec[k] for k in ("origin", "seconds")}), flush=True)
            del model
        (out / "done").write_text("")


def cmd_lightgbm(a):
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.results) / f"paper2_lightgbm_{cfg['name']}"
    out.mkdir(parents=True, exist_ok=True)
    for origin, train, test in windows(cfg):
        if (out / f"audit_origin{origin}.json").exists():
            continue
        model = build_model(cfg["model"]["name"], **cfg["model"]["params"]).fit(train)
        rec, lifts = audit_model(model, train, test, cfg, randomise=False)
        shap = model.shap(test.drop(columns=[UNITS]))
        price_cols = [c for c in shap.columns if c.startswith("price")]
        lifts["shap_price"] = shap[price_cols].sum(axis=1).to_numpy()
        for c in rec["levers"]:
            lifts[f"shap_{c}"] = shap[c].to_numpy()
        rec.update({"config": cfg["name"], "origin": origin})
        (out / f"audit_origin{origin}.json").write_text(json.dumps(rec, indent=1, default=float))
        lifts.to_csv(out / f"lifts_origin{origin}.csv.gz", index=False)
        print(f"{cfg['name']} origin {origin}: {rec['seconds']} s", flush=True)


def cmd_zones(a):
    from rwm.audit.zones import SCORED_WEEKS, Episode, did
    from rwm.data import load_dataset
    from rwm.data.dominicks import WEEK_1_START

    eps = pd.read_csv(a.episodes)
    eps = eps[eps["scored"]]
    wcfg, lcfg = yaml.safe_load(open(a.world_config)), yaml.safe_load(open(a.lightgbm_config))
    out = Path(a.results) / "paper2_zones"
    out.mkdir(parents=True, exist_ok=True)
    week_date = lambda w: WEEK_1_START + pd.Timedelta(days=7 * (int(w) - 1))
    for cat, ce in eps.groupby("category"):
        if a.categories and cat not in a.categories:
            continue
        panel = load_dataset("dominicks", categories=[cat]).panel
        units = pd.DataFrame({"store": panel[STORE].astype(str).astype(int),
                              "week": ((panel[DATE] - WEEK_1_START).dt.days // 7 + 1).astype(int),
                              "units": panel[UNITS]})
        # one origin per cluster of episode starts: the earliest start, taking the
        # episodes that start within the next three weeks
        starts, clusters = sorted(ce["first_week"].unique()), []
        for s in starts:
            if not clusters or s > clusters[-1][0] + 3:
                clusters.append([s])
            else:
                clusters[-1].append(s)
        for cl in clusters:
            origin = int(cl[0])
            tag = f"{cat}_week{origin}"
            if (out / f"{tag}.json").exists():
                continue
            train = panel[panel[DATE] < week_date(origin)]
            test = panel[(panel[DATE] >= week_date(origin)) & (panel[DATE] < week_date(origin + SCORED_WEEKS))]
            test = test.reset_index(drop=True)
            fitted = {}
            for label, cfg in (("world", wcfg), ("lightgbm", lcfg)):
                t0 = time.time()
                fitted[label] = build_model(cfg["model"]["name"], **cfg["model"]["params"]).fit(train)
                print(f"{tag} {label} fitted in {time.time() - t0:.0f} s", flush=True)
            recs = []
            plan = test.drop(columns=[UNITS])
            base = {k: m.predict(plan) for k, m in fitted.items()}
            for _, e in ce[ce["first_week"].isin(cl)].iterrows():
                last = origin + SCORED_WEEKS - 1
                ep = Episode(int(e["zone"]), int(e["store"]), int(e["first_week"]), int(e["last_week"]),
                             float(e["price_log_ratio"]), [int(x) for x in str(e["controls"]).split()])
                truth = did(units, ep, last_week=last)
                mine = (plan[STORE].astype(str).astype(int) == ep.store) & (plan[DATE] >= week_date(ep.first_week))
                cf = plan.copy()
                cf.loc[mine, PRICE] = cf.loc[mine, PRICE] / np.exp(ep.log_ratio)
                rec = {"category": cat, "origin_week": origin, "store": ep.store, "zone": ep.zone,
                       "first_week": ep.first_week, "scored_last_week": last, "price_log_ratio": ep.log_ratio,
                       "did": truth}
                for k, m in fitted.items():
                    alt = m.predict(cf)
                    rec[f"{k}_implied"] = float(np.log(base[k][mine].sum() / max(alt[mine].sum(), 1e-9)))
                recs.append(rec)
                show = {k: rec[k] for k in ("store", "first_week", "world_implied", "lightgbm_implied")}
                print(json.dumps({**show, "did": round(truth["estimate"], 3)}, default=float), flush=True)
            (out / f"{tag}.json").write_text(json.dumps(recs, indent=1, default=float))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkpoints")
    c.add_argument("--results", required=True)
    c.add_argument("--pattern", nargs="*", default=["*"])
    l = sub.add_parser("lightgbm")
    l.add_argument("--config", required=True)
    l.add_argument("--results", required=True)
    z = sub.add_parser("zones")
    z.add_argument("--episodes", required=True)
    z.add_argument("--world-config", required=True)
    z.add_argument("--lightgbm-config", required=True)
    z.add_argument("--results", required=True)
    z.add_argument("--categories", nargs="*")
    a = p.parse_args(argv)
    {"checkpoints": cmd_checkpoints, "lightgbm": cmd_lightgbm, "zones": cmd_zones}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
