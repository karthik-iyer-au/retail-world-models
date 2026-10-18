"""Every training-loss change tried on the world model and the lift model, scored
against the configuration it changed (paper 1, loss section).

  python scripts/loss_ledger.py

A loss change is a configuration whose loss settings differ from the defaults
(Tweedie with power 1.5 on sales relative to the item's level, no extra terms).
Its reference is the configuration with the same data, windows and every other
setting, and default loss settings. Scores are each run's overall WRMSSE (mean of
the twelve hierarchy levels) on the development windows, from the audit of the
saved runs (results/paper2/summary/versions.csv), averaged over the seeds and
windows that both configurations ran. The loss weight after a promotion (study
E) is taken from results/paper1_revision/tables/afterpromo_stage1.csv, where the
world model was rerun in the same job. A change whose runs are not in the records
is listed as not run on full data.

Writes results/paper1_revision/tables/loss_ledger.csv.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"likelihood": "tweedie", "tweedie_power": 1.5, "latent_weight": 0.0, "pretrain_steps": 0,
            "total_weight": 0.0, "volume_weight": 0.0, "peak_weight": 0.0, "after_promo_weight": 0.0,
            "state_spread_weight": 0.0}
# settings that only qualify a loss term (they mean nothing without it)
QUALIFIERS = {"split_weight", "rollout_discount", "volume_by", "finetune", "latent_ema", "encoder_lr_scale"}


def params(c: dict) -> dict:
    m = c["model"]
    p = m.get("params", {})
    p = p.get("params", p) if m["name"] == "seed_average" else p
    return {k: v for k, v in p.items() if k != "seed"}


def loss_keys(p: dict) -> dict:
    return {k: v for k, v in p.items() if k in DEFAULTS and v != DEFAULTS[k]} | \
           {k: v for k, v in p.items() if k in QUALIFIERS and v is not None}


def change_name(k: dict) -> str:
    if k.get("likelihood") == "negative_binomial":
        return "Negative binomial likelihood"
    if "after_promo_weight" in k:
        return f"Weight {1 + k['after_promo_weight']:g} on the periods after a promotion"
    if "peak_weight" in k:
        return "Peak weight, dollar weight and Tweedie power 1.1"
    if "volume_weight" in k:
        by = "dollar" if k.get("volume_by") == "dollars" else "unit"
        return f"{by.capitalize()}-volume weight and Tweedie power 1.1"
    if "pretrain_steps" in k and "total_weight" in k:
        return "Latent pretraining, then forecast with a totals term"
    if "pretrain_steps" in k and "latent_weight" in k:
        return "Latent pretraining, then forecast with the latent term"
    if "pretrain_steps" in k:
        return f"Latent pretraining, encoder {k.get('finetune', 'full').replace('_', ' ')} afterwards"
    if "state_spread_weight" in k:
        return "Latent term with a spread penalty and discount"
    if "latent_weight" in k:
        return f"Latent term, weight {k['latent_weight']:g}"
    if "total_weight" in k:
        return "Totals term"
    return ", ".join(f"{a}={b}" for a, b in k.items())


def main():
    cfg = {}
    for f in glob.glob(str(ROOT / "configs" / "experiments" / "*.yaml")):
        c = yaml.safe_load(open(f))
        if c.get("model", {}).get("name") not in ("state_model", "seed_average"):
            continue
        cfg[c["name"]] = (c["dataset"], c["evaluation"], params(c))
    v = pd.read_csv(ROOT / "results" / "paper2" / "summary" / "versions.csv")
    v["base"] = v["config"].str.replace(r"_seed\d+$", "", regex=True)
    strip = lambda p: {k: x for k, x in p.items() if k not in DEFAULTS and k not in QUALIFIERS}  # noqa: E731
    rows = []
    for name, (d, e, p) in sorted(cfg.items()):
        keys = loss_keys(p)
        if not keys or "after_promo_weight" in keys:
            continue
        refs = [m for m, (d2, e2, p2) in cfg.items()
                if m != name and d2 == d and e2 == e and strip(p2) == strip(p) and not loss_keys(p2)
                and m in set(v["base"])]
        row = {"change": change_name(keys), "config": name, "dataset": d["name"],
               "category": (d.get("params", {}).get("category") or d.get("params", {}).get("code") or ""),
               "settings": "; ".join(f"{a}={b}" for a, b in keys.items())}
        a = v[v["base"] == name]
        if a.empty or not refs:
            rows.append({**row, "status": "not run on full data" if a.empty else "no matched reference"})
            continue
        b = v[v["base"] == refs[0]]
        m = a.merge(b, on=["seed", "origin"], suffixes=("", "_ref"))
        if m.empty:
            rows.append({**row, "reference": refs[0], "status": "no shared seed and window"})
            continue
        rows.append({**row, "reference": refs[0], "runs": len(m), "wrmsse": m["wrmsse"].mean(),
                     "wrmsse_ref": m["wrmsse_ref"].mean(), "status": "scored"})
    ap = pd.read_csv(ROOT / "results" / "paper1_revision" / "tables" / "afterpromo_stage1.csv")
    for w in ("afterpromo_w2", "afterpromo_w5"):
        for cat, g in ap.groupby("category"):
            x, r = g[g["variant"] == w], g[g["variant"] == "world_model"]
            if len(x) and len(r):
                rows.append({"change": change_name({"after_promo_weight": int(w[-1])}),
                             "config": f"dominicks_{cat}_dev_{w}", "dataset": "dominicks", "category": cat,
                             "settings": f"after_promo_weight={w[-1]}",
                             "reference": f"world model, same job", "runs": 1,
                             "wrmsse": float(x["overall"].iloc[0]), "wrmsse_ref": float(r["overall"].iloc[0]),
                             "status": "scored"})
    out = pd.DataFrame(rows)
    out["diff"] = out["wrmsse"] - out["wrmsse_ref"]
    out.to_csv(ROOT / "results" / "paper1_revision" / "tables" / "loss_ledger.csv", index=False, float_format="%.5g")
    print(out.groupby(["change", "status"]).size())


if __name__ == "__main__":
    main()
