"""Every one-setting change to the world model and lift model that has saved runs,
scored against the configuration it changed (paper 1, what was tried).

  python scripts/change_ledger.py

Two configurations form a pair when they share the data set, the evaluation and
every model setting but one change (a change may move two settings that only work
together, such as the level window and how it is grouped). Scores are each run's
overall WRMSSE on its windows from the audit of the saved runs
(results/paper2/summary/versions.csv), averaged over the seeds and windows both
configurations ran. Configurations without saved runs are not in that file and
are not scored here; results/README.md records them.

Writes results/paper1_revision/tables/change_ledger.csv (one row per pair) and
change_ledger_summary.csv (one row per change).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# change name -> (settings it may move, the setting that says which side has it)
CHANGES = {
    "Roll-forward through the plan (world model against lift model)": ({"rollout"}, "rollout"),
    "Normalised roll-forward": ({"roll_norm"}, "roll_norm"),
    "Own forecast fed into the roll-forward": ({"feedback"}, "feedback"),
    "Brand switching": ({"switching"}, "switching"),
    "Averaging three fits": ({"seed_average"}, "seed_average"),
    "Longer item level window (8 against 4 periods)": ({"scale_window"}, "scale_window"),
    "Level window set by item role": ({"scale_window", "scale_window_by"}, "scale_window_by"),
    "Recent and longer level together": ({"scale_window", "level_views"}, "level_views"),
    "Past-year level": ({"long_level"}, "long_level"),
    "Year-ago sales as an input": ({"year_ago"}, "year_ago"),
    "Plan of earlier periods as inputs": ({"plan_lags"}, "plan_lags"),
    "Plan split into regular and promoted parts": ({"plan_split"}, "plan_split"),
    "Weekday as seven inputs": ({"weekday_flags"}, "weekday_flags"),
    "Event kinds as inputs": ({"extra", "weekday_flags"}, "extra"),
    "More training steps": ({"steps"}, "steps"),
    "Regular price held for more periods": ({"regular_hold"}, "regular_hold"),
    "Readout dropout and weight decay": ({"readout_dropout", "weight_decay"}, "readout_dropout"),
    "Recorded base price as the regular price": ({"levers", "regular_column"}, "regular_column"),
    "Unknown-item labels in training": ({"unknown_label_rate"}, "unknown_label_rate"),
    "New-item start from similar items": ({"cold_start", "cold_neighbours"}, "cold_start"),
    "Longer history and training period": ({"history", "train_periods"}, "history"),
}


def load() -> dict:
    from loss_ledger import params  # noqa: E402  (same folder)
    out = {}
    for f in glob.glob(str(ROOT / "configs" / "experiments" / "*.yaml")):
        c = yaml.safe_load(open(f))
        m = c.get("model", {}).get("name")
        if m not in ("state_model", "seed_average"):
            continue
        p = params(c)
        p["seed_average"] = m == "seed_average"
        out[c["name"]] = (json.dumps(c["dataset"], sort_keys=True), json.dumps(c["evaluation"], sort_keys=True),
                          p, c["dataset"]["name"])
    return out


def has(v) -> bool:
    return v not in (None, False, 0, [], "")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    cfg = load()
    v = pd.read_csv(ROOT / "results" / "paper2" / "summary" / "versions.csv")
    v["base"] = v["config"].str.replace(r"_seed\d+$", "", regex=True)
    scored = sorted(set(v["base"]) & set(cfg))
    rows = []
    for i, a in enumerate(scored):
        for b in scored[i + 1:]:
            da, ea, pa, ds = cfg[a]
            db, eb, pb, _ = cfg[b]
            if da != db or ea != eb:
                continue
            moved = {k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
            for name, (allowed, key) in CHANGES.items():
                if not moved or not moved <= allowed or key not in moved:
                    continue
                if key in ("scale_window", "steps", "history"):
                    on, off = (a, b) if (pa.get(key) or 0) > (pb.get(key) or 0) else (b, a)
                else:
                    on, off = (a, b) if has(pa.get(key)) else (b, a)
                m = v[v["base"] == on].merge(v[v["base"] == off], on=["seed", "origin"], suffixes=("", "_ref"))
                if m.empty:
                    continue
                rows.append({"change": name, "with": on, "without": off, "dataset": ds,
                             "setting": f"{key}={cfg[on][2].get(key)} against {cfg[off][2].get(key)}",
                             "runs": len(m), "wrmsse": m["wrmsse"].mean(), "wrmsse_ref": m["wrmsse_ref"].mean()})
                break
    d = pd.DataFrame(rows)
    # configurations copied under another name share their runs; count each comparison once
    d = d.drop_duplicates(["change", "dataset", "wrmsse", "wrmsse_ref"])
    d["diff"] = d["wrmsse"] - d["wrmsse_ref"]
    tab = ROOT / "results" / "paper1_revision" / "tables"
    d.to_csv(tab / "change_ledger.csv", index=False, float_format="%.5g")
    s = (d.groupby("change").agg(pairs=("diff", "size"), lower=("diff", lambda x: int((x < 0).sum())),
                                 mean_diff=("diff", "mean"), datasets=("dataset", lambda x: ", ".join(sorted(set(x)))))
         .reset_index().sort_values("pairs", ascending=False))
    s.to_csv(tab / "change_ledger_summary.csv", index=False, float_format="%.4g")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
