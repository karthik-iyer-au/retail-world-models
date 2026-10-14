"""Paper 1 revision: the small tables the paper reads, from the analysis CSVs
written by scripts/revision_analysis.py and scripts/mechanism_tests.py
(docs/PAPER1_REVISION_PLAN.md). The full CSVs stay with the run folders; these
tables are committed.

  python scripts/revision_tables.py --rev <results>/paper1_revision --out results/paper1_revision/tables

Reads every analysis folder under --rev (weekly, weekly_<key>, reconcile_<key>,
m5, reconcile_m5) and the mechanism CSVs, and writes:
  accuracy.csv      mean WRMSSE over windows by data set, window set, model and level, and the share of windows won
  paired.csv        A minus B by level: mean over windows with a 95% interval from a bootstrap over windows,
                    and the windows in which the item-bootstrap interval of that window excludes zero
  mcs.csv           the 90% model confidence set by data set and level (rolling windows)
  intervals.csv     conformal and native interval scores, mean over windows
  decisions.csv     decision costs summed over windows, and as a ratio to seasonal naive
  reconcile.csv     reconciliation by method and level, mean over windows
  mechanism.csv     plan permutation, persistence and state probe, mean over windows by data set and model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# reconciliation under the amendment of 14 October 2026 lives in reconcile_<key>;
# the reconciliation in the first Frat folder (weekly/) used the earlier rule and is not read
RECONCILE_FOLDERS = "reconcile_*"


def gather(rev: Path, part: str, folders: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(rev.glob(f"{folders}/{part}.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("key", "set", "origin", "model", "a", "b", "level", "method", "decision") if c in df]
    return df.drop_duplicates(subset=keys, keep="last")


def accuracy(sc: pd.DataFrame) -> pd.DataFrame:
    best = sc.groupby(["key", "set", "origin", "level"])["wrmsse"].transform("min")
    sc = sc.assign(won=np.isclose(sc["wrmsse"], best))
    out = sc.groupby(["key", "set", "model", "level"]).agg(
        wrmsse=("wrmsse", "mean"), sd=("wrmsse", "std"), windows=("wrmsse", "size"), wins=("won", "sum")).reset_index()
    return out


def paired(pr: pd.DataFrame, boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (key, wset, a, b, lv), g in pr.groupby(["key", "set", "a", "b", "level"]):
        d = g["diff"].to_numpy()
        n = len(d)
        if n > 1:
            means = d[rng.integers(0, n, size=(boot, n))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
        else:
            lo, hi = g["lo"].iloc[0], g["hi"].iloc[0]  # one window: its item-bootstrap interval
        rows.append({"key": key, "set": wset, "a": a, "b": b, "level": lv, "windows": n, "mean_diff": d.mean(),
                     "lo": lo, "hi": hi, "a_better_windows": int((g["hi"] < 0).sum()),
                     "b_better_windows": int((g["lo"] > 0).sum())})
    return pd.DataFrame(rows)


def intervals(iv: pd.DataFrame) -> pd.DataFrame:
    if iv.empty:
        return iv
    cols = [c for c in ("spl", "coverage_50", "coverage_80", "coverage_95", "native_spl") if c in iv]
    return iv.groupby(["key", "set", "model", "level"])[cols].mean().reset_index().assign(
        windows=iv.groupby(["key", "set", "model", "level"]).size().to_numpy())


def decisions(dc: pd.DataFrame) -> pd.DataFrame:
    if dc.empty:
        return dc
    tot = dc.groupby(["key", "set", "decision", "model"]).agg(cost=("cost", "sum"), windows=("origin", "nunique")).reset_index()
    sn = tot[tot["model"] == "seasonal_naive"][["key", "set", "decision", "cost"]].rename(columns={"cost": "naive_cost"})
    tot = tot.merge(sn, on=["key", "set", "decision"], how="left")
    tot["ratio_to_naive"] = tot["cost"] / tot["naive_cost"]
    return tot.drop(columns="naive_cost")


def reconcile(rc: pd.DataFrame) -> pd.DataFrame:
    if rc.empty:
        return rc
    return rc.groupby(["key", "set", "method", "level"]).agg(
        wrmsse=("wrmsse", "mean"), windows=("wrmsse", "size")).reset_index()


def mechanism(rev: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f).assign(source=f.stem) for f in sorted(rev.glob("mechanism_*.csv"))]
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    m["model"] = np.where(m["rollout"], "world_model", "lift")
    m["data"] = m["config"].str.extract(r"(dominicks_\w{3}|frat|m5)", expand=False)
    m["windows_set"] = np.where(m["config"].str.contains("rolling"), "rolling", "final")
    m["bottom_change"] = m["bottom_permuted"] / m["bottom_true"] - 1
    m["total_change"] = m["total_permuted"] / m["total_true"] - 1
    keep = ["bottom_change", "total_change"] + sorted(c for c in m if c.startswith("persist_h")) + \
        [c for c in ("probe_before_r2", "probe_after_r2") if c in m]
    out = m.groupby(["data", "windows_set", "model"])[keep].mean().reset_index()
    out["windows"] = m.groupby(["data", "windows_set", "model"]).size().to_numpy()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rev", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rev, out = Path(a.rev), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {}
    for part in ("scores", "paired", "mcs", "intervals", "decisions"):
        frames = [gather(rev, part, f) for f in ("weekly", "weekly_*", "m5")]
        frames = [f for f in frames if not f.empty]
        parts[part] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tables = {
        "accuracy": accuracy(parts["scores"]) if not parts["scores"].empty else pd.DataFrame(),
        "paired": paired(parts["paired"]) if not parts["paired"].empty else pd.DataFrame(),
        "mcs": parts["mcs"],
        "intervals": intervals(parts["intervals"]),
        "decisions": decisions(parts["decisions"]),
        "reconcile": reconcile(gather(rev, "reconcile", RECONCILE_FOLDERS)),
        "mechanism": mechanism(rev),
    }
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(out / f"{name}.csv", index=False, float_format="%.5g")
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
