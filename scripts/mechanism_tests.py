"""Paper 1 revision: does the world model's transition do anything the lift
model does not? Tests on saved checkpoints, with no refitting
(docs/PAPER1_REVISION_PLAN.md, "Mechanism tests").

  python scripts/mechanism_tests.py --results <runs folder> --out results/paper1_revision/mechanism.csv \
      --pattern "*rev_*_rolling_world_model_seed1_*" "*rev_*_rolling_lift_seed1_*" ...

For each run and window:
1. timing: the plan of the forecast periods (price and levers together) is
   permuted within each series, keeping the same exposure; accuracy against
   the true plan;
2. persistence: for series with nothing planned in the window, a promotion
   (and a 20% cut) in the first period; the forecast change in each period
   of the window, as a share of the forecast without it;
3. beyond sales (world model, windows with data after them): a ridge probe
   from the rolled state, against the state before the window, to log sales
   over the periods after the window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_readback import ridge_r2  # noqa: E402

from rwm.audit.runs import load_checkpoint, model_params, plan_levers, windows  # noqa: E402
from rwm.data import load_dataset  # noqa: E402
from rwm.data.schema import DATE, PRICE, SERIES, UNITS  # noqa: E402
from rwm.evaluation import compare as C  # noqa: E402


def permuted(plan: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = plan.copy()
    for _, idx in plan.groupby(SERIES).indices.items():
        perm = rng.permutation(idx)
        out.loc[out.index[idx], cols] = plan.iloc[perm][cols].to_numpy()
    return out


def persistence(model, plan: pd.DataFrame, levers: list[str], regular: pd.Series) -> list[float]:
    reg = plan[SERIES].map(regular)
    quiet = (plan[PRICE] >= 0.95 * reg).groupby(plan[SERIES]).transform("all")
    for c in levers:
        quiet &= (plan[c].fillna(0) == 0).groupby(plan[SERIES]).transform("all")
    q = plan[quiet.to_numpy()].copy()
    if q.empty:
        return []
    first = q[DATE] == q[DATE].min()
    f0 = model.predict(q)
    q2 = q.copy()
    q2.loc[first, PRICE] = q2.loc[first, PRICE] * 0.8
    if levers and "promo" in levers:
        q2.loc[first, "promo"] = 1
    f1 = model.predict(q2)
    by = pd.DataFrame({"d": q[DATE].to_numpy(), "f0": f0, "f1": f1}).groupby("d").sum()
    return list(((by["f1"] - by["f0"]) / by["f0"].clip(lower=1e-9)).to_numpy())


def states(model, test: pd.DataFrame):
    """State before the window and after the last period, with each row's series index."""
    import torch

    _, fut = model._future(test.drop(columns=[UNITS], errors="ignore"))
    rows_out, before, after = [], [], []
    model._net.eval()
    with torch.no_grad():
        for rows, picked in model._batches():
            start = torch.full_like(rows, model._past["price"].shape[1])
            got, _, _, _ = model._forward(model._past, {**fut, "start": torch.zeros_like(rows)}, rows, start, picked, full=True)
            real = torch.ones_like(rows, dtype=torch.bool) if picked is None else picked.reshape(-1) >= 0
            rows_out.append(rows[real].cpu().numpy())
            before.append(got["state"][real].float().cpu().numpy())
            after.append(None if got["rolled"] is None else got["rolled"][real][:, -1].float().cpu().numpy())
    return np.concatenate(rows_out), np.concatenate(before), (None if after[0] is None else np.concatenate(after))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pattern", nargs="+", required=True)
    a = p.parse_args(argv)
    res = Path(a.results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = pd.read_csv(a.out) if Path(a.out).exists() else pd.DataFrame(columns=["run", "origin"])
    seen = set(zip(done["run"], done["origin"]))
    rows = done.to_dict("records")
    for d in sorted({x for pat in a.pattern for x in res.glob(pat) if (x / "manifest.json").exists()}):
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        if not list(d.glob("checkpoint_origin*.pt")):
            continue
        params = model_params(cfg)
        levers = [c for c in plan_levers(cfg) if c != "cost"]
        fr = C.frame(cfg)
        full = None
        for origin, train, test in windows(cfg):
            if (d.name, origin) in seen or not (d / f"checkpoint_origin{origin}.pt").exists():
                continue
            model = load_checkpoint(d, origin)
            plan = test.drop(columns=[UNITS])
            f0 = model.predict(plan)
            fc = lambda f: C.to_matrix(test.assign(forecast=f), SERIES, DATE, "forecast", list(np.sort(test[DATE].unique())), fr.names)
            s0 = C.level_scores(fr, origin, fc(f0))
            cols = [PRICE] + [c for c in levers if c in plan]
            sp = C.level_scores(fr, origin, fc(model.predict(permuted(plan, cols, seed=origin))))
            regular = train[train[DATE].isin(np.sort(train[DATE].unique())[-12:])].groupby(SERIES)[PRICE].max()
            pers = persistence(model, plan, levers, regular)
            rec = {"run": d.name, "config": cfg["name"], "origin": origin,
                   "rollout": bool(params.get("rollout")),
                   "bottom_true": s0[list(s0)[-1]], "bottom_permuted": sp[list(sp)[-1]],
                   "total_true": s0["total"], "total_permuted": sp["total"],
                   **{f"persist_h{h + 1}": v for h, v in enumerate(pers)}}
            # 3. probe to sales after the window, where the data go on
            if params.get("rollout"):
                if full is None:
                    full = load_dataset(cfg["dataset"]["name"], **cfg["dataset"].get("params", {})).panel
                end = test[DATE].max()
                later = np.sort(full[DATE].unique())
                later = later[later > end][: len(np.unique(test[DATE]))]
                if len(later):
                    y = full[full[DATE].isin(later)].groupby(full[SERIES].astype(str))[UNITS].sum()
                    idx, before, after = states(model, test)
                    names = model._names[idx]
                    target = np.log1p(y.reindex(names).fillna(0).to_numpy())
                    rng = np.random.default_rng(origin)
                    half = rng.random(len(names)) < 0.5
                    rec["probe_after_r2"] = ridge_r2(after[half], target[half], after[~half], target[~half])
                    rec["probe_before_r2"] = ridge_r2(before[half], target[half], before[~half], target[~half])
            rows.append(rec)
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "run"}), flush=True)
            del model


if __name__ == "__main__":
    main()
