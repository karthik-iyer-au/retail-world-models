"""The research loop: a fixed search for better world-model settings, run in
the manner of Karpathy's autoresearch with the rules of its first use (the
companion work's loop, after M. Dabbeeru's review), tightened where that first
use showed a weakness.

The rules are fixed in a spec file (configs/research/*.yaml) before the first
try and are never changed by the loop:

- One change at a time. Each round, every change on the menu not yet kept is
  tried on top of the best settings so far, and the best of them is kept if
  it clears the bar (forward selection). The loop stops when none does.
- A locked score: the mean WRMSSE over the spec's development windows, which
  lie before every window the paper reports. Lower is better.
- A held-out window, the latest in the spec, is scored for every kept change
  and logged, but never used to decide, so the loop cannot learn its test.
- The keep bar is the seed spread, not a fixed small number: the range of the
  starting settings' scores over the spec's seeds (the first use kept changes
  that beat the best by 0.1 point when the seed spread was 1.4). A change is
  kept only if its mean over every seed beats the best's mean by at least the
  bar.
- Successive halving, the synchronous form of ASHA (Li et al., MLSys 2020):
  each round's changes run first at `rung_fraction` of the training steps on
  one seed; the best third go on to full steps on every seed. Only full-step,
  every-seed scores decide.
- Every try is logged, kept or not, with its config, so the whole search can
  be published with the paper.

This module holds the search only. Running a try is handed in as `evaluate`,
a function from a list of configs to a list of results (each the run's
metrics, or None if it failed), so the same loop runs on Modal
(`cloud/modal_app.py research`) or locally, and is tested with a stand-in.
"""
import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from rwm.experiments.run import with_seed
from rwm.utils.hashing import sha256_obj
from rwm.utils.paths import REPO_ROOT

Evaluate = Callable[[list[dict]], list[dict | None]]


def load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load((REPO_ROOT / path).read_text())
    for key in ("name", "base", "evaluation", "seeds", "menu"):
        if key not in spec:
            raise ValueError(f"research spec needs '{key}'")
    if len(spec["seeds"]) < 2:
        raise ValueError("at least two seeds, so the keep bar can be measured")
    names = [c["name"] for c in spec["menu"]]
    if len(names) != len(set(names)):
        raise ValueError("menu names must be unique")
    return spec


def merge(params: dict, change: dict) -> dict:
    """The model settings with one change applied: keys are model parameters,
    and a value of None removes that parameter (back to the model's default)."""
    out = copy.deepcopy(params)
    for k, v in change.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_for(spec: dict, params: dict, seed: int, label: str, step_fraction: float = 1.0) -> dict:
    base = yaml.safe_load((REPO_ROOT / spec["base"]).read_text())
    p = dict(params)
    if step_fraction < 1.0:
        p["steps"] = max(1, int(round(p.get("steps", 20000) * step_fraction)))
    config = {
        "name": f"research_{spec['name']}_{label}",
        "seed": 0,
        "dataset": base["dataset"],
        "model": {"name": base["model"]["name"], "params": p},
        "evaluation": spec["evaluation"],
    }
    return with_seed(config, seed)


def scores(metrics: dict) -> tuple[float, float]:
    """The locked score (mean WRMSSE over the development windows) and the
    held-out window's WRMSSE. The held-out window is the latest one."""
    splits = sorted(metrics["splits"], key=lambda s: s["test_start"])
    dev, held = splits[:-1], splits[-1]
    return float(np.mean([s["wrmsse"] for s in dev])), float(held["wrmsse"])


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        with open(self.path, "a") as f:  # opened and closed on every row, so nothing waits in a buffer
            f.write(json.dumps(row, default=str) + "\n")


def search(spec: dict, evaluate: Evaluate, log: Log, max_rounds: int | None = None) -> dict:
    seeds = list(spec["seeds"])
    rung = float(spec.get("rung_fraction", 1 / 3))
    promote = float(spec.get("promote_share", 1 / 3))
    base_params = yaml.safe_load((REPO_ROOT / spec["base"]).read_text())["model"].get("params", {})

    def full(params: dict, label: str) -> tuple[float, float, list] | None:
        """Mean locked score and held-out score over every seed, at full steps."""
        got = evaluate([config_for(spec, params, s, f"{label}_s{s}") for s in seeds])
        if any(g is None for g in got):
            return None
        each = [scores(g) for g in got]
        return float(np.mean([e[0] for e in each])), float(np.mean([e[1] for e in each])), [e[0] for e in each]

    start = full(base_params, "start")
    if start is None:
        raise RuntimeError("the starting settings did not run")
    bar = max(float(np.ptp(start[2])), float(spec.get("min_bar", 0.0)))
    best = {"params": base_params, "score": start[0], "held_out": start[1], "kept": []}
    log.write({"kind": "start", "score": start[0], "held_out": start[1], "seed_scores": start[2], "keep_bar": bar,
               "config_sha256": sha256_obj(base_params)})
    open_menu = list(spec["menu"])
    round_no = 0
    while open_menu and (max_rounds is None or round_no < max_rounds):
        round_no += 1
        trials = [(c, merge(best["params"], c["change"])) for c in open_menu]
        # rung 1: a share of the steps, one seed
        # at full steps the screening run is the first seed's full run, named the same, so it is read back and not repeated
        rung_label = (lambda c: f"r{round_no}_{c['name']}_s{seeds[0]}") if rung >= 1.0 else (lambda c: f"r{round_no}_{c['name']}_rung")
        cheap = evaluate([config_for(spec, p, seeds[0], rung_label(c), rung) for c, p in trials])
        ranked = []
        for (c, p), g in zip(trials, cheap):
            row = {"kind": "rung", "round": round_no, "change": c["name"], "params_changed": c["change"]}
            if g is None:
                log.write({**row, "failed": True})
                continue
            s, h = scores(g)
            log.write({**row, "score": s})
            ranked.append((s, c, p))
        ranked.sort(key=lambda r: r[0])
        n_up = max(1, int(np.ceil(promote * len(ranked)))) if ranked else 0
        # rung 2: full steps, every seed, for the best of rung 1
        results = []
        for s_cheap, c, p in ranked[:n_up]:
            got = full(p, f"r{round_no}_{c['name']}")
            row = {"kind": "try", "round": round_no, "change": c["name"], "params_changed": c["change"]}
            if got is None:
                log.write({**row, "failed": True})
                continue
            log.write({**row, "score": got[0], "held_out": got[1], "seed_scores": got[2],
                       "gain": best["score"] - got[0], "keep_bar": bar})
            results.append((got, c, p))
        if not results:
            break
        results.sort(key=lambda r: r[0][0])
        (score, held, seed_scores), c, p = results[0]
        if best["score"] - score < bar:
            log.write({"kind": "stop", "round": round_no, "reason": "no change beat the best by the keep bar"})
            break
        best = {"params": p, "score": score, "held_out": held, "kept": best["kept"] + [c["name"]]}
        log.write({"kind": "kept", "round": round_no, "change": c["name"], "score": score, "held_out": held})
        # forward selection: every other change is tried again, on top of the new best
        open_menu = [m for m in open_menu if m["name"] != c["name"]]
    log.write({"kind": "end", "kept": best["kept"], "score": best["score"], "held_out": best["held_out"]})
    return best
