"""The research loop's rules, checked with a stand-in for running a try."""
import json

import yaml

from rwm.research import loop


def _spec(tmp_path, menu, seeds=(0, 1, 2), min_bar=0.0):
    base = {"name": "base", "seed": 0, "dataset": {"name": "synthetic"},
            "model": {"name": "state_model", "params": {"steps": 300, "lr": 0.001}},
            "evaluation": {"horizon": 7, "n_origins": 3}}
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base))
    return {"name": "t", "base": str(path), "evaluation": base["evaluation"], "seeds": list(seeds),
            "menu": menu, "min_bar": min_bar}


def _fake(effects, held_effects=None, noise=0.01, calls=None):
    """Score = 1 minus the effect of each setting present, plus seed noise. The
    held-out window gets its own effects, which must never decide."""
    held_effects = held_effects or {}

    def evaluate(configs):
        out = []
        for c in configs:
            if calls is not None:
                calls.append(c)
            p = c["model"]["params"]
            s = 1.0 - sum(v for k, v in effects.items() if p.get(k)) + noise * (c["seed"] % 3)
            h = 1.0 - sum(v for k, v in held_effects.items() if p.get(k))
            out.append({"splits": [{"test_start": "2020-01-01", "wrmsse": s}, {"test_start": "2020-02-01", "wrmsse": s},
                                   {"test_start": "2020-03-01", "wrmsse": h}]})
        return out
    return evaluate


def test_keeps_a_change_only_when_it_beats_the_best_by_the_seed_spread(tmp_path):
    menu = [{"name": "big", "change": {"big": True}}, {"name": "tiny", "change": {"tiny": True}}]
    log = loop.Log(tmp_path / "log.jsonl")
    best = loop.search(_spec(tmp_path, menu), _fake({"big": 0.1, "tiny": 0.005}), log)
    assert best["kept"] == ["big"]  # tiny (0.005) is inside the seed spread (0.02)
    rows = [json.loads(r) for r in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "start" and abs(rows[0]["keep_bar"] - 0.02) < 1e-9
    assert rows[-1]["kind"] == "end"


def test_the_held_out_window_never_decides(tmp_path):
    menu = [{"name": "looks_good_on_test_only", "change": {"x": True}}]
    best = loop.search(_spec(tmp_path, menu), _fake({}, held_effects={"x": 0.5}), loop.Log(tmp_path / "l.jsonl"))
    assert best["kept"] == []


def test_changes_are_tried_again_on_top_of_each_new_best(tmp_path):
    menu = [{"name": "a", "change": {"a": True}}, {"name": "b", "change": {"b": True}}]
    best = loop.search(_spec(tmp_path, menu), _fake({"a": 0.2, "b": 0.1}, noise=0.0, ), loop.Log(tmp_path / "l.jsonl"))
    assert best["kept"] == ["a", "b"]
    assert best["params"]["a"] and best["params"]["b"]


def test_cheap_rung_runs_fewer_steps_on_one_seed(tmp_path):
    calls = []
    menu = [{"name": f"m{i}", "change": {f"m{i}": True}} for i in range(6)]
    loop.search(_spec(tmp_path, menu), _fake({}, calls=calls), loop.Log(tmp_path / "l.jsonl"), max_rounds=1)
    rung = [c for c in calls if c["name"].endswith("_rung_seed0")]
    assert len(rung) == 6 and all(c["model"]["params"]["steps"] == 100 for c in rung)
    full = [c for c in calls if "_rung" not in c["name"] and "_start_" not in c["name"]]
    assert len(full) == 2 * 3  # the best third of six, on every seed


def test_a_change_of_none_goes_back_to_the_default():
    assert loop.merge({"a": 1, "b": 2}, {"a": None, "c": 3}) == {"b": 2, "c": 3}


def test_at_full_steps_the_screening_run_is_the_first_seeds_run(tmp_path):
    calls = []
    spec = {**_spec(tmp_path, [{"name": "a", "change": {"a": True}}]), "rung_fraction": 1.0, "promote_share": 1.0}
    loop.search(spec, _fake({"a": 0.2}, calls=calls), loop.Log(tmp_path / "l.jsonl"), max_rounds=1)
    names = [c["name"] for c in calls]
    assert names.count("research_t_r1_a_s0_seed0") == 2  # screened, then read back as seed 0 of the full run
