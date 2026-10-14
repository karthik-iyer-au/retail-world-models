"""scripts/revision_tables.py on small hand-made analysis CSVs."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import revision_tables as T  # noqa: E402


def test_tables_from_two_folders(tmp_path):
    rev = tmp_path / "rev"
    for folder, key in (("weekly", "frat"), ("weekly_dominicks_che", "dominicks_che")):
        d = rev / folder
        d.mkdir(parents=True)
        rows = [{"key": key, "set": "rolling", "origin": o, "model": m, "level": "total", "wrmsse": w + 0.01 * o}
                for o in range(3) for m, w in (("world_model", 0.5), ("lightgbm", 0.6), ("seasonal_naive", 1.0))]
        pd.DataFrame(rows).to_csv(d / "scores.csv", index=False)
        pd.DataFrame([{"key": key, "set": "rolling", "origin": o, "a": "lightgbm", "b": "world_model", "level": "total",
                       "diff": 0.1, "lo": 0.05, "hi": 0.15} for o in range(3)]).to_csv(d / "paired.csv", index=False)
        pd.DataFrame([{"key": key, "set": "rolling", "origin": o, "model": m, "decision": "replenish_0.8", "cost": c}
                      for o in range(1, 3) for m, c in (("world_model", 80.0), ("seasonal_naive", 100.0))]
                     ).to_csv(d / "decisions.csv", index=False)
    # an old Frat reconciliation (earlier rule) that must not be read, and the new one
    pd.DataFrame([{"key": "frat", "set": "final", "origin": 0, "method": "ets_ols", "level": "total", "wrmsse": 9.0}]
                 ).to_csv(rev / "weekly" / "reconcile.csv", index=False)
    (rev / "reconcile_frat").mkdir()
    pd.DataFrame([{"key": "frat", "set": "final", "origin": 0, "method": "ets_ols", "level": "total", "wrmsse": 0.7}]
                 ).to_csv(rev / "reconcile_frat" / "reconcile.csv", index=False)
    out = tmp_path / "tables"
    T.main(["--rev", str(rev), "--out", str(out)])
    acc = pd.read_csv(out / "accuracy.csv")
    wm = acc[(acc["key"] == "frat") & (acc["model"] == "world_model")].iloc[0]
    assert abs(wm["wrmsse"] - 0.51) < 1e-9 and wm["windows"] == 3 and wm["wins"] == 3
    pr = pd.read_csv(out / "paired.csv")
    assert (pr["b_better_windows"] == 3).all() and (pr["lo"] > 0).all()
    dc = pd.read_csv(out / "decisions.csv")
    assert abs(dc[dc["model"] == "world_model"]["ratio_to_naive"].iloc[0] - 0.8) < 1e-9
    rc = pd.read_csv(out / "reconcile.csv")
    assert len(rc) == 1 and abs(rc["wrmsse"].iloc[0] - 0.7) < 1e-9
