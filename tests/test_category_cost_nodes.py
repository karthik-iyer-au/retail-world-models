"""The per-node category costs rebuild the reported category-plan cost."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import category_cost_nodes as CC  # noqa: E402

from rwm.evaluation import compare as C  # noqa: E402
from tests.test_ensemble_ablation import fr  # noqa: E402,F401  (fixture)


def test_nodes_rebuild_the_reported_cost(fr):  # noqa: F811
    rng = np.random.default_rng(3)
    cols = ["store_id"]
    for sp in fr.splits:
        t0, t1, _ = fr.cut(sp.origin)
        fc = fr.units[:, t0:t1] * np.exp(rng.normal(0, 0.4, (len(fr.names), t1 - t0)))
        d = CC.node_rows(fr, sp.origin, fc, cols, daily=False)
        cost = (d["dollars"] * d["abs_period"] / d["actual_window"]).sum() / d["dollars"].sum()
        assert np.isclose(cost, C.category_planning_loss(fr, sp.origin, fc, cols))
        assert np.allclose(d["abs_week"], d["abs_period"])  # weekly data
        assert (d["abs_half"] <= d["abs_period"] + 1e-9).all()


def test_uncertainty_summary_signs():
    import category_cost_uncertainty as CU
    rng = np.random.default_rng(4)
    rows = []
    for o in range(3):
        for node in range(40):
            a = rng.uniform(50, 100)
            for m, s in (("world_model", 0.05), ("lightgbm", 0.3)):
                e = abs(rng.normal(0, s)) * a
                rows.append(dict(key="toy", origin=o, model=m, node=node, abs_period=e, actual_window=a,
                                 fc_window=a + e, abs_half=e / 2, actual_half=a / 2, abs_week=e,
                                 actual_week_sum=a, dollars=a, units_before=a))
    d = pd.DataFrame(rows)
    s = CU.summarise(d, boot=200)
    r = s[(s.variant == "dollars") & (s.model == "lightgbm")].iloc[0]
    assert r.hi < 0 and r.world_lower_windows == 3 and r.lo <= r.world_minus <= r.hi
    # a node that sold nothing must not dominate the corrected cost
    d.loc[(d.node == 0), "actual_window"] = 0.0
    s2 = CU.summarise(d, boot=50)
    assert s2[(s2.variant == "dollars")].cost.max() < 1
    assert s2[(s2.variant == "as_first_run")].cost.max() > 1e3
