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
