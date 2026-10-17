"""Every revision number the manuscript quotes is in the committed result tables.

The prose quotes revision results only through lookup commands (\\rv, \\rpair, ...)
defined by paper/figs/make_revision.py from results/paper1_revision/tables. This
test rebuilds those values and checks that every lookup in the manuscript's
sources resolves, so a renamed or missing result fails here and not only in the
LaTeX build.
"""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ARITY = {"rv": 3, "rf": 3, "rpair": 5, "rab": 4, "rw": 3, "rdec": 3, "rint": 4, "rintold": 4, "rrec": 3,
         "rper": 5, "rmech": 4, "rsf": 3, "rtail": 4, "rrank": 4, "rpan": 5, "raft": 3, "rcount": 1, "rwin": 3, "rreg": 3}


def _values():
    spec = importlib.util.spec_from_file_location("make_revision", PAPER / "figs" / "make_revision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.vals.clear()
    acc = mod.rolling_accuracy()
    sets = mod.mcs_sets()
    mod.accuracy_values(acc)
    p = mod.paired_values()
    ab = mod.ablation_values()
    mod.other_values()
    mod.studyf_values()
    mod.regret_values()
    mod.window_values()
    mod.pantry_values()
    mod.count_values(acc, sets, p, ab)
    return dict(mod.vals)


def _lookups(text: str):
    for cmd, n in ARITY.items():
        pat = re.compile(r"\\" + cmd + r"(?![A-Za-z])" + r"\{([^{}]*)\}" * n)
        for m in pat.finditer(text):
            yield "@".join([cmd, *m.groups()])


def test_every_quoted_revision_value_exists():
    vals = _values()
    sources = [PAPER / "abstract.tex", PAPER / "body.tex", PAPER / "results.tex"]
    used = {k for f in sources for k in _lookups(f.read_text())}
    assert len(used) > 100
    missing = sorted(k for k in used if k not in vals)
    assert not missing, missing


def test_counts_follow_the_tables():
    vals = _values()
    # the ensemble is in the model confidence set at the total on all nine data sets
    assert vals["rcount@ensinsettotal"] == "nine"
    # equal-weight ensemble: one third each, so the sweep's equal point exists for every data set
    assert vals["rv@mfive@ens@ixs"] == "0.834"
