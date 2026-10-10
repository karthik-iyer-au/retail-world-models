import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwm.data import load_dataset
from rwm.data.schema import DATE, UNITS
from rwm.forecaster import build_model
from rwm.utils import resume

SETTINGS = dict(
    horizon=8, history=16, train_periods=120, categorical=["item_id", "store_id"], extra=["promo"],
    d_model=16, layers=1, heads=2, steps=40, batch=64, lr=2e-3, device="cpu", validation_periods=8, seed=3,
)


@pytest.fixture(scope="module")
def split():
    p = load_dataset("synthetic", n_stores=2, n_items=4, n_periods=200, seed=2).panel
    cut = np.sort(p[DATE].unique())[-9]
    return p[p[DATE] <= cut], p[p[DATE] > cut].reset_index(drop=True)


def test_training_cut_short_goes_on_from_its_last_save(split, tmp_path, monkeypatch):
    train, test = split
    future = test.drop(columns=[UNITS])
    whole = build_model("state_model", **SETTINGS).fit(train)

    calls = {"n": 0}
    real_save = resume.save

    def cut_after_five(state, tag):
        real_save(state, tag)
        calls["n"] += 1
        if calls["n"] == 5:  # saves come every 2 steps, so this stops after step 10 of 40
            raise KeyboardInterrupt

    monkeypatch.setattr(resume, "save", cut_after_five)
    with resume.scope(tmp_path, "origin0"):
        with pytest.raises(KeyboardInterrupt):
            build_model("state_model", **SETTINGS).fit(train)
    monkeypatch.setattr(resume, "save", real_save)
    assert torch.load(tmp_path / "origin0_training.pt", weights_only=False)["step"] == 10

    with resume.scope(tmp_path, "origin0"):
        again = build_model("state_model", **SETTINGS).fit(train)
    assert again.validation_log == whole.validation_log
    np.testing.assert_allclose(again.predict(future), whole.predict(future), rtol=1e-5)


def test_finished_training_is_not_repeated(split, tmp_path):
    train, test = split
    with resume.scope(tmp_path, "origin0"):
        first = build_model("state_model", **SETTINGS).fit(train)
    assert torch.load(tmp_path / "origin0_training.pt", weights_only=False)["step"] == SETTINGS["steps"]
    with resume.scope(tmp_path, "origin0"):
        second = build_model("state_model", **SETTINGS).fit(train)
    future = test.drop(columns=[UNITS])
    np.testing.assert_allclose(second.predict(future), first.predict(future), rtol=1e-6)


def test_each_averaged_member_saves_on_its_own(split, tmp_path):
    with resume.scope(tmp_path, "origin1"):
        build_model("seed_average", model="state_model", members=2, params={**SETTINGS, "steps": 4}).fit(split[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["origin1_member0_training.pt", "origin1_member1_training.pt"]


def test_nothing_is_saved_outside_a_scope(split, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_model("state_model", **{**SETTINGS, "steps": 4}).fit(split[0])
    assert resume.path("training") is None
    assert list(tmp_path.iterdir()) == []



class _OnGpu:
    """Stands in for a random state that was loaded onto a GPU: torch refuses it
    as a random state until .cpu() brings it back."""
    def __init__(self, t):
        self.t = t

    def cpu(self):
        return self.t


def test_saved_random_states_load_onto_another_device(split, tmp_path, monkeypatch):
    train, _ = split
    with resume.scope(tmp_path, "origin0"):
        build_model("state_model", **{**SETTINGS, "steps": 4}).fit(train)
    real_load = resume.load

    def load_on_gpu(tag, device=None):
        s = real_load(tag, device)
        s["gen"], s["rng"], s["step"] = _OnGpu(s["gen"]), _OnGpu(s["rng"]), 2
        return s

    monkeypatch.setattr(resume, "load", load_on_gpu)
    with resume.scope(tmp_path, "origin0"):
        build_model("state_model", **{**SETTINGS, "steps": 4}).fit(train)  # goes on from step 2
