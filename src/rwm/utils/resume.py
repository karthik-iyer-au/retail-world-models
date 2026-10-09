"""Training saves partway through a run, so a run that was cut short goes on
from its last save instead of starting again.

A run opens a scope with a folder and a name for the window it is fitting;
a model that fits several members (seed_average) adds the member to the name;
the model saves and loads under a tag for each stretch of training. Outside
any scope, nothing is saved or loaded.

    with resume.scope(folder, "origin1"):
        with resume.scope(None, "member0"):
            resume.save(state, "training")   # folder/origin1_member0_training.pt
"""
import os
from contextlib import contextmanager
from pathlib import Path

_folder: Path | None = None
_parts: list[str] = []


@contextmanager
def scope(folder: Path | None, name: str):
    """Save under `folder` (or the folder already in use) with `name` added to the file names."""
    global _folder
    before = _folder
    if folder is not None:
        _folder = Path(folder)
    _parts.append(name)
    try:
        yield
    finally:
        _parts.pop()
        _folder = before


def path(tag: str) -> Path | None:
    if _folder is None:
        return None
    return _folder / ("_".join([*_parts, tag]) + ".pt")


def save(state: dict, tag: str) -> None:
    """Write `state` for `tag`; written to a temporary file first so a cut never leaves half a file."""
    import torch

    p = path(tag)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, p)


def load(tag: str, device=None) -> dict | None:
    import torch

    p = path(tag)
    if p is None or not p.exists():
        return None
    return torch.load(p, map_location=device, weights_only=False)
