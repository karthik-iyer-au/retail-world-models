"""Dataset loaders are looked up by name from the experiment config."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from rwm.data.schema import validate


@dataclass
class Dataset:
    name: str
    panel: pd.DataFrame
    levers: list[str]
    files: list[Path] = field(default_factory=list)  # raw files read, for checksums


_LOADERS: dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str):
    def wrap(fn):
        _LOADERS[name] = fn
        return fn

    return wrap


def load_dataset(name: str, **params) -> Dataset:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset '{name}'. Known: {sorted(_LOADERS)}")
    ds = _LOADERS[name](**params)
    ds.panel = validate(ds.panel, ds.levers)
    return ds
