"""The interface every model implements, prior work and ours alike.

`fit` sees history, including units, ordered by series then date. `predict` sees the future rows with the
decisions planned for them (price, promotion and so on) and no units. That is
the question the paper asks: given these decisions, what will sell.
"""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd

_MODELS: dict[str, Callable[..., "Forecaster"]] = {}


class Forecaster(ABC):
    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Forecaster": ...

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> np.ndarray:
        """One forecast per row of `future`, in the same row order."""

    # A model whose fit only stores history (the naive and zero-shot models)
    # learns nothing from it, so taking in new history is fitting again.
    stores_history_only = False

    def refresh(self, history: pd.DataFrame) -> "Forecaster":
        """Take in `history`, which may hold series the model was not trained
        on, for forecasting from the end of it, without learning from it: the
        fitted weights stay as they are. Used by the transfer test, which
        trains without some stores or items and then forecasts them from their
        own history."""
        if self.stores_history_only:
            return self.fit(history)
        raise NotImplementedError(f"{type(self).__name__} cannot take in new history without training")


def register_model(name: str):
    def wrap(cls):
        _MODELS[name] = cls
        return cls

    return wrap


def build_model(name: str, **params) -> Forecaster:
    import rwm.model  # noqa: F401  (registers our model)
    import rwm.prior_work  # noqa: F401  (registers prior-work models)
    import rwm.seed_average  # noqa: F401  (registers the seed-average wrapper)
    import rwm.combination  # noqa: F401  (registers the forecast combination)

    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(_MODELS)}")
    return _MODELS[name](**params)
