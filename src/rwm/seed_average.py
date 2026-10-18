"""Average of one model fitted several times with different seeds.

A standard way to steady any model whose result depends on its random
start. It wraps any registered model, prior work or ours, so both sides of a
comparison can be given the same treatment.

  model:
    name: seed_average
    params:
      model: state_model      # the model to repeat
      members: 5              # how many times
      seed: 0                 # member i is fitted with seed 1000 * seed + i
      params: {...}           # the wrapped model's own settings
"""
import inspect

import numpy as np
import pandas as pd

from rwm.forecaster import _MODELS, Forecaster, build_model, register_model
from rwm.utils import resume


@register_model("seed_average")
class SeedAverage(Forecaster):
    def __init__(self, model: str, params: dict | None = None, members: int = 5, seed: int = 0):
        if members < 1:
            raise ValueError("members must be at least 1")
        build_model(model, **(params or {}))  # also makes sure the model is registered
        takes_seed = "seed" in inspect.signature(_MODELS[model].__init__).parameters
        self.models = [
            build_model(model, **{**(params or {}), **({"seed": 1000 * seed + i} if takes_seed else {})})
            for i in range(members)
        ]

    def fit(self, train: pd.DataFrame) -> "SeedAverage":
        for i, m in enumerate(self.models):
            with resume.scope(None, f"member{i}"):  # each member saves its own training
                m.fit(train)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(future) for m in self.models], axis=0)

    def breakdown(self, future: pd.DataFrame) -> pd.DataFrame:
        """Baseline and lift averaged over the members, as the forecast is."""
        parts = [m.breakdown(future) for m in self.models]
        return sum(parts) / len(parts)

    def nothing_planned(self, future: pd.DataFrame) -> pd.DataFrame:
        """The members' "nothing planned" plan. Every member is fitted on the
        same data with the same settings, so they share it."""
        return self.models[0].nothing_planned(future)

    def save(self, path) -> None:
        """All members in one file, each as its own model would save it."""
        import io

        import torch

        if not all(hasattr(m, "save") for m in self.models):
            raise TypeError("members cannot be saved")
        blobs = []
        for m in self.models:
            buf = io.BytesIO()
            m.save(buf)
            blobs.append(buf.getvalue())
        name = next(k for k, v in _MODELS.items() if isinstance(self.models[0], v))
        torch.save({"model": name, "members": blobs}, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "SeedAverage":
        import io

        import torch

        saved = torch.load(path, map_location="cpu", weights_only=False)
        member = _MODELS[saved["model"]]
        out = cls.__new__(cls)
        out.models = [member.load(io.BytesIO(b), device=device) for b in saved["members"]]
        return out
