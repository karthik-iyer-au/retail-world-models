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
        for m in self.models:
            m.fit(train)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(future) for m in self.models], axis=0)
