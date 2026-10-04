# Prior work

Methods published by other authors. They serve two purposes: benchmarks to
compare against, and building blocks our model may use.

Rules for adding one:

1. One module or folder per method.
2. Start from the authors' own code or official package where one exists.
   Record the paper, the source, its licence and the exact version or commit
   in the table below and in `docs/PROVENANCE.md`.
3. Implement the `Forecaster` interface in `rwm/forecaster.py` and register a
   name, so the runner and the scoring treat it like every other model.
4. Any change made to the authors' code is listed in the module docstring.

| Method | Reference | Module |
|---|---|---|
| Seasonal naive, recent average | Standard benchmarks, as used in M5 | `naive.py` |
| LightGBM, one model per store, direct | Ke et al. (2017); design of the top M5 entries, Makridakis, Spiliotis, Assimakopoulos (2022) | `lightgbm_direct.py` |
