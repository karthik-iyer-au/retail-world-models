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
| TimesFM 2.5, zero-shot; plan through its in-context linear model | Das et al. (2024); `timesfm` package, checkpoint `google/timesfm-2.5-200m-pytorch` | `timesfm_zero_shot.py` |
| TimesFM 3.0, zero-shot; plan read as covariates | google-research/timesfm (August 2026); `timesfm` 3.0.2, checkpoint `google/timesfm-3.0-pytorch` (non-commercial licence) | `timesfm3_zero_shot.py` |
| Chronos-Bolt, zero-shot, history only | Ansari et al. (2024); `chronos-forecasting`, checkpoint `amazon/chronos-bolt-base` | `chronos_zero_shot.py` |
| Chronos-2, zero-shot; plan read as covariates | Ansari et al. (2025), arXiv 2510.15821; `chronos-forecasting` 2.3.2, checkpoint `amazon/chronos-2` | `chronos2_zero_shot.py` |
| Timestral 1.0, zero-shot; plan read as covariates | Mistral (October 2026); github.com/mistralai/timestral at a534bdb | `timestral_zero_shot.py` |
