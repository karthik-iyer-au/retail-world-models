# retail-world-models

Forecasting retail demand under price, promotion, display and feature
decisions, using public data only.

The question this repository answers: which way of representing a retail
category gives the most accurate and most explainable forecasts when
commercial decisions change. Several published methods are tested against
each other and against one new model. The results decide which one is
reported as best.

## Layout

| Folder | What it holds | Rule |
|---|---|---|
| `data/` | Raw files (not in git) and their checksums (in git) | Raw files are never edited |
| `src/rwm/data/` | One loader per dataset, all producing the same table shape | No modelling here |
| `src/rwm/prior_work/` | Methods published by others | Each one cites its paper and source code |
| `src/rwm/model/` | Our model | Nothing from prior work is copied in; it is imported |
| `src/rwm/evaluation/` | Splits and accuracy measures | Shared by every model, so all are scored the same way |
| `src/rwm/experiments/` | The runner | One config in, one recorded result out |
| `configs/` | One file per dataset and per experiment | A result is defined by its config |
| `results/` | Run outputs | Each run has a manifest that lets it be repeated |
| `docs/` | Decision log, provenance, protocol, how to reproduce | Updated in the same commit as the change |
| `tests/` | Checks on every piece above | Must pass before any commit |

## Quick start

```
make install
make test
make smoke
```

`make smoke` runs the whole pipeline on generated data and needs no downloads.

## How a result is made repeatable

Every run writes a `manifest.json` holding the code commit, the full config,
the checksum of every data file read, the seed and the installed package
versions. Reportable runs use `--strict`, which refuses to start if the code
is not committed or the data does not match its registered checksum.
See `docs/REPRODUCE.md`.
