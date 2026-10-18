# retail-world-models

Forecasting retail demand under price, promotion, display and feature
decisions, using public data only.

The question this repository answers: at which grain of the product, store
and time hierarchy do which forecasting models work best, when the price and
promotion plan is known? Store orders are decided for an item in a store,
supplier funding for an item across stores, category performance by category
and store, and the financial plan at the chain total. The work covers price and
promotion, two of the four Ps, on public data; it does not forecast product or
place, or solve the four together.

## What was found (Paper 1, on these data sets only)

On M5, six Dominick's categories, Breakfast at the Frat and Corporación
Favorita, over rolling forecast origins with paired tests:

- **Item by store and item across stores:** an equal-weight average of a world
  model, LightGBM and TimesFM given the plan had the lowest error on 7 and 8 of
  9 data sets, and the 90% model confidence set kept one or two models.
- **Category by store, category and chain total:** the lowest-error model
  changed from data set to data set, and four to six windows did not separate a
  median of 4 of 9 models.
- **Category-by-store planning cost:** the average or LightGBM was lowest on 7
  of 8 data sets, ahead of the world model by a paired interval; the world model
  was lowest only on M5. (A first version of this cost put the world model first
  on 6 of 8; it let category-store nodes with no sales in a window dominate on
  Dominick's and was corrected in study G.)
- LightGBM was the strongest single model at the fine grains; averaging added
  to it, and weights fitted on earlier windows rarely beat equal weights.

A retailer's own data, or a forecast solved across all four Ps, may order the
models differently. Each claim is linked to its table, script and runs in
[`docs/RESULTS.md`](docs/RESULTS.md); the manuscript is `paper/main.pdf`, *World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets*.
Paper 2 (`paper2/main.pdf`), *Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Causal Validity*, audits the plan-lift explanations of the 494 saved
world-model and lift-model versions: no version passed every check on M5, and
on Dominick's only 6 of 540 version-windows did once C1 failures of the audit's
own definition of nothing planned are set aside, both the world model and LightGBM state about twice the measured
response to Dominick's zone price changes, and the number of checks a version
passed was not clearly related to its accuracy.

## Layout

| Folder | What it holds | Rule |
|---|---|---|
| `data/` | Raw files (not in git) and their checksums (in git) | Raw files are never edited |
| `src/rwm/data/` | One loader per dataset, all producing the same table shape | No modelling here |
| `src/rwm/prior_work/` | Methods published by others | Each one cites its paper and source code |
| `src/rwm/model/` | Our model | Nothing from prior work is copied in; it is imported |
| `src/rwm/evaluation/` | Splits and accuracy measures | Shared by every model. The M5 score is checked against the organisers' published numbers |
| `src/rwm/experiments/` | The runner | One config in, one recorded result out |
| `configs/` | One file per dataset, per experiment, and per job (a list of runs) | A result is defined by its config |
| `results/` | Run outputs | Each run has a manifest that lets it be repeated |
| `notebooks/` | A launcher for running a config on Colab | Calls the repository's commands only; holds no model or data code |
| `docs/` | Decision log, provenance, protocol, how to reproduce | Updated in the same commit as the change |
| `tests/` | Checks on every piece above | Must pass before any commit |

## Quick start

```
make install
make test
make smoke
python -m rwm.data.fetch m5        # public data, checked against registered checksums
```

`make smoke` runs the whole pipeline on generated data and needs no downloads.

## How a result is made repeatable

Every run writes a `manifest.json` holding the code commit, the full config,
the checksum of every data file read, the seed and the installed package
versions. Reportable runs use `--strict`, which refuses to start if the code
is not committed or the data does not match its registered checksum.
See `docs/REPRODUCE.md`.

## Licence

Non-commercial use only. Industry and commercial use of the code or either
manuscript, or any part of them, including use inside a company, is prohibited
unless approved in writing by the authors.

- Code, including the figure and table scripts in `paper/figs/` and `paper2/figs/`: [PolyForm Noncommercial 1.0.0](LICENSE).
- The manuscript text, figures and tables in `paper/` and `paper2/`: [CC BY-NC 4.0](LICENSE-docs).
- Data is not included. M5, Dominick's and Breakfast at the Frat are each obtained from their providers under the providers' own terms.

