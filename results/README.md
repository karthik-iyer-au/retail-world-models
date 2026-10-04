# Results

Each run writes `results/<time>_<name>_<config hash>/` with `manifest.json`,
`metrics.json`, `forecasts.csv` and, for models that can be saved, one
checkpoint file per window. Run folders are not committed.

A run that is quoted in the paper is copied to `results/promoted/` (manifest
and metrics only) and committed, so every number in the paper points to a
manifest in git. Checkpoints and forecast files are too large for git. They
are kept outside it, and `metrics.json` records each checkpoint's checksum so
the stored file can be proven to be the one the run produced.

## M5 benchmarks

WRMSSE over the 12 levels of the M5 hierarchy. Lower is better. Each row
links to a folder in `results/promoted/` holding its manifest and scores.

| Method | 25 Apr to 22 May 2016 | 23 May to 19 Jun 2016 (official test period) |
|---|---|---|
| Seasonal naive | 0.870 | 0.847 |
| LightGBM, one model per store | 0.715 | 0.552 |
| Our model, item encoder and readout only, one run | 0.708 | not run |

Published scores on the official test period, from the organisers' files
(`data/raw/m5_reference/`): seasonal naive 0.847, best statistical benchmark
(exponential smoothing, bottom-up) 0.671, 50th-ranked entry 0.576,
10th-ranked entry 0.547, winning entry 0.520.

Notes:

- The LightGBM settings were fixed before the test-period run and were not
  tuned on either window.
- In the April to May window the LightGBM forecasts are 7.7% below actual
  sales in total. In the test period they are 2.1% below.
- Our model's row is a single training run on a GPU, which does not repeat
  exactly. The difference from LightGBM is smaller than can be claimed from
  one run. The test period is left unseen until the model design is final.

| Run | Folder |
|---|---|
| Seasonal naive, April to May | `20261004T045330_m5_validation_seasonal_naive_a67817a2` |
| LightGBM, April to May | `20261004T034345_m5_validation_lightgbm_11b5432b` |
| Seasonal naive, test period | `20261004T045417_m5_test_seasonal_naive_54f6d687` |
| LightGBM, test period | `20261004T041739_m5_test_lightgbm_9d232fa7` |
| Our model, April to May | `20261004T063152_m5_validation_state_model_ec8ad17b` |

## Breakfast at the Frat benchmarks

Average over the last three 8-week windows of the data (27 July 2011 to
4 January 2012). Lower is better. "Overall" is WRMSSE over six levels (total,
category, store, category by store, item, item by store). The other columns
are item-by-store accuracy on the weeks where that lever was on, and on weeks
with no promotion at all. Lever values for the forecast weeks are taken as
planned in advance.

| Method | Overall | On display | In circular | Tag-only price cut | No promotion |
|---|---|---|---|---|---|
| Seasonal naive | 0.832 | 1.471 | 1.510 | 0.869 | 0.872 |
| Average of last 8 weeks | 0.756 | 1.383 | 1.381 | 0.680 | 0.540 |
| LightGBM, price only | 0.554 | 1.025 | 1.015 | 0.668 | 0.431 |
| LightGBM, all recorded levers | 0.447 | 0.793 | 0.805 | 0.558 | 0.424 |

| Run | Folder |
|---|---|
| Seasonal naive | `20261004T070636_frat_seasonal_naive_02ce19d4` |
| Average of last 8 weeks | `20261004T070720_frat_recent_average_d9fdc7b6` |
| LightGBM, price only | `20261004T070804_frat_lightgbm_price_only_78aebb07` |
| LightGBM, all recorded levers | `20261004T070912_frat_lightgbm_0dbcdc4c` |

## Development runs over several seeds

These runs are saved in the Drive results folder; their manifests have not
yet been copied into `results/promoted/`. Scores are averages over five
seeds, lower is better.

M5, 25 April to 22 May 2016: our model (item encoder and readout, mixed
precision, A100) scores 0.699 on average, range 0.669 to 0.725, against
LightGBM at 0.715. Four of the five runs beat LightGBM.

Breakfast at the Frat, the two 8-week windows before the held-out final 8
weeks:

| Method | Overall | Range | On display | In circular | Tag-only price cut | No promotion |
|---|---|---|---|---|---|---|
| LightGBM, all recorded levers | 0.421 | 0.416 to 0.427 | 0.817 | 0.822 | 0.508 | 0.405 |
| Our model, items drawn at random | 0.424 | 0.407 to 0.438 | 0.819 | 0.862 | 0.554 | 0.413 |
| Our model, whole stores drawn | 0.430 | 0.407 to 0.453 | 0.818 | 0.851 | 0.563 | 0.415 |
| + item-to-item, conditional similarity | 0.476 | 0.430 to 0.508 | 0.861 | 0.909 | 0.580 | 0.419 |
| + item-to-item, attention | 0.465 | 0.416 to 0.505 | 0.871 | 0.907 | 0.568 | 0.419 |
| + item-to-item, both | 0.453 | 0.419 to 0.498 | 0.840 | 0.901 | 0.562 | 0.420 |

Reading: with all 55 products in a store able to affect each other, the
item-to-item part makes the forecast worse in every version, and makes it
vary more from seed to seed. It is not kept in this form.
