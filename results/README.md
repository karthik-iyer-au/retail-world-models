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

Published scores on the official test period, from the organisers' files
(`data/raw/m5_reference/`): seasonal naive 0.847, best statistical benchmark
(exponential smoothing, bottom-up) 0.671, 50th-ranked entry 0.576,
10th-ranked entry 0.547, winning entry 0.520.

Notes:

- The LightGBM settings were fixed before the test-period run and were not
  tuned on either window.
- In the April to May window the LightGBM forecasts are 7.7% below actual
  sales in total. In the test period they are 2.1% below.

| Run | Folder |
|---|---|
| Seasonal naive, April to May | `20261004T045330_m5_validation_seasonal_naive_a67817a2` |
| LightGBM, April to May | `20261004T034345_m5_validation_lightgbm_11b5432b` |
| Seasonal naive, test period | `20261004T045417_m5_test_seasonal_naive_54f6d687` |
| LightGBM, test period | `20261004T041739_m5_test_lightgbm_9d232fa7` |

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
