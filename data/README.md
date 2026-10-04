# Data

Raw files go in `data/raw/<dataset>/` and are not committed. Their checksums
are committed in `data/manifests/<dataset>.json`.

## Adding a dataset

1. Download the files from the source named in `configs/datasets/<dataset>.yaml`
   under that source's own terms of use.
2. Put them, unchanged, in `data/raw/<dataset>/`.
3. Run `make data-register DATASET=<dataset>` and commit the manifest it writes.

After that, `make data-verify DATASET=<dataset>` confirms on any machine that
the files are byte-identical to the ones used here.

## Datasets

| Dataset | Folder | Levers recorded |
|---|---|---|
| M5 (Walmart) | `data/raw/m5/` | price |
| M5 published results | `data/raw/m5_reference/` | used only to check our scoring |
| Dominick's Finer Foods | `data/raw/dominicks/` | price, promotion, cost |
| Breakfast at the Frat | `data/raw/breakfast_at_the_frat/` | price, promotion, display, feature |

The files each loader expects are listed in `configs/datasets/`.

## M5 files and where they come from

From the Kaggle competition "M5 Forecasting - Accuracy": `calendar.csv`,
`sell_prices.csv`, `sales_train_evaluation.csv`.

From the organisers' public folder, linked from
github.com/Mcompetitions/M5-methods:

- `Dataset/sales_test_evaluation.csv` goes in `data/raw/m5/`. It holds the
  actual sales for the final 28-day test period.
- These go in `data/raw/m5_reference/`, under the names in
  `data/manifests/m5_reference.json`: the published weights, the published
  scores per level, and the organisers' forecast files for two benchmarks and
  the winning entry. `tests/test_m5_reference.py` uses them to confirm that
  our WRMSSE gives the published numbers.

No public dataset used here records shelf space. Range changes are visible
only as items entering and leaving stores.

`synthetic` is generated in code for tests and the smoke run. It is never
used for a reported result.
