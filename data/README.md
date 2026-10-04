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
| Dominick's Finer Foods | `data/raw/dominicks/` | price, promotion, cost |
| Breakfast at the Frat | `data/raw/breakfast_at_the_frat/` | price, promotion, display, feature |

The files each loader expects are listed in `configs/datasets/`.

No public dataset used here records shelf space. Range changes are visible
only as items entering and leaving stores.

`synthetic` is generated in code for tests and the smoke run. It is never
used for a reported result.
