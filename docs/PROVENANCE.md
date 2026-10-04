# Provenance

## Statement

All code in this repository was written for this repository. It contains no
code, data, trained weights, configuration or results from any employer or
client of the author. Every dataset used is public and is obtained from its
original source under that source's terms.

## Data sources

| Dataset | Source | Access |
|---|---|---|
| M5 | Kaggle, "M5 Forecasting - Accuracy" | Free with a Kaggle account |
| Dominick's Finer Foods | Kilts Center for Marketing, Chicago Booth | Free for academic research |
| Breakfast at the Frat | dunnhumby source files | Free with registration |

Checksums of the exact files used are in `data/manifests/`.

## Prior-work code

One row per method.

| Method | Paper | Code source | Licence | Version or commit | Changes made |
|---|---|---|---|---|---|
| Seasonal naive, recent average | Standard benchmarks | Written here | n/a | n/a | n/a |

## Author's own prior published work

The evaluation approach (leakage-resistant splits, sanity checks on
explanations) follows the author's earlier work on molecular representation
learning (M-JEPA, github.com/Kar488/M-JEPA). The method carries over. No
weights or data do.
