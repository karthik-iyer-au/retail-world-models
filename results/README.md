# Results

Each run writes `results/<time>_<name>_<config hash>/` with `manifest.json`,
`metrics.json` and `forecasts.csv`. Run folders are not committed.

A run that is quoted in the paper is copied to `results/promoted/` (manifest
and metrics only) and committed, so every number in the paper points to a
manifest in git.
