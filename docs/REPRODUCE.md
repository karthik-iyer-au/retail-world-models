# How to reproduce a result

1. Find the run's `manifest.json` (under `results/promoted/` for paper numbers).
2. Check out the commit in `git.commit`.
3. Install the package versions listed in `packages`.
4. Download the data with `python -m rwm.data.fetch <name>`. It ends by checking every file against the registered checksums and must pass.
5. Save the `config` block as a YAML file and run
   `make run CONFIG=<that file>`.
6. The new `metrics.json` should match the original.

## What a strict run guarantees

`make run` uses `--strict`. It refuses to start if:

- the working tree has uncommitted changes, or
- the raw data has no registered manifest or does not match it.

So a strict result is always tied to one commit, one config and one exact set
of data files.

## Known limits

- Package versions are recorded per run, not pinned in advance.
- GPU runs can differ slightly between hardware even with a fixed seed. Each manifest records the device used.

## Running on Colab

Open `notebooks/colab_run.ipynb` in Colab, set the job file and commit in its first cell, and run all cells. A job file in `configs/jobs/` names the data to fetch and the runs to make. The notebook only calls the repository's commands and writes each run folder to Google Drive.

## Running on Modal

`cloud/modal_app.py` runs a job file on Modal with every run on its own
machine at the same time, so a job takes about as long as its longest run
rather than the sum of all of them. Like the Colab notebook, it holds no model
or data code: it checks out one commit and calls `rwm.experiments.job`.

```
modal run cloud/modal_app.py --job configs/jobs/paper1_revision_m5.yaml --list   # plan only
modal run cloud/modal_app.py --job configs/jobs/paper1_revision_m5.yaml --out paper1_revision
modal volume get rwm-results paper1_revision results/paper1_revision_runs
```

- **Committed code only.** The launcher refuses to start if the working tree
  has changes or the commit is not on GitHub. Each machine checks out that
  commit, so every run is strict and its manifest names the commit.
- **Machines.** Models that use a GPU get an L4 (`--gpu A10G` or `--gpu A100`
  to change) and 8 processor cores; LightGBM and the naive models get 16 cores
  and no GPU. M5 and Favorita runs get 64 GB of memory, the others 16 GB
  (`src/rwm/experiments/plan.py`).
- **Data.** One machine fetches the job's data into the `rwm-data` volume
  first. Files already there are kept, and every run checks them against
  `data/manifests/`.
- **Restarts.** Training saves are written to the `rwm-results` volume every
  five minutes. A machine that is taken away is retried up to twice and goes
  on from its last save; runs already saved are never repeated.
- **Part of a job.** `--only 3,7` runs those entries of the `--list` output.

One-time setup:

1. `pip install modal` and `modal setup` (opens a browser to sign in).
2. `modal secret create kaggle KAGGLE_API_TOKEN=<token>` for M5 and Favorita.
3. Breakfast at the Frat has no public download. Upload its files once:
   `modal volume put rwm-data <folder with the files> raw/breakfast_at_the_frat`.

Run times on Modal differ from the Colab times in the papers (`tab_compute`)
because the machines differ; each manifest records its machine. GPU results
can differ slightly between GPU types, so runs compared in one table should
use one GPU type.
