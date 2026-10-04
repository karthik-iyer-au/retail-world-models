# How to reproduce a result

1. Find the run's `manifest.json` (under `results/promoted/` for paper numbers).
2. Check out the commit in `git.commit`.
3. Install the package versions listed in `packages`.
4. Place the raw data and run `make data-verify DATASET=<name>`. It must pass.
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
- GPU runs can differ slightly between hardware even with a fixed seed.
