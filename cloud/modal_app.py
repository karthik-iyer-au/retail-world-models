"""Run a job file on Modal, every run on its own machine, all at once.

  modal run cloud/modal_app.py --job configs/jobs/paper1_revision_m5.yaml --out paper1_revision
  modal run cloud/modal_app.py --job <job file> --list        # what would run, and where; nothing starts

This file holds no model or data code. Like the Colab notebook, it checks out
one commit of this repository and calls the repository's own commands:

1. The commit is the local HEAD. The launcher refuses to start unless the
   working tree is clean and that commit is on GitHub, so every run is strict
   and tied to code anyone can check out.
2. One machine fetches the job's data into the `rwm-data` volume, once.
   Files already there are kept, and every run verifies their checksums.
3. Each run in the job (each seed counted separately) gets its own machine:
   a GPU for models that use one, a many-core processor for LightGBM and the
   naive models. Runs already saved in the output folder are not repeated.
4. Results go to the `rwm-results` volume, one folder per run with its
   manifest, as on Colab. Training saves are written to the volume every few
   minutes, so a machine that is taken away resumes from its last save.

Get the results with
  modal volume get rwm-results <out> results/<out>

One-time setup (see docs/REPRODUCE.md): `pip install modal`, `modal setup`,
a Kaggle token stored as the Modal secret `kaggle` (key KAGGLE_API_TOKEN) for
M5 and Favorita, and the Breakfast at the Frat files, which have no public
download, uploaded once with
  modal volume put rwm-data <folder with the Frat files> raw/breakfast_at_the_frat
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal
import yaml

REPO_URL = "https://github.com/karthik-iyer-au/retail-world-models"
LOCAL_REPO = Path(__file__).resolve().parents[1]
DATA = "/vol/data"
RESULTS = "/vol/results"
CHECKOUT = "/root/repo"

# GPU for deep and foundation models. The L4 is about two to three times a
# Colab T4 on this work and has 24 GB; change with --gpu (for example A10G, A100).
DEFAULT_GPU = "L4"
GPU_CPUS = 8  # Colab gave 2; data preparation and batching are processor-bound
CPU_CPUS = 16  # LightGBM uses every core it is given
TIMEOUT_S = 12 * 3600
SAVE_EVERY_S = 300

app = modal.App("retail-world-models")
data_volume = modal.Volume.from_name("rwm-data", create_if_missing=True)
results_volume = modal.Volume.from_name("rwm-results", create_if_missing=True)

# Dependencies only, from pyproject.toml, so the image is rebuilt only when
# they change. The code itself is checked out at the pinned commit in each run.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "p7zip-full")
    .pip_install_from_pyproject(
        str(LOCAL_REPO / "pyproject.toml"),
        optional_dependencies=["dev", "trees", "deep", "fetch", "foundation"],
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


def _checkout(commit: str) -> None:
    """The repository at exactly `commit`, installed without touching its dependencies."""
    subprocess.run(["git", "init", "-q", CHECKOUT], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "fetch", "-q", "--depth", "1", REPO_URL, commit], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "checkout", "-q", "FETCH_HEAD"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", CHECKOUT], check=True)
    # Raw data lives on the volume and is linked in at data/raw. The repository
    # ignores data/raw/ as a folder, not a link, so the link is excluded locally
    # (in .git/info/exclude, not the repository) and strict runs see a clean tree.
    raw = Path(CHECKOUT) / "data" / "raw"
    Path(DATA, "raw").mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raw.symlink_to(Path(DATA, "raw"))
    with open(Path(CHECKOUT) / ".git" / "info" / "exclude", "a") as f:
        f.write("/data/raw\n")
    status = subprocess.run(["git", "-C", CHECKOUT, "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError(f"checkout is not clean:\n{status.stdout}")


def _job(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "rwm.experiments.job", *args], cwd=CHECKOUT)


@app.function(image=image, volumes={DATA: data_volume}, secrets=[modal.Secret.from_name("kaggle")],
              cpu=4, memory=16384, timeout=4 * 3600)
def fetch(job, commit: str) -> None:
    """The data a job names: a job file in the repository, or a job's dataset list itself."""
    _checkout(commit)
    if isinstance(job, str):
        _job("--job", job, "--fetch-only")
    else:
        sys.path.insert(0, f"{CHECKOUT}/src")
        os.chdir(CHECKOUT)
        from rwm.experiments.job import fetch_datasets

        fetch_datasets(job)
    data_volume.commit()


def _dataset_job(config: dict) -> dict:
    """The datasets entry of a job for one config: its data set, and its categories for Dominick's."""
    d = config["dataset"]
    entry = {"name": d["name"]}
    if d["name"] == "dominicks" and d.get("params", {}).get("categories"):
        entry["codes"] = list(d["params"]["categories"])
    return {"datasets": [entry]}


@app.function(image=image, volumes={DATA: data_volume, RESULTS: results_volume},
              timeout=TIMEOUT_S, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_entry(job_file: str, index: int, commit: str, out: str) -> int:
    """One run of the job, on this machine. Returns the run's exit code."""
    _checkout(commit)
    data_volume.reload()
    stop = threading.Event()

    def keep_saving():  # training saves reach the volume, so a retry resumes from them
        while not stop.wait(SAVE_EVERY_S):
            try:
                results_volume.commit()
            except Exception as e:  # a missed save only costs time on a retry
                print(f"could not save to the volume: {e}", flush=True)

    saver = threading.Thread(target=keep_saving, daemon=True)
    saver.start()
    try:
        done = _job("--job", job_file, "--no-fetch", "--index", str(index), "--out", f"{RESULTS}/{out}")
    finally:
        stop.set()
        results_volume.commit()
    return done.returncode


@app.function(image=image, volumes={DATA: data_volume, RESULTS: results_volume},
              timeout=TIMEOUT_S, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_config(config: dict, commit: str, out: str) -> dict | None:
    """One run of a config made by the research loop, strict, at `commit`.
    Returns its metrics, or None if it failed. A run already saved for exactly
    this config is read back, not repeated."""
    import json
    import traceback

    _checkout(commit)
    data_volume.reload()
    sys.path.insert(0, f"{CHECKOUT}/src")
    from rwm.experiments.job import finished_run
    from rwm.experiments.run import run

    folder = Path(RESULTS) / out
    folder.mkdir(parents=True, exist_ok=True)
    try:
        done = finished_run(folder, config) or run(config, strict=True, out_root=folder)
    except Exception:
        traceback.print_exc()
        return None
    finally:
        results_volume.commit()
    return json.loads((done / "metrics.json").read_text())


def _git_state() -> tuple[str, bool] | None:
    """The local commit and whether the tree has changes, or None inside a machine (no checkout there)."""
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(LOCAL_REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# Taken when the launch starts, before the image is built, so edits made while
# it builds cannot change which code the runs are checked against.
LAUNCH = _git_state()


def _local_commit() -> str:
    git = lambda *a: subprocess.run(["git", "-C", str(LOCAL_REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    if LAUNCH is None or LAUNCH[1]:
        raise SystemExit("Commit your changes first: cloud runs are strict and use the committed code only.")
    commit = LAUNCH[0]
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise SystemExit("The code changed while the launch was starting; launch again from a clean tree.")
    if commit not in git("ls-remote", REPO_URL):
        git("fetch", "-q", REPO_URL)
        if not git("branch", "-r", "--contains", commit):
            raise SystemExit(f"Push {commit[:8]} to GitHub first: each machine checks the code out from there.")
    return commit


def _summary(job_file: str, out: str, configs: list[dict]) -> None:
    """One line per run, read from the results volume, as the job runner prints them."""
    sys.path.insert(0, str(LOCAL_REPO / "src"))
    from rwm.experiments.job import summary_row
    from rwm.utils.hashing import sha256_obj

    print(f"\nSummary of {job_file} (average over each run's windows; lower is better)")
    folders = {e.path.split("/")[-1] for e in results_volume.listdir(out)}
    for config in configs:
        tag = f"_{config['name']}_{sha256_obj(config)[:8]}"
        match = sorted(f for f in folders if f.endswith(tag))
        if not match:
            print(f"{config['name']:<48} NOT SAVED, see that run's log")
            continue
        local = Path("/tmp") / "rwm_summary" / match[-1]
        local.mkdir(parents=True, exist_ok=True)
        with open(local / "metrics.json", "wb") as f:
            for chunk in results_volume.read_file(f"{out}/{match[-1]}/metrics.json"):
                f.write(chunk)
        print(summary_row(local))


@app.local_entrypoint()
def main(job: str, out: str = "runs", gpu: str = DEFAULT_GPU, list: bool = False, only: str = ""):
    """--job a job file; --out the folder on the results volume; --only a
    comma-separated list of run numbers from --list, to run part of a job."""
    sys.path.insert(0, str(LOCAL_REPO / "src"))
    from rwm.experiments import plan

    os.chdir(LOCAL_REPO)
    runs = plan.expand(plan.load(job))
    chosen = [int(i) for i in only.split(",")] if only else range(len(runs))
    for i in chosen:
        r = runs[i]
        where = gpu if plan.machine(r["config"]) == "gpu" else f"{CPU_CPUS} cores"
        print(f"{i:>3}  {r['config']['name']:<56} {where}, {plan.memory_gb(r['config'])} GB")
    if list:
        return

    commit = _local_commit()
    print(f"\nrunning commit {commit} on Modal; results in rwm-results/{out}")
    fetch.remote(job, commit)

    calls = []
    for i in chosen:
        config = runs[i]["config"]
        on_gpu = plan.machine(config) == "gpu"
        options = dict(gpu=gpu if on_gpu else None, cpu=GPU_CPUS if on_gpu else CPU_CPUS,
                       memory=plan.memory_gb(config) * 1024)
        calls.append((i, config, run_entry.with_options(**options).spawn(job, i, commit, out)))

    failed = []
    for i, config, call in calls:
        try:
            code = call.get()
        except Exception as e:
            code = f"error: {e}"
        if code != 0:
            failed.append(f"{i} {config['name']} ({code})")
    _summary(job, out, [runs[i]["config"] for i in chosen])
    if failed:
        print("\nDid not finish:\n  " + "\n  ".join(failed))


@app.local_entrypoint()
def research(spec: str, gpu: str = DEFAULT_GPU, max_rounds: int = 0):
    """The research loop (src/rwm/research/loop.py) with every try on its own
    machine. The log is written to results/research/<name>/log.jsonl, to be
    committed with the paper's records.

      modal run cloud/modal_app.py::research --spec configs/research/<name>.yaml
    """
    sys.path.insert(0, str(LOCAL_REPO / "src"))
    from rwm.experiments import plan
    from rwm.research import loop

    os.chdir(LOCAL_REPO)
    commit = _local_commit()
    s = loop.load_spec(spec)
    base = yaml.safe_load((LOCAL_REPO / s["base"]).read_text())
    fetch.remote(_dataset_job(base), commit)
    out = f"research/{s['name']}"

    def evaluate(configs):
        calls = []
        for c in configs:
            on_gpu = plan.machine(c) == "gpu"
            options = dict(gpu=gpu if on_gpu else None, cpu=GPU_CPUS if on_gpu else CPU_CPUS,
                           memory=plan.memory_gb(c) * 1024)
            calls.append(run_config.with_options(**options).spawn(c, commit, out))
        got = []
        for call in calls:
            try:
                got.append(call.get())
            except Exception as e:
                print(f"a try failed: {e}", flush=True)
                got.append(None)
        return got

    log = loop.Log(LOCAL_REPO / "results" / "research" / s["name"] / "log.jsonl")
    best = loop.search(s, evaluate, log, max_rounds=max_rounds or None)
    print(f"kept: {best['kept']}; score {best['score']:.4f}; held out {best['held_out']:.4f}")
