"""A job queue for Karthik's own machine (the NUC with the RTX 5090), with Modal as the base.

  modal deploy cloud/local_app.py                                    # a separate app; deploying it
                                                                     # never stops other loops
  python cloud/start_local.py --config <config.yaml> --out <folder>  # queue one run
  python cloud/start_local.py status                                 # queue and recent jobs
  bash cloud/nuc_worker.sh                                           # on the NUC, in Ubuntu

Why: the NUC's RTX 5090 costs nothing per hour. Whole pieces of work go there
(a new loop, or the final runs of kept settings), never the second half of a
loop that started on Modal, because a loop's keep bar is measured on one kind
of machine.

How a job runs:
1. `start_local.py` puts {config, commit, out} on the `rwm-local-jobs` queue.
2. The worker on the NUC asks `jobs` for the next one (POST /claim). A job
   whose run is already saved on `rwm-results` is skipped there.
3. The worker checks out the commit, downloads the data set from the
   `rwm-vast` bridge (the same token, the same `rwm-data` volume), runs the
   config strictly, and uploads the run folder to the bridge, which writes it
   to `rwm-results`, exactly as a rented Vast machine does.
4. It reports the end (POST /finish) and asks for the next job.

Only public or licensed research data goes through the bridge; Woolworths
data stays where it is. The bridge token (Modal secret `rwm-bridge`) is the
only secret the NUC holds, in ~/rwm/bridge_token, written there by Karthik.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

RESULTS = "/vol/results"
REPO_URL = "https://github.com/karthik-iyer-au/retail-world-models"
CHECKOUT = "/root/repo"
POLL_S = 120
QUEUE = "rwm-local-jobs"
STATE = "rwm-local-state"

app = modal.App("rwm-local")
results_volume = modal.Volume.from_name("rwm-results")
bridge_secret = modal.Secret.from_name("rwm-bridge")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("fastapi[standard]", "pyyaml").env({"PYTHONUNBUFFERED": "1"}))


def job_id(job: dict) -> str:
    return f"{job['out']}/{job['config']['name']}"


def _finished(folder: Path, config: dict) -> Path | None:
    """Same rule as rwm.experiments.job.finished_run, with the same config hash."""
    import hashlib

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    tag = f"_{config['name']}_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
    for f in sorted(folder.glob(f"*{tag}")):
        if (f / "metrics.json").exists() and (f / "manifest.json").exists():
            return f
    return None


@app.function(image=image, volumes={RESULTS: results_volume}, secrets=[bridge_secret], cpu=0.25, memory=512)
@modal.asgi_app()
def jobs():
    from fastapi import FastAPI, HTTPException, Request, Response

    web = FastAPI()
    token = os.environ["BRIDGE_TOKEN"]
    queue = modal.Queue.from_name(QUEUE, create_if_missing=True)
    state = modal.Dict.from_name(STATE, create_if_missing=True)

    def check(request: Request):
        if request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(status_code=403)

    @web.post("/claim")
    async def claim(request: Request):
        check(request)
        worker = (await request.json()).get("worker", "?")
        while True:
            job = queue.get(block=False)
            if job is None:
                return Response(status_code=204)
            results_volume.reload()
            if _finished(Path(RESULTS) / job["out"], job["config"]):
                state[job_id(job)] = {"status": "already saved", "at": time.time()}
                continue
            state[job_id(job)] = {"status": "running", "worker": worker, "at": time.time(), "commit": job["commit"]}
            return job

    @web.post("/finish")
    async def finish(request: Request):
        check(request)
        body = await request.json()
        state[body["id"]] = {"status": "done" if body.get("ok") else "failed", "at": time.time(),
                             "worker": body.get("worker", "?"), "seconds": body.get("seconds")}
        return {"recorded": True}

    return web


@app.function(image=image, volumes={RESULTS: results_volume}, timeout=24 * 3600, cpu=0.5, memory=2048)
def local_loop(spec: str, commit: str, max_rounds: int) -> dict:
    """The research loop (src/rwm/research/loop.py) with every try run on the NUC.

    Each try is put on the queue once (a loop restarted after its daily limit
    finds its tries already queued or running and waits for them) and read back
    from the results volume when the NUC has saved it. A try the NUC reports as
    failed is queued once more; a second failure counts as a failed try."""
    subprocess.run(["git", "init", "-q", CHECKOUT], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "fetch", "-q", "--depth", "1", REPO_URL, commit], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "checkout", "-q", "FETCH_HEAD"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", CHECKOUT], check=True)
    sys.path.insert(0, f"{CHECKOUT}/src")
    os.chdir(CHECKOUT)
    from rwm.research import loop

    s = loop.load_spec(spec)
    out = f"research/{s['name']}"
    queue = modal.Queue.from_name(QUEUE, create_if_missing=True)
    state = modal.Dict.from_name(STATE, create_if_missing=True)

    def evaluate(configs):
        jobs = [{"config": c, "commit": commit, "out": out} for c in configs]
        tries = {job_id(j): 0 for j in jobs}
        results_volume.reload()
        for j in jobs:
            st = state.get(job_id(j), {}).get("status")
            if not _finished(Path(RESULTS) / out, j["config"]) and st not in ("queued", "running"):
                queue.put(j)
                state[job_id(j)] = {"status": "queued", "at": time.time()}
        got = {}
        while len(got) < len(jobs):
            time.sleep(POLL_S)
            results_volume.reload()
            for j in jobs:
                k = job_id(j)
                if k in got:
                    continue
                done = _finished(Path(RESULTS) / out, j["config"])
                if done:
                    got[k] = json.loads((done / "metrics.json").read_text())
                elif state.get(k, {}).get("status") == "failed":
                    tries[k] += 1
                    if tries[k] >= 2:
                        print(f"{k} failed twice on the NUC", flush=True)
                        got[k] = None
                    else:
                        queue.put(j)
                        state[k] = {"status": "queued", "at": time.time()}
        return [got[job_id(j)] for j in jobs]

    class SavedLog(loop.Log):
        def write(self, row):
            results_volume.reload()
            super().write(row)
            print(row, flush=True)
            results_volume.commit()

    best = loop.search(s, evaluate, SavedLog(Path(RESULTS) / out / "log.jsonl"), max_rounds=max_rounds or None)
    return {k: best[k] for k in ("kept", "score", "held_out")}
