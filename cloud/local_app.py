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
import time
from pathlib import Path

import modal

RESULTS = "/vol/results"
QUEUE = "rwm-local-jobs"
STATE = "rwm-local-state"

app = modal.App("rwm-local")
results_volume = modal.Volume.from_name("rwm-results")
bridge_secret = modal.Secret.from_name("rwm-bridge")
image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]").env({"PYTHONUNBUFFERED": "1"})


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
