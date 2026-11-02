"""Run research-loop tries on rented Vast.ai GPUs, with Modal as the base.

  modal deploy cloud/vast_app.py                       # a separate app: deploying it never stops
                                                       # the loops of cloud/modal_app.py
  python cloud/start_vast.py configs/research/<spec>.yaml     # start a loop
  python cloud/start_vast.py --config <config.yaml> --out <folder>   # one run (a test)

Why: a five-copy Favorita run takes about 25 hours of an L4 on Modal; an RTX
4090 on Vast.ai is faster and costs a fraction as much per hour
(docs/PAPER1_REVISION_PLAN.md, study R). Only public or licensed research data
goes to Vast (M5, Dominick's, Favorita, Breakfast at the Frat); Woolworths data
never does.

How a try runs:
1. `vast_try` (a small Modal machine) reads back a saved run of exactly this
   config from the `rwm-results` volume, as on Modal, and stops there if it finds one.
2. Otherwise it rents the cheapest machine that matches the query (an RTX 4090,
   enough memory, CUDA 13, verified host, reliability over 0.98) and starts the
   worker script below on it, passing the config, the commit and a one-purpose
   token for the bridge.
3. The worker checks out the commit, installs the repository with the pinned
   torch, downloads the data set's files from the `bridge` endpoint (which
   serves the `rwm-data` volume to holders of the token), runs the config
   strictly (checksums verified, clean tree, manifest written with the GPU it ran on),
   and uploads the run folder to the bridge, which writes it to `rwm-results`.
4. `vast_try` sees the run on the volume, destroys the machine, and returns the
   metrics. A machine that fails or stops is destroyed and the try is started
   once more on another machine.

Secrets (Modal): `vast` (VAST_API_KEY) and `rwm-bridge` (BRIDGE_TOKEN). Neither
is printed; the bridge token is the only secret given to a rented machine.
"""
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_URL = "https://github.com/karthik-iyer-au/retail-world-models"
DATA = "/vol/data"
RESULTS = "/vol/results"
CHECKOUT = "/root/repo"
TORCH = "2.14.1"  # the version the Modal runs used, recorded in their manifests
MAX_MACHINES = 6  # rented machines at once
RUN_HOURS = 40  # a try still going after this long is stopped
POLL_S = 120

# Machines: one RTX 4090, CUDA 13 drivers (for torch 2.14 cu130), a verified
# host with reliability above 0.98, and memory for the data set.
QUERY = ("gpu_name=RTX_4090 num_gpus=1 cuda_vers>=13.0 reliability>0.98 verified=true rentable=true "
         "disk_space>=120 inet_down>=300 cpu_cores_effective>=8")
MEMORY_GB = {"favorita": 96, "m5": 64}
DEFAULT_MEMORY_GB = 48

app = modal.App("rwm-vast")
data_volume = modal.Volume.from_name("rwm-data")
results_volume = modal.Volume.from_name("rwm-results")
bridge_secret = modal.Secret.from_name("rwm-bridge")
vast_secret = modal.Secret.from_name("vast")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("fastapi[standard]", "vastai", "pyyaml").env({"PYTHONUNBUFFERED": "1"}))

WORKER = r"""#!/bin/bash
# Runs one config on a rented machine; written by cloud/vast_app.py.
set -uo pipefail
exec > >(tee -a /work.log) 2>&1
fail() { echo "FAILED: $1"; curl -fsS -X PUT -H "Authorization: Bearer $BRIDGE_TOKEN" --data-binary @/work.log "$BRIDGE_URL/failed/$OUT/$RUN_NAME" || true; sleep infinity; }
apt-get update -qq && apt-get install -y -qq git curl p7zip-full >/dev/null || fail "apt"
nvidia-smi || fail "no GPU"
mkdir -p /root/repo && cd /root/repo && git init -q && git fetch -q --depth 1 "$REPO_URL" "$COMMIT" && git checkout -q FETCH_HEAD || fail "checkout"
pip install -q "torch==$TORCH" && pip install -q -e ".[trees,deep]" || fail "install"
mkdir -p "data/raw/$DATASET"
for f in $(curl -fsS -H "Authorization: Bearer $BRIDGE_TOKEN" "$BRIDGE_URL/data/$DATASET"); do
  curl -fsS -H "Authorization: Bearer $BRIDGE_TOKEN" -o "data/raw/$DATASET/$f" "$BRIDGE_URL/data/$DATASET/$f" || fail "data $f"
done
mkdir -p /out && python - <<'EOF' || fail "run"
import base64, json, os
from pathlib import Path
from rwm.experiments.run import run
run(json.loads(base64.b64decode(os.environ["CONFIG_B64"])), strict=True, out_root=Path("/out"))
EOF
rm -rf /out/_partial
tar czf /run.tgz -C /out . || fail "tar"
curl -fsS -X PUT -H "Authorization: Bearer $BRIDGE_TOKEN" --data-binary @/run.tgz "$BRIDGE_URL/results/$OUT" || fail "upload"
echo "DONE"
sleep infinity
"""


@app.function(image=image, volumes={DATA: data_volume, RESULTS: results_volume}, secrets=[bridge_secret],
              timeout=3600, cpu=2, memory=4096)
@modal.asgi_app()
def bridge():
    """Data out to, and run folders in from, rented machines that hold the token."""
    import io
    import tarfile

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, PlainTextResponse

    web = FastAPI()
    token = os.environ["BRIDGE_TOKEN"]

    def check(request: Request):
        if request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(status_code=403)

    def safe(*parts: str) -> Path:
        if any(".." in p or p.startswith("/") for p in parts):
            raise HTTPException(status_code=400)
        return Path(*parts)

    @web.get("/data/{dataset}")
    def listing(dataset: str, request: Request):
        check(request)
        data_volume.reload()
        folder = Path(DATA, "raw") / safe(dataset)
        return PlainTextResponse("\n".join(sorted(p.name for p in folder.iterdir() if p.is_file())))

    @web.get("/data/{dataset}/{name}")
    def data(dataset: str, name: str, request: Request):
        check(request)
        return FileResponse(Path(DATA, "raw") / safe(dataset, name))

    @web.put("/results/{out:path}")
    async def results(out: str, request: Request):
        check(request)
        body = await request.body()
        dest = Path(RESULTS) / safe(out)
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as t:
            t.extractall(dest, filter="data")
        results_volume.commit()
        return {"saved": len(body)}

    @web.put("/failed/{out:path}")
    async def failed(out: str, request: Request):
        check(request)
        log = Path(RESULTS) / "_vast_failed" / safe(out.replace("/", "__") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(await request.body())
        results_volume.commit()
        return {"saved": True}

    return web


def _vast(*args: str) -> str:
    key = Path.home() / ".config" / "vastai" / "vast_api_key"
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(os.environ["VAST_API_KEY"])
        key.chmod(0o600)
    return subprocess.run(["vastai", *args, "--raw"], check=True, capture_output=True, text=True).stdout


def _finished(folder: Path, config: dict) -> Path | None:
    """Same rule as rwm.experiments.job.finished_run, with the same config hash."""
    import hashlib

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    tag = f"_{config['name']}_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
    for f in sorted(folder.glob(f"*{tag}")):
        if (f / "metrics.json").exists() and (f / "manifest.json").exists():
            return f
    return None


def _offer(memory_gb: int, exclude: set) -> dict:
    offers = json.loads(_vast("search", "offers", f"{QUERY} cpu_ram>={memory_gb}", "-o", "dph_total"))
    for o in offers:
        if o["machine_id"] not in exclude:
            return o
    raise RuntimeError("no machine matches the query")


@app.function(image=image, volumes={RESULTS: results_volume}, secrets=[vast_secret, bridge_secret],
              timeout=24 * 3600, cpu=0.25, memory=1024, max_containers=MAX_MACHINES)
def vast_try(config: dict, commit: str, out: str) -> dict | None:
    """One try on a rented machine; returns its metrics, or None if it failed twice."""
    folder = Path(RESULTS) / out
    results_volume.reload()
    done = _finished(folder, config)
    if done:
        return json.loads((done / "metrics.json").read_text())
    dataset = config["dataset"]["name"]
    url = modal.Function.from_name("rwm-vast", "bridge").get_web_url()
    env = {"REPO_URL": REPO_URL, "COMMIT": commit, "TORCH": TORCH, "DATASET": dataset, "OUT": out,
           "RUN_NAME": config["name"], "BRIDGE_URL": url, "BRIDGE_TOKEN": os.environ["BRIDGE_TOKEN"],
           "CONFIG_B64": base64.b64encode(json.dumps(config).encode()).decode(),
           "WORKER_B64": base64.b64encode(WORKER.encode()).decode()}
    tried: set = set()
    for attempt in range(2):
        offer = _offer(MEMORY_GB.get(dataset, DEFAULT_MEMORY_GB), tried)
        tried.add(offer["machine_id"])
        made = json.loads(_vast(
            "create", "instance", str(offer["id"]), "--image", "python:3.12-slim", "--disk", "120",
            "--label", f"rwm {config['name']}"[:60],
            "--env", " ".join(f"-e {k}={v}" for k, v in env.items()),
            "--onstart-cmd", "echo $WORKER_B64 | base64 -d > /worker.sh && bash /worker.sh"))
        iid = made["new_contract"]
        print(f"{config['name']}: machine {iid} ({offer['geolocation']}, US${offer['dph_total']:.3f}/h)", flush=True)
        started = time.time()
        try:
            while time.time() - started < RUN_HOURS * 3600:
                time.sleep(POLL_S)
                results_volume.reload()
                done = _finished(folder, config)
                if done:
                    return json.loads((done / "metrics.json").read_text())
                if (Path(RESULTS) / "_vast_failed" / f"{out.replace('/', '__')}__{config['name']}.log").exists():
                    print(f"{config['name']}: failed on machine {iid}", flush=True)
                    break
                state = json.loads(_vast("show", "instance", str(iid)))
                if state.get("actual_status") in ("exited", "offline") or state.get("intended_status") == "stopped":
                    print(f"{config['name']}: machine {iid} {state.get('actual_status')}", flush=True)
                    break
        finally:
            gone = subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, text=True)  # -y: no prompt
            print(f"{config['name']}: machine {iid} destroyed ({gone.returncode})", flush=True)
    return None


@app.function(image=image, volumes={RESULTS: results_volume}, timeout=24 * 3600, cpu=0.5, memory=2048)
def vast_loop(spec: str, commit: str, max_rounds: int) -> dict:
    """The research loop (src/rwm/research/loop.py) with every try on a rented machine."""
    subprocess.run(["git", "init", "-q", CHECKOUT], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "fetch", "-q", "--depth", "1", REPO_URL, commit], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "checkout", "-q", "FETCH_HEAD"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", CHECKOUT], check=True)
    sys.path.insert(0, f"{CHECKOUT}/src")
    os.chdir(CHECKOUT)
    from rwm.research import loop

    s = loop.load_spec(spec)
    out = f"research/{s['name']}"

    def evaluate(configs):
        calls = [vast_try.spawn(c, commit, out) for c in configs]
        got = []
        for call in calls:
            try:
                got.append(call.get())
            except Exception as e:
                print(f"a try failed: {e}", flush=True)
                got.append(None)
        return got

    class SavedLog(loop.Log):
        def write(self, row):
            results_volume.reload()
            super().write(row)
            print(row, flush=True)
            results_volume.commit()

    best = loop.search(s, evaluate, SavedLog(Path(RESULTS) / out / "log.jsonl"), max_rounds=max_rounds or None)
    return {k: best[k] for k in ("kept", "score", "held_out")}
