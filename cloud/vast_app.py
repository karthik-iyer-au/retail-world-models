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
# The app name comes from RWM_VAST_APP at deploy time (`RWM_VAST_APP=<name> modal deploy
# cloud/vast_app.py`), so a new version can be deployed beside one whose machines are mid-run.
# Since 4 November each loop watches all its machines from its own container (_manage), so an
# app needs one container per loop plus its bridge, whatever the number of machines.
APP_NAME = os.environ.get("RWM_VAST_APP", "rwm-vast")
MAX_MACHINES = 6 if APP_NAME == "rwm-vast" else 12  # rented machines at once
RUN_HOURS = 40  # a try still going after this long is stopped
POLL_S = 120
STALE_S = 15 * 60  # a try's marker older than this means its manager has stopped

# Machines: one RTX 4090, CUDA 13 drivers (for torch 2.14 cu130), a verified
# host with reliability above 0.98, and memory for the data set.
QUERY_GPU = ("gpu_name=RTX_4090 num_gpus=1 cuda_vers>=13.0 reliability>0.99 verified=true rentable=true "
         "disk_space>=120 inet_down>=300 cpu_cores_effective>=8")
# LightGBM uses no GPU, so its scores do not depend on the machine: the cheapest verified host
# with at least 16 processor cores (the 16 threads its configs ask for) will do. Reliability above
# 0.99 since 3 November: on cheaper hosts LightGBM segfaulted or read memory out of bounds.
QUERY_CPU = "num_gpus>=1 cpu_cores_effective>=16 reliability>0.99 verified=true rentable=true disk_space>=120 inet_down>=200"
CPU_MODELS = {"lightgbm_direct", "seasonal_naive", "recent_average"}  # as rwm.experiments.plan
MEMORY_GB = {"favorita": 96, "m5": 64}
DEFAULT_MEMORY_GB = 48

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("rwm-data")
results_volume = modal.Volume.from_name("rwm-results")
bridge_secret = modal.Secret.from_name("rwm-bridge")
vast_secret = modal.Secret.from_name("vast")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("fastapi[standard]", "vastai", "pyyaml").env({"PYTHONUNBUFFERED": "1", "RWM_VAST_APP": APP_NAME}))

WORKER = r"""#!/bin/bash
# Runs one config on a rented machine; written by cloud/vast_app.py.
set -uo pipefail
exec > >(tee -a /work.log) 2>&1
# on failure: send the log, then end the container, so vast_try sees the machine stop and destroys it
fail() { echo "FAILED: $1"; curl -fsS --retry 5 --retry-all-errors -X PUT -H "Authorization: Bearer $BRIDGE_TOKEN" --data-binary @/work.log "$BRIDGE_URL/failed/$OUT/$RUN_NAME" || true; exit 1; }
apt-get update -qq && apt-get install -y -qq git curl p7zip-full >/dev/null || fail "apt"
nvidia-smi || fail "no GPU"
mkdir -p /root/repo && cd /root/repo && git init -q
for t in 1 2 3 4 5; do git fetch -q --depth 1 "$REPO_URL" "$COMMIT" && break; sleep 30; done  # a host's network can be slow to come up
git checkout -q FETCH_HEAD || fail "checkout"
# hosts' links to the package index drop now and then (3 November: read timeouts, broken connections): retry
PIP="pip install -q --retries 10 --timeout 60"
# pretrained forecasters compile kernels at run time (Timestral through triton): they need a C compiler
if [ -n "${EXTRAS:-}" ]; then for t in 1 2 3; do apt-get update -qq && apt-get install -y -qq gcc >/dev/null && break; sleep 30; done; fi
for t in 1 2 3; do $PIP "torch==$TORCH" && $PIP -e ".[trees,deep${EXTRAS:+,$EXTRAS}]" && break; [ $t = 3 ] && fail "install"; sleep 30; done
python cloud/fetch_data.py "$BRIDGE_URL" "$DATASET" "data/raw/$DATASET" || fail "data"  # in parts: one long request is cut off
# training saves partway (about every 5% of steps) under /out/_partial; copy them to the results
# volume every five minutes and start from the last copy, so a lost machine costs minutes, not the run
mkdir -p /out
AUTHH="Authorization: Bearer $BRIDGE_TOKEN"
if curl -fsS -H "$AUTHH" -o /partial.tgz "$BRIDGE_URL/partial/$OUT/$RUN_NAME" 2>/dev/null; then
  tar xzf /partial.tgz -C /out && echo "resuming from the saved partway state"
elif [ -n "${BASE_NAME:-}" ] && curl -fsS -H "$AUTHH" -o /partial.tgz "$BRIDGE_URL/partial/$OUT/$BASE_NAME" 2>/dev/null; then
  # a piece of a split run starts from what the run saved before it was split
  tar xzf /partial.tgz -C /out && echo "resuming from the unsplit run's partway state"
fi
# the last run of a split run: every piece's finished fit, merged into one save folder
PIECE_LIST=${PIECES:-}
for P in ${PIECE_LIST//:/ }; do  # the list is joined with ":" (a space would end the value in Vast's env string)
  curl -fsS --retry 5 --retry-all-errors -H "$AUTHH" -o /piece.tgz "$BRIDGE_URL/partial/$OUT/$P" || fail "piece $P"
  # only that piece's own fit (key ...__o<window>m<copy>), so no stale copy overwrites another's
  rm -rf /piece && mkdir /piece && tar xzf /piece.tgz -C /piece || fail "piece $P"
  K=${P##*__o}; PO=${K%m*}; PM=${K#*m}
  find /piece -type f ! \( -name "origin${PO}_member${PM}_*" -o \( -name "origin${PO}_*" ! -name "*_member*" \) \) -delete
  [ -d /piece/_partial ] && cp -r /piece/_partial /out/ || fail "piece $P"
done
( while sleep 300; do
    [ -d /out/_partial ] && tar czf /partial.new --exclude='*.tmp' -C /out _partial && mv /partial.new /partial.up \
      && curl -fsS --retry 3 -X PUT -H "$AUTHH" --data-binary @/partial.up "$BRIDGE_URL/partial/$OUT/$RUN_NAME" >/dev/null
  done ) &
SAVER=$!
python - <<'EOF' || { kill $SAVER; fail "run"; }
import base64, json, os
from pathlib import Path
from rwm.experiments.run import run
run(json.loads(base64.b64decode(os.environ["CONFIG_B64"])), strict=True, out_root=Path("/out"))
EOF
kill $SAVER 2>/dev/null
if [ -n "${RWM_PIECE:-}" ]; then
  # a piece's finished fit is what the last run needs: save it, and keep it on the bridge
  # only this piece's own saves: copies of other fits it started from (a split run's earlier
  # saves) would overwrite those pieces' finished fits when the last run merges them (10 November)
  O=${RWM_PIECE%,*}; M=${RWM_PIECE#*,}
  find /out/_partial -type f ! \( -name "origin${O}_member${M}_*" -o \( -name "origin${O}_*" ! -name "*_member*" \) \) -delete
  tar czf /partial.up --exclude='*.tmp' -C /out _partial || fail "piece tar"
  curl -fsS --retry 5 --retry-all-errors -X PUT -H "$AUTHH" --data-binary @/partial.up "$BRIDGE_URL/partial/$OUT/$RUN_NAME" >/dev/null || fail "piece save"
fi
rm -rf /out/_partial
tar czf /run.tgz -C /out . || fail "tar"
curl -fsS --retry 5 --retry-all-errors -X PUT -H "Authorization: Bearer $BRIDGE_TOKEN" --data-binary @/run.tgz "$BRIDGE_URL/results/$OUT" || fail "upload"
if [ -z "${RWM_PIECE:-}" ]; then
  curl -fsS -X DELETE -H "$AUTHH" "$BRIDGE_URL/partial/$OUT/$RUN_NAME" >/dev/null || true
  for P in ${PIECE_LIST//:/ }; do curl -fsS -X DELETE -H "$AUTHH" "$BRIDGE_URL/partial/$OUT/$P" >/dev/null || true; done
fi
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
    from fastapi.responses import FileResponse, PlainTextResponse, Response

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

    @web.get("/size/{dataset}/{name}")
    def size(dataset: str, name: str, request: Request):
        check(request)
        return PlainTextResponse(str((Path(DATA, "raw") / safe(dataset, name)).stat().st_size))

    @web.get("/part/{dataset}/{name}")
    def part(dataset: str, name: str, start: int, length: int, request: Request):
        """One part of a file (cloud/fetch_data.py): a whole large file in one
        request outlasts Modal's limit on a web request."""
        check(request)
        if start < 0 or not 0 < length <= 256 << 20:
            raise HTTPException(status_code=400)
        with open(Path(DATA, "raw") / safe(dataset, name), "rb") as f:
            f.seek(start)
            return Response(f.read(length), media_type="application/octet-stream")

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

    def partial_file(out: str) -> Path:
        return Path(RESULTS) / "_vast_partial" / safe(out.replace("/", "__") + ".tgz")

    @web.put("/partial/{out:path}")
    async def partial_put(out: str, request: Request):
        """A run's partway training state (cloud/vast_app.py worker), replaced each time."""
        check(request)
        f = partial_file(out)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        tmp.write_bytes(await request.body())
        tmp.replace(f)
        results_volume.commit()
        return {"saved": True}

    @web.get("/partial/{out:path}")
    def partial_get(out: str, request: Request):
        check(request)
        results_volume.reload()
        f = partial_file(out)
        if not f.exists():
            raise HTTPException(status_code=404)
        return FileResponse(f)

    @web.delete("/partial/{out:path}")
    def partial_delete(out: str, request: Request):
        check(request)
        partial_file(out).unlink(missing_ok=True)
        results_volume.commit()
        return {"deleted": True}

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
    args = list(args)
    at = args.index("--entrypoint") if "--entrypoint" in args else len(args)  # --args must come last
    cmd = ["vastai", *args[:at], "--raw", *args[at:]]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _first_json(text: str) -> dict:
    """The JSON object in the CLI's output, which may have lines before it."""
    return json.loads(text[text.index("{"):])


def _finished(folder: Path, config: dict) -> Path | None:
    """Same rule as rwm.experiments.job.finished_run, with the same config hash."""
    import hashlib

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    tag = f"_{config['name']}_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
    for f in sorted(folder.glob(f"*{tag}")):
        if (f / "metrics.json").exists() and (f / "manifest.json").exists():
            return f
    return None


def _uses_gpu(model: dict) -> bool:
    """False when the model, and the model a five-copy average wraps, run on the processor only."""
    names = {model.get("name")}
    inner = model.get("params", {}).get("model")
    names.add(inner.get("name") if isinstance(inner, dict) else inner)
    return bool(names - {"seed_average", None} - CPU_MODELS)


def _offer(memory_gb: int, exclude: set, gpu: bool = True) -> dict:
    query = QUERY_GPU if gpu else QUERY_CPU
    offers = json.loads(_vast("search", "offers", f"{query} cpu_ram>={memory_gb}", "-o", "dph_total"))
    for o in offers:
        if o["machine_id"] not in exclude:
            return o
    raise RuntimeError("no machine matches the query")


BAD = Path(RESULTS) / "_vast_bad_machines.json"  # hosts where a run failed; never rented again


def _bad() -> set:
    return set(json.loads(BAD.read_text())) if BAD.exists() else set()


CODE_FAULTS = ("ModuleNotFoundError", "ImportError", "Failed to find C compiler")  # the image's fault, not the host's


def _host_fault(log: Path) -> bool:
    """Whether a failure should keep its host out of later rentals: not when the run failed for want of a
    package or compiler on the image (10 November 2026: such failures had excluded 32 of 33 offers)."""
    try:
        text = log.read_text(errors="replace")[-20000:]
    except OSError:
        return True
    return not any(f in text for f in CODE_FAULTS)


def _mark_bad(machine_id) -> None:
    results_volume.reload()
    BAD.write_text(json.dumps(sorted(_bad() | {machine_id})))
    results_volume.commit()


EXTRA_PACKAGES = {"timesfm": "foundation", "chronos": "foundation", "chronos2": "foundation", "timestral": "foundation",
                  "tft": "benchmarks"}  # the optional packages a model needs on the machine (pyproject.toml)


def _extras(model: dict) -> str:
    inner = model.get("params", {}).get("model")
    names = {model.get("name"), inner.get("name") if isinstance(inner, dict) else inner}
    return ",".join(sorted({EXTRA_PACKAGES[n] for n in names if n in EXTRA_PACKAGES}))


def _rent(config: dict, env: dict, dataset: str, tried: set, label: str | None = None) -> tuple[int, int]:
    """Rent the cheapest matching machine not tried yet and start the worker on it."""
    offer = _offer(MEMORY_GB.get(dataset, DEFAULT_MEMORY_GB), tried | _bad(), _uses_gpu(config["model"]))
    env = {**env, **({"EXTRAS": _extras(config["model"])} if _extras(config["model"]) else {})}
    tried.add(offer["machine_id"])
    made = _first_json(_vast(
        "create", "instance", str(offer["id"]), "--image", "python:3.12-slim", "--disk", "120",
        "--label", f"rwm {label or config['name']}"[:60],
        "--env", " ".join(f"-e {k}={v}" for k, v in env.items()),
        # the image's own command, not Vast's ssh start-up, which needs packages a slim image lacks
        "--entrypoint", "bash", "--args", "-c", "echo $WORKER_B64 | base64 -d > /worker.sh && bash /worker.sh"))
    iid = made["new_contract"]
    print(f"{label or config['name']}: machine {iid} ({offer['geolocation']}, US${offer['dph_total']:.3f}/h)", flush=True)
    return iid, offer["machine_id"]


@app.function(image=image, volumes={RESULTS: results_volume}, secrets=[vast_secret, bridge_secret],
              timeout=24 * 3600, cpu=0.25, memory=1024, max_containers=MAX_MACHINES)
def vast_try(config: dict, commit: str, out: str) -> dict | None:
    """One try on a rented machine; returns its metrics, or None if it failed twice."""
    folder = Path(RESULTS) / out
    results_volume.reload()
    done = _finished(folder, config)
    if done:
        return json.loads((done / "metrics.json").read_text())
    # A loop restarted after its daily time limit asks again for tries whose
    # machines are still working: wait for those instead of renting a second machine.
    marker = Path(RESULTS) / "_vast_running" / out.replace("/", "__") / f"{config['name']}.json"
    while marker.exists() and time.time() - json.loads(marker.read_text())["at"] < STALE_S:
        time.sleep(POLL_S)
        results_volume.reload()
        done = _finished(folder, config)
        if done:
            return json.loads((done / "metrics.json").read_text())

    def mark(iid) -> None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"at": time.time(), "machine": iid}))
        results_volume.commit()

    dataset = config["dataset"]["name"]
    url = modal.Function.from_name(APP_NAME, "bridge").get_web_url()  # this app's bridge, at this app's code
    env = {"REPO_URL": REPO_URL, "COMMIT": commit, "TORCH": TORCH, "DATASET": dataset, "OUT": out,
           "RUN_NAME": config["name"], "BRIDGE_URL": url, "BRIDGE_TOKEN": os.environ["BRIDGE_TOKEN"],
           "CONFIG_B64": base64.b64encode(json.dumps(config).encode()).decode(),
           "WORKER_B64": base64.b64encode(WORKER.encode()).decode()}
    # a manager that stopped without destroying its machine (a restart of the app) leaves the
    # machine working: carry on with that machine rather than start the run again
    adopt = None
    if marker.exists():
        old = json.loads(marker.read_text()).get("machine")
        try:
            if json.loads(_vast("show", "instance", str(old))).get("actual_status") in ("running", "loading"):
                adopt = old
        except Exception:
            pass
    tried: set = set()
    failed_log = Path(RESULTS) / "_vast_failed" / f"{out.replace('/', '__')}__{config['name']}.log"
    for attempt in range(3):
        # a failure log from an earlier machine would end this attempt at once (3 November): keep it under another name
        if failed_log.exists():
            failed_log.rename(failed_log.with_suffix(f".{int(time.time())}.log"))
            results_volume.commit()
        machine = None
        if adopt is not None and attempt == 0:
            iid = adopt
            print(f"{config['name']}: carrying on with machine {iid}", flush=True)
        else:
            iid, machine = _rent(config, env, dataset, tried)
        started = time.time()
        try:
            while time.time() - started < RUN_HOURS * 3600:
                mark(iid)
                time.sleep(POLL_S)
                results_volume.reload()
                done = _finished(folder, config)
                if done:
                    return json.loads((done / "metrics.json").read_text())
                if failed_log.exists():
                    print(f"{config['name']}: failed on machine {iid}", flush=True)
                    if machine is not None and _host_fault(failed_log):
                        _mark_bad(machine)
                    break
                state = json.loads(_vast("show", "instance", str(iid)))
                if state.get("actual_status") in ("exited", "offline") or state.get("intended_status") == "stopped":
                    print(f"{config['name']}: machine {iid} {state.get('actual_status')}", flush=True)
                    break
        finally:
            gone = subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, text=True)  # -y: no prompt
            marker.unlink(missing_ok=True)
            results_volume.commit()
            print(f"{config['name']}: machine {iid} destroyed ({gone.returncode})", flush=True)
    return None


def _destroy(iid) -> None:
    """Destroy a machine and check it is gone; a destroy call can fail on a passing API
    error, which left finished machines running with no one watching them (4 November)."""
    for attempt in range(5):
        subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, text=True)
        time.sleep(5)
        shown = subprocess.run(["vastai", "show", "instance", str(iid), "--raw"], capture_output=True, text=True)
        if shown.returncode != 0 or "{" not in shown.stdout:
            return
        try:
            if json.loads(shown.stdout[shown.stdout.index("{"):]).get("actual_status") in (None, "destroyed"):
                return
        except Exception:
            return
        time.sleep(20 * (attempt + 1))
    print(f"machine {iid} could not be destroyed", flush=True)


def _sha8(config: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:8]


def _members(config: dict) -> int:
    m = config["model"]
    return m.get("params", {}).get("members", 5) if m.get("name") == "seed_average" else 1


def _splittable(config: dict) -> bool:
    """A model on the graphics card (alone or as a five-copy average) over more than one window or copy."""
    return _uses_gpu(config["model"]) and config["evaluation"].get("n_origins", 1) * _members(config) > 1


def _manage(configs: list[dict], commit: str, out: str, limit: int, split: bool = False) -> list[dict | None]:
    """Run each config on a rented machine, at most `limit` machines at once, watched from
    this one container (4 November: one Modal container per machine reached the
    workspace's limit of 100 containers). A run already saved is read back; a run whose
    marker another manager is refreshing is waited for; a stale marker whose machine is
    still working is carried on with. A run that fails is retried on another host, up to
    three hosts, and a host where a run failed is never rented again. Machines are not
    destroyed if this container is stopped: their markers go stale and the next manager
    carries on with them.

    With `split` (9 November 2026) a five-copy average over several windows is run as one
    piece per (window, copy), each on its own machine, then one last run that loads every
    piece's finished fit and scores (rwm.experiments.run.piece_of); the last run's result
    is the run's result, identical to running it in one go (tests/test_split_run.py)."""
    folder = Path(RESULTS) / out
    url = modal.Function.from_name(APP_NAME, "bridge").get_web_url()
    st = {}

    def marker(name):
        return Path(RESULTS) / "_vast_running" / out.replace("/", "__") / f"{name}.json"

    def failed_log(name):
        return Path(RESULTS) / "_vast_failed" / f"{out.replace('/', '__')}__{name}.log"

    def piece_done(c, o, m):
        for f in sorted(folder.glob(f"*_{c['name']}_{_sha8(c)}_piece_o{o}_m{m}")):
            if (f / "metrics.json").exists():
                return f
        return None

    results_volume.reload()
    for c in configs:
        name = c["name"]
        whole = {"config": c, "state": "pending", "attempts": 0, "tried": set(), "result": None,
                 "env": {}, "deps": [], "done": (lambda c=c: _finished(folder, c))}
        if split and _splittable(c) and not _finished(folder, c):
            n_o = c["evaluation"].get("n_origins", 1)
            n_m = _members(c)
            began = any(piece_done(c, o, m) for o in range(n_o) for m in range(n_m)) or \
                any(marker(f"{name}__o{o}m{m}").exists() for o in range(n_o) for m in range(n_m))
            if began or not marker(name).exists():  # an unsplit machine already working is left to finish
                keys = []
                for o in range(n_o):
                    for m in range(n_m):
                        k = f"{name}__o{o}m{m}"
                        keys.append(k)
                        st[k] = {"config": c, "state": "pending", "attempts": 0, "tried": set(), "result": None,
                                 "env": {"RWM_PIECE": f"{o},{m}", "BASE_NAME": name}, "deps": [],
                                 "done": (lambda c=c, o=o, m=m: piece_done(c, o, m))}
                whole.update(env={"PIECES": ":".join(keys)}, deps=keys)  # no spaces: Vast splits its env string on them (10 November)
        st[name] = whole

    def finish(name, iid, result=None, bad=False, again=False):
        r = st[name]
        if iid is not None:
            _destroy(iid)
        marker(name).unlink(missing_ok=True)
        if bad and r.get("machine") is not None:
            _mark_bad(r["machine"])
        if result is not None:
            r.update(state="done", result=result)
        elif again and r["attempts"] < 3:
            r.update(state="pending")
        else:
            r.update(state="gone")
        results_volume.commit()

    while any(r["state"] not in ("done", "gone") for r in st.values()):
        results_volume.reload()
        for name, r in st.items():
            if r["state"] in ("done", "gone"):
                continue
            if any(st[d]["state"] == "gone" for d in r["deps"]):
                print(f"{name}: a piece failed three times", flush=True)
                r["state"] = "gone"
                continue
            done = r["done"]()
            if done:
                if r["state"] == "running":
                    print(f"{name}: done on machine {r['iid']}", flush=True)
                finish(name, r.get("iid") if r["state"] == "running" else None,
                       result=json.loads((done / "metrics.json").read_text()))
                continue
            if r["state"] == "running":
                iid = r["iid"]
                if failed_log(name).exists():
                    print(f"{name}: failed on machine {iid}", flush=True)
                    host = _host_fault(failed_log(name))
                    failed_log(name).rename(failed_log(name).with_suffix(f".{int(time.time())}.log"))
                    finish(name, iid, bad=host, again=True)
                    continue
                try:
                    info = json.loads(_vast("show", "instance", str(iid)))
                except Exception:
                    info = {"actual_status": "gone"}
                if info.get("actual_status") in ("exited", "offline", "gone") or info.get("intended_status") == "stopped" \
                        or time.time() - r["started"] > RUN_HOURS * 3600:
                    print(f"{name}: machine {iid} {info.get('actual_status')} (intended {info.get('intended_status')})", flush=True)
                    # a host that stops a machine before it has started (seen 4 November) is not rented again
                    finish(name, iid, again=True, bad=info.get("actual_status") in ("loading", "created"))
                    continue
                marker(name).parent.mkdir(parents=True, exist_ok=True)
                marker(name).write_text(json.dumps({"at": time.time(), "machine": iid}))
            elif r["state"] in ("pending", "waiting") and marker(name).exists():
                m = json.loads(marker(name).read_text())
                if time.time() - m["at"] < STALE_S:
                    r["state"] = "waiting"  # another manager has it
                    continue
                try:
                    alive = json.loads(_vast("show", "instance", str(m["machine"]))).get("actual_status") in ("running", "loading")
                except Exception:
                    alive = False
                if alive:
                    print(f"{name}: carrying on with machine {m['machine']}", flush=True)
                    r.update(state="running", iid=m["machine"], machine=None, started=time.time())
                else:
                    # the machine may still exist (a passing error from the Vast API reads as
                    # "not alive"): destroy it before renting another, so no run has two
                    # machines and none is left with no one watching it (4 November)
                    _destroy(m["machine"])
                    marker(name).unlink(missing_ok=True)
                    r["state"] = "pending"
            elif r["state"] == "waiting":
                r["state"] = "pending"
        running = sum(1 for r in st.values() if r["state"] == "running")
        for name, r in st.items():
            if running >= limit:
                break
            if r["state"] != "pending" or any(st[d]["state"] != "done" for d in r["deps"]):
                continue
            c = r["config"]
            env = {"REPO_URL": REPO_URL, "COMMIT": commit, "TORCH": TORCH, "DATASET": c["dataset"]["name"], "OUT": out,
                   "RUN_NAME": name, "BRIDGE_URL": url, "BRIDGE_TOKEN": os.environ["BRIDGE_TOKEN"],
                   "CONFIG_B64": base64.b64encode(json.dumps(c).encode()).decode(),
                   "WORKER_B64": base64.b64encode(WORKER.encode()).decode(), **r["env"]}
            if failed_log(name).exists():
                failed_log(name).rename(failed_log(name).with_suffix(f".{int(time.time())}.log"))
            try:
                iid, machine = _rent(c, env, c["dataset"]["name"], r["tried"], label=name)
            except Exception as e:
                print(f"{name}: could not rent a machine: {e}", flush=True)
                break
            r["tried"].add(machine)
            r.update(state="running", iid=iid, machine=machine, started=time.time(), attempts=r["attempts"] + 1)
            marker(name).parent.mkdir(parents=True, exist_ok=True)
            marker(name).write_text(json.dumps({"at": time.time(), "machine": iid}))
            running += 1
        results_volume.commit()
        time.sleep(POLL_S)
    return [st[c["name"]]["result"] for c in configs]


@app.function(image=image, volumes={RESULTS: results_volume}, secrets=[vast_secret, bridge_secret],
              timeout=24 * 3600, cpu=0.5, memory=2048)
def vast_batch(configs: list[dict], commit: str, out: str, limit: int = 20, split: bool = False) -> list:
    """Several runs at once (the final runs, entry of 10 November 2026), each on rented machines, split by
    (window, copy) when asked; results in rwm-results/<out>. Runs already finished there are read back."""
    return _manage(configs, commit, out, limit, split)


@app.function(image=image, volumes={RESULTS: results_volume}, secrets=[vast_secret, bridge_secret],
              timeout=24 * 3600, cpu=0.5, memory=2048)
def vast_loop(spec: str, commit: str, max_rounds: int, limit: int = 6, split: bool = False) -> dict:
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
        return _manage(configs, commit, out, limit, split)

    class SavedLog(loop.Log):
        def write(self, row):
            results_volume.reload()
            super().write(row)
            print(row, flush=True)
            results_volume.commit()

    best = loop.search(s, evaluate, SavedLog(Path(RESULTS) / out / "log.jsonl"), max_rounds=max_rounds or None)
    return {k: best[k] for k in ("kept", "score", "held_out")}
