#!/bin/bash
# The NUC worker (cloud/local_app.py): runs queued research jobs on the NUC's GPU.
#
#   bash nuc_worker.sh setup     # once: Python environment in ~/rwm, GPU check
#   bash nuc_worker.sh           # run jobs one at a time until stopped (Ctrl-C); keep the Ubuntu window open
#   bash nuc_worker.sh 3         # run up to three jobs side by side (Ctrl-C stops all three)
#   touch ~/rwm/PAUSE            # finish the current jobs, then wait;  rm ~/rwm/PAUSE to carry on
#
# Everything lives in ~/rwm, away from ww. Jobs run at low priority (nice 10) so
# other work on the NUC comes first. Each side-by-side job has its own folder
# (~/rwm/slot<n>: code at the job's commit, data, output); they share the Python
# environment, whose packages are installed once per commit. Needs
# ~/rwm/bridge_token (the rwm-bridge token, written by Karthik, readable only by him).
set -uo pipefail
HOME_DIR="${RWM_HOME:-$HOME/rwm}"
REPO_URL="https://github.com/karthik-iyer-au/retail-world-models"
JOBS_URL="https://karthik-iyer-au--rwm-local-jobs.us-east.modal.run"
BRIDGE_URL="https://karthik-iyer-au--rwm-vast-bridge.us-east.modal.run"
TORCH="2.14.1"  # the version the Modal and Vast runs used
PY="$HOME_DIR/venv/bin/python"
mkdir -p "$HOME_DIR"

if [ "${1:-}" = "setup" ]; then
  command -v git >/dev/null && command -v curl >/dev/null && command -v flock >/dev/null && python3 -m venv --help >/dev/null 2>&1 \
    || { echo "Run first: sudo apt-get update && sudo apt-get install -y git curl python3-venv"; exit 1; }
  nvidia-smi || { echo "No GPU visible in Ubuntu: check the NVIDIA driver in Windows"; exit 1; }
  [ -x "$PY" ] || python3 -m venv "$HOME_DIR/venv"
  "$PY" -m pip install -q --upgrade pip && "$PY" -m pip install -q "torch==$TORCH" || exit 1
  "$PY" -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print(torch.cuda.get_device_name(0), 'works:', float((x @ x).sum()) == float((x @ x).sum()))"
  [ -s "$HOME_DIR/bridge_token" ] || echo "Now put the bridge token in $HOME_DIR/bridge_token (then: chmod 600 $HOME_DIR/bridge_token)"
  exit 0
fi

# side by side: start one worker per slot in this window and stop them all on Ctrl-C
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  trap 'kill $(jobs -p) 2>/dev/null; exit 0' INT TERM
  for n in $(seq 1 "$1"); do bash "$0" --slot "$n" & sleep 5; done
  wait
  exit 0
fi
SLOT=1
[ "${1:-}" = "--slot" ] && SLOT="$2"

[ -s "$HOME_DIR/bridge_token" ] || { echo "Missing $HOME_DIR/bridge_token"; exit 1; }
chmod 600 "$HOME_DIR/bridge_token"
TOKEN="$(tr -d '[:space:]' < "$HOME_DIR/bridge_token")"
AUTH="Authorization: Bearer $TOKEN"
WORK="$HOME_DIR/slot$SLOT"
REPO="$WORK/repo"
WORKER="nuc-$(hostname)-$SLOT"
mkdir -p "$WORK" "$HOME_DIR/logs"
echo "NUC worker $SLOT started $(date -u '+%d %b %H:%M UTC'); Ctrl-C stops it, touch $HOME_DIR/PAUSE pauses it"

while true; do
  if [ -e "$HOME_DIR/PAUSE" ]; then sleep 60; continue; fi
  JOB="$WORK/job.json"
  code=$(curl -sS -o "$JOB" -w '%{http_code}' -X POST -H "$AUTH" -H 'Content-Type: application/json' \
         -d "{\"worker\": \"$WORKER\"}" "$JOBS_URL/claim") || { sleep 120; continue; }
  if [ "$code" = "204" ]; then sleep 60; continue; fi
  if [ "$code" != "200" ]; then echo "queue answered $code"; sleep 300; continue; fi
  read -r ID COMMIT OUT NAME DATASET < <("$PY" -c "import json,sys; j=json.load(open(sys.argv[1])); c=j['config']; print(j['out']+'/'+c['name'], j['commit'], j['out'], c['name'], c['dataset']['name'])" "$JOB")
  LOG="$HOME_DIR/logs/${ID//\//__}.log"
  echo "$(date -u '+%d %b %H:%M UTC')  [$SLOT] start $ID at ${COMMIT:0:8}"
  START=$(date +%s)
  (
    set -e
    mkdir -p "$REPO" && cd "$REPO"
    [ -d .git ] || git init -q
    git fetch -q --depth 1 "$REPO_URL" "$COMMIT" && git checkout -q FETCH_HEAD
    # the packages the code needs, once per commit for all slots; the code itself is
    # read from this slot's checkout (PYTHONPATH), so slots at different commits do not clash
    (
      flock 9
      if [ "$(cat "$HOME_DIR/installed" 2>/dev/null)" != "$COMMIT" ]; then
        "$PY" -m pip install -q ".[trees,deep]" && "$PY" -m pip uninstall -q -y rwm && echo "$COMMIT" > "$HOME_DIR/installed"
      fi
    ) 9> "$HOME_DIR/install.lock"
    BRIDGE_TOKEN="$TOKEN" "$PY" cloud/fetch_data.py "$BRIDGE_URL" "$DATASET" "data/raw/$DATASET"
    rm -rf "$WORK/out" && mkdir -p "$WORK/out"
    PYTHONPATH="$REPO/src" nice -n 10 "$PY" - "$JOB" "$WORK/out" <<'EOF'
import json, sys
from pathlib import Path
from rwm.experiments.run import run
run(json.load(open(sys.argv[1]))["config"], strict=True, out_root=Path(sys.argv[2]))
EOF
    rm -rf "$WORK/out/_partial"
    tar czf "$WORK/run.tgz" -C "$WORK/out" .
    curl -fsS --retry 5 --retry-all-errors -X PUT -H "$AUTH" --data-binary @"$WORK/run.tgz" "$BRIDGE_URL/results/$OUT" >/dev/null
  ) > "$LOG" 2>&1
  OK=$?
  SECONDS_TAKEN=$(( $(date +%s) - START ))
  if [ $OK -ne 0 ]; then
    curl -sS -X PUT -H "$AUTH" --data-binary @"$LOG" "$BRIDGE_URL/failed/$OUT/$NAME" >/dev/null || true
    echo "$(date -u '+%d %b %H:%M UTC')  [$SLOT] FAILED $ID (see $LOG)"
  else
    echo "$(date -u '+%d %b %H:%M UTC')  [$SLOT] done $ID in $((SECONDS_TAKEN / 60)) min"
  fi
  curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
    -d "{\"id\": \"$ID\", \"ok\": $([ $OK -eq 0 ] && echo true || echo false), \"worker\": \"$WORKER\", \"seconds\": $SECONDS_TAKEN}" \
    "$JOBS_URL/finish" >/dev/null || true
done
