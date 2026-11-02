#!/bin/bash
# The NUC worker (cloud/local_app.py): runs queued research jobs on the NUC's GPU, one at a time.
#
#   bash nuc_worker.sh setup     # once: Python environment in ~/rwm, GPU check
#   bash nuc_worker.sh           # run jobs until stopped (Ctrl-C); keep the Ubuntu window open
#   touch ~/rwm/PAUSE            # finish the current job, then wait;  rm ~/rwm/PAUSE to carry on
#
# Everything lives in ~/rwm, away from ww. Jobs run at low priority (nice 10) so
# other work on the NUC comes first. Needs ~/rwm/bridge_token (the rwm-bridge
# token, written by Karthik, readable only by him).
set -uo pipefail
HOME_DIR="${RWM_HOME:-$HOME/rwm}"
REPO_URL="https://github.com/karthik-iyer-au/retail-world-models"
JOBS_URL="https://karthik-iyer-au--rwm-local-jobs.us-east.modal.run"
BRIDGE_URL="https://karthik-iyer-au--rwm-vast-bridge.us-east.modal.run"
TORCH="2.14.1"  # the version the Modal and Vast runs used
WORKER="nuc-$(hostname)"
PY="$HOME_DIR/venv/bin/python"
mkdir -p "$HOME_DIR"

if [ "${1:-}" = "setup" ]; then
  command -v git >/dev/null && command -v curl >/dev/null && python3 -m venv --help >/dev/null 2>&1 \
    || { echo "Run first: sudo apt-get update && sudo apt-get install -y git curl python3-venv"; exit 1; }
  nvidia-smi || { echo "No GPU visible in Ubuntu: check the NVIDIA driver in Windows"; exit 1; }
  [ -x "$PY" ] || python3 -m venv "$HOME_DIR/venv"
  "$PY" -m pip install -q --upgrade pip && "$PY" -m pip install -q "torch==$TORCH" || exit 1
  "$PY" -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print(torch.cuda.get_device_name(0), 'works:', float((x @ x).sum()) == float((x @ x).sum()))"
  [ -s "$HOME_DIR/bridge_token" ] || echo "Now put the bridge token in $HOME_DIR/bridge_token (then: chmod 600 $HOME_DIR/bridge_token)"
  exit 0
fi

[ -s "$HOME_DIR/bridge_token" ] || { echo "Missing $HOME_DIR/bridge_token"; exit 1; }
chmod 600 "$HOME_DIR/bridge_token"
AUTH="Authorization: Bearer $(tr -d '[:space:]' < "$HOME_DIR/bridge_token")"
REPO="$HOME_DIR/repo"
echo "NUC worker started $(date -u '+%d %b %H:%M UTC'); Ctrl-C stops it, touch $HOME_DIR/PAUSE pauses it"

while true; do
  if [ -e "$HOME_DIR/PAUSE" ]; then sleep 60; continue; fi
  JOB="$HOME_DIR/job.json"
  code=$(curl -sS -o "$JOB" -w '%{http_code}' -X POST -H "$AUTH" -H 'Content-Type: application/json' \
         -d "{\"worker\": \"$WORKER\"}" "$JOBS_URL/claim") || { sleep 120; continue; }
  if [ "$code" = "204" ]; then sleep 60; continue; fi
  if [ "$code" != "200" ]; then echo "queue answered $code"; sleep 300; continue; fi
  read -r ID COMMIT OUT NAME DATASET < <("$PY" -c "import json,sys; j=json.load(open(sys.argv[1])); c=j['config']; print(j['out']+'/'+c['name'], j['commit'], j['out'], c['name'], c['dataset']['name'])" "$JOB")
  LOG="$HOME_DIR/logs/${ID//\//__}.log"; mkdir -p "$HOME_DIR/logs"
  echo "$(date -u '+%d %b %H:%M UTC')  start $ID at ${COMMIT:0:8} (log $LOG)"
  START=$(date +%s)
  (
    set -e
    mkdir -p "$REPO" && cd "$REPO"
    [ -d .git ] || git init -q
    git fetch -q --depth 1 "$REPO_URL" "$COMMIT" && git checkout -q FETCH_HEAD
    if [ "$(cat "$HOME_DIR/installed" 2>/dev/null)" != "$COMMIT" ]; then
      "$PY" -m pip install -q -e ".[trees,deep]" && echo "$COMMIT" > "$HOME_DIR/installed"
    fi
    BRIDGE_TOKEN="${AUTH#Authorization: Bearer }" "$PY" cloud/fetch_data.py "$BRIDGE_URL" "$DATASET" "data/raw/$DATASET"
    rm -rf "$HOME_DIR/out" && mkdir -p "$HOME_DIR/out"
    nice -n 10 "$PY" - "$JOB" "$HOME_DIR/out" <<'EOF'
import json, sys
from pathlib import Path
from rwm.experiments.run import run
run(json.load(open(sys.argv[1]))["config"], strict=True, out_root=Path(sys.argv[2]))
EOF
    rm -rf "$HOME_DIR/out/_partial"
    tar czf "$HOME_DIR/run.tgz" -C "$HOME_DIR/out" .
    curl -fsS -X PUT -H "$AUTH" --data-binary @"$HOME_DIR/run.tgz" "$BRIDGE_URL/results/$OUT" >/dev/null
  ) > "$LOG" 2>&1
  OK=$?
  SECONDS_TAKEN=$(( $(date +%s) - START ))
  if [ $OK -ne 0 ]; then
    curl -sS -X PUT -H "$AUTH" --data-binary @"$LOG" "$BRIDGE_URL/failed/$OUT/$NAME" >/dev/null || true
    echo "$(date -u '+%d %b %H:%M UTC')  FAILED $ID (see $LOG)"
  else
    echo "$(date -u '+%d %b %H:%M UTC')  done $ID in $((SECONDS_TAKEN / 60)) min"
  fi
  curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
    -d "{\"id\": \"$ID\", \"ok\": $([ $OK -eq 0 ] && echo true || echo false), \"worker\": \"$WORKER\", \"seconds\": $SECONDS_TAKEN}" \
    "$JOBS_URL/finish" >/dev/null || true
done
