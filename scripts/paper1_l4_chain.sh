#!/usr/bin/env bash
# Paper 1 revision on its own runtime (colab_paper1.ipynb), after a lost runtime:
# fetch the data, the analyses one data set at a time (each in its own folder,
# saved after every window), reconciliation as a separate pass, the records
# snapshot, then the M5 months and their tests.
# Frat's analysis is already in <OUT>/paper1_revision/weekly.
# A watchdog stops an analysis when free memory falls below 6 GB, so one step
# cannot take the runtime down (the runtime was lost twice to memory).
# Needs KAGGLE_API_TOKEN in the environment for M5.
# Usage: bash scripts/paper1_l4_chain.sh <results folder on Drive> [data folder on Drive]
set -u
OUT="$1"
DATA="${2:-/content/drive/MyDrive/retail-world-model/data}"
REV="$OUT/paper1_revision"
KEYS="dominicks_che dominicks_sna dominicks_sdr dominicks_bjc dominicks_ana dominicks_fre"

guarded() {  # run a command; stop it if available memory falls below 6 GB
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local avail
    avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    if [ "$avail" -lt 6000 ]; then
      echo "memory watchdog: stopped $* with ${avail} MB available"
      pkill -TERM -P "$pid"; kill -TERM "$pid"; sleep 5; pkill -KILL -P "$pid"; kill -KILL "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 1
    fi
    sleep 5
  done
  wait "$pid"
}

python -m rwm.data.fetch breakfast_at_the_frat --from "$DATA"
python -m rwm.data.fetch dominicks che sna sdr bjc ana fre
for key in $KEYS; do
  guarded python scripts/revision_analysis.py --results "$OUT" --out "$REV/weekly_$key" --keys "$key" \
    --parts scores paired mcs intervals decisions || echo "analysis FAILED for $key"
done
python scripts/records_snapshot.py --results "$OUT" --out "$REV/snapshot.json" || echo "snapshot FAILED"
for key in frat $KEYS; do  # Frat again: base forecasts under the amendment of 14 October 2026
  guarded python scripts/revision_analysis.py --results "$OUT" --out "$REV/reconcile_$key" --keys "$key" \
    --parts reconcile || echo "reconciliation FAILED for $key"
done
python -m rwm.experiments.job --job configs/jobs/paper1_revision_m5.yaml --out "$OUT"
guarded python scripts/mechanism_tests.py --results "$OUT" --out "$REV/mechanism_m5.csv" \
  --pattern "*rev_m5_rolling_world_model_seed1_*" "*rev_m5_rolling_lift_seed1_*" "*m5_test_world_model_roll_norm_seed1_*" "*m5_test_lift_seed1_*"
guarded python scripts/revision_analysis.py --results "$OUT" --out "$REV/m5" --keys m5 \
  --parts scores paired mcs intervals decisions || echo "analysis FAILED for m5"
guarded python scripts/revision_analysis.py --results "$OUT" --out "$REV/reconcile_m5" --keys m5 \
  --parts reconcile || echo "reconciliation FAILED for m5"
echo "paper 1 revision chain finished"
