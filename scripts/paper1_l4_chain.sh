#!/usr/bin/env bash
# Paper 1 revision on its own runtime (colab_paper1.ipynb), after a lost runtime:
# the analyses one data set at a time (each in its own folder, so a crash loses
# one), the records snapshot, then the M5 months and their tests.
# Frat's analysis is already in <OUT>/paper1_revision/weekly.
# Needs KAGGLE_API_TOKEN in the environment for M5.
# Usage: bash scripts/paper1_l4_chain.sh <results folder on Drive>
set -u
OUT="$1"
REV="$OUT/paper1_revision"
for key in dominicks_che dominicks_sna dominicks_sdr dominicks_bjc dominicks_ana dominicks_fre; do
  if [ -f "$REV/weekly_$key/scores.csv" ] && grep -q ",rolling," "$REV/weekly_$key/scores.csv"; then
    echo "already analysed $key"; continue
  fi
  python scripts/revision_analysis.py --results "$OUT" --out "$REV/weekly_$key" --keys "$key" || echo "analysis FAILED for $key"
done
python scripts/records_snapshot.py --results "$OUT" --out "$REV/snapshot.json" || echo "snapshot FAILED"
python -m rwm.experiments.job --job configs/jobs/paper1_revision_m5.yaml --out "$OUT"
python scripts/mechanism_tests.py --results "$OUT" --out "$REV/mechanism_m5.csv" \
  --pattern "*rev_m5_rolling_world_model_seed1_*" "*rev_m5_rolling_lift_seed1_*" "*m5_test_world_model_roll_norm_seed1_*" "*m5_test_lift_seed1_*"
python scripts/revision_analysis.py --results "$OUT" --out "$REV/m5" --keys m5
echo "paper 1 revision chain finished"
