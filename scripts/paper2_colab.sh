#!/usr/bin/env bash
# Paper 2 audit, every step, on Colab. Usage: bash scripts/paper2_colab.sh <results folder on Drive>
# Each step skips work already saved, so the script can be started again after a dropped runtime.
set -u
OUT="$1"
python -m rwm.experiments.job --job configs/jobs/paper2_refit.yaml --out "$OUT"
python scripts/audit_runs.py checkpoints --results "$OUT" --pattern "*paper2_refit_*" "*frat_*" "*dominicks_*" "*m5_*"
for c in che sna sdr bjc ana fre; do
  python scripts/audit_runs.py lightgbm --config configs/experiments/dominicks_${c}_final_lightgbm.yaml --results "$OUT"
done
python scripts/audit_runs.py lightgbm --config configs/experiments/frat_final_lightgbm.yaml --results "$OUT"
python scripts/audit_runs.py zones --episodes results/paper2/zone_episodes.csv \
  --world-config configs/experiments/dominicks_ana_final_world_model_switching_avg3_roll_norm.yaml \
  --lightgbm-config configs/experiments/dominicks_ana_final_lightgbm.yaml --results "$OUT"
python scripts/audit_runs.py lightgbm --config configs/experiments/m5_test_lightgbm.yaml --results "$OUT"
python scripts/audit_summary.py --results "$OUT" --out "$OUT/paper2_summary"
echo "paper 2 audit finished"
