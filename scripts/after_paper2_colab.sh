#!/usr/bin/env bash
# After scripts/paper2_colab.sh: the checkpoint audit again with the loading fix,
# the summary, then the paper 1 revision M5 months. Run on the runtime that has
# the Kaggle token (colab_run.ipynb). Usage: bash scripts/after_paper2_colab.sh <results folder on Drive>
set -u
OUT="$1"
git pull -q origin main && pip install -q -e ".[trees,deep,fetch,foundation,benchmarks]"
python scripts/audit_runs.py checkpoints --results "$OUT" --pattern "*paper2_refit_*" "*frat_*" "*dominicks_*" "*m5_*"
python scripts/audit_summary.py --results "$OUT" --out "$OUT/paper2_summary"
echo "paper 2 audit summary written"
python -m rwm.experiments.job --job configs/jobs/paper1_revision_m5.yaml --out "$OUT"
python scripts/mechanism_tests.py --results "$OUT" --out "$OUT/paper1_revision/mechanism_m5.csv" \
  --pattern "*rev_m5_rolling_world_model_seed1_*" "*rev_m5_rolling_lift_seed1_*" "*m5_test_world_model_roll_norm_seed1_*" "*m5_test_lift_seed1_*"
python scripts/revision_analysis.py --results "$OUT" --out "$OUT/paper1_revision/m5" --keys m5
echo "paper 1 revision M5 finished"
