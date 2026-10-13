#!/usr/bin/env bash
# After the paper 1 revision weekly job: mechanism tests and analyses for
# Dominick's and Frat. Usage: bash scripts/after_paper1_weekly.sh <results folder on Drive>
set -u
OUT="$1"
git pull -q origin main && pip install -q -e ".[trees,deep,fetch,foundation,benchmarks]"
python scripts/mechanism_tests.py --results "$OUT" --out "$OUT/paper1_revision/mechanism_weekly.csv" \
  --pattern "*rev_dominicks_*_rolling_world_model_seed1_*" "*rev_dominicks_*_rolling_lift_seed1_*" \
            "*rev_frat_rolling_world_model_seed1_*" "*rev_frat_rolling_lift_seed1_*" "*frat_final_world_model_seed1_*" "*frat_final_lift_seed1_*"
python scripts/revision_analysis.py --results "$OUT" --out "$OUT/paper1_revision/weekly" \
  --keys frat dominicks_che dominicks_sna dominicks_sdr dominicks_bjc dominicks_ana dominicks_fre
echo "paper 1 revision weekly analyses finished"
