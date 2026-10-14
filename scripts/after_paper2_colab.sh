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
# the paper 1 revision M5 months now run on the paper 1 runtime (scripts/paper1_l4_chain.sh)
