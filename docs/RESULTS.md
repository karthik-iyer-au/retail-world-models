# Results overview (Paper 1)

Each main claim of `paper/main.pdf`, with the table it rests on, the script that
makes the table, and the configs and jobs whose runs fed it. Every claim is about
the four public data sets used here (M5, six Dominick's categories, Breakfast at
the Frat, Corporación Favorita) and covers price and promotion only, not all four
Ps. The pre-registered plan and every dated amendment are in
`docs/PAPER1_REVISION_PLAN.md`.

Run folders (forecasts, checkpoints) are kept outside git; each run's
`manifest.json` records its commit, config and data checksums. Rolling-window
runs are named `rev_<data>_rolling_<model>` and sit under the Drive results
folder named in `notebooks/colab_run.ipynb`.

| Claim (paper section) | Table or figure (by name; numbers move as the paper changes) | Result file | Made by | Configs and jobs |
|---|---|---|---|---|
| The model with the lowest error depends on the grain; few models are level at item grains, about four at category and total (§6.1) | grain table, grain and profile figures | `results/paper1_revision/tables/grain.csv` (from `accuracy.csv`, `mcs.csv`, `frat_mcs.csv`, `favorita_mcs.csv`) | `paper/figs/make_revision.py` (`grain_table`), `scripts/mcs_from_scores.py` for Frat | `configs/experiments/rev_*_rolling_*.yaml`; jobs `paper1_revision_weekly`, `paper1_revision_m5`, `favorita_a`, `favorita_b` |
| Item by store, the equal-weight average had the lowest error on 7 of 9 data sets (§6.2) | rolling table, heatmap figure | `accuracy.csv`, `paired.csv`, `mcs.csv` | `scripts/revision_analysis.py`, `scripts/revision_tables.py` | as above |
| What each member adds to the average; fitted weights rarely beat equal weights (§6.3) | ablation table and figure | `ablation_*.csv`, `favorita_ablation_*.csv` | `scripts/ensemble_ablation.py` | saved forecasts of the rolling runs |
| The three core models window by window (§6.4) | core-model figure | `results/paper1_revision/scores/studyF_scores_*.csv` | `scripts/revision_analysis.py` (study F rerun of the scores) | as above |
| All trained models use the plan; the world model carries a promotion forward as lift (§6.5) | perturbation table, mechanism figure | `perturbation.csv`, `mechanism.csv`, `../mechanism_weekly.csv` | `scripts/input_perturbation.py`, `scripts/perturbation_tables.py`, `scripts/mechanism_tests.py` | saved checkpoints of the rolling runs |
| Pretrained models need the plan; Timestral behind TimesFM with the plan (§6.6) | Timestral table | `studyF_*.csv`, `studyF_timestral_vs_timesfm.csv`, `studyF_tails.csv` | `scripts/make_timestral_configs.py`, `scripts/paired_from_scores.py`, `scripts/timestral_tails.py` | jobs `timestral_*` |
| Favorita, kept out of development (§6.7) | grain and rolling tables | `favorita_mcs.csv`, `favorita_ablation_*.csv` | `scripts/revision_analysis.py` | jobs `favorita_a`, `favorita_b` |
| Intervals and ranking (§6.8) | interval table, decisions figure | `intervals_fixed.csv`, `intervals.csv`, `studyF_ranks.csv` | `src/rwm/evaluation/compare.py` | as above |
| Decisions: the world model lowest category-plan cost on 6 of 8; promotion regret (§6.9) | decisions table | `decisions.csv`, `../promo_regret.csv` | `scripts/revision_analysis.py`, `scripts/promo_regret.py` | as above; paper 2 known-answer worlds |
| Reconciliation (§6.9) | reconciliation table | `reconcile.csv`, `../reconcile/reconcile_*.csv` | `src/rwm/evaluation/reconcile.py` | as above |
| Final windows of the first version (§6.10) | M5, final-window and new-model final tables | `results/promoted/`, `results/records/snapshot.json` | `paper/figs/make_tables.py` | jobs `final_*`, `m5_confirm*` |
| Tried and dropped (§7) | dropped and ledger tables | `pantry_stage*.csv`, `afterpromo_stage1.csv`; ledger in `results/README.md` | `scripts/make_pantry_configs.py`, `scripts/make_afterpromo_configs.py` | jobs `pantry_*`, `afterpromo_weight` |

Rebuild every number, table and figure: `make` in `paper/` (about 30 seconds).
`tests/test_paper_revision_values.py` checks that every revision number the paper
quotes exists in these files.
