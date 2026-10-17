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
| The model with the lowest error depends on the grain; the confidence set kept one or two models at item grains and about four at category and total (four to six windows) (§6.1) | grain table, grain and profile figures | `results/paper1_revision/tables/grain.csv` (from `accuracy.csv`, `mcs.csv`, `frat_mcs.csv`, `favorita_mcs.csv`) | `paper/figs/make_revision.py` (`grain_table`), `scripts/mcs_from_scores.py` for Frat | `configs/experiments/rev_*_rolling_*.yaml`; jobs `paper1_revision_weekly`, `paper1_revision_m5`, `favorita_a`, `favorita_b` |
| Item by store, the equal-weight average had the lowest error on 7 of 9 data sets (§6.2) | rolling table, heatmap figure | `accuracy.csv`, `paired.csv`, `mcs.csv` | `scripts/revision_analysis.py`, `scripts/revision_tables.py` | as above |
| What each member adds to the average; fitted weights rarely beat equal weights (§6.3) | ablation table and figure | `ablation_*.csv`, `favorita_ablation_*.csv` | `scripts/ensemble_ablation.py` | saved forecasts of the rolling runs |
| The three core models window by window (§6.4) | core-model figure | `results/paper1_revision/scores/studyF_scores_*.csv` | `scripts/revision_analysis.py` (study F rerun of the scores) | as above |
| All trained models use the plan; the world model carries a promotion forward as lift (§6.5) | perturbation table, mechanism figure | `perturbation.csv`, `mechanism.csv`, `../mechanism_weekly.csv` | `scripts/input_perturbation.py`, `scripts/perturbation_tables.py`, `scripts/mechanism_tests.py` | saved checkpoints of the rolling runs |
| Pretrained models need the plan; Timestral behind TimesFM with the plan (§6.6) | Timestral table | `studyF_*.csv`, `studyF_timestral_vs_timesfm.csv`, `studyF_tails.csv` | `scripts/make_timestral_configs.py`, `scripts/paired_from_scores.py`, `scripts/timestral_tails.py` | jobs `timestral_*` |
| Favorita, kept out of development (§6.7) | grain and rolling tables | `favorita_mcs.csv`, `favorita_ablation_*.csv` | `scripts/revision_analysis.py` | jobs `favorita_a`, `favorita_b` |
| Intervals and ranking (§6.8) | interval table, decisions figure | `intervals_fixed.csv`, `intervals.csv`, `studyF_ranks.csv` | `src/rwm/evaluation/compare.py` | as above |
| Decisions: the average or LightGBM lowest category-plan cost on 7 of 8, with paired intervals and sensitivity (corrected in study G); store orders; promotion regret (§6.9) | decisions and category-cost tables | `category_cost.csv`, `category_cost_nodes_zero.csv`, `decisions.csv` (store orders), `../promo_regret.csv` | `scripts/category_cost_nodes.py`, `scripts/category_cost_uncertainty.py`, `scripts/revision_analysis.py`, `scripts/promo_regret.py` | as above, job `study_g`; paper 2 known-answer worlds |
| Frat lift model with the world model's regularisation (§6.3) | text | `studyG_frat_lift_matched.csv`, Frat rows `lift_matched_for_world_model` in `ablation_*.csv` | `scripts/ensemble_ablation.py` | `rev_frat_rolling_lift_matched`, job `study_g` |
| Reconciliation, with the MinT-shrink fix (§6.9) | reconciliation table | `reconcile.csv`, `../reconcile/reconcile_*.csv` (Frat MinT rerun: `reconcile_frat_mint_fixed.csv`) | `src/rwm/evaluation/reconcile.py`, `scripts/mint_rerun.py`, `scripts/apply_mint_fix.py` | as above |
| Final windows of the first version (§6.10) | M5, final-window and new-model final tables | `results/promoted/`, `results/records/snapshot.json` | `paper/figs/make_tables.py` | jobs `final_*`, `m5_confirm*` |
| Tried and dropped (§7) | dropped and ledger tables | `pantry_stage*.csv`, `afterpromo_stage1.csv`; ledger in `results/README.md` | `scripts/make_pantry_configs.py`, `scripts/make_afterpromo_configs.py` | jobs `pantry_*`, `afterpromo_weight` |

Rebuild every number, table and figure: `make` in `paper/` (about 30 seconds).
`tests/test_paper_revision_values.py` checks that every revision number the paper
quotes exists in these files.

## Paper 2

| Claim | Table or figure | Result file | Made by |
|---|---|---|---|
| Pass counts of every check over 933 version-windows of 494 versions | pass-count table, checks figure | `results/paper2/summary/versions.csv`, `stability.csv`, `agreement.csv` | `scripts/audit_runs.py`, `scripts/audit_summary.py`, `paper2/figs/real.py` |
| Diagnosis of C1 and C4 on 14 runs | text | `results/paper2/diagnose/audit_diagnose.csv` | `scripts/audit_diagnose.py` (runs listed in `results/paper2/diagnose_runs.txt`) |
| Stated against measured response to zone price episodes (C7) | C7 figure | `results/paper2/summary/zones.csv`, `results/paper2/zone_episodes.csv` | `scripts/audit_runs.py zones`, `scripts/zone_episodes.py` |
| Checks passed against accuracy (main test) | main-test table and figure | `paper2/generated/real_main_test*.csv` | `paper2/figs/real.py` |
