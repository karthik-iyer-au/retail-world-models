# Ledger: open promises, decisions and asks for paper 1's revision

One row per item: date, who owes it, status. Read before answering "what's left". Update in the same turn as
any new promise, decision or finished item. The research record behind it is docs/PAPER1_REVISION_PLAN.md.

## Running

| Date | Item | Owner | Status |
|---|---|---|---|
| 3 Nov | Study S: Dominick's on cleaned prices, both models, six categories | Claude | Done (all six pairs reported 5 Nov; logs to 88f6597) |
| 1 Nov | Study R: Favorita loops, both models, on Vast | Claude | Running; world model limited to two rounds (Karthik 5 Nov: "2 rounds is good"), plan entry dcb9d99 |
| 28 Oct | Study M: M5 world-model loop on Modal, two rounds | Claude | Running, round 2 |
| 4 Nov | Study T: loss round after each world-model loop | Claude | Frat, five Dominick's categories done (snacks kept item-spread NB); frozen entrees running; M5 and Favorita when their loops end |
| 5 Nov | 3-hourly checks of Vast and Modal loops | Claude | Scheduled tasks running |

## To do once the searches end (Karthik 5 Nov: "ledger it and pick up everything at the end")

| Date | Item | Owner | Status |
|---|---|---|---|
| 5 Nov | Final runs: tuned settings of both models on the paper's reported rolling and held-out windows, paired tests at every grain | Claude | Open; cost first; NUC 5090 is the place to run them |
| 5 Nov | Ensemble rebuilt from the tuned world model and tuned LightGBM, TimesFM with the plan unchanged; untuned ensemble reported beside it (Karthik: "Yes we should"; plan entry 0ab4b46) | Claude | Open, part of the final runs |
| 5 Nov | Rerun the paper's Dominick's comparison of all models on cleaned prices | Claude | Open; cost first |
| 5 Nov | Paper writes up ablations, occlusion and perturbation, mechanism tests, calibration (Karthik 5 Nov) | Claude | Open; results in results/paper1_revision/tables (ablation_*, perturbation, mechanism, intervals_fixed) |
| 5 Nov | The research loop gets its own section, with journal-quality seaborn-style figures, no bar or lollipop charts (Karthik 5 Nov): search path step lines with every try and the keep band; change by data set heatmap; seed-noise densities, one copy against five; search against held-out scatter; before and after tuning slopes | Claude | Open; draft from all logs at the end |
| 5 Nov | Title chosen after the final results | Karthik and Claude | Open |

## Reminders

| Date | Item | Owner | Status |
|---|---|---|---|
| 4 Nov | Saturday reminder about the NUC's 96 GB RAM | Claude | Redundant: the RAM arrived 5 Nov |
