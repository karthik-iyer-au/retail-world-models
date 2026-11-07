# Ledger: open promises, decisions and asks for paper 1's revision

One row per item: date, who owes it, status. Read before answering "what's left". Update in the same turn as
any new promise, decision or finished item. The research record behind it is docs/PAPER1_REVISION_PLAN.md.

## Running

| Date | Item | Owner | Status |
|---|---|---|---|
| 3 Nov | Study S: Dominick's on cleaned prices, both models, six categories | Claude | Done (all six pairs reported 5 Nov; logs to 88f6597) |
| 1 Nov | Study R: Favorita loops, both models, on Vast | Claude | Running; world model limited to two rounds (Karthik 5 Nov: "2 rounds is good"), plan entry dcb9d99; world model round 1 kept latent_overshooting (JEPA check over seven steps), 0.547 -> 0.484, held-out 0.600 -> 0.560 (reported 7 Nov); LightGBM kept min_data_in_leaf_4x, objective_poisson, learning_rate_half (0.580 / 0.573) |
| 28 Oct | Study M: M5 world-model loop on Modal, two rounds | Claude | Running, round 2 |
| 4 Nov | Study T: loss round after each world-model loop | Claude | Frat and all six Dominick's categories done (only snacks kept a change: item-spread NB); M5 and Favorita when their loops end |
| 5 Nov | 3-hourly checks of Vast and Modal loops | Claude | Scheduled tasks running |

## To do once the searches end (Karthik 5 Nov: "ledger it and pick up everything at the end")

| Date | Item | Owner | Status |
|---|---|---|---|
| 5 Nov | Final runs: tuned settings of both models on the paper's reported rolling and held-out windows, paired tests at every grain | Claude | Open; cost first; on Vast by default, the NUC only when ww is not using it (Karthik 6 Nov: "U will compete with ww for NUc") |
| 5 Nov | Ensemble rebuilt from the tuned world model and tuned LightGBM, TimesFM with the plan unchanged; untuned ensemble reported beside it (Karthik: "Yes we should"; plan entry 0ab4b46) | Claude | Open, part of the final runs |
| 5 Nov | Rerun the paper's Dominick's comparison of all models on cleaned prices | Claude | Open; cost first |
| 5 Nov | Paper writes up ablations, occlusion and perturbation, mechanism tests, calibration (Karthik 5 Nov) | Claude | Open; results in results/paper1_revision/tables (ablation_*, perturbation, mechanism, intervals_fixed) |
| 5 Nov | The research loop gets its own section, with journal-quality seaborn-style figures, no bar or lollipop charts (Karthik 5 Nov): search path step lines with every try and the keep band; change by data set heatmap; seed-noise densities, one copy against five; search against held-out scatter; before and after tuning slopes | Claude | Open; draft from all logs at the end |
| 6 Nov | Offered (Karthik asked about M-JEPA's hybrid fine-tuning): study U, JEPA pretraining then hybrid fine-tuning (forecast loss with the latent loss kept on, encoder slowed) on top of each data set's final settings, (a) pretrained on its own data, (b) pretrained on all data sets pooled; one round, same keep rule, written down before it runs. Fairness (Karthik 6 Nov: "we want jepa to win without favoring it"): recipes fixed from M-JEPA before running, same try budget and keep rule as any change, all JEPA tries counted and reported; (b) only with LightGBM also trained on the same pooled data, and reported beside the foundation models | Claude | Karthik agreed the terms (6 Nov: "perfect look forward to resuts"); next: cost it and write the plan entry before it runs, alongside the final runs |
| 7 Nov | Wall-clock time and spend for every study in the paper (Karthik: "remember all this runtime so we can update wall clock time in paper"): per-run wall_seconds and device from each run's manifest, Modal billing and Vast invoices saved in results/compute (first snapshot 7 Nov; gitignored, so kept in the workspace and copied to the rwm-results volume, not GitHub), table built by script | Claude | Open; refresh snapshots at the end of the searches and after the final runs |
| 5 Nov | Title chosen after the final results | Karthik and Claude | Open |
| 5 Nov | The reviewer said the paper reads as AI-written; the whole text is rewritten to fix it, not patched (Karthik 5 Nov: "Remember th reviewer provided feedback this is ai written so it needs to be fixed too"). In Karthik's voice, following the technical-paper-house-style skill: conclusions first, plain words, no em dashes, no hedging, no dramatic or clever headings; and none of the usual tells (lists of three, "not X but Y" turns, signposting such as "notably" or "crucially", stock words such as "robust", "leverage", "delve", every paragraph the same shape, summaries that repeat the paragraph). Then checked by a fresh reader that has not seen the drafting, and by Karthik, before submission | Claude, then Karthik | Open |

## Reminders

| Date | Item | Owner | Status |
|---|---|---|---|
| 4 Nov | Saturday reminder about the NUC's 96 GB RAM | Claude | Closed: memory fitted and the NUC is in use by ww (Karthik 7 Nov); the stale reminder fired once on 7 Nov and will not repeat |
