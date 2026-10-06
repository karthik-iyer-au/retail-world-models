# Literature used

One row per paper or source the work leans on: what we took from it, where it
sits in the code, and what happened when we tested it. Results are on the
development windows; numbers are in `results/README.md`. Entries marked
"to verify" have not been checked against the paper itself and must be before
they are cited.

## Benchmarks and scoring

| Source | What we took | Where | Outcome |
|---|---|---|---|
| Makridakis, Spiliotis, Assimakopoulos (2022), the M5 accuracy competition | WRMSSE over the hierarchy; the 28-day M5 test design | `rwm.evaluation.hierarchy` | In use |
| Ke et al. (2017), LightGBM | The tree benchmark, one direct model per store, Tweedie loss | `rwm.prior_work.lightgbm_direct` | The benchmark to beat: Frat 0.421, Dominick's analgesics 0.715 |
| Bates and Granger (1969), combining forecasts | Weighted combination of our model and LightGBM, weights set on a validation window | `rwm.combination` | Best Frat result, 0.388; no gain on Dominick's |

## The neural model

| Source | What we took | Where | Outcome |
|---|---|---|---|
| Vaswani et al. (2017), the transformer | Encoder over an item's history to give its state | `state_model` | In use |
| Tweedie loss for sales counts (as used across M5 entries) | Loss on sales relative to the item's own level | `state_model` | In use; negative binomial gave no material difference |
| Veit, Belongie, Karaletsos (2017), conditional similarity networks | Item-to-item similarity under several conditions | `neighbours: similarity` | Worse than no neighbours on Frat (0.453 to 0.476 against 0.406). Not yet used for new items |
| Srivastava et al. (2014), dropout; Loshchilov and Hutter (2019), decoupled weight decay | Regularisation of the readout | `readout_dropout`, `weight_decay` | Steadier across seeds, no accuracy gain |

## State prediction and world models

| Source | What we took | Where | Outcome |
|---|---|---|---|
| LeCun (2022), a path towards autonomous machine intelligence; Assran et al. (2023), I-JEPA | Predict the future state, not the future data; a slow-moving copy of the encoder as the target | `latent_weight`, `latent_ema`, `pretrain_steps`, `finetune` | Level with no state check on Frat (0.407 to 0.415); frozen encoder worse (0.428) |
| Author's own M-JEPA (github.com/Kar488/M-JEPA) | Method only: fine-tuning recipes compared one by one; leakage-resistant splits | JEPA study configs | Best recipe: long pretraining, slowed encoder (0.408) |
| Cho et al. (2014), the GRU | The step that moves the state on one period under the plan | `rollout` | World model level with the lift model on both: Frat 0.408 against 0.406, Dominick's 0.667 against 0.676 |
| Hansen, Su, Wang (2024), TD-MPC2 | Near periods weighted more in the rollout check | `rollout_discount` | No gain (Frat 0.427; Dominick's 0.685 against 0.667) |
| Bardes, Ponce, LeCun (2022), VICReg | Variance and covariance penalty against a collapsed state | `state_spread_weight` | No gain (same runs) |
| "DWM: separating world effects from actions in latent world models" (arXiv 2607.18715) | Plan effect kept apart from the no-plan state | `plan_split` | No overall gain (Frat 0.417 against 0.408; Dominick's 0.704 against 0.667). Better item-by-store and promoted-week accuracy on Dominick's, worse totals |
| Hafner et al. (2023), DreamerV3 | A state with randomness; squashed targets | Not built | Candidate for unrecorded circular placement |
| Assran et al. (2025), V-JEPA 2 (action-conditioned variant) | Frozen encoder, train the predictor only, rollout loss | Compared in the JEPA study | Frozen was worse for us. Details to verify |
| LeWorldModel and SIGReg | One-term penalty against collapse | Not built (VICReg form used) | To verify |
| WorldTS (arXiv 2609.31162) | Related work: latent dynamics first, then a frozen readout, for forecasting with known future inputs | Related work only | Accuracy claims not verified |
| AD-WM (arXiv 2609.30264) | Not read (site rate-limited) | n/a | To read |

## Still to add

- The PepsiCo paper named as a target to beat: full reference and its reported numbers.
- TimesFM, GraphSAGE and HGT, when those baselines are built from the published papers and author code.
- Explanation methods for the faithfulness paper (Shapley values for LightGBM and the sanity checks in `docs/PROTOCOL.md`).
- Data sources are listed in `docs/PROVENANCE.md`.
