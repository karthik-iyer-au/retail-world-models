# Notes for the two papers

A running list of what each paper has to state, kept here so nothing found
along the way is lost. Paper 1 is about forecast accuracy under commercial
decisions. Paper 2 is about whether the model's explanations are faithful,
and whether faithfulness and accuracy go together. Results are in
`results/README.md`; rules for scoring are in `PROTOCOL.md`.

## How data is split

Each split answers a different question, and both papers must say which
split a number came from.

| Split | What is held out | Question it answers | Status |
|---|---|---|---|
| By time, rolling | The weeks after each cut-off date | How good is the forecast of weeks the model has not seen | In use for every result |
| Final window | The last 8 weeks (28 days on M5) of each dataset | The reported result; touched once, at the end | Locked, unused |
| Validation window | The last weeks of the training data | When to stop training and how to weight a combination of models | To build |
| By product (scaffold-style) | Whole products or whole brands | Can a new item be forecast from its attributes | To build |
| By lever | Scores taken only on weeks with a lever on, or off | Is the gain on promoted weeks or ordinary weeks | In use |
| By level | Item-store up to chain total, shown separately | Is the error on single items or does it add up | In use from the Dominick's follow-up onward |
| By horizon | Week 1 against week 8 | Which model is weak far ahead | To build |

One exception to record: on Breakfast at the Frat the three benchmarks
(seasonal naive, average of last 8 weeks, LightGBM) were scored on the last
three windows, which include the final window, before the final window was
locked. Our model has never been run on it. The benchmarks must not be
tuned any further on that dataset before the final run.

Rules that apply to every comparison:
- A benchmark gets every input our model gets (calendar, crowding, base price).
- Averaging several fits, when used, is applied to both sides.
- Results are averages over several seeds, with the range shown.
- Training batches are drawn at random today; drawing promoted weeks more
  often is a variant to test, with a correction for the changed mix.
- A random split of rows is never used: it would let the model see the
  future of the same item.

## Faithfulness checks (paper 2)

Checks that hold by construction, each with a test in `tests/`:
- The lift is exactly zero when nothing is planned.
- The baseline does not move when the plan changes.
- Baseline plus lift equals the forecast.
- A column not named as a lever is left alone by the "nothing planned" plan.
- Item-to-item weights stay inside the group, never point at the item
  itself, and sum to at most one.

Checks against the data, to run and report:
- Lift against what happened, by lever and lever combination (done once on
  Breakfast at the Frat: close for every lever except display and circular
  together, which is 13% low).
- Lift against what happened, by category and by event (frozen pizza
  under-lifted, cold cereal over-lifted).
- Direction and size: a deeper cut should never lower the forecast; turning
  a lever on should move the forecast the way the data moves.
- Plan A against plan B from the same starting point, compared side by side.
- The dip after a promotion (pantry loading) counted inside the lift.
- Stability: the same explanation from different seeds.
- Agreement between our baseline-and-lift split and Shapley values from
  LightGBM on the same weeks.

Limits to state plainly:
- Merchants promote in weeks they expect to sell well, so part of the
  measured lift is good timing. Lift is what the forecast attributes to the
  plan, not proof of what the plan caused. Dominick's ran pricing
  experiments across stores (Hoch et al.); those are the place to test the
  lift against a real cause.
- About two-thirds of the spread in circular lift in a store is shared
  across the chain for that item and week, and the recorded clues explain
  about 2% of it. Something set centrally and not recorded, most likely
  placement, drives it. Planned: a labelled "if placement were known" test.
- No stock data: a sell-out looks like low demand.
- Weather is not known 8 weeks ahead; it can explain past weeks so a
  heatwave is not credited to a promotion.
- A familiarity (energy) score is planned, to say how far a plan sits from
  anything the model has seen.

The question paper 2 asks: across model versions, does a version whose
explanations pass more of these checks also forecast better?

## Negative results to report

- Item-to-item by attention or conditional similarity over a whole store:
  worse on Breakfast at the Frat and on Dominick's analgesics.
- Same week last year: no gain; only about one circular slot in five repeats.
- Seasonal calendar as on-off columns: worse for our model and for LightGBM
  on both datasets.
- Recorded base price on Breakfast at the Frat: worse; the column is a
  weekly shelf price, not a slow-moving regular price.
- Training four times longer on Dominick's: far worse at summed-up levels,
  unchanged at item-store level.

## Findings on the main hypotheses so far

- World model against direct readout, Breakfast at the Frat: level
  (0.408 against 0.406 overall, five seeds each). The world model is not
  more accurate on this dataset. What it adds is the rollout itself: a
  promotion reaching later weeks through the state, and two plans compared
  from the same starting state.
- Latent (JEPA-style) loss: on the direct model no recipe improves the
  overall score; an encoder trained on the latent loss alone and then frozen
  still reaches 0.428, so the signal is informative. On the world model,
  long pretraining then a slowed encoder is level overall and best on
  display weeks; the latent check trained alongside the forecast loss is
  worse.
- The recipe matters: the first one tried was among the weakest.

## Hypotheses still to test

- A true world model: the state rolled forward week by week under the plan,
  with the forecast read from each rolled state and the JEPA-style check on
  each step, against the direct readout and LightGBM.
- Which way of using the latent loss works (study running).
- Losses: Tweedie with an added term on summed-up totals; negative binomial;
  a two-part model for slow sellers.
- A combination of our model and LightGBM, reported separately from our
  model alone.
- TimesFM and a graph model on product attributes as further comparisons.

## New-item test (built, not yet run)

- Split: a random one in ten products is kept out of training in every window
  (`evaluation.new_items`), same products for every model and seed. Scored at
  item-by-store level, apart from known items.
- Our model: a new item borrows the state and scale of the ten known items in
  the same store with the most labels in common, and is run with its own
  labels and its own plan (`cold_start`). Similarity here is a count of shared
  labels, not a learned similarity. A learned one is still to build.
- Limit: the error scale for a new item uses its real history, which the model
  never saw. This is for scoring only.

## Dominick's training fixes and world model (recorded 6 Oct)

- Negative result: a loss term on store totals worsens the total (1.042) by pulling forecasts 9% low.
- Validation stopping removes the long-training failure (0.683 against 1.473) but does not beat the short run (0.676).
- World model on Dominick's: 0.729 to 0.732 overall, behind LightGBM (0.715) and the lift model (0.676). Item-by-store accuracy is level (0.699 to 0.704); the gap is under-forecasting at summed levels.

## Scope of the world-model claim (agreed 6 Oct)

- Tests so far cover one item's own price and promotion plan, scored on accuracy only. Parity with LightGBM there is the entry ticket, not the result.
- The claim to test: one model that holds up when the plan changes. Tests, in order: (1) plan A against plan B ranking, (2) unusual lever mixes (`evaluation.rare_plans`, built), (3) range change (new items built; delisting to build), (4) category totals under a changed plan (needs the item-to-item model fixed).
- Limits to state: no public data on shelf space or stock; placement within display and circular is not recorded.

## Combination and likelihood (recorded 6 Oct)

- Lift model + LightGBM combination: Frat 0.388 (LightGBM 0.421, lift 0.406), better than both parts on display (0.800) and circular (0.814) weeks. Dominick's 0.670, level with lift (0.676). The two models make different errors on Frat promoted weeks.
- Negative binomial against Tweedie: no material difference (Frat 0.410 against 0.406; Dominick's 0.665 against 0.676).

## World-model literature check (6 Oct)

- Matches the literature: state rolled forward under the plan and checked against a slow-copy encoder's state for the real future (TD-MPC2, V-JEPA 2-AC); pretrain then fine-tune.
- Added from the literature, to test: plan effect kept apart from the no-plan state (DWM, arXiv 2607.18715; ours is separate by construction and exactly zero until something is planned); variance and covariance penalty against collapse (VICReg; SIGReg in LeWorldModel is the related one-term form); near periods weighted more in the rollout check (TD-MPC2).
- Not built: a state with randomness (DreamerV3), relevant to unrecorded circular placement.
- Related work to cite: WorldTS (arXiv 2609.31162), latent dynamics first then a frozen readout, for forecasting with known future inputs. Accuracy claims not verified.
- Not read: AD-WM (arXiv 2609.30264), rate-limited.
- New-item result: borrowing from similar items takes new-item error from 1.510 (zero forecast) to 0.890 on Frat. LightGBM had no new-item route in that run; rerun with one queued.
