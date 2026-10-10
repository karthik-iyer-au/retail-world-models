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

## Plan A against plan B (built 6 Oct, `evaluation.plan_pairs`)

- For the same item and store, every pair of weeks in the test window whose lever mix differs. Two scores: share of pairs where the forecast picks the week that really sold more, and the average miss on the size of the change (log scale).
- This uses real weeks only, so it is a check on ranking real plans, not a causal claim: the two weeks also differ in season and in whatever else changed.
- Fairness: same pairs for every model.

## World-model tricks: results (6 Oct)

- Correction to an earlier note: the world model is not behind on Dominick's. Without the totals loss it scores 0.667 (lift 0.676, LightGBM 0.715). The 0.73 figures were caused by the totals loss.
- Plan split, collapse penalty and week weighting give no overall gain on either dataset at one untuned setting each. The plan split improves item-by-store and promoted-week accuracy on Dominick's (0.693, 1.276) and worsens totals (0.793): it trades level for shape.
- Plan A against plan B result (Frat): all models pick the better week about 78% of the time; LightGBM 78.2%, ours 77.4% to 77.5%. No advantage for the world model on this test.

## Overnight part 1 (recorded 6 Oct)

- Five seeds on Dominick's analgesics: world model 0.666, LightGBM 0.702, lift model 0.718. The lift model's earlier 0.676 was a three-seed figure that did not hold; report five seeds and the spread. The world model is the steadier of our two.
- World model + LightGBM combination: Frat 0.390 (lift + LightGBM 0.388), Dominick's 0.668 (world model alone 0.666). On Dominick's the combination adds nothing.
- Negative result: new-item borrowing fails on Dominick's (2.999 against LightGBM 0.923) because there are no attribute labels to match on. The new-item claim holds only where item attributes exist; a learned similarity over real attributes (size, brand from the item description) is needed before claiming more.

## All Dominick's categories (recorded 7 Oct)

- 28 categories, one seed, settings from analgesics unchanged. LightGBM wins 14, lift 5, world model 9. Mean overall: LightGBM 0.688, lift 0.784, world model 0.747. On promoted weeks LightGBM wins 24 of 27.
- The analgesics result does not generalise. Analgesics was the category used to set our models, so reporting it alone would be selective. The paper must report all categories.
- Our models fail badly in a few categories (snacks, soft drinks, bath soap), which suggests a stability problem (scaling of large-volume items, or the regular-price estimate) to diagnose before any more tuning.
- The first Colab session ended after about 23 hours; the queue resumed from saved runs in a new session.

## Diagnosis of the Dominick's failures (7 Oct)

- Where the error is: item-by-store accuracy is close to LightGBM in every bad category (for example snacks 0.74 against 0.67). The loss is at the category total, where our models over-forecast by 20% to 40% in one window.
- Cause, reproduced on a six-store sample of snacks (lift model, first window, forecast over actual 1.63): the over-forecast sits in items with a partial, spiky recent history. Items recorded in 8 to 20 of the last 26 weeks are 8% of sales and are forecast at 9 times actual. These are in-and-out and seasonal lines that sold heavily for a few weeks (one week in September 1996 had category sales 3.5 times normal on a 10 cent promotion) and then fell away. Our model sets each item's level from its 26-week average, so it holds them at the old level.
- Capping extreme weeks in the level (`scale_cap`) helps little (1.63 to 1.53). Taking the level from the latest weeks (`scale_window`) removes most of it: 1.31 with 8 weeks, 1.14 with 4 weeks. Sample result, one seed, 800 steps; to be confirmed on full data.
- LightGBM does not have this problem because it sees short recent averages directly.
- Also found: the Dominick's loader fails on cigarettes under the local pandas version (mixed types in the SALE column); it ran on Colab. To fix.

## M5 with the lift model and the world model (7 Oct)

- Practice window, one seed: lift model 0.683, world model 0.692, plain model 0.699 (five seeds), LightGBM 0.715. More seeds needed before ranking our three. Both new runs under-forecast by about 5%.

## Level fix confirmed in part (7 Oct)

- Full-data check on eight categories, one seed: a latest-weeks level cuts the snacks error (lift 0.959 to 0.728, world model 1.463 to 0.874) and brings the world model level with LightGBM on soft drinks and bath soap. Controls are unharmed (analgesics lift 0.718 to 0.649). Cheese and cigarettes do not improve, and the lift model on soft drinks and bath soap does not improve.
- So in-and-out items are one cause, not the only one. What remains is a category-total bias that differs by window and by model. Next diagnosis: cheese (under-forecast 16% in the first window) and cigarettes (over-forecast 58% in the second window for the lift model).
- The world model with an 8-week level is the steadiest version so far across these eight.

## Second diagnosis: cheese (7 Oct, six-store sample, lift model)

- First window (Thanksgiving and Christmas): promoted weeks are under-forecast by about a third, and the two holiday weeks by 25% to 45%. LightGBM on the same sample has the same miss (promoted weeks 0.68 of actual, against 0.66 for ours). This is not a fault particular to our model. Year-ago inputs, the holiday calendar, the world-model rollout and longer training do not fix it.
- Second window (January, after the holidays): our model over-forecasts ordinary weeks by 11%; LightGBM is level. A level from non-promoted weeks, year-ago inputs and the calendar do not fix it.
- A structural difference from LightGBM: our loss is taken on sales relative to each item's own level, so every item counts alike; LightGBM's loss is on units, so big sellers count for more and category totals are held closer. Weighting our loss by volume (`volume_weight`) with a Tweedie power of 1.1 (LightGBM's) cuts the January over-forecast from 1.068 to 1.044 and the snacks first-window over-forecast from 1.63 to 1.39. It helps, it is not the whole gap, and combined with the 4-week level on snacks it is no better than the 4-week level alone (1.20 against 1.14).
- Sample results, one seed, 800 steps. Full-data job `dominicks_units_loss` is ready.

## Cheese, continued: seasonal inputs and the world model in January (7 Oct, six-store sample, 800 steps)

- Cheese peaks do not sit on a fixed week relative to the holiday (1993 and 1994: mid December; 1995: mid November and mid to late December; 1996: Thanksgiving week and the week of 26 December). They follow the chain's promotion timing. Event flags alone cannot place them.
- January window, forecast over actual (LightGBM 0.96 overall, 1.00 on ordinary weeks): lift 1.07, world-model rollout 1.08, calendar with after-event flags 1.06, smooth time-of-year input 1.09. None closes it at this training length. The over-forecast lasts all eight weeks, so it is a seasonal level (cheese sells more in November and December), not a one-week dip.
- The world-model rollout was tested on the January window and gives no gain there.
- Volume weighting now has a dollar form (`volume_by: dollars`) so items sold by weight and by pack are on one footing.
- Open: whether full-length training on all years with these inputs learns the seasonal level. Job `dominicks_seasonal_long` tests that.
- Related platform, not our method: stable-worldmodel (arXiv 2605.21800), a library and benchmark for world models in control and video, with tests under controlled changes in the environment. Our "unusual plan" tests are the same idea applied to retail.

## Peaks and dips: what was missing and what was added (7 Oct)

- Gap found in the world-model version: the roll-forward was fed each week's plan but not each week's sales, so it was autoregressive on actions and not on outcomes. Added `feedback`: each week's sales are fed into the next step (real sales for a share of training examples, the model's own forecast otherwise and always when forecasting; the no-plan path always uses its own). On generated data it still gives zero lift with nothing planned and learns the rise and the dip.
- Added `peak_weight`: weeks far from the item's usual level count for more in the loss. Risk to check: over-calling promotions and worse ordinary weeks.
- Added a peaks-and-dips score (`evaluation.after_promo`): forecast over actual and error in promoted weeks, in the weeks just after a promotion ended, and in ordinary weeks. This is the direct measure of whether a model sees the dip.
- Job `dominicks_peaks_and_dips` (30 runs, six categories) compares LightGBM, the world model, feedback, weighting, and both.
- Clarified against stable-worldmodel (arXiv 2605.21800): its solver-and-cost loop is how a trained model is used to choose actions; it does not train the model. The matching piece for us would be a promotion optimiser, which is not built.

## 8-week level across all Dominick's categories (7 Oct)

- World model, mean overall: 0.747 with the 26-week level, 0.717 with the 8-week level; LightGBM 0.688. Median 0.689, 0.736, 0.676. Wins against LightGBM: 14 then 10 of 28. The change removes the worst failures and is not a uniform gain; it should become the default only together with whatever the peaks-and-dips job supports.

## Peaks and dips: does the world model see the week after a promotion (Dominick's, six categories, one seed)

Job `configs/jobs/dominicks_peaks_and_dips.yaml`. Development windows, 8-week level. Weeks are split into promoted, the weeks just after a promotion, and ordinary. f/a is forecast over actual (1.0 is right), err is RMSSE. Lower error is better.

| Category | Model | Overall | Total | Item | Promoted f/a | Promoted err | After promo f/a | After promo err | Ordinary f/a | Ordinary err |
|---|---|---|---|---|---|---|---|---|---|---|
| Cheese | LightGBM | 0.7421 | 0.8461 | 0.6560 | 0.800 | 1.068 | 0.969 | 0.374 | 0.977 | 0.539 |
| Cheese | World model | 1.1163 | 1.4361 | 0.7417 | 1.207 | 1.514 | 1.035 | 0.405 | 1.025 | 0.551 |
| Cheese | + sales feedback | 1.0721 | 1.3494 | 0.7312 | 1.081 | 1.379 | 1.085 | 0.407 | 1.050 | 0.564 |
| Cheese | + dollar and peak weighting | 0.8137 | 0.9316 | 0.6959 | 1.001 | 1.221 | 1.009 | 0.402 | 1.055 | 0.555 |
| Cheese | + both | 0.8697 | 1.0172 | 0.7063 | 0.978 | 1.246 | 1.029 | 0.401 | 1.064 | 0.561 |
| Snack crackers | LightGBM | 0.5871 | 0.5132 | 0.6709 | 0.908 | 0.937 | 0.921 | 0.386 | 1.022 | 0.471 |
| Snack crackers | World model | 0.8743 | 0.8619 | 0.8434 | 1.195 | 1.279 | 1.329 | 0.585 | 1.132 | 0.608 |
| Snack crackers | + sales feedback | 0.9409 | 0.9443 | 0.8362 | 1.277 | 1.223 | 1.348 | 0.536 | 1.173 | 0.563 |
| Snack crackers | + dollar and peak weighting | 0.9079 | 0.9718 | 0.7389 | 1.317 | 1.077 | 1.179 | 0.434 | 1.111 | 0.500 |
| Snack crackers | + both | 1.2938 | 1.6536 | 0.7285 | 1.458 | 1.074 | 1.223 | 0.442 | 1.150 | 0.498 |
| Soft drinks | LightGBM | 0.7469 | 0.9343 | 0.5398 | 0.768 | 0.721 | 0.904 | 0.110 | 0.863 | 0.193 |
| Soft drinks | World model | 0.7467 | 0.8690 | 0.6135 | 1.010 | 0.867 | 1.316 | 0.180 | 0.964 | 0.239 |
| Soft drinks | + sales feedback | 1.0499 | 1.3268 | 0.7199 | 0.974 | 1.111 | 0.997 | 0.133 | 0.890 | 0.202 |
| Soft drinks | + dollar and peak weighting | 0.7767 | 0.8699 | 0.6959 | 1.102 | 0.996 | 1.235 | 0.165 | 1.089 | 0.215 |
| Soft drinks | + both | 0.7814 | 0.8567 | 0.7164 | 0.956 | 1.044 | 1.333 | 0.165 | 1.159 | 0.219 |
| Bottled juice | LightGBM | 0.9920 | 1.1805 | 0.7989 | 0.714 | 1.926 | 0.957 | 0.477 | 0.946 | 0.586 |
| Bottled juice | World model | 1.0998 | 1.2552 | 0.9084 | 0.924 | 2.502 | 1.035 | 0.557 | 0.918 | 0.590 |
| Bottled juice | + sales feedback | 1.0457 | 1.1467 | 0.9055 | 0.936 | 2.418 | 1.153 | 0.588 | 0.958 | 0.607 |
| Bottled juice | + dollar and peak weighting | 1.0625 | 1.1623 | 0.9459 | 1.136 | 2.580 | 1.050 | 0.572 | 0.961 | 0.605 |
| Bottled juice | + both | 1.2215 | 1.3590 | 1.0591 | 1.299 | 3.179 | 1.181 | 0.636 | 0.982 | 0.604 |
| Analgesics | LightGBM | 0.7006 | 0.8093 | 0.6879 | 0.887 | 1.270 | 0.927 | 0.500 | 1.028 | 0.588 |
| Analgesics | World model | 0.6793 | 0.7381 | 0.7043 | 0.923 | 1.297 | 0.966 | 0.563 | 1.047 | 0.597 |
| Analgesics | + sales feedback | 0.8379 | 1.0787 | 0.7218 | 0.978 | 1.384 | 0.978 | 0.575 | 1.099 | 0.606 |
| Analgesics | + dollar and peak weighting | 0.8265 | 1.0427 | 0.7166 | 1.027 | 1.310 | 0.966 | 0.578 | 1.115 | 0.609 |
| Analgesics | + both | 0.9008 | 1.2129 | 0.7239 | 1.015 | 1.373 | 0.930 | 0.579 | 1.138 | 0.609 |
| Frozen entrees | LightGBM | 0.4167 | 0.3674 | 0.5009 | 0.840 | 0.732 | 0.679 | 0.385 | 0.948 | 0.281 |
| Frozen entrees | World model | 0.3932 | 0.3250 | 0.4973 | 0.881 | 0.742 | 0.705 | 0.387 | 0.887 | 0.272 |
| Frozen entrees | + sales feedback | 0.4113 | 0.3534 | 0.4999 | 0.820 | 0.744 | 0.696 | 0.390 | 0.892 | 0.277 |
| Frozen entrees | + dollar and peak weighting | 0.4001 | 0.3295 | 0.5031 | 0.878 | 0.736 | 0.596 | 0.410 | 0.855 | 0.282 |
| Frozen entrees | + both | 0.3769 | 0.2934 | 0.5106 | 0.982 | 0.785 | 0.727 | 0.392 | 0.966 | 0.276 |

Findings.

- The unchanged world model does not see the dip where the dip is large. After a promotion it forecasts 1.33 times actual in snack crackers and 1.32 in soft drinks, against 1.13 and 0.96 in ordinary weeks. LightGBM is at 0.92 and 0.90. In cheese, bottled juice and analgesics the world model is within 4 percent after a promotion. In frozen entrees every model, LightGBM included, forecasts about 0.70 of actual after a promotion, so there the week after is stronger than any model expects.
- Feeding the model's own sales forecast back into the roll-forward fixes the soft drinks dip (1.32 to 1.00, error 0.180 to 0.133) and nothing else. It makes the overall score worse in four of six categories, mostly through the category total.
- Dollar and peak weighting fixes the cheese promoted weeks (f/a 1.21 to 1.00, overall 1.116 to 0.814) and cuts the snack crackers after-promotion error (0.585 to 0.434) and item error (0.843 to 0.739). It is worse overall in snack crackers, soft drinks and analgesics and does not hurt ordinary-week error by more than 0.02 anywhere.
- Both together is not better than either alone, except frozen entrees (0.377, best of all models).
- LightGBM stays best overall in cheese, snack crackers and bottled juice. The world model is best or level in analgesics, soft drinks and frozen entrees.
- One seed. None of these changes becomes a default on this evidence.

## Recent and longer level together, and roll-forward normalisation (Dominick's, development windows, one seed)

Jobs `configs/jobs/dominicks_level_views.yaml` and `configs/jobs/dominicks_roll_norm.yaml`.

Level views: the model is scaled by the 26-week level and is also shown the 8-week and 4-week levels as inputs, so it can learn which to trust per item. Overall score, lower is better.

| Category | LightGBM | World model, 26-week level | World model, 8-week level | World model, both levels |
|---|---|---|---|---|
| che | 0.742 | 0.996 | 1.116 | 0.883 |
| sna | 0.587 | 1.463 | 0.874 | 1.515 |
| sdr | 0.747 | 0.864 | 0.747 | 1.069 |
| bjc | 0.992 | 1.229 | 1.100 | 1.159 |
| ana | 0.702 | 0.666 | 0.679 | 0.648 |
| fre | 0.417 | 0.382 | 0.393 | 0.399 |
| cig | 0.534 | 0.690 | 0.643 | 0.440 |
| tpa | 0.731 | 0.779 | 0.857 | 0.831 |
| ber | 0.454 | 0.470 | 0.505 | 0.490 |
| did | 0.344 | 0.393 | 0.406 | 0.388 |

Showing both levels is the best world-model version in 4 of 10 categories (cheese, analgesics, cigarettes, dish detergent) and beats LightGBM in 3 (analgesics, frozen entrees, cigarettes). It does not fix the categories the 8-week level fixed: snack crackers (1.515, promoted weeks forecast at 1.86 times actual) and soft drinks (1.069) stay at the 26-week failure. The model does not learn to switch to the recent level for in-and-out items when the recent level is only an input and the 26-week level still sets the scale.

Roll-forward normalisation (layer normalisation on the rolled state), against the same model without it (8-week level):

| Category | World model | World model + normalisation | With sales feedback | With sales feedback + normalisation |
|---|---|---|---|---|
| che | 1.116 | 0.963 | 1.072 | 1.019 |
| sna | 0.874 | 0.933 | 0.941 | 1.268 |
| sdr | 0.747 | 0.882 | 1.050 | 1.007 |
| bjc | 1.100 | 1.026 | 1.046 | 1.049 |
| ana | 0.679 | 0.790 | 0.838 | 0.849 |
| fre | 0.393 | 0.396 | 0.411 | 0.394 |

Normalisation helps in two categories, hurts in three and is level in one. It is ruled out as a general fix. With sales feedback and normalisation the snack crackers after-promotion forecast is right (0.97 of actual, error 0.419 against 0.585) and item error is the lowest of any world-model version (0.702), but the category total gets much worse (1.60). The M5 run with normalisation is recorded separately when it finishes.

M5 practice window, world model with roll-forward normalisation (`m5_validation_world_model_roll_norm`, one seed, only change is `roll_norm: true`): overall 0.646, total 0.510, item 0.835, forecast/actual 0.962. Without it the world model scored 0.692 (total 0.600, item 0.833), the lift model 0.683 and LightGBM 0.715. The gain is in the totals, not the item level. Normalisation helps on the 28-step daily roll-forward and not on the 8-step weekly one on Dominick's. One seed; needs more seeds before it is claimed.

## Item roles (Dominick's, six categories, development windows, one seed)

Job `configs/jobs/dominicks_item_roles.yaml`. Each item gets one role from its own sales before the test windows (`src/rwm/data/roles.py`): new line, in-and-out, volume driver (top 5% of items by units that sell on deal), long tail (slowest items making the last 5% of units), hi-lo (half or more of units on deal), promo responsive (at least twice ordinary sales on deal), core. Volume driver is not called a KVI: a KVI list rests on shopper price perception and basket data, which these datasets do not have.

Overall score, lower is better. Baselines are the peaks-and-dips rows above (world model with an 8-week level).

| Category | LightGBM | LightGBM + role | World model | World model + role | World model + role picks the level |
|---|---|---|---|---|---|
| Cheese | 0.742 | 0.754 | 1.116 | 1.107 | 1.074 |
| Snack crackers | 0.587 | 0.577 | 0.874 | 1.163 | 2.422 |
| Soft drinks | 0.747 | 0.734 | 0.747 | 0.919 | 1.046 |
| Bottled juice | 0.992 | 0.987 | 1.100 | 1.062 | 1.140 |
| Analgesics | 0.701 | 0.694 | 0.679 | 0.639 | 0.655 |
| Frozen entrees | 0.417 | 0.423 | 0.393 | 0.411 | 0.415 |

Forecast over actual in promoted weeks and the weeks just after:

| Category | World model promoted | + role | + role picks level | World model after promo | + role | + role picks level |
|---|---|---|---|---|---|---|
| Cheese | 1.207 | 1.072 | 1.079 | 1.035 | 1.070 | 1.062 |
| Snack crackers | 1.195 | 1.438 | 2.419 | 1.329 | 1.436 | 1.188 |
| Soft drinks | 1.010 | 0.977 | 1.102 | 1.316 | 1.255 | 1.333 |

Findings.

- The role label does almost nothing for LightGBM: within 0.013 of the plain run in every category, better in four. LightGBM already reads the same information from its sales and price history.
- For the world model the label moves results by more than LightGBM's but in both directions: best world-model result so far on analgesics (0.639) and better on bottled juice, worse on snack crackers and soft drinks. Single-seed world-model runs have varied by up to about 0.07 on analgesics, so only the snack crackers and soft drinks losses are clearly larger than run-to-run variation.
- Letting the role pick the level does not fix snack crackers or soft drinks. Snack crackers collapses (2.422): item-level error improves (0.725 against 0.843) but promoted weeks are forecast at 2.4 times actual and the category total breaks (3.45). This is the same pattern as sales feedback with normalisation: better items, a broken total in promoted weeks.
- The after-promotion dip in soft drinks (about 1.3 times actual) is not fixed by any role variant.
- No role variant becomes a default. The role table itself is kept for reporting results by item type.

## M5 practice window: roll-forward normalisation, three seeds

| Run | Seed 1 | Seed 2 | Seed 3 | Mean | Range |
|---|---|---|---|---|---|
| World model + normalisation, overall | 0.646 | 0.675 | 0.647 | 0.656 | 0.646 to 0.675 |
| World model, overall | 0.692 | 0.680 | 0.750 | 0.707 | 0.680 to 0.750 |
| World model + normalisation, total | 0.510 | 0.577 | 0.518 | 0.535 | |
| World model, total | 0.600 | 0.579 | 0.722 | 0.634 | |
| World model + normalisation, item by store | 0.835 | 0.833 | 0.835 | 0.834 | |
| World model, item by store | 0.833 | 0.830 | 0.830 | 0.831 | |

References: lift model 0.683 and LightGBM 0.715 (one seed each in this table's setting; LightGBM is deterministic).

The gain holds on three seeds: 0.656 against 0.707, and the worst normalised seed (0.675) is better than the best plain seed (0.680). All of it is in the totals; item-by-store error is unchanged (0.834 against 0.831). Normalisation also makes the model steadier across seeds (range 0.029 against 0.070). It is the best M5 result so far and ahead of LightGBM by 0.059. On Dominick's (8 weekly steps) it did not help; on M5 (28 daily steps) it does, which fits the reading that it stops the rolled state drifting over a long roll.

## LightGBM against the world model on three seeds (Dominick's, six categories, development windows)

Job `configs/jobs/dominicks_peaks_seeds.yaml` adds seeds 2 and 3 to the peaks-and-dips runs (world model with an 8-week level). Overall score, mean of three seeds (range), lower is better.

| Category | LightGBM | World model | Seeds where the world model is better |
|---|---|---|---|
| Cheese | 0.749 (0.742 to 0.759) | 1.088 (0.981 to 1.167) | 0 of 3 |
| Snack crackers | 0.586 (0.581 to 0.590) | 0.813 (0.751 to 0.874) | 0 of 3 |
| Soft drinks | 0.759 (0.747 to 0.775) | 0.902 (0.747 to 1.123) | 1 of 3 |
| Bottled juice | 0.987 (0.979 to 0.992) | 1.074 (1.033 to 1.100) | 0 of 3 |
| Analgesics | 0.701 (0.689 to 0.712) | 0.711 (0.655 to 0.800) | 2 of 3 |
| Frozen entrees | 0.419 (0.417 to 0.421) | 0.406 (0.391 to 0.434) | 2 of 3 |

Findings.

- LightGBM is steady across seeds (ranges within 0.03). The world model is not: its range within a category is 0.04 to 0.38 (soft drinks 0.747 to 1.123, cheese 0.981 to 1.168).
- On the mean, LightGBM is ahead in five of six categories. The world model is ahead only on frozen entrees (0.406 against 0.419) and level on analgesics (0.711 against 0.701).
- The world model's single-seed wins on analgesics and soft drinks earlier were within its seed spread. Run-to-run instability is the main weakness on Dominick's, and the case for seed averaging or a steadier model (roll-forward normalisation fixed this on M5) is strong.

## Brand switching between items in a store-week (Dominick's, six categories, three seeds)

Job `configs/jobs/dominicks_switching.yaml`. The world model (8-week level) is run as before; in each store and week its item forecasts are rescaled so that the store's total for the category is the baseline total plus a learned share of the summed promotion lift. The share is learned on the validation weeks (one per run and origin); the rest of each item's lift is treated as taken from other items. Same seeds as the world model rows, so each pair differs only in the switching step.

Overall score, mean of three seeds (range); learned share of lift that is new sales, range over runs.

| Category | LightGBM | World model | World model + switching | Seeds where switching is better | Share of lift that is new sales |
|---|---|---|---|---|---|
| Cheese | 0.749 | 1.088 | 1.111 (1.005 to 1.198) | 0 of 3 | 0.70 to 1.00 |
| Snack crackers | 0.586 | 0.813 | 0.752 (0.727 to 0.776) | 3 of 3 | 0.00 to 0.66 |
| Soft drinks | 0.759 | 0.902 | 0.789 (0.711 to 0.895) | 3 of 3 | 0.64 to 1.00 |
| Bottled juice | 0.987 | 1.074 | 1.066 (1.009 to 1.101) | 1 of 3, 1 level | 0.69 to 1.00 |
| Analgesics | 0.701 | 0.711 | 0.707 (0.661 to 0.778) | 1 of 3 | 0.71 to 1.00 |
| Frozen entrees | 0.419 | 0.406 | 0.416 (0.398 to 0.434) | 0 of 3, 1 level | 0.68 to 1.00 |

Findings.

- Switching fixes what it was built for. In snack crackers and soft drinks, the two categories where promoted weeks broke the total, it is better on every seed: snack crackers 0.813 to 0.752, soft drinks 0.902 to 0.789. Promoted-week forecast over actual in snack crackers falls from 1.20 to 0.98 on seed 1, and the item-level error improves too (0.843 to 0.803), so it does not trade item accuracy for the total.
- The learned share is itself a finding and reads as merchants would expect: in snack crackers most of a promotion's lift is taken from other crackers (the share of new sales is 0.0 to 0.66), while in bottled juice, analgesics and frozen entrees the lift is mostly new to the category (about 0.9). Where the share comes out at 1.0 the step changes nothing, which is a built-in check (frozen entrees and bottled juice seed 3 match the world model exactly).
- Elsewhere it is level or slightly worse (cheese +0.02, frozen entrees +0.01): one share per run is too blunt where the switching is small.
- LightGBM is still ahead on the mean in five of six categories; switching closes the soft drinks gap to 0.03 and leaves the instability across seeds as the main weakness. A three-seed average inside each run is queued next.

## Averaging three fits inside each run (Dominick's, six categories)

Job `configs/jobs/dominicks_switching_avg3.yaml`: the world model with brand switching, each run the average of three fits with different seeds (one run per category). Overall score, lower is better. The other columns are three-seed means of single fits.

| Category | LightGBM | World model | World model + switching | World model + switching, average of 3 fits |
|---|---|---|---|---|
| Cheese | 0.749 | 1.088 | 1.111 | 0.921 |
| Snack crackers | 0.586 | 0.813 | 0.752 | 0.722 |
| Soft drinks | 0.759 | 0.902 | 0.789 | 0.816 |
| Bottled juice | 0.987 | 1.074 | 1.066 | 1.170 |
| Analgesics | 0.701 | 0.711 | 0.707 | **0.651** |
| Frozen entrees | 0.419 | 0.406 | 0.416 | **0.411** |

Findings.

- Averaging three fits gives the best world-model result so far on cheese (0.921), snack crackers (0.722) and analgesics (0.651, ahead of LightGBM by 0.050). It beats LightGBM in two categories (analgesics, frozen entrees).
- It is not uniformly better: bottled juice gets worse (1.170) and soft drinks is slightly worse than the single-fit mean (0.816 against 0.789). One averaged run per category is itself one draw, so these two differences are within the spread seen earlier.
- LightGBM stays ahead on cheese, snack crackers, soft drinks and bottled juice. The gap is smallest in soft drinks (0.06) and largest in cheese (0.17).
- Item-level error with averaging is the best of any world-model version in cheese (0.696) and snack crackers (0.700); the remaining gap to LightGBM in those categories is mostly in the category totals.

## Brand switching with the normalised roll-forward (Dominick's, six categories, three seeds)

Job `configs/jobs/dominicks_switching_roll_norm.yaml` (commit 73c6686): the world model with brand switching and layer normalisation in the roll-forward (`roll_norm`), seeds 1 to 3. Overall score, three-seed mean with the range across seeds in brackets; lower is better.

| Category | LightGBM | World model | World model + switching | World model + switching + normalised roll-forward |
|---|---|---|---|---|
| Cheese | 0.749 | 1.088 (0.981 to 1.167) | 1.111 (1.005 to 1.198) | 1.028 (0.963 to 1.067) |
| Snack crackers | 0.586 | 0.813 (0.751 to 0.874) | 0.752 (0.727 to 0.776) | 0.830 (0.713 to 0.979) |
| Soft drinks | 0.759 | 0.902 (0.747 to 1.123) | 0.789 (0.711 to 0.895) | 0.827 (0.749 to 0.887) |
| Bottled juice | 0.987 | 1.074 (1.033 to 1.100) | 1.066 (1.009 to 1.101) | 1.047 (1.020 to 1.080) |
| Analgesics | 0.701 | 0.711 (0.655 to 0.800) | 0.707 (0.661 to 0.778) | **0.696** (0.661 to 0.760) |
| Frozen entrees | 0.419 | 0.406 (0.391 to 0.434) | 0.416 (0.398 to 0.434) | **0.404** (0.386 to 0.419) |

Three-seed means of the other measures with normalisation: category total 1.275, 0.850, 0.991, 1.175, 0.774, 0.349; item 0.732, 0.770, 0.647, 0.903, 0.707, 0.494; promoted-week forecast over actual 1.08, 1.05, 0.92, 0.96, 0.92, 0.83 (cheese to frozen entrees).

Findings.

- Normalisation narrows the spread across seeds in five of six categories, as it did on M5: cheese 0.19 to 0.10, soft drinks 0.18 to 0.14, bottled juice 0.09 to 0.06, analgesics 0.12 to 0.10, frozen entrees 0.04 to 0.03 (against switching alone). Snack crackers is the exception: seed 3 drifts to 0.979 (total 1.12, promoted and after-promotion weeks over-forecast by 16 and 20 per cent), so its range widens from 0.05 to 0.27.
- It improves the mean over switching alone in four categories (cheese by 0.08, bottled juice by 0.02, analgesics and frozen entrees by 0.01) and worsens it in snack crackers (by 0.08, all from seed 3) and soft drinks (by 0.04).
- With normalisation the world model beats LightGBM on the three-seed mean in analgesics (0.696 against 0.701) and frozen entrees (0.404 against 0.419). LightGBM stays ahead in cheese, snack crackers, soft drinks and bottled juice; the gaps are 0.28, 0.24, 0.07 and 0.06.
- After-promotion weeks are still over-forecast in snack crackers and soft drinks (1.15 and 1.13): the dip after a deal is under-learned there, which neither switching nor normalisation addresses.
- Next: averaging fits with normalisation on, which combines the two changes that each reduce seed noise.

## Lift model on M5, three seeds

Job `configs/jobs/m5_lift_seeds.yaml` (commit 7e5c0f4): seeds 2 and 3 of `m5_validation_lift`, added to seed 1. Overall WRMSSE on the M5 practice window, lower is better.

| Model | Seeds | Mean | Range |
|---|---|---|---|
| LightGBM | 1 | 0.715 | |
| Lift model | 3 | 0.713 | 0.683 to 0.734 |
| World model | 3 | 0.707 | 0.680 to 0.750 |
| World model + normalised roll-forward | 3 | **0.656** | 0.646 to 0.675 |

Lift model by seed: 0.683, 0.721, 0.734 (category total 0.641 and 0.664 for seeds 2 and 3; item-store 0.830 and 0.833).

Findings.

- The lift model's first seed (0.683) was a good draw. Over three seeds it ties LightGBM (0.713 against 0.715) and is level with the plain world model.
- The world model with the normalised roll-forward is the only model clearly ahead on M5: 0.057 better than the lift model on the mean, and its worst seed (0.675) beats the lift model's mean.
- Normalisation is now the M5 default for the world model in the paper comparison.

## Averaging three fits with switching and the normalised roll-forward (Dominick's, seed 1)

Job `configs/jobs/dominicks_switching_avg3_roll_norm.yaml` (commit edad17d): the world model with brand switching and the normalised roll-forward, each run the average of three fits. One run per category; frozen entrees did not finish because the runtime stopped. Overall score, lower is better.

| Category | LightGBM | Switching + normalisation (3-seed mean) | Switching, average of 3 fits | Switching + normalisation, average of 3 fits |
|---|---|---|---|---|
| Cheese | 0.749 | 1.028 | 0.921 | 0.983 |
| Snack crackers | 0.586 | 0.830 | 0.722 | 1.806 |
| Soft drinks | 0.759 | 0.827 | 0.816 | 0.770 |
| Bottled juice | 0.987 | 1.047 | 1.170 | 1.061 |
| Analgesics | 0.701 | 0.696 | 0.651 | **0.625** |
| Frozen entrees | 0.419 | 0.404 | 0.411 | not finished |

Category total by window (first, second): cheese 1.33, 1.10; snack crackers 0.93, 4.20; soft drinks 1.02, 0.82; bottled juice 1.10, 1.26; analgesics 0.73, 0.52. Item-store error by window: cheese 0.74, 0.69; snack crackers 0.81, 0.61; soft drinks 0.58, 0.66; bottled juice 0.72, 1.12; analgesics 0.67, 0.72.

Findings.

- Analgesics reaches 0.625, the best result on any Dominick's category so far and 0.076 ahead of LightGBM. Soft drinks reaches 0.770, the best world-model result there and 0.011 behind LightGBM.
- Snack crackers fails in the second window: promoted weeks are forecast at 2.6 times actual and the category total error is 4.20, while its item-store error (0.61) is the best of any version. The error is in the category total, not the item forecasts, so it sits in how promotion lift adds up across items in the store; the learned switching share for this run has not yet been checked. This is the same over-forecast of promoted weeks seen on seed 3 with normalisation alone.
- Cheese and bottled juice do not improve on the best earlier versions.
- One run per category is one draw; seeds 2 and 3 of this setup are queued next to separate the change from the noise.

## Averaged switching with the normalised roll-forward, three seeds (Dominick's)

Jobs `dominicks_switching_avg3_roll_norm.yaml` (seed 1, commit edad17d), `dominicks_fre_switching_avg3_roll_norm.yaml` (frozen entrees seed 1) and `dominicks_switching_avg3_roll_norm_seeds.yaml` (seeds 2 and 3, commit 9a0b8a6). Each run is the average of three fits. Overall score, three-seed mean with the range across seeds in brackets; lower is better.

| Category | LightGBM | Switching + normalisation, single fit (3-seed mean) | Switching + normalisation, average of 3 fits (3-seed mean) | Seeds 1, 2, 3 |
|---|---|---|---|---|
| Cheese | 0.749 | 1.028 | 0.971 (0.943 to 0.988) | 0.983, 0.943, 0.988 |
| Snack crackers | 0.586 | 0.830 | 1.092 (0.730 to 1.806) | 1.806, 0.740, 0.730 |
| Soft drinks | 0.759 | 0.827 | **0.743** (0.713 to 0.770) | 0.770, 0.713, 0.746 |
| Bottled juice | 0.987 | 1.047 | 1.048 (1.019 to 1.065) | 1.061, 1.019, 1.065 |
| Analgesics | 0.701 | 0.696 | **0.622** (0.618 to 0.625) | 0.625, 0.622, 0.618 |
| Frozen entrees | 0.419 | 0.404 | **0.391** (0.386 to 0.398) | 0.386, 0.388, 0.398 |

Findings.

- The world model now beats LightGBM on the three-seed mean in three of six categories: analgesics by 0.079, frozen entrees by 0.028 and soft drinks by 0.016. Every seed beats LightGBM in analgesics and frozen entrees; two of three do in soft drinks.
- Averaging three fits with normalisation is the steadiest world-model setup so far. The spread across seeds is 0.05 or less in five categories (analgesics 0.007, frozen entrees 0.012, bottled juice 0.046, cheese 0.045, soft drinks 0.057), against up to 0.38 for the plain world model.
- Snack crackers seed 1 (1.806) is a single failed run: seeds 2 and 3 score 0.740 and 0.730, the best world-model results in that category. The failure is in the second window's category total (4.20) with promoted weeks forecast at 2.6 times actual, while item-store error stays normal. It shows the summed promotion lift can still run away on one fit in three; a guard on the store-level lift is the fix to test.
- LightGBM stays ahead in cheese (0.22), snack crackers (0.14 on the median seed) and bottled juice (0.06).
- This setup (brand switching, normalised roll-forward, average of three fits) is the candidate world model for the Dominick's comparison in the paper.

## Compute and reproducibility (for the paper's reproducibility statement)

- Every run writes `manifest.json` next to its scores: code commit (runs refuse to start on uncommitted code), the full config and its checksum, the seed, checksums of every data file read, Python and package versions (including PyTorch), operating system, and the device.
- From commit "Record hardware and run times" on, the manifest also records the machine (processor, logical cores, memory, GPU name and memory, GPU count, CUDA and cuDNN versions), the start and finish time, total wall-clock seconds, and fit and forecast seconds for each window.
- Runs saved before that commit have their start time in the manifest and their finish time from when `metrics.json` was written; `scripts/run_times.py <results folder>` builds one table of every run (device, start, finish, wall minutes, score) and the total GPU-hours, marking which times are recorded and which come from file times.
- All GPU runs so far used Google Colab Pro+ with one NVIDIA T4 (16 GB); LightGBM runs used the Colab CPU.
- Development runs (choosing the setup) use the rolling windows before the held-out final 8 weeks; the final 8 weeks are not loaded by any development run. The paper should state how many setups were tried on the development windows (the results README lists every one) and report the locked final test once.

## Final tests, fixed on 9 October 2026 before any of them ran

Chosen on the development windows only; the configs are copies of the development configs with only the name and the scored window changed (checked by `tests/test_final_configs.py`). Each is run once with seeds 1, 2 and 3 and reported as it comes out, with no reruns or changes chosen after seeing the results.

| Dataset | Window | Models |
|---|---|---|
| M5 | official test period, 23 May to 19 June 2016 (`include_test`) | world model with normalised roll-forward; LightGBM, one model per store; lift model |
| Dominick's (cheese, snack crackers, soft drinks, bottled juice, analgesics, frozen entrees) | last 8 weeks | world model with brand switching, normalised roll-forward, average of three fits; LightGBM |
| Breakfast at the Frat | last 8 weeks | world model; lift model with readout dropout and weight decay; LightGBM with all recorded levers |

Jobs: `final_m5.yaml`, `final_dominicks.yaml`, `final_frat.yaml`, `final_m5_lift.yaml`, run in that order after the averaged M5 practice run.

The Frat benchmarks (seasonal naive, average of last 8 weeks, LightGBM) were scored once on windows that included the final 8 weeks before the window was locked (see "How data is split"); the final LightGBM here uses the development settings, which were not tuned on that score.

## Terms for the two models (agreed 9 October 2026)

- Lift model: a direct multi-horizon forecaster, conditioned on the plan. One encoded state, and each future week forecast from it with that week's plan; no transition between weeks.
- World model: an action-conditioned latent dynamics model (latent rollout). A learned transition, next state = f(state, this week's plan), applied week by week, with baseline and lift read out from each state. It is autoregressive in the latent state, not in sales: forecast sales are never fed back as inputs.
- The forecast is the predicted consequence of the plan (baseline plus the lift the plan adds); what sold is the observed consequence, and training compares the two. The world model's predicted consequence also includes the next state, so a plan's effect carries into later weeks; the lift model's stops at the week it is in.
- The loop, in these terms: state + action (this week's plan) -> predicted consequence (sales, and for the world model the next state) -> compared with the observed consequence. The comparison between the two models measures what carrying the consequence forward adds.

## Averaging three fits on M5 (practice window, seed 1)

Job `configs/jobs/m5_world_model_roll_norm_avg3.yaml` (commit 8c10468): the world model with normalised roll-forward, the average of three fits in one run. Wall clock about 5 h on one T4 (three fits of about 80 minutes each, plus data loading and scoring).

| Model | Overall | Category total | Item by store |
|---|---|---|---|
| LightGBM | 0.715 | | |
| Lift model, 3-seed mean | 0.713 | | |
| World model, 3-seed mean | 0.707 | | |
| World model + normalisation, single fits, 3-seed mean (range) | **0.656** (0.646 to 0.675) | | |
| World model + normalisation, average of 3 fits | 0.665 | 0.574 | 0.828 |

Findings.

- Averaging three fits does not improve the world model on M5: 0.665 sits inside the range of single fits and above their mean. On M5 the single fits are already steady (range 0.03), so there is little seed noise for averaging to remove, unlike Dominick's.
- The M5 final test therefore uses single fits of the world model with normalisation, as fixed in the final-test plan before this result was read.

## Final M5 test result (official test period, three seeds, read 10 October 2026)

- LightGBM 0.547 (range 0.547 to 0.548), equal to the 10th-ranked entry. World model with normalised roll-forward 0.638 (range 0.627 to 0.647), between the best statistical benchmark (0.671) and the 50th-ranked entry (0.576).
- The practice-window lead (0.656 against 0.715) did not hold on the test period. The paper reports both windows and says so.
- The loss is at the aggregate levels (total 0.379 against 0.247). At item by store the world model is slightly better (0.897 against 0.905). The world model over-forecasts by about 2.6%, LightGBM under-forecasts by about 2.1%.
- What the paper can claim on M5: the world model is competitive with published mid-table entries and better than the statistical benchmarks, and is not better than a tuned gradient-boosted baseline on the official period. The case for the world model rests on Dominick's and Frat, where plans and promotions carry the signal, and on what it does that LightGBM cannot (rolling a plan forward week by week).
- Compute: one Tesla T4 (14.6 GB), 2-core Xeon 2.0 GHz, 12.7 GB memory, torch 2.11.0 with CUDA 13.0. About 101 minutes per world model seed, 78 per LightGBM seed.

### M5 by level: where the world model leads (test period, three seeds each; practice window for comparison)

| Level | World model, test | LightGBM, test | World model, practice | LightGBM, practice |
|---|---|---|---|---|
| Overall (official measure) | 0.638 | **0.547** | **0.656** | 0.715 |
| Total | 0.379 | **0.247** | **0.535** | 0.659 |
| Item | **0.999** | 1.002 | | 0.860 |
| Item by state | **0.951** | 0.954 | | 0.853 |
| Item by store | **0.897** | 0.905 | **0.834** | 0.847 |

- At item by store, the level a store orders at, the world model is ahead on both windows: 0.897 against 0.905 on the test period (no overlap across seeds: 0.896 to 0.898 against 0.905 to 0.906) and 0.834 against 0.847 on the practice window. The margin is about 1% to 1.5%.
- LightGBM wins the official measure on the test period because the world model is far worse at the totals. The world model forecasts about 2.6% above actual sales; its item-level errors lean the same way and add up at the top of the hierarchy.
- Wording for the paper: LightGBM scores better on the official measure; the world model is slightly more accurate at item by store on both windows and loses at the totals through an upward bias. The official measure was named as primary before the runs, so it stays the headline.
- Correcting the bias is future work, to be tested on Dominick's and Frat, not tuned on M5.

### M5 test period: where the world model's forecast goes wrong (three-seed average forecasts against LightGBM)

From `scripts/compare_forecasts.py` on the six final-test runs; tables saved to Drive `results/_analysis/m5_test_world_model_vs_lightgbm/`. 28 days, 1,229,140 units sold.

| | Actual | World model | LightGBM |
|---|---|---|---|
| Units | 1,229,140 | 1,261,570 (+2.6%) | 1,203,490 (-2.1%) |
| Error of the daily company total, % of units | | 3.7 | 2.8 |
| Error item by item, % of units | | 75.7 | 76.6 |
| Item-store pairs where this model is closer | | 53.8% | 46.2% |

1. The over-forecast is in Foods. Foods is +3.9% (+32,000 units, the whole net excess); Foods 3 alone is +4.5% (+25,000). Hobbies (-1.4%) and Household (+0.7%) are close. LightGBM is under in all three.
2. One store carries the most: CA_3 +7.5% (+12,900 units). TX_2 +4.5%, WI_2 +3.3%. Two stores are under (CA_4 -1.3%, WI_3 -1.6%).
3. Sundays and Mondays. Monday +6.4%, Sunday +5.1%, Tuesday and Wednesday about 0 (-0.4%, -0.6%). The worst days are Sunday 29 May (+11.7%, Memorial Day weekend), Monday 6 June (+11.6%), Thursday 2 June (+10.5%) and Monday 13 June (+7.4%). The model carries the weekend level into Monday.
4. It does not grow with the horizon: +4.1%, +2.1%, +4.4%, -0.1% in weeks 1 to 4. The rolled-forward state is not drifting.
5. LightGBM's better total is partly opposite errors cancelling. By each pair's sales in the test window: both models over-forecast slow sellers by the same amount (under 0.5 a day: +37% and +40%; 0.5 to 1: +8% both). On items selling over 10 a day, LightGBM is 17.5% under (48,000 units) and the world model 7.1% under (19,500 units); the world model is closer on 60% of these pairs. LightGBM's shortfall on fast sellers offsets its excess on slow sellers. Caveat: bands are set by test-window sales, which pushes the top band towards under-forecasting for any model, so the comparison between models is the reading, not the level.

What to fix (to be tested on Dominick's and Frat and on the M5 practice window, not on the M5 test period): the day-of-week pattern after weekends and holidays, and the level in Foods. A daily total check on the rolled level (does the sum of item forecasts match a store-level forecast) is the direct way to stop item errors stacking.

## Revised world model: plan (written 10 October 2026, before any revised run)

The first final test (version 1) stays reported as it came out. The revision answers the M5 error analysis above and is judged on data it was not built on.

Changes, each a single switch:

- A. Day of the week as seven separate inputs (`weekday_flags`). Version 1 gives the day as a smooth weekly cycle, on which Sunday sits next to Monday, which fits the Sunday-into-Monday over-forecast. LightGBM takes the day as a number its trees can split anywhere.
- B. M5 events by kind (national, religious, cultural, sporting) as inputs next to the single event flag. Version 1 sees Memorial Day and a sporting final as the same signal.
- C. Twice the training steps (60,000), the training-length check.
- Not taken up: the totals term in the loss. On Dominick's it made the total worse (9% under) and it hurt on Frat.

Order of work:

1. Screening on the M5 practice window (25 April to 22 May 2016), seed 1: A, A+B, A+B+C. Version 1 seed 1 scores 0.646 there. A change is kept if it lowers the overall score; C is kept only if it also lowers it by more than the seed-to-seed range of version 1 (0.029), since it doubles the cost.
2. Confirmation on two M5 windows not used for any choice (29 February to 27 March and 28 March to 24 April 2016), three seeds: version 1, version 2 and LightGBM. Version 2 is the better model only if it beats version 1 on both windows.
3. Version 2 on the official test period, three seeds, reported next to version 1 and labelled as made after seeing the test-period errors.
4. The changes are checked on the Dominick's and Frat development windows only where they apply (B and C; A is daily-only).

Also: the fast-seller comparison repeated with items banded by their sales in the 28 days before the test period (`compare_forecasts.py --band-from` with the practice-window runs), to remove the pull towards under-forecasting that banding by test-period sales gives.

### Fast sellers, banded by sales before the test period (10 October 2026)

Items put in bands by their average daily sales over 25 April to 22 May 2016, the 28 days before the test period (`--band-from` the practice-window run `20261007T153614_m5_validation_world_model_roll_norm_seed1_59a9247d`). Tables on Drive `results/_analysis/m5_test_world_model_vs_lightgbm_banded_before/`.

| Sales before the test | Units sold | World model bias | LightGBM bias | World model item error | LightGBM item error | Pairs where the world model is closer |
|---|---|---|---|---|---|---|
| No sales | 22,347 | -22.8% | -38.9% | 103.5% | 101.0% | 74.9% |
| Under 0.5 a day | 111,113 | -7.1% | -4.7% | 138.0% | 139.8% | 55.5% |
| 0.5 to 1 | 134,383 | +0.4% | +1.1% | 107.8% | 108.8% | 52.0% |
| 1 to 3 | 347,280 | +2.8% | +0.8% | 80.7% | 81.4% | 51.8% |
| 3 to 10 | 360,844 | +5.4% | -0.6% | 60.6% | 60.5% | 49.5% |
| Over 10 | 253,175 | +6.1% | -5.4% | 43.6% | 46.1% | 53.1% |

- Banded this way, the earlier fast-seller reading does not hold as stated. Banding by test-period sales had pulled both models towards under-forecasting the top band.
- The world model's excess comes from items that were selling well just before the test: +5.4% on items at 3 to 10 a day and +6.1% on items over 10 a day (about 35,000 of its 32,000 net extra units; slow sellers are under). It carries a recent high level forward. LightGBM sits close to actual on these items.
- Item by item the world model is still slightly better on the fastest items (43.6% against 46.1%) and level on 3 to 10 a day.
- Candidate change D for the revision, added to the plan before any revised run: the level the model starts from should lean less on the latest weeks for items selling above their longer-run rate. To be defined and screened on the practice window with A to C.
