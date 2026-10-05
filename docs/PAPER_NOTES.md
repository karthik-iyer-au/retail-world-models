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
