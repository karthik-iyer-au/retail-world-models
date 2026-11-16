# Paper 3 (ICWM 2027, deadline 22 December 2026): plan, written before any result

Status: DRAFT for Karthik. Nothing below has been run. Once Karthik agrees the plan it is committed as fixed, and
every later change is a dated amendment, as for papers 1 and 2.

## Question

When does a world model that forecasts well become a trustworthy simulator of alternative plans? Concretely: across
model versions and generated worlds, does forecast accuracy on the plans merchants actually make predict how
accurately the model simulates plans they never make, and how much a planner loses by acting on its simulation?

## Why this is new and not papers 1 or 2

- Paper 1 (under review at IJPE) measures forecast accuracy on observed plans. Paper 3 cites it and reuses none of
  its results.
- Paper 2 (on arXiv, to be submitted to TMLR) tests whether the model's explanation of a forecast, the baseline and
  the lift, is right. TMLR forbids reusing its results in a paper under review elsewhere, so paper 3 reuses no
  number, table or figure from paper 2. It may use the same data generator (code, not a result) and cites paper 2.
- Paper 3's measurements are new: forecasts of plans held out by design (interventions), their error against the
  generator's truth, decision losses from acting on them, and the relation between forecast accuracy and all of
  these across many fitted versions.

## Data

The generated store data of `src/rwm/data/synthetic_levers.py`, whose `expected_units` gives the expected sales of
any plan. New worlds for this paper (new seeds, so no fitted model or result of paper 2 is reused):

- Merchants plan into high-demand weeks at several strengths (confounding 0, 0.5, 1, 1.5).
- Each world is generated with three seeds.

## Plans held out by design (the interventions)

Models are trained only on the plans merchants make in each world. They are then asked to forecast, from the same
origin, plans of four kinds that never occur in training:

1. Deeper cuts: the planned cut deepened to depths beyond any in training (to 40% and 50% off).
2. Lasting regular-price changes: the regular price moved 5% and 10% up and down for the whole horizon.
3. Off-pattern promotions: promotions and displays placed in the lowest-demand weeks instead of the highest.
4. No plan: every lever off (the baseline a merchant would compare against).

The observed plan itself is scored too, so each model has a forecast error and four intervention errors.

## Models

- The world model, at the settings of the companion forecasting study, and versions of it from a small grid (width,
  training length, loss) so that forecast accuracy varies across versions: about 30 versions per world.
- LightGBM with the plan as features, at a matching grid.
- A log-linear price and promotion model of the kind retailers have long used (log sales on log price relative to
  regular, promotion and display flags, item, store and week effects), fitted by least squares.
- The truth (`expected_units`) as the ceiling.

## Measures (fixed now)

- Forecast accuracy: WRMSSE on the observed plan's held-out weeks.
- Intervention accuracy: for each intervention, the error of the predicted change in sales (intervention minus
  observed plan) against the true change, as a ratio of summed predicted to summed true change, and its rank
  correlation across item-weeks.
- Decision losses: (a) choosing the half of item-weeks to promote by predicted lift, scored as true lift captured
  over the best possible; (b) ordering stock to the predicted sales of an intervention, scored as absolute
  over- and under-order over true sales; (c) choosing the regular-price move (from the four in intervention 2) that
  maximises predicted revenue, scored as true revenue lost against the best move.

## The test of the question

Across all fitted versions in all worlds, the rank correlation between forecast accuracy and each intervention
measure, with a bootstrap over versions. Reading fixed now: if forecast accuracy predicts intervention accuracy
and decision loss, better forecasting is a fair guide to planning; if the correlation is near zero or reversed,
especially in worlds where plans follow demand, it is not, and the paper says which interventions and decisions
break the link. The log-linear model is the reference a world model has to beat on interventions.

## Real-data check (short, not evidence)

The Dominick's zone price episodes are the only real interventions available; they are noisy and are reported as a
check of direction only, with their interval, and cited to the companion explanation study for how they were found.

## Compute and timing

CPU on Modal: about 4 worlds x 3 seeds x (30 + 30 + 1) fits, a few minutes each, tens of US dollars. Runs in the
week after the plan is fixed; writing in the two weeks after; submission by 22 December.
