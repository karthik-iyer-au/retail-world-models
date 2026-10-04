# Evaluation protocol

Changes to this protocol are made only through a dated entry in
`DECISIONS.md`, and never after seeing test results.

## Accuracy

- Rolling-origin splits. Train on everything before the cut, forecast the
  next horizon, repeat at several cuts.
- M5: 28-day horizon, RMSSE per series and WRMSSE, as in the competition.
- The final test window of each dataset is used once, at the end. All tuning
  uses earlier windows.
- Every model is scored by the same code in `rwm/evaluation/`.
- Differences between models are tested for significance across series and
  origins, not read off a single average.

## Lever effects

- Error is reported separately for periods where a lever changed (price,
  promotion, display, feature) and periods where nothing changed.
- On Dominick's, estimated price effects are compared with the chain's
  in-store pricing experiments.

## Explanations

Each forecast is broken down into baseline plus the contribution of each
lever, computed by running the model with the lever on and off.

- Removal test: remove the lever the breakdown says mattered most. The
  forecast should move by about the stated amount.
- Randomisation test: scramble the trained weights. The breakdown should change.
- Stability test: retrain with different seeds. The breakdowns should agree.

## Components

A component of our model stays only if removing it makes the result
measurably worse on tuning windows. Each removal is a config and a recorded run.

## The three claims are checked separately

Accuracy, lever effects and explanation faithfulness are three checks. Passing
one says nothing about the other two.
