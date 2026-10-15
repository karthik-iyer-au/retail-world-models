# Paper 1 revision: plan, written before any new result (13 October 2026)

An outside review (ChatGPT, shared 13 October 2026) recommended major revision
before submission to the International Journal of Forecasting. Every point it
lists is taken on. This file fixes what is run and how it is scored before any
new run starts. Changes are dated entries at the end, made before the results
they affect are read.

## What the review asked for, and what answers it

| Review point | Answer |
|---|---|
| P0 No significance analysis | Paired comparisons with bootstrap intervals (below), on the final windows and on the rolling windows |
| P0 Too few forecast origins | Rolling-origin evaluation: 6 more windows on Dominick's, 4 on Frat, 4 on M5 |
| P0 Weak evidence for the world-model mechanism | The lift model is the same network with the transition removed (same encoder, size and steps): run it in every window; plus plan-perturbation tests on saved world models |
| P0 Forecasting versus causal plan response | Wording throughout; the switching parameter described as a forecast-implied split of lift; reference to paper 2's measured and known-answer results |
| P1 No reconciliation benchmark | Bottom-up, top-down and MinT-family reconciliation of ETS base forecasts, and of each model's bottom level combined with ETS above it |
| P1 Point accuracy only | Interval forecasts for every model by rolling conformal calibration; native quantiles where a model has them; weighted interval score and coverage |
| P1 No decision value | A replenishment simulation on item-store forecasts; a category planning loss; promotion-decision regret in paper 2's generated world where true lifts are known |
| P1 Benchmarks | Chronos-Bolt (zero-shot foundation model), the Temporal Fusion Transformer, and an equal-weight ensemble |
| P1 Hand-copied figure data | `paper/figs/rwdata.py` generated from run records (`results/runs_index.csv`), with a test that every value in the paper traces to a record |
| P2 Framing and length | Development history moved to a supplement; the main text organised around hypotheses; title chosen after the results |

## Rolling-origin evaluation

Windows are taken before the development windows, so none was used to choose
any setting: `holdout_periods` skips the final window and the two development
windows (24 weeks on Dominick's and Frat), and on M5 the four months already
reported (112 days).

| Data | New windows | Horizon | Step |
|---|---|---|---|
| Dominick's, the six final categories | 6 | 8 weeks | 8 weeks |
| Frat | 4 | 8 weeks | 8 weeks |
| M5 | 4 (November 2015 to February 2016) | 28 days | 28 days |

Models in every window, settings of the companion paper's final configurations:
world model, lift model (the same with `rollout` off: the transition ablation),
LightGBM, seasonal naive, TimesFM (mean, and with the plan), Chronos-Bolt
(base, zero-shot, mean of its quantiles), the Temporal Fusion Transformer
(Dominick's and Frat; on M5 its cost is out of reach and this is stated), and
the equal-weight mean of the world model, LightGBM and TimesFM with the plan,
fixed in advance with no weights fitted. To fit the compute, neural models are
one fit (seed 1) in the rolling windows rather than the average of three used
in the final windows; the final windows keep their reported scores.

The lift model, Chronos-Bolt, TFT and the ensemble are also run on each final
window so they can be compared there.

## Significance

For each data set, level and pair of models:
- per window: the difference in WRMSSE with a 95% interval from 1,000
  bootstrap resamples of items (all of an item's series and the aggregates they
  feed are recomputed), so the interval reflects variation across series;
- across windows: the mean difference with a 95% interval from a bootstrap over
  windows, and the share of windows each model wins;
- the 90% model confidence set (Hansen, Lunde and Nason 2011, range statistic,
  bootstrap over windows) of each level.
A model is said to beat another only when the interval excludes zero.

## Mechanism tests on saved world models

On the saved world-model checkpoints (Frat, M5, and paper 2's Dominick's
refits), with no refitting:
1. Timing: the plan for the 8 (or 28) forecast periods is permuted within each
   series, keeping the same total promotion exposure; the change in accuracy
   against the true plan.
2. Persistence: one promotion added at period h; the forecast change at h and at
   each later period of the window, against LightGBM's (which has no state).
3. Beyond sales: a linear probe from the rolled state to sales in the 8 periods
   after the window, against the same probe on the state before the window.

## Uncertainty

Each model's interval for a series, horizon and level comes from the errors of
the same model in the earlier windows of the same data set (scaled by the
series' scale), quantiles 2.5, 10, 25, 75, 90, 97.5; the first rolling window
has no earlier window and is not scored for intervals. Scored by weighted
interval score and empirical coverage at 50%, 80% and 95%. Where a model gives
quantiles of its own (TimesFM, Chronos-Bolt, TFT) they are scored too.

## Decisions

- Replenishment: order the model's interval quantile at the critical ratio
  (0.5, 0.8, 0.95) for each item-store-period; cost = overage + underage x
  ratio / (1 - ratio), in units; reported against seasonal naive.
- Category planning: absolute error of category-store-period totals, weighted
  by dollar sales.
- Promotion decisions: in paper 2's generated worlds, promote an item-week when
  the model's predicted lift times margin exceeds a fixed promotion cost; regret
  against the decision made with the true lift.
All are simulations under stated assumptions, not business results.

## Reconciliation

Base forecasts at every node by AutoETS (statsforecast) on the history before
each window. Reconciled by bottom-up, top-down (historical proportions), OLS,
WLS with structural scaling, WLS with residual variances, and MinT-shrink where
the hierarchy has at most 6,000 nodes. Also each model's bottom level with ETS
above it, reconciled by WLS with residual variances. Scored by WRMSSE at every
level.

## Rules

- No new result changes a setting of a model already reported.
- Every model, window and test is reported, including those that go against
  the paper's earlier claims.
- The title is chosen after these results.

## Amendments

**14 October 2026, before any reconciliation result was read.** Base
forecasts for reconciliation: seasonal AutoETS (season 52 weeks, or 7 days on
M5) is fitted at the aggregate nodes only; the item-by-store nodes take a
non-seasonal AutoETS. Seasonal ETS costs about ten times as much per node and
the bottom level holds most nodes (about 9,000 per Dominick's category), which
put the reconciliation of the rolling windows beyond the compute available.
Frat's reconciliation, computed earlier with seasonal ETS at every node, is
recomputed under this rule so every data set uses the same one; the earlier
numbers are not used. The item bootstrap is now computed in closed form at the
levels where each node holds one item and by one matrix product elsewhere; the
numbers are the same as before (tested against the direct computation).

**15 October 2026: two studies added after the rolling-window results were
read, written down before either ran.** The rolling windows put the
equal-weight ensemble first, so the claim now rests on what the world model
adds to it. Both studies use the rolling windows and the seed-1 runs already
made; no model is retrained except LightGBM in study B, refitted with its
recorded settings. They are reported as added after the main results.

*A. Ensemble ablation* (`scripts/ensemble_ablation.py`, saved forecasts only).
1. Variants, each an equal-weight mean: the ensemble (world model, LightGBM,
   TimesFM with the plan); each pair left when one of the three is removed;
   and the ensemble with the lift model in place of the world model, which
   separates the carried state from adding any third model.
2. Weights: on a grid over the three weights in steps of 0.05, with equal
   weights added as one more point, (a) the weights
   with the lowest mean WRMSSE over the earlier windows of the data set,
   scored on the next window (out of sample; the first window has no earlier
   window and is not scored), chosen separately for item by store, for the
   total and for the mean over all levels; and (b) the full curve of WRMSSE
   against the world model's weight, the other two sharing the rest equally.
3. Tests as in the plan: each variant against the ensemble by the item
   bootstrap per window and the bootstrap over windows, and the 90% model
   confidence set among the variants. The world model is said to add to the
   ensemble at a level only where the ensemble beats the pair without it.

*B. Input perturbation* (`scripts/input_perturbation.py`). For the world
model, the lift model and LightGBM, on the saved checkpoints (LightGBM
refitted on the same training data), the forecast is recomputed with one plan
input changed at a time: price, and each lever of the data set (promotion on
Dominick's; promotion, display, feature and price reduction only on Frat;
events and SNAP days on M5). Each input is (a) occluded: set to nothing
planned (price at the regular price, lever off) and (b) permuted within each
series across the window. Read: the change in WRMSSE at item by store and at
the total, and the change in forecast units in the periods where the input
was on. For the world model and the lift model only, history occlusion: the
periods before the last 4, 13 or 26 (M5: 28, 56) are set to missing, as for
an item not yet on sale. All windows for the world model and the lift model;
LightGBM on all windows of Dominick's and Frat and the last two of M5.
