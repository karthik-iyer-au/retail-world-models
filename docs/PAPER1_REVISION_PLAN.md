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
