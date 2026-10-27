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

**15 October 2026, later: study C, pantry loading, written down before it
ran.** After a promotion the world model forecasts 1.32 and 1.33 times
actual in soft drinks and snack crackers, the two categories where shoppers
stock up, against 0.90 and 0.92 for LightGBM (development notes, 7 October),
and the persistence test shows the rolled state carrying lift forward rather
than a dip. The tested model rolls its state forward on the plan alone, and
nothing fixes the sign of a carried effect. Study C tests two ways to give
the dip a route with the right direction (`scripts/make_pantry_configs.py`,
job `configs/jobs/pantry_loading.yaml`):
1. `stock`: one number per item carried from period to period, built by sales
   above the item's level (real sales in history; in the forecast, the
   plan's extra sales as the model forecasts them, the same in training as in
   forecasting), decaying by a learnt share, and lowering later sales by a
   learnt amount that cannot be negative.
2. `plan_lags: 2` on the lift model: each period sees the plan of the two
   periods before it, as LightGBM does.
Five variants on the development windows of the six Dominick's categories,
each one fit with seed 1: the world model, the world model with `stock`, the
lift model, the lift model with `plan_lags`, and the lift model with `stock`.
Measures: forecast over actual and error in the two periods after a
promotion (the after-promotion score), and the overall, total and item
scores. Read: soft drinks and snack crackers are the categories where a dip
is expected; the other four show whether a change costs anything where it
is not. Stage 2, only for a variant that brings the after-promotion forecast
of soft drinks and snack crackers within 10% of actual without making the
overall score worse in more than two categories: seeds 2 and 3 for it and
for the world model, then the rolling windows. Nothing becomes a default
without stage 2.

**15 October 2026, later: study D, a data set kept out of development,
written down before its data were downloaded.** Every source so far had
development windows of its own, and every Dominick's category appears in at
least one development config, so the results show how the models hold up
over time, not on a new retailer. Study D applies the recipe, frozen, once to
a source never used: Corporación Favorita (Ecuador, daily, Kaggle
competition `favorita-grocery-sales-forecasting`). Nothing is tuned on it and
it is reported as it comes out.
1. Data (`rwm.data.favorita`): train.csv from 1 January 2015 to its last day
   (15 August 2017), with items.csv, stores.csv and holidays_events.csv. A
   missing row is a day with no sales; negative sales (returns) are set to
   zero. The series are the 10 stores with the most units sold in 2016,
   crossed with the items sold in every one of those 10 stores in 2016.
   Favorita records no prices, so price is constant and the revenue weights
   of WRMSSE become unit weights; this is stated with every result.
2. Plan inputs, all known in advance: the promotion flag (`onpromotion`,
   missing before April 2014 and set to off), national holidays and events
   (not transferred), and paydays (the 15th and the last day of each month,
   when public-sector wages are paid).
3. Hierarchy: total, store, family, store by family, item, item by store.
4. Windows: four rolling 28-day windows, the last ending on 15 August 2017,
   each forecast from everything earlier.
5. Models, each with the settings of its M5 rolling config (the other daily
   source) and only the data set's columns changed: seasonal naive,
   LightGBM, the world model, the lift model, TimesFM with the plan, and the
   equal-weight ensemble; one fit each (seed 1). The world model with the
   stock term is added only if it passes stage 1 of study C, with no other
   change.
6. Tests as in the plan: scores by level, paired comparisons with the item
   bootstrap and the bootstrap over windows, the model confidence set,
   decisions and intervals, and the ensemble ablation of study A.

**15 October 2026, after stage 1 of study C, before stage 2 ran.** Stage 1
(seed 1, `results/paper1_revision/tables/pantry_stage1.csv`): only the lift
model with the stock term passes the stage-1 rule (after-promotion forecast
over actual 0.93 in soft drinks and in snack crackers; overall score worse
than the world model's in two categories). The world model with the stock
term and the lift model with the lagged plan are worse than the world model
in three categories each, and the plain lift model is at 1.22 after
promotion in soft drinks. One correction to the reason given for study C:
the 1.32 and 1.33 after promotion came from an earlier world-model version
(8-week level, 7 October); the final world model is at 1.00 and 1.04 on
these windows. Stage 2 as written: seeds 2 and 3 for the lift model with the
stock term and for the world model on the development windows; the stage-1
rule is applied again to the mean of the three seeds. If it still passes,
the lift model with the stock term runs on the six Dominick's rolling
windows with its settings unchanged (one fit, seed 1) and is scored and
tested as the other models; it is not added to study D.

**16 October 2026, stage 2 of study C, results.** Table:
`results/paper1_revision/tables/pantry_stage2.csv`.

- Seeds 2 and 3 were the same fit. The pantry configs are single fits taken
  out of a seed average, and their model settings carry no seed, so the run
  seed never reached the network (it fits with seed 0 every time). Seeds 2
  and 3 give identical scores in all twelve runs. Seed 1 differs from them
  only because it ran on another runtime. Stage 2 therefore has two
  replicates, not three, and the difference between them is runtime noise,
  not seed noise. It is large: the world model's overall score in snack
  crackers is 0.911 on one runtime and 0.749 on the other. No other job in
  the repository repeats a seed on a config of this kind.
- The rule, applied as written to the mean of the three runs: the lift model
  with the stock term is at 0.97 (soft drinks) and 0.93 (snack crackers)
  after promotion, and worse overall than the world model in two categories
  (analgesics, soft drinks). It passes. On the second replicate alone it is
  worse in four categories and would fail.
- Rolling windows (one run each): the stock term does not fix the
  after-promotion forecast. Soft drinks: 1.27 with it, 1.28 for the plain
  lift model, 1.19 for the world model. Snack crackers: 1.16 with it, 1.03
  for the plain lift model, 1.17 for the world model. Total error is lower
  than the plain lift model's in four of six categories and higher in
  snack crackers (1.51 against 1.03) and analgesics. It beats LightGBM at
  total in none.
- Decision: the stock term does not become a default. The paper reports it
  as tried and not kept, with the runtime noise stated next to the
  development-window differences.

**16 October 2026, study E, written down before it runs: a loss weight on
the periods just after a promotion.** The three changes of study C changed
what the model can express; none fixed the after-promotion forecast without
costing accuracy elsewhere. Study E leaves the model as it is and changes
only what training counts: a forecast period with no promotion, but with one
in either of the two periods before it (the rule of the after-promotion
check), counts 1 + w times in the loss, with the total weight kept the same.
w = 2 and w = 5, on the world model of the pantry study, on the six
Dominick's development windows, seed 1. The world model is rerun in the same
job, because stage 2 of study C showed that runs on different runtimes
differ by as much as the variants do. Rule: a weight passes if, against the
world model of the same job, the after-promotion forecast over actual in
soft drinks and snack crackers is within 10% of actual and its error there is
no higher, and the overall score is worse in at most two categories. A weight
that passes runs on the six rolling windows with nothing else changed; it
becomes a default only if it holds there. The model's seed still does not
reach these single fits (study C, stage 2), so this is one fit per config.

**16 October 2026, study E, results.** Table:
`results/paper1_revision/tables/afterpromo_stage1.csv` (one fit each, all on
one runtime; the world model rerun matches its stage-2 runs of study C to the
fourth decimal). Neither weight passes. With w = 2 the error in the periods
after a promotion falls in all six categories (soft drinks 0.174 to 0.150,
snack crackers 0.477 to 0.413) and the forecast there stays within 10% of
actual, but the overall score is worse than the world model's in four
categories (snack crackers 0.749 to 0.857). With w = 5 the overall score is
also worse in four categories and snack crackers after a promotion goes to
1.11. Decision: no default changes and no rolling-window runs. The paper
reports that weighting those periods lowers their error but costs more
elsewhere than it gains.

**16 October 2026, seed fix.** From this commit a run's seed reaches the
model whenever the model takes a seed, also when its config names none.
Before it, single fits of the state model, LightGBM and TFT whose configs
named no seed trained with seed 0 for every run seed. The results already
recorded are those of seed 0 and stay as they are (each manifest records its
commit); seed averages were not affected, since their configs name a seed.

**16 October 2026, interval check, written down before it runs.** The
conformal intervals cover far less than planned: on the rolling windows the
95% interval holds 91% of actual sales at item by store but about 36% at the
total, and the 50% interval about 20% at every level. Two causes are visible
in the code. First, the error quantiles are taken separately for each
horizon, and a level with one node (the total) has only one earlier error per
horizon per window, so with one to four earlier windows the 2.5% and 97.5%
quantiles are the smallest and largest of one to four numbers. Second, sales
are whole units, and an interval such as 0.3 to 0.8 holds no possible value,
which lowers coverage of the narrow 50% interval at item by store. The fix,
applied to the saved forecasts of every model with nothing else changed:
where a level has fewer than 50 earlier errors for each horizon, all horizons
are pooled into one set of quantiles, and interval ends are rounded outwards
to whole units (except on Favorita, where some items are sold by weight).
Both versions are reported. The pinball score does not change.

**16 October 2026, interval check, result.** Pooling fixes the aggregate
levels. Mean over rolling windows, world model, old to new coverage at the
total for 50%, 80% and 95%: Dominick's 0.22, 0.35, 0.43 to 0.48, 0.77, 0.88;
Frat 0.00, 0.04, 0.08 to 0.42, 0.83, 0.88; M5 0.11, 0.18, 0.19 to 0.42, 0.76,
0.86. LightGBM and the ensemble move the same way (ensemble 95% at the total:
0.86, 0.96, 0.94). The pinball score at the total falls by about a third
because pooling also widens the quantiles that were too narrow (world model:
Dominick's 0.48 to 0.32, Frat 0.28 to 0.15, M5 0.30 to 0.19). At item by store
the whole-unit rounding lands near the targets on Dominick's (0.46, 0.76,
0.94) and on Favorita, which is not rounded (0.48, 0.79, 0.95), but overshoots
on Frat (0.64, 0.87, 0.97) and on M5 (0.84, 0.94, 0.98), where most daily sales
are zero or one and rounding a narrow interval outwards takes in a whole unit.
So the honest bottom-level coverage on M5 and Frat lies between the old and the
rounded figures, and the pinball score, unchanged by rounding, is the measure
to report there. The three models stay within 0.02 of each other on coverage
at item by store and within 0.03 on the pinball score, so intervals do not
separate them. Table: results/paper1_revision/tables/intervals_fixed.csv.

**17 October 2026, study F, Timestral, written down before it runs.**
Mistral released Timestral 1.0 on 15 October 2026, a pretrained forecaster
that reads known-future inputs itself and that its authors report ahead of
TimesFM and Chronos on public benchmarks. A reader will ask whether the
ensemble result holds against it. Timestral-209M (code commit a534bdb of
github.com/mistralai/timestral) is run zero-shot on the rolling windows of
every data set, in two forms that mirror TimesFM's: history only
(`timestral_mean`) and with the plan as known-future inputs, the same inputs
TimesFM's plan form gets (`timestral_plan`). The forecast is the mean of its
21 quantiles. Two ensembles are added to the analysis: Timestral's plan form
in place of TimesFM's (`ensemble_timestral`), and all four together
(`ensemble_4`). Questions and pass rules, judged on the mean over rolling
windows with the paired item bootstrap: (1) does `timestral_plan` beat
`timesfm_plan`, and does it beat the ensemble on its own, at the total and at
item by store; (2) does `ensemble_timestral` beat the current ensemble. If the
ensemble still beats Timestral alone, the main result stands. If
`ensemble_timestral` is better, it is reported as such and the paper's
ensemble is not changed after the fact; the swap is reported as a robustness
result. Timestral's own quantiles are scored with the pinball score beside the
conformal intervals.

**17 October 2026, ranking check, written down before it runs.** Accuracy
scores say how close a forecast is, not whether it puts items in the right
order, which is what a merchant uses it for when choosing which items to back.
For every window and model: the Spearman rank correlation, across item by
store series with sales in the window before, between the forecast change
(horizon total over the previous horizon's actual total) and the actual
change. Also across the category by store nodes. With it, three more measures
from the same saved forecasts: the top-decile hit rate (of the tenth of nodes
the forecast puts highest for change, the share that are in the actual top
tenth; 0.10 by chance); a promotion ranking, the Spearman correlation across
promoted item by store periods of the horizon between forecast and actual
units, each over the series' mean in its unpromoted periods of the weight
window before (data sets that record promotions: Dominick's, Frat, Favorita);
and bias, forecast over actual for the whole horizon, in total and as the
median across nodes of the level. Reported as the mean over windows with the
spread across windows; no pass rule, they are descriptive.

**17 October 2026, study F, model size changed before any result.**
Timestral-209M proved too slow for the study: the first Dominick's
category (47,000 series, six windows, with the plan as inputs) had not
finished after 80 minutes on a T4, which puts the study at about a day of GPU
time. No Timestral forecast had been scored. The study now uses Timestral-28M,
which its authors report ahead of Chronos-2, with the same inputs, windows and
questions. Model size is checked rather than assumed: on Frat, the smallest
data set, the plan form is also run with 209M (`timestral209_plan`) and the
two sizes are compared. If 209M is clearly better there, the paper says that
the reported Timestral results are a lower bound for the larger model.

**17 October 2026, study F result.** The ensemble beats Timestral's plan form on every
data set, at the total and at item by store, in every Dominick's category and
on Frat, Favorita and M5, with the paired interval clear of zero in 17 of 18
comparisons (Dominick's cheese at item by store just touches zero). Timestral's
history-only form is closer and loses on the mean everywhere except Dominick's
cheese at item by store, where it is level.
(1) Timestral's plan form does not beat TimesFM's plan form anywhere except
the M5 total, where they tie (WRMSSE 1.03 against 1.04). (2) Swapping
Timestral in for TimesFM makes the ensemble worse on Dominick's and Frat,
leaves it level on Favorita (item by store 0.003 worse), and makes it better on the M5 total only (0.61
against 0.71; item by store slightly worse). Per the rule above, the paper's
ensemble is unchanged and the M5 swap is reported as a robustness result. The
size check runs the wrong way for Timestral: on Frat, 209M is worse than 28M
(item by store 1.04 against 0.86, total 1.39 against 1.05), so 28M is not
understating it.

Why Timestral does badly on Dominick's was checked before reporting, on the
saved forecasts. It is not a scale or input bug: its median forecast sums to
0.8 to 1.3 of actual sales in every window on analgesics and soft drinks, and
the history-only form's mean sums to 0.9 to 1.4. The failure is in its upper tail when it is given the plan. On
analgesics its 99% quantile sums to 57 to 107 times actual sales, and because
the point forecast is the mean of the quantiles, as written down above, a
handful of series carry the error: the top 1% of series by excess hold 75 to
88% of it. One series that sold 2 units in a window was forecast 7,586 (median
289). The same happens, more mildly, on soft drinks (99% quantile 14 to 53
times actual) and Frat (3 to 7 times), and not on the daily data sets. The
forecast rule is not changed after the result; the paper reports the finding
and notes that Timestral's median would sit much closer, which is a reason a
merchant should not use a foundation model's plan-conditioned tails unchecked.

Timestral's own quantiles score a pinball loss at item by store of 0.31
(Dominick's), 0.28 (Frat, Favorita) and 0.37 (M5), behind TFT where it was run
(Dominick's 0.20, Frat 0.19) and the ensemble's conformal intervals
(Dominick's 0.22, Frat 0.20, Favorita 0.24, M5 0.32).

Ranking check. The ensemble ranks item by store changes best or level with the
best on every data set (Spearman 0.61 Dominick's, 0.46 Favorita, 0.83 Frat,
0.43 M5; the four-model ensemble is 0.47 on Favorita) and has the best or
level-best top-decile hit rate (0.57, 0.51, 0.68, 0.53 against 0.10 by
chance). On promoted periods, LightGBM and the ensemble lead (Spearman 0.67 and
0.66 on Dominick's; the ensemble leads on Frat, 0.74, and Favorita, 0.58);
TimesFM's plan form trails the ensemble by 0.03 to 0.11. The ensemble's total bias is within 6% of actual sales on
every data set; Timestral's plan form over-forecasts Dominick's by 3 times on
average and Frat by 17%. Tables: results/paper1_revision/tables/studyF_accuracy.csv,
studyF_paired.csv, studyF_ranks.csv.

**17 October 2026, study F, the paired test asked for in question (1).** The
Colab analysis compared each Timestral form with the ensemble only. Timestral
against TimesFM, in the same form, is computed from the saved per-window scores
(`scripts/paired_from_scores.py`, bootstrap over windows as in
`scripts/revision_tables.py`; the per-window item bootstrap needs the saved
forecasts and is not computed). With the plan, TimesFM is more accurate in 16 of
18 comparisons and level at the Favorita and M5 totals; Timestral is more
accurate in none. Table: results/paper1_revision/tables/studyF_timestral_vs_timesfm.csv.

**17 October 2026, reconciliation on Dominick's.** The reconciliation of the
six Dominick's categories ran on the rolling and final windows on 14 and 15
October (Drive folders `reconcile_dominicks_*`) but was left out when the tables
were gathered, so the first draft said it was not run. Its per-window files are
now under results/paper1_revision/reconcile/ and in reconcile.csv; the snack
cracker final rows already in the table agree to 0.001. Result: combining a
trained model's item forecasts with ETS above them raises LightGBM's total
error in all six categories, and the best reconciled ETS total is behind
LightGBM and the ensemble in every category.

**17 October 2026, study G, the second review, written down before it runs.**
A second reader of the revised paper asked for four checks. (1) The category-plan
cost, quoted as point estimates, gets paired uncertainty: per category-by-store
node errors are saved for every model, window and data set
(`scripts/category_cost_nodes.py`), and the cost difference between the world
model and each other model is bootstrapped over nodes within each window
(5000 resamples, seed 0) and summarised by the mean over windows. The world
model's lead on a data set is called clear only when the 95% interval of that
mean difference lies below zero against every other model. The same comparison is
repeated with three other choices, reported whatever they show: the first half of
the window only; error of the window total instead of the sum of period errors;
and nodes weighted by units sold and equally, in place of dollar sales. (2) The
Frat lift model ran with readout dropout 0.2 and weight decay 0.01, carried over
from the companion paper's final Frat config, while the world model ran with
neither; on the other data sets the two share every setting but the transition.
The Frat lift model is rerun with the world model's settings
(`rev_frat_rolling_lift_matched`, job `study_g`), and the ensemble
ablation adds `lift_matched_for_world_model`. Both numbers are reported; the
matched one is used for the claim that only the transition differs. (3) The
MinT-shrink failure on Frat is diagnosed from the ETS fits, which need no saved
forecasts, before any reconciliation conclusion is drawn. (4) Wording: a
difference not detected is not called equal.

**17 October 2026, study G, an error found in the category-plan cost.** The
per-node rerun showed that the cost as implemented divided each
category-by-store node's error by its sales in the window, floored at 1e-9. On
Dominick's some nodes sold nothing in a window (a store closed or not
reporting), so they entered at about 10^9 times their error and decided every
Dominick's result: the costs in `decisions.csv` are of the order of 10^10. Frat
and M5 have no such nodes and are unaffected. The plan's definition, absolute
error of category-store-period totals weighted by dollar sales, is now computed
as the error in dollars over sales in dollars, each node's units valued at its
average price in the weight window (`scripts/category_cost_uncertainty.py`,
variant `dollars`), and the paper's category-plan results are restated on it.
The first cost is kept in the table as `as_first_run`. The sensitivity variants
fixed before the run (first half, window total, unit weights, equal weights)
are computed on the corrected cost; equal weights become the mean over nodes
that sold in the window of each node's error over its sales.

**27 October 2026, study H, the latest foundation models, written down before it runs.**
The comparison gave the foundation models unequal footing. TimesFM ran as
version 2.5 (September 2025), which takes the plan only through a separate
linear model fitted in context; Chronos ran as Chronos-Bolt (2024) on unit
history alone; Timestral 1.0 (October 2026) read the plan itself, but as its
28M model, because 209M needed about a day of GPU time run by run (study F).
Since then TimesFM 3.0 (August 2026) reads past and future covariates itself,
and Chronos-2 (October 2025) reads known-future covariates; and runs now go
to one machine each (`cloud/modal_app.py`), so the larger Timestral is
affordable. Study H runs, zero-shot on the rolling windows of every data set,
each with the windows, context and plan inputs of the matching TimesFM config:
TimesFM 3.0 history only and with the plan (`timesfm3_mean`, `timesfm3_plan`),
Chronos-2 likewise (`chronos2_mean`, `chronos2_plan`), and Timestral-209M with
the plan (`timestral209_plan`). Each forecast is the mean of the model's
quantiles. Questions and pass rules, judged on the mean over rolling windows
with the paired item bootstrap, at the total and at item by store, as in
study F: (1) does the strongest foundation model with the plan beat the
ensemble on its own; (2) does the ensemble with that model in place of
TimesFM 2.5 beat the current ensemble. If the ensemble still beats every
foundation model alone, the main result stands. If the swapped ensemble is
better, it is reported as a robustness result and the paper's ensemble is not
changed after the fact. The earlier versions' results stay in the paper,
labelled with their versions. The `foundation` dependencies move from
`timesfm` 2.0.2 to 3.0.2, which still holds TimesFM 2.5; each manifest records
the version used.

**27 October 2026, study I, transfer to stores and items not trained on, written down before it runs.**
A merchant adds a range or opens a store, and the forecast has to work for
lines and stores the model was not trained on. A trained network reads any
series' state from its own history, while trees that split on a store's or
item's code have nothing to go on for a code they never saw. Study I tests
that on public data. In every rolling window a fifth of stores, or a fifth of
items (seed 0, the same fifth in every window), is kept out of training; the
fitted model then reads the kept-out series' own history without learning
from it (`refresh`; no weight changes) and forecasts them. Each kept-out run
has a comparison run trained on everything and scored on the same fifth.
Data: M5 and Dominick's cheese and snacks, the rolling windows and settings of
each data set's revision configs. Models: the world model (three seeds),
LightGBM as in the paper for items, and for stores one LightGBM over all
stores with the store as an input (one model per store cannot forecast a store
it never saw), and seasonal naive, which learns nothing and so marks what the
history alone gives. The measure is the cost of not training on a series: the
kept-out score minus the comparison run's score on the same fifth (WRMSSE,
lower is better, so a smaller cost is better). Question and pass rule: is the
world model's cost smaller than LightGBM's, judged on the mean over windows
(and seeds) with a paired interval over windows that excludes zero, on each
data set and for stores and items separately. Reported whatever it shows,
with forecast over actual on the kept-out fifth. The world model is not
changed for this study; it runs at the revision settings.

**27 October 2026, study J, the research loop on the world model, written down before it runs.**
The world model ran at settings chosen by hand during development. Study J
searches its settings with a fixed procedure (`src/rwm/research/loop.py`,
spec `configs/research/dominicks_che_world_model.yaml`): one change at a time
from a menu fixed in the spec (20 changes: loss weights, the roll-forward,
training, weight averaging, latent overshooting and test-time adaptation),
each tried on top of the best settings so far; the best is kept only if its
mean over three seeds beats the best's mean by at least the seed spread of the
starting settings; the loop stops when no change clears that bar. Changes are
screened at a third of the training steps on one seed, and the best third go
on to full steps on every seed (successive halving). The score is the mean
WRMSSE over three windows that lie before every window the paper reports; a
fourth, later window is scored for every kept change and never used to decide.
Every try is logged with its config (`results/research/<spec>/log.jsonl`) and
published with the paper. The settings the loop keeps are then run, unchanged,
on the paper's rolling windows beside the revision settings, and both are
reported; the paper's other results are not changed after the fact. The loop
is run on Dominick's cheese first; whether the kept settings carry over to the
other categories and data sets is tested, not assumed.

**27 October 2026, study H result, M5.** Mean over the four rolling windows,
WRMSSE averaged over levels (lower is better): TimesFM 3.0 0.963 history only
and 0.976 with the plan, against TimesFM 2.5 0.964 and 0.982; Chronos-2 1.058
and 1.054, against Chronos-Bolt 1.184; Timestral-209M with the plan 2.845,
forecasting 17% to 61% above actual sales in every window, against
Timestral-28M 1.038. The world model scores 0.745, LightGBM 0.751 and the
ensemble 0.760. At the total the foundation models score 1.00 to 1.13 (and
Timestral-209M 3.14) against 0.61 to 0.71 for the trained models. Per the rule
above the main result stands on M5: no foundation model with the plan comes
near the ensemble. Timestral-209M's over-forecast is checked against its
authors' instructions for covariates before it is reported as its accuracy.
The swapped ensembles and the other data sets follow.

**27 October 2026, study I result, Dominick's cheese and snacks.** The cost of
not training on a series (kept-out score minus the comparison run's score on the
same fifth, mean over six windows and three seeds) and the world model's cost
against LightGBM's, paired by window with a 90% bootstrap interval:

| Data | Kept out | World model | LightGBM | Difference (90% interval) |
|---|---|---|---|---|
| Cheese | items | 0.014 | 0.065 | -0.050 (-0.080 to -0.013) |
| Cheese | stores | see below | 0.049 | see below |
| Snacks | items | 0.012 | 0.152 | -0.140 (-0.176 to -0.108) |
| Snacks | stores | -0.003 | 0.069 | -0.072 (-0.095 to -0.050) |

By the rule fixed before the run, the world model's cost is smaller in all
four. In the earliest cheese window the world model failed in every one of its
runs at seeds 1 to 3, trained on everything or not (WRMSSE 35 to 1,481,
forecasts up to 215 times actual sales; identical between the store and item
comparison runs, so a property of the model in that window, not of the test).
That window decides the cheese-stores row on its own, so the row is not read
as a result; over the five other windows the world model's cost is 0.029
against LightGBM's 0.049. On the kept-out fifth LightGBM under-forecasts
(forecast over actual 0.69 to 0.94) while the world model stays near actual
(0.98 to 1.08). The failure in the earliest cheese window, and the high scores
in the last two snacks windows (2.7 and 1.1 to 2.4 against 0.4 to 0.7 in the
others, seeds 1 to 3), are a reliability problem of the world model at the
revision settings and are reported as such; study J is where settings that
remove them are searched for. M5 is not yet run.

**27 October 2026, study J result.** The loop stopped after its first round
with no change kept: the starting settings scored 0.429 (seeds 0.413 to
0.452, so a keep bar of 0.040), and the best changes scored 0.406 to 0.407
(a higher learning rate, and loss weighted by volume at 0.5), short of the
bar. On the held-out window, never used to decide, the starting settings
scored 1.65; volume weighting 0.82 and the plan split 0.77, with seed ranges
of 0.014 and 0.008 on the development windows against 0.039; the higher
learning rate 3.93. One weakness of the design is recorded: changes were
screened at a third of the training steps, which disadvantages weight
averaging (a decay of 0.999 averages over most of a short run), so it was not
fairly tested.

**27 October 2026, study K, the world model's failures, exploratory, written down before it runs.**
Study K comes from looking at study I and study J's results, so it is
exploratory and is reported as such; it does not change any result fixed
before. Question: do the changes that study J pointed to remove the world
model's failures in the rolling windows (the earliest cheese window, the last
two snacks windows), without making the other windows worse? On the paper's
rolling windows of Dominick's cheese and snacks, three seeds each, at full
training steps: the revision settings as they are; volume weighting at 0.5
with the plan split (`fix`); `fix` with weight averaging (decay 0.999);
`fix` with one AdaJEPA step; and weight averaging and AdaJEPA alone.
Measures, per data set and variant: the number of window-seed runs with WRMSSE
above 2 (a failure), the worst window-seed score, the mean over windows and
seeds, and the mean over the windows where no variant failed. No pass rule is
set; the numbers are reported as they come. If a variant removes the failures
and is no worse elsewhere, it becomes a candidate for a confirmatory run on
data sets it was not chosen on (M5, Favorita, the other Dominick's
categories), written down before that run.
