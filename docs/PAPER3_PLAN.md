# Paper 3 (ICWM 2027, deadline 22 December 2026): plan, written before any result

Status: FIXED 16 November 2026, before any run (Karthik, 22:35: "yes we can start paper 3"). Every later change is a
dated amendment below, as for papers 1 and 2.

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
- Each world is generated with three seeds (101, 102, 103; paper 2 used seed 7). 18 stores in three price zones,
  20 items, 260 weeks; models are fitted on weeks up to 252 and forecast weeks 253 to 260.

## Plans held out by design (the interventions)

Models are trained only on the plans merchants make in each world. They are then asked to forecast, from the same
origin, plans of four kinds that never occur in training:

1. Deeper cuts: every planned cut deepened to 50% and to 60% off (cuts in training run from 10% to 40%).
2. Lasting regular-price changes: the regular price moved 5% and 10% up and down for the whole horizon.
3. Off-pattern promotions: promotions and displays placed in the lowest-demand weeks instead of the highest.
4. No plan: every lever off (the baseline a merchant would compare against).

The observed plan itself is scored too, so each model has a forecast error and four intervention errors.

## Models

- The world model at the settings of the known-answer generator's study, in 12 versions per world so that forecast
  accuracy varies: width 16, 32 and 64; 500 and 2,000 training steps; 13 and 26 weeks of history.
- LightGBM with the plan as features, in 12 versions per world: 7, 31 and 127 leaves; 100 and 500 rounds; at least
  20 and 100 rows per leaf.
- A log-linear price and promotion model of the kind retailers have long used: Poisson regression of sales on the
  series, the season (yearly sine and cosine), log price over the 12-week highest price (the regular price), log
  regular price over the series' mean regular price, promotion, display, both together, and the week after a planned
  week. The regular price for a plan is the 12-week highest price over history and plan, as for the other models.
- The truth (`expected_units`) as the ceiling.

## Measures (fixed now)

- Forecast accuracy: RMSSE per item by store on the observed plan's held-out weeks, averaged over series (scale:
  mean squared one-week change in training).
- Intervention accuracy: for each intervention, the error of the predicted change in sales (intervention minus
  observed plan) against the true change, as a ratio of summed predicted to summed true change, and its rank
  correlation across item-weeks.
- Decision losses: (a) promotion choice: the predicted lift of switching the promotion on in each item-week of the
  horizon; keep the half with the largest predicted lift; score the true lift captured over the true lift of the
  best half; (b) stock: ordering to the predicted sales of each plan, scored as the summed absolute difference from
  expected sales over summed expected sales; (c) regular price: per series, the move among -10%, -5%, 0, +5% and
  +10% with the largest predicted revenue over the horizon; score the true revenue given up against the best move,
  summed over series, over the true revenue of the best moves.

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

CPU on Modal: 4 strengths x 3 seeds x (12 + 12 + 1) = 300 fits, a few minutes each, tens of US dollars. Runs in the
week after the plan is fixed; writing in the two weeks after; submission by 22 December.

## Amendments

(none yet)

**Amendment 1, 16 November 2026, after the first results and before any of these runs.** Karthik, 23:20 ("yes"),
after asking why the world model does not learn the effect of a new plan when it was meant to carry what it learns
to new situations, and that any change to the data be given to every model. Added, labelled in the paper as added
after the first results:

1. World model versions trained to learn the effect of actions, each on the settings of `wm_d32_s2000_h26` with one
   change: JEPA pretraining (500 steps on the latent loss first, latent weight 0.1); the latent check (latent weight
   0.1); the latent check seven steps ahead; and adaptation at the plan date (three steps, as in the companion
   forecasting study's search). Scored on the same held-out plans.
2. The recorded regular price. Retailers know whether a price change is a temporary cut or a new regular price; the
   models so far had to infer the regular price from past prices, so a lasting change looks like a cut. In versions
   marked `_rr` every model is given the planned regular price as an input: the world model `wm_d32_s2000_h26`
   (as its recorded regular price), LightGBM `lgb_l31_r500_m20` (as a feature) and the log-linear model (in place
   of the 12-week highest price). The held-out regular-price plans move the recorded regular price with the price.
3. A world whose effects are not log-linear, so the log-linear model's form does not match the truth by
   construction: a temporary cut lifts sales by 1 + 1.015(1 - exp(-depth / 0.15)) (equal to the log-linear world at a
   20% cut, flattening beyond it); the regular-price response is linear in the price ratio, 1 - 0.5(ratio - 1); and a
   promotion with a display adds less than both alone (joint effect -0.10). Four planning strengths, three seeds,
   every version above.

The test is unchanged: the link between forecast accuracy and intervention accuracy and decision loss, now also
for these versions and in the new world, and whether any of the changes in 1 and 2 make the world model's
intervention accuracy better.

**Amendment 2, 17 November 2026, after the results of amendment 1 and before any of these runs.** Karthik, 00:46:
"paper 3 should answer what can we do about the what if not state it does not work". Amendment 1 found that no change
to the world model's training made its simulation of new plans better, and that the recorded regular price helped
the log-linear model most. Two remedies a retailer can apply are added, labelled in the paper as added after the
first results, and given to every model on the same data:

1. Split the job. The forecaster (world model `wm_rr`, or LightGBM `lgb_rr`) forecasts the week as it would be at
   the current regular price with nothing planned; the log-linear model with the recorded regular price
   (`loglinear_rr`) supplies the change the plan makes, as the ratio of its forecast under the plan to its forecast
   at the current regular price with nothing planned. The current regular price is the last recorded regular price
   before the plan date. Versions `wm_ll` and `lgb_ll`, on every world of amendment 1.
2. Test and learn. Retailers can run a few test plans that do not follow demand. New worlds in which 5% of the
   training item-weeks of every series carry a random plan, drawn apart from demand: a cut with probability one half,
   10% to 60% off; a promotion and a display each with probability one half. Everything else in the world is as
   before (the test plans come from their own random stream). The held-out plans and the scoring are unchanged, so
   the deeper cuts are now inside the range tested. Versions `wm_d32_s2000_h26`, `wm_rr`, `lgb_l31_r500_m20`,
   `lgb_rr`, `loglinear`, `loglinear_rr`, `wm_ll` and `lgb_ll`; both world shapes, four planning strengths, three
   seeds.

Reading fixed now: a remedy works if it lowers the intervention error and the regular-price and promotion choice
losses against the same model without it, in the same worlds, across seeds. Compute: about 240 fits on Modal CPU,
about US$35 (Karthik, 23:42: "run what u need for paper3 if the charges are small").

**Amendment 2, revised 17 November 2026, 01:00, before any test-and-learn result.** Karthik, 00:56: test and learn
"should learn the overall shape but still ensure you can't clear the house with a 100% discount depth or for a price
change"; "there is no experiment and exploit if it's just learning what it did in the past"; the model has "to
understand the underlying meaning and then transfer to unseen data", as M-JEPA did. Two changes; the test-and-learn
runs launched at 00:50 were cancelled before any result was read:

1. The test plans stay inside the merchants' own range (cuts of 10% to 40%) and differ only in being placed at random,
   apart from demand. The deeper cuts of 50% and 60% and the lasting regular-price moves stay unseen, so the
   held-out plans test whether what the tests teach carries over, not whether the model remembers them.
2. A world model with the shape of the price response built in (`wm_shape`, on the settings of `wm_rr`): the
   plan's price enters only through a curve whose shape is fixed and whose size is learned per item and store, so
   sales never fall as a cut deepens and the gain is bounded; and a lasting regular price lowers sales through a
   learned elasticity. The rest of the network does not see the price. Run in the worlds with and without tests.

**Amendment 3, 17 November 2026, 12:45, after the results of amendment 2 and before any of these runs.** Karthik
passed on a reviewer's challenge: the world model with a price shape may win only because its shape matches the
worlds' shapes, and the term $-a_2\log(1-c)$ is not bounded (only the 95% cap on the cut keeps it finite). Two
controls are added, labelled in the paper as added after the first results:

1. The same price shape in the simple model (`loglinear_shape`): the log-linear model with the recorded regular
   price, its log-price term replaced by the shape of the world model (the cut through $1-e^{-c/b}$ for
   $b \in \{0.05, 0.1, 0.2, 0.4\}$ and $-\log(1-c)$, cut capped at 95%, all coefficients non-negative) and its
   regular-price coefficient kept at or below zero. Fitted by Poisson maximum likelihood with those bounds. Run on
   every world. If the world model with the price shape beats it, its learned state adds to the shape; if not, the
   gain is the shape's.
2. A third world shape that neither model's shape contains (`kinked`): a cut adds nothing up to 10%, then sales rise
   in a straight line to 1.6 times at 40% and stay there for deeper cuts; the regular-price response and the
   promotion-with-display effect are those of the bent world. Four planning strengths, three seeds; versions
   `wm_d32_s2000_h26`, `wm_rr`, `wm_shape`, `wm_ll`, `lgb_l31_r500_m20`, `lgb_rr`, `lgb_ll`, `loglinear`,
   `loglinear_rr`, `loglinear_shape`.

The text is corrected: the price shape keeps sales from falling as a cut deepens; the saturating part is bounded,
the logarithmic part is not and is held finite only by the cap on the cut. Compute: about 130 fits, about US$20.

**Amendment 3, addition, 17 November 2026, 12:45, before any of its runs.** Karthik passed on the same reviewer's
second suggestion: give LightGBM the same rules too. Added `lgb_shape`: `lgb_rr` with every price feature and the
recorded regular price constrained so that predicted sales cannot rise with price (LightGBM monotone constraints);
trees already stay flat beyond the deepest cut seen. Run on every world of amendment 3.

**Amendment 4, 17 November 2026, 13:15, after the results of amendment 3 and before any of these runs.** Karthik
passed on a reviewer's points (13:12): the controls of amendment 3 used one version of each model, while the
original models had twelve; the bootstrap resamples versions that share data, so its intervals overstate the
evidence; and the kinked-world result covers only the simpler models tested. Added, labelled in the paper as added:

1. Equal tuning for the three models with the price rules, in the bent and kinked worlds (four planning strengths,
   seeds 101 to 103): the world model with the rules at the 12 settings of the world model's search
   (`wm_shape_d*_s*_h*`), LightGBM with the rules at the 12 settings of its search (`lgb_shape_l*_r*_m*`), and the
   log-linear model with the rules at six settings (penalty 1e-4, 1e-3, 1e-2; the grid of b values 0.05, 0.1, 0.2,
   0.4 or the finer grid 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5) (`ll_shape_a*_b*`). Each model is reported at the median
   over its versions and at its best version chosen on the held-out plans, so each gets the same chance.
2. More independent worlds for the head-to-head in the kinked world: seeds 104 to 110 at each planning strength
   (28 more worlds), for the three models with the rules at their base settings.
3. Statistics with the world as the unit: for each head-to-head, the number of worlds in which one model is ahead,
   with a two-sided sign test, and a 95% interval for the median paired difference from a bootstrap over worlds.
   The rank-correlation intervals over versions are kept and described as variation over versions within fixed
   data, not over independent replications.

Compute: about 300 world model fits and 600 cheaper fits, about US$40 (Karthik, 23:42 on 16 November: "run what u
need for paper3 if the charges are small").

**Amendment 5, 17 November 2026, 13:40, before any of these runs.** Karthik, 13:36: "yes in addition to syth we
should do this" (a real-data check). Real data hold no answer for a plan never run, but some rare plans were run.
Added, labelled in the paper as added after the first results:

1. Breakfast at the Frat (77 stores, 55 items, 156 weeks; the data record each week's regular price as BASE_PRICE,
   which is the recorded regular price). Origins at weeks 125, 133, 141 and 149, forecasting 8 weeks each. Every
   item-week with a cut deeper than 30% off the regular price is removed from training, so no model sees a cut that
   deep; 30% is near the 95th percentile of cuts. Scored on the held-out weeks in three groups: item-weeks with a
   cut deeper than 30% (the unseen deep cuts), item-weeks whose regular price differs by 10% or more from the series'
   last regular price before the origin (lasting regular-price moves), and the rest.
2. M5, FOODS_3 in stores CA_1, TX_1 and WI_1, summed to weeks (no promotion records). The regular price is not
   recorded, so it is constructed: a price that holds for four weeks or more becomes the regular price, and the
   previous regular price holds otherwise. Origins 32, 24, 16 and 8 weeks before the end, forecasting 8 weeks.
   Scored on item-weeks whose regular price differs by 5% or more from the last regular price before the origin,
   and the rest.
3. Models: the world model (`wm_d32_s2000_h26` settings), with the recorded regular price, and with the price rules;
   LightGBM (`lgb_l31_r500_m20`), with the recorded regular price, and with the rules; the log-linear model, with the
   recorded regular price, and with the rules.
4. Measures, per group and origin: summed forecast over summed sales (1 is right) and its |log|, and summed absolute
   error over summed sales. Comparisons between models: by origin, and the paired difference in absolute error with
   a 95% interval from a bootstrap over series.

Reading fixed now: if the models with the rules and the recorded regular price have lower error on the unseen deep
cuts and the lasting regular-price moves than the same models without them, the synthetic finding carries to real
data; if not, the paper says so. This is a check on sales that happened, not on counterfactual plans. Compute: about
US$30 to 50.

**Amendment 6, 17 November 2026, 14:15, after the summary results of amendment 5 and before any of the analyses
below.** Karthik, 14:12: "how do we fix the weak?" The real-data check of amendment 5 was weak in three ways: deep
cuts mostly ran with a circular or display, which the models had seen; the Frat's recorded regular price moves often,
so a 10% difference is not always a lasting move; and four origins give few replications. Added, labelled as added:

1. Pure price cuts: the Frat deep cuts split into those with no circular and no display (TPR_ONLY) and the rest,
   scored from the saved forecasts.
2. Clean lasting moves: Frat item-weeks whose recorded regular price is 10% or more from its last value before the
   origin and the same in every held-out week of the series, after holding at least four weeks before the origin;
   scored from the saved forecasts.
3. A harder and larger test: every cut deeper than 20% removed from training, scored on cuts deeper than 20% (and
   pure price cuts deeper than 20%), and eight origins (weeks 101 to 149, every 8 weeks) for both thresholds; the
   same nine models.

Reading fixed now: the same as amendment 5. Whatever these show is reported, including no gain.
Compute: about US$30.

**Amendment 7, 17 November 2026, 14:30, before any of these runs.** Karthik, 14:19: "we already downloaded the
complete journey for the paper 1 work". The Complete Journey (dunnhumby) records for every purchase the shelf price
and the loyalty discount taken, so the regular shelf price is known exactly, and it records by product, store and
week whether the product was on display or in the mailer. Added, labelled as added:

1. Data: purchases by the 2,500 households summed to product by week over all stores (102 weeks), for the 767
   products bought in at least 95 weeks. (Clarified before any run, after a smoke test on one origin showed fuel
   and fees among them: only food and household departments are kept (grocery, drug and general merchandise,
   produce, meat, packaged meat, deli, pastry, nutrition, seafood, frozen), and product-weeks paid at less than 5% of
   the shelf price are dropped as recording errors; the product count is reported.) Price: amount paid over units; recorded regular price: shelf price before
   the loyalty and coupon-match discounts over units; promotion: in the mailer in any store that week; display: on
   display in any store that week.
2. Test: origins at weeks 63, 71, 79, 87 and 95, forecasting 8 weeks; every product-week with a cut deeper than 30%
   (about the 90th percentile) removed from training. Scored on cuts deeper than 30%, on those with no mailer and no
   display (pure price cuts), on product-weeks whose regular shelf price is 10% or more from its last value before
   the origin, and the rest.
3. The same nine models and measures as amendment 5.

Reading fixed now: as in amendments 5 and 6; every result is reported. Compute: about US$20.

**Amendment 8, 17 November 2026, 18:15, before any of these runs.** Karthik, 18:06 to 18:09: a second domain to
show the test is not specific to retail ("any other one u can think of?"), run on Vast ("where cost is cheap").
Added, labelled in the paper as added:

1. A dosing world, generated (no real patient data; allowed on Vast). 360 patients in 18 clinics, 260 days. A
   persistent severity state (AR(1), persistence 0.8) lowers each patient's daily response count. The drug
   accumulates: concentration C_t = C_(t-1) exp(-ln 2 / 3) + dose_t (half-life 3 days). The expected response is
   base x exp(-0.5 severity) x (1 + C / (C + 0.5)) x weekly pattern, and the count is Poisson. Doctors dose by
   indication: a dose is given with probability sigmoid(-1.4 + strength x severity), sized 10% to 40% of the maximum
   dose and larger when severity is high. Strengths 0, 0.5, 1, 1.5; seeds 101 to 103.
2. The action reaches the models through the price columns, so every model and rule applies unchanged: price =
   1 - 0.95 x dose (dose as a fraction of the maximum), regular price 1. The price rules then say that response
   never falls as a dose rises and that the gain may flatten. Accumulation over days is not in the rules; models
   must learn it from the data.
3. Held-out plans from the same origin (fit on days 1 to 252, forecast days 253 to 260): every planned dose raised
   to 50% and to 60% of the maximum; a steady daily dose of 20% every day; doses moved to the days of lowest
   severity; no dose. Measures: forecast accuracy and the error on new plans of the main study.
4. Models: the 25 versions of the main study (the link test), plus the world model, LightGBM and the log-linear model
   with the rules (the rules control). Statistics with the world as the unit.

Reading fixed now: if the dosing world repeats the retail pattern (accuracy guides kinds of model but not versions
where dosing follows severity; the rules improve the error on new doses), the test is shown in a second domain; if
not, the paper reports what differs. Compute: one rented Vast machine, about US$5 to 15, destroyed after the runs.
