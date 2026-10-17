# Paper 2: plan, written before any audit result (13 October 2026)

Working title: *Do faithful explanations go with accurate forecasts? An audit of
plan-lift explanations in retail forecasting.*

Same format as MolSanity and paper 1: LaTeX in `paper2/`, figures and tables
generated from run records, every check and every model version reported.

## Question

The world model and the lift model explain each forecast as a baseline (what
sells with nothing planned) times a lift (what the plan adds), and the lift can
be split by lever. A merchant reads that split as "the promotion added 40
units". The paper asks:

1. Are these explanations faithful to the model, stable, and sensible?
2. Do they match what happened, and, where a price change was not chosen for
   the week, what a price change causes?
3. Across model versions, does a version whose explanations pass more checks
   also forecast better? (MolSanity found that, for molecular attributions,
   faithfulness and correctness come apart.)

## What is explained

For each item, store and forecast period: forecast, baseline, total lift, and
the contribution of each lever (forecast with the plan as made, minus forecast
with that lever switched off and the rest as made). For LightGBM the same
on/off contributions, plus TreeSHAP values on the lever inputs.

## Model versions (no retraining in the first pass)

Every saved checkpoint of the world model and the lift model on the
development and final windows of Dominick's (six categories), Frat and M5,
with LightGBM runs as the comparison. Each version's accuracy is already in its
`metrics.json`. Versions are grouped by configuration; seeds within a
configuration give stability.

## Checks (each scored per version, all reported)

| # | Check | Score | Source of the idea |
|---|---|---|---|
| C1 | By construction: zero lift with nothing planned; baseline unchanged when the plan changes; baseline times lift equals the forecast | pass / fail, largest violation | repository tests |
| C2 | Randomisation: re-initialise the readout, then the transition, then the encoder; the lifts should change | 1 minus Spearman between original and randomised lifts | Adebayo et al. 2018 |
| C3 | Seed stability: same configuration, different seeds | Spearman of item-week lifts between seeds, against the same for forecasts | |
| C4 | Price direction: a deeper cut should not lower the forecast | share of item-weeks where it does, over a grid of cuts | |
| C5 | Lift against what happened: on promoted item-weeks, predicted lift against sales over the model's own baseline, by lever and lever mix | bias and rank correlation | PAPER_NOTES "checks against the data" |
| C6 | Agreement with LightGBM: per-lever contributions against LightGBM on/off contributions and TreeSHAP | rank correlation | |
| C7 | Ground truth by quasi-experiment (Dominick's): response to sustained zone price deviations | model's implied response against a difference-in-differences estimate and its interval | Hoch, Drèze and Purk 1994; Abrams, Gui and Hortaçsu 2018 |
| C8 | Plan read-back from the rolled state (world model) | R squared against control | paper 1 |

## C7 in detail (fixed before any model lift is computed)

The public Dominick's files do not record which stores or weeks were in the
chain's pricing experiments. The section is therefore called *zone price
deviations* and is described as a quasi-experiment, never as the experiments.

1. Rows with `OK = 1`, stores with a price zone; unit price = PRICE / QTY;
   weeks with a SALE flag or a temporary cut against the item's recent highest
   price removed, leaving regular prices.
2. Zone reference: for each zone, UPC and week, the modal regular price across
   the zone's stores.
3. A store-category is a deviation episode when, for at least 12 consecutive
   weeks, the ratio of its regular prices to the zone reference has the same
   sign for most UPCs and a median of at least 5% in size. Controls: stores in
   the same zone and category whose median ratio is within 1% in those weeks.
4. Response: difference-in-differences on log category units, episode store
   against same-zone controls, 26 weeks before against the episode weeks, store
   and week fixed effects, errors clustered by store.
5. The model's implied response: forecast the episode weeks with the prices as
   they were, and again with each item's price set to the zone reference, all
   else unchanged; compare the log ratio with the estimate in 4.
6. Score: whether the model's response is inside the 95% interval, its sign,
   and its error. Every episode found is reported, including those with wide
   intervals.

Caveat to state: these deviations may include zone pricing changes that were
not experiments, and the model was trained on data that include them.

## Rules

- No audit result changes any model setting. The audit reads models; it does
  not tune them.
- Every version and every check is reported, including versions or checks that
  fail.
- The episode detector in C7 is written and its episodes listed before any
  model lift is computed for C7.
- Thresholds above are fixed here; any change is a dated entry in this file,
  made before the affected results are read.

## Order of work

1. Audit code and tests on synthetic data (`rwm/audit/`), here.
2. Episode detector for C7 on the Dominick's files, and its episode list.
3. Colab job: list every saved checkpoint, run C1 to C6 and C8 per version,
   write one record per version.
4. C7 on the Dominick's versions.
5. Tables, figures, manuscript.

## Amendments

**13 October 2026, before any model lift was computed.** On analgesics the
first detector (each store against the zone's modal price) found one episode.
Looking at store prices showed why: in the chain's experiments about a third of
a zone's stores moved up, a third down and a third stayed, so the zone's mode
is often itself a moved price. The detector now measures each store against
its zone peers: for each store and item, the log of the week's regular price
over that item's median regular price in that store over all weeks, minus the
median of the same across the zone's stores that week. An episode needs at
least 12 consecutive weeks each at least 3% away in the same direction with at
least 60% of items moved that way, and a median of at least 5% over the
episode. Controls, the DiD and every other rule are unchanged.

**13 October 2026, after the episode list and before any model lift.** The
detector found 54 episodes in the six categories (`results/paper2/zone_episodes.csv`),
most in Dominick's weeks 190 to 240, where the chain's experiments ran. Two
rules are added. An episode needs at least three control stores for its
interval to be estimated; episodes with fewer are listed but not scored. The
models forecast at most 8 weeks ahead, so the model is fitted on data up to the
week before an episode and scored on the episode's first 8 weeks, against a
difference-in-differences restricted to the same 8 weeks (26 weeks before
against those 8). The full-episode estimate stays in the list.

**13 October 2026, before any audit result (C1 to C6, C8) or model lift for
C7 has been read.** Three additions.

*Known-answer study.* The checks are first run where the answer is known:
generated weekly store data (`rwm.data.synthetic_levers`) with a temporary-cut
elasticity of -2.5, a regular-price elasticity of -0.5, a promotion effect of
+40%, a display effect of +25%, +15% more when both run, and a 15% dip the
week after a planned week. 3 zones of 6 stores, 20 items, 260 weeks; in each
zone one store's regular prices go up 10% and one down 10% for 16 weeks at
random weeks during training, and one up and one down starting at the
forecast origin (week 253). Two worlds: plans at random (clean), and plans
made more often in weeks with high unobserved demand (confounded, strength 1).
In each world: the world model (3 seeds), the world model fitted with prices
shuffled within each series (blind to price), the world model fitted with
promotion and display shuffled within each series (blind to the levers),
LightGBM, and LightGBM with prices shuffled. Settings are those of the
Dominick's final models except width 32 and 2000 steps (processor only). Every
check is run, and each model's explanations are also scored against the true
contributions: rank agreement and summed-lift bias on planned item-weeks,
overall and by lever, and the stated response to the origin price spells
against the true -0.5 and the difference-in-differences estimate.

*C7 additions, data only.* An event study for each scored episode (weekly
difference-in-differences coefficients, 26 weeks before to 8 after, the week
before as reference) to show pre-trends; a placebo: the same estimator with the
start moved 26 weeks earlier, where no effect is expected; the episode count
and pooled slope for WEEK_SIZE in {2, 3, 4}% and MIN_SIZE in {4, 5, 7}%; the
pooled slope by category; and the 8-week against the full-episode estimate.

*Pass thresholds and the main test, fixed now.* A version passes C1 if every
violation is below 0.001 of the largest forecast; C2 if the rank agreement of
its lifts after the lift readout is re-initialised is below 0.5; C3 if its
configuration's seed agreement of lifts is at least 0.7; C4 if fewer than 1%
of item-weeks fall after a deeper cut; C5 if, on promoted or price-cut
item-weeks, summed forecast over summed sales is within 0.8 to 1.25 and the
rank agreement of predicted with observed lift is at least 0.3; C6 if the rank
agreement of its total lift with LightGBM's on the same window is at least 0.5.
The main test: within each data set, the Spearman correlation across versions
between the number of checks passed (C1 to C6, those that apply) and
item-by-store WRMSSE, and between each check's continuous score and WRMSSE;
pooled by a permutation test that shuffles scores within data set (10,000
permutations). C7 and C8 are reported per model, not per version.

**13 October 2026, after the C7 robustness results and before any model lift
for C7.** The event study shows signed sales drifting in the weeks before the
detected start, and the placebo is not clean: its pooled slope is -0.26
(95% interval -0.72 to 0.19, heteroskedasticity-robust across episodes)
against -0.35 (-0.69 to -0.02) for the real starts, and 42% of placebo
intervals exclude zero. Two exploratory analyses, added after seeing this and
labelled as such in the paper: (1) the price path of each episode store from
52 weeks before the start to 8 after (`scripts/zone_price_path.py`), which
shows prices already moving in the episode's direction in about half of the
store-weeks of the 8 weeks before the start, so the comparison weeks of the
8-week estimator are partly treated; (2) the pooled slope with a
heteroskedasticity-robust interval across episodes, for the 8-week estimate,
the full-episode estimate and the placebo. Per-episode intervals clustered by
store are too narrow (median standard error 0.02 against a placebo spread of
0.12), so the pre-registered per-episode score (inside the interval) is still
reported but the comparison of models with the measured response is made on
the pooled slope and its interval. Nothing else changes.

**13 October 2026, correction, before any model lift for C7.** An episode is
scored only if the 26 comparison weeks before its start exist in the data, as
the estimator specifies. Four scored episodes started within the first 26
weeks: one soft-drink episode starting in week 1 had no comparison weeks at all
(its interval had zero width) and three bottled-juice episodes had
only some. They are now listed but not scored, leaving 38. The robustness analyses are rerun. Also corrected: the chain's experiments
reported a 10% regular price change moving category volume about 3% over 16
weeks, an elasticity near -0.3, not -0.4 as first written here.

**17 October 2026, after the audit results were read: diagnosis only.** The
audit of the saved versions (933 version-windows of 494 versions) finished on
15 October. Two results needed a cause before they could be written up: C1
failed on most Dominick's versions although baseline plus lift equals the
forecast exactly in every version, and C4 failed on almost every Dominick's and
M5 version. `scripts/audit_diagnose.py` reruns both on fourteen saved runs
(`results/paper2/diagnose_runs.txt`, chosen by rule: the three Dominick's runs
with the largest C1 violation and six others drawn at random from the six
categories fetched, three M5 runs at random, the two Frat runs with the largest
C1 violation). For C1 it uses the regular price the model itself stores, and
leaves a missing price missing; for C4 it cuts a random tenth of series instead
of every price at once, and reports drops larger than 5%. The pre-registered
scores, pass counts and main test are reported unchanged; the diagnosis is
reported beside them as added after the results. A local check on a Frat world
model before this ran: with the model's own regular price the largest C1
violation fell from 0.08% to 0.02% of the largest forecast. The main test is
computed per version (mean over a version's windows), as the plan states
("across versions"); the first table code computed it per version-window.
