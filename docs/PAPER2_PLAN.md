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
