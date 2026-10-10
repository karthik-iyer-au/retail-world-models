# Results

Each run writes `results/<time>_<name>_<config hash>/` with `manifest.json`,
`metrics.json`, `forecasts.csv` and, for models that can be saved, one
checkpoint file per window. Run folders are not committed.

A run that is quoted in the paper is copied to `results/promoted/` (manifest
and metrics only) and committed, so every number in the paper points to a
manifest in git. Checkpoints and forecast files are too large for git. They
are kept outside it, and `metrics.json` records each checkpoint's checksum so
the stored file can be proven to be the one the run produced.

## M5 benchmarks

WRMSSE over the 12 levels of the M5 hierarchy. Lower is better. Each row
links to a folder in `results/promoted/` holding its manifest and scores.

| Method | 25 Apr to 22 May 2016 | 23 May to 19 Jun 2016 (official test period) |
|---|---|---|
| Seasonal naive | 0.870 | 0.847 |
| LightGBM, one model per store | 0.715 | 0.552 |
| Our model, item encoder and readout only, one run | 0.708 | not run |
| LightGBM, final test, seeds 1 to 3 (mean) | | 0.547 |
| World model with normalised roll-forward, final test, seeds 1 to 3 (mean) | 0.656 | 0.638 |

Published scores on the official test period, from the organisers' files
(`data/raw/m5_reference/`): seasonal naive 0.847, best statistical benchmark
(exponential smoothing, bottom-up) 0.671, 50th-ranked entry 0.576,
10th-ranked entry 0.547, winning entry 0.520.

Notes:

- The LightGBM settings were fixed before the test-period run and were not
  tuned on either window.
- In the April to May window the LightGBM forecasts are 7.7% below actual
  sales in total. In the test period they are 2.1% below.
- Our model's row is a single training run on a GPU, which does not repeat
  exactly. The difference from LightGBM is smaller than can be claimed from
  one run. The test period is left unseen until the model design is final.

| Run | Folder |
|---|---|
| Seasonal naive, April to May | `20261004T045330_m5_validation_seasonal_naive_a67817a2` |
| LightGBM, April to May | `20261004T034345_m5_validation_lightgbm_11b5432b` |
| Seasonal naive, test period | `20261004T045417_m5_test_seasonal_naive_54f6d687` |
| LightGBM, test period | `20261004T041739_m5_test_lightgbm_9d232fa7` |
| Our model, April to May | `20261004T063152_m5_validation_state_model_ec8ad17b` |

## Breakfast at the Frat benchmarks

Average over the last three 8-week windows of the data (27 July 2011 to
4 January 2012). Lower is better. "Overall" is WRMSSE over six levels (total,
category, store, category by store, item, item by store). The other columns
are item-by-store accuracy on the weeks where that lever was on, and on weeks
with no promotion at all. Lever values for the forecast weeks are taken as
planned in advance.

| Method | Overall | On display | In circular | Tag-only price cut | No promotion |
|---|---|---|---|---|---|
| Seasonal naive | 0.832 | 1.471 | 1.510 | 0.869 | 0.872 |
| Average of last 8 weeks | 0.756 | 1.383 | 1.381 | 0.680 | 0.540 |
| LightGBM, price only | 0.554 | 1.025 | 1.015 | 0.668 | 0.431 |
| LightGBM, all recorded levers | 0.447 | 0.793 | 0.805 | 0.558 | 0.424 |

| Run | Folder |
|---|---|
| Seasonal naive | `20261004T070636_frat_seasonal_naive_02ce19d4` |
| Average of last 8 weeks | `20261004T070720_frat_recent_average_d9fdc7b6` |
| LightGBM, price only | `20261004T070804_frat_lightgbm_price_only_78aebb07` |
| LightGBM, all recorded levers | `20261004T070912_frat_lightgbm_0dbcdc4c` |

## Development runs over several seeds

These runs are saved in the Drive results folder; their manifests have not
yet been copied into `results/promoted/`. Scores are averages over five
seeds, lower is better.

M5, 25 April to 22 May 2016: our model (item encoder and readout, mixed
precision, A100) scores 0.699 on average, range 0.669 to 0.725, against
LightGBM at 0.715. Four of the five runs beat LightGBM.

Breakfast at the Frat, the two 8-week windows before the held-out final 8
weeks:

| Method | Overall | Range | On display | In circular | Tag-only price cut | No promotion |
|---|---|---|---|---|---|---|
| LightGBM, all recorded levers | 0.421 | 0.416 to 0.427 | 0.817 | 0.822 | 0.508 | 0.405 |
| Our model, items drawn at random | 0.424 | 0.407 to 0.438 | 0.819 | 0.862 | 0.554 | 0.413 |
| Our model, whole stores drawn | 0.430 | 0.407 to 0.453 | 0.818 | 0.851 | 0.563 | 0.415 |
| + item-to-item, conditional similarity | 0.476 | 0.430 to 0.508 | 0.861 | 0.909 | 0.580 | 0.419 |
| + item-to-item, attention | 0.465 | 0.416 to 0.505 | 0.871 | 0.907 | 0.568 | 0.419 |
| + item-to-item, both | 0.453 | 0.419 to 0.498 | 0.840 | 0.901 | 0.562 | 0.420 |
| + item-to-item, both, same category only | 0.443 | 0.412 to 0.470 | 0.849 | 0.890 | 0.552 | 0.420 |
| Our model, items drawn at random, 10,000 steps | 0.432 | 0.411 to 0.450 | 0.829 | 0.871 | 0.567 | 0.420 |
| Lever step: regular price | 0.429 | 0.419 to 0.442 | 0.845 | 0.890 | 0.550 | 0.413 |
| Lever step: regular price, baseline times lift | 0.406 | 0.401 to 0.414 | 0.840 | 0.863 | 0.537 | 0.406 |
| Lever step: regular price, baseline times lift, latent loss | 0.411 | 0.401 to 0.422 | 0.829 | 0.868 | 0.548 | 0.407 |
| Lever step: latent loss alone | 0.428 | 0.414 to 0.440 | 0.825 | 0.867 | 0.556 | 0.412 |
| Lever step: regular price, baseline times lift, 52 weeks of history | 0.413 | 0.408 to 0.418 | 0.836 | 0.875 | 0.542 | 0.407 |
| Lift model + the plan for the four weeks before each forecast week | 0.411 | 0.406 to 0.418 | 0.833 | 0.846 | 0.551 | 0.409 |
| Lift model + the same week last year and its neighbours | 0.408 | 0.394 to 0.420 | 0.848 | 0.877 | 0.544 | 0.410 |
| Lift model + both of the above | 0.418 | 0.414 to 0.424 | 0.882 | 0.900 | 0.539 | 0.413 |
| Lift model + recorded base price as the regular price | 0.434 | 0.418 to 0.460 | 0.897 | 0.933 | 0.558 | 0.418 |
| Lift model + US seasonal calendar | 0.425 | 0.416 to 0.434 | 0.847 | 0.871 | 0.543 | 0.409 |
| Lift model + crowding | 0.406 | 0.400 to 0.414 | 0.836 | 0.868 | 0.540 | 0.406 |
| Lift model + base price, calendar and crowding | 0.446 | 0.425 to 0.459 | 0.856 | 0.893 | 0.547 | 0.422 |
| LightGBM + calendar and crowding | 0.426 | 0.413 to 0.435 | 0.817 | 0.834 | 0.510 | 0.407 |
| Lift model + latent loss alongside, weight 0.1 | 0.411 | 0.406 to 0.417 | 0.840 | 0.871 | 0.542 | 0.406 |
| Lift model + latent loss alongside, weight 2 | 0.410 | 0.407 to 0.415 | 0.816 | 0.869 | 0.546 | 0.406 |
| Lift model, 3,000 steps latent-only pretraining, then all trained | 0.407 | 0.400 to 0.412 | 0.836 | 0.869 | 0.542 | 0.406 |
| Lift model, pretraining, then encoder at a tenth of the rate | 0.414 | 0.402 to 0.425 | 0.842 | 0.882 | 0.547 | 0.406 |
| Lift model, pretraining, then encoder frozen | 0.428 | 0.416 to 0.441 | 0.836 | 0.901 | 0.547 | 0.417 |
| Lift model, pretraining, encoder at a tenth, latent loss kept on | 0.415 | 0.402 to 0.425 | 0.835 | 0.874 | 0.546 | 0.405 |
| Lift model, 10,000 steps pretraining, then encoder at a tenth | 0.408 | 0.400 to 0.422 | 0.826 | 0.849 | 0.543 | 0.408 |
| Lift model, stopped at its best point on the last 8 training weeks | 0.425 | 0.414 to 0.437 | 0.863 | 0.894 | 0.554 | 0.423 |
| Lift model + totals term in the loss | 0.414 | 0.395 to 0.433 | 0.856 | 0.892 | 0.553 | 0.418 |
| Lift model + readout dropout and stronger weight decay | 0.409 | 0.405 to 0.412 | 0.831 | 0.867 | 0.528 | 0.407 |
| Lift model + all three of the above | 0.445 | 0.438 to 0.453 | 0.943 | 0.990 | 0.570 | 0.446 |
| World model (state rolled forward weekly), baseline and lift | 0.408 | 0.396 to 0.418 | 0.845 | 0.868 | 0.539 | 0.404 |
| World model + latent check alongside the forecast loss | 0.417 | 0.401 to 0.426 | 0.870 | 0.910 | 0.538 | 0.404 |
| World model, 10,000 steps latent-only pretraining, then encoder at a tenth | 0.411 | 0.392 to 0.423 | 0.813 | 0.872 | 0.543 | 0.402 |
| World model, that recipe, + validation stop and totals term | 0.458 | 0.427 to 0.527 | 0.942 | 1.035 | 0.583 | 0.440 |
| LightGBM, average of five fits (three repeats) | 0.417 | 0.408 to 0.425 | 0.811 | 0.821 | 0.508 | 0.402 |
| Lever step: regular price, baseline times lift, average of five fits (three repeats) | 0.400 | 0.397 to 0.404 | 0.813 | 0.839 | 0.535 | 0.402 |

Reading: with all 55 products in a store able to affect each other, the
item-to-item part makes the forecast worse in every version, and makes it
vary more from seed to seed. Limiting it to products in the same category
narrows the gap but does not close it. It is not kept in this form. Training
for longer does not help either, so the first version was not stopped early.

Calendar, crowding and base price: none of them helps either model on
these two windows. The windows run from late July to early November 2011
and hold only Labor Day and Halloween, and training sees each event once
or twice, so this is a weak test of the calendar. The recorded base price
changes about twelve times a year per item and equals the shelf price in
every week without a promotion, so it is not a slow-moving regular price.

Latent (JEPA-style) loss on the direct model: no recipe beats the lift
model overall (0.406). Long pretraining followed by a slowed encoder is
level overall and the best of the seven on display and circular weeks.
With the encoder frozen after latent-only pretraining the model still
reaches 0.428, so the pretrained state alone carries most of the signal.

Training fixes on Breakfast at the Frat: total forecast over total sales is
within 3% of 1 in every run, so there is no summed-up bias to fix on this
dataset. The validation stop costs accuracy here, because it takes 8 of
only 90 training weeks out of the targets. Dropout with weight decay is
level overall, steadier across seeds, and a little better on every lever
column.

World model on Breakfast at the Frat: level with the direct lift model
(0.408 against 0.406), neither better nor worse, on the overall score and
on every lever column. Trained with long latent-only pretraining it gives
the best display-week score of any version (0.813, against 0.817 for
LightGBM) with circular weeks unchanged. The validation stop and totals
term hurt here, as they do for the direct model.

Lever step: building the forecast as baseline times lift is the part that
helps. It beats LightGBM overall in all five runs and halves the spread
between seeds. The overall score covers every level from item-store up to
the total; the lever columns are item-store weeks only, and there LightGBM
is still ahead. Regular price alone and the latent loss do not help here.

Dominick's analgesics, the two 8-week windows before the held-out final 8
weeks, one run each:

| Method | Overall | Weeks with a promotion code | Other weeks |
|---|---|---|---|
| Seasonal naive | 1.274 | 1.798 | 1.002 |
| Average of last 8 weeks | 0.723 | 1.492 | 0.684 |
| LightGBM with price, promotion and cost | 0.715 | 1.259 | 0.587 |
| Our model, item encoder and readout, three seeds | 0.692 (0.679 to 0.705) | 1.362 | 0.605 |
| Our model with regular price and baseline times lift, three seeds | 0.676 (0.665 to 0.682) | 1.339 | 0.603 |

| Our model with item-to-item (both) among analgesics in a store, three seeds | 0.697 (0.681 to 0.707) | 1.372 | 0.606 |

| LightGBM + US seasonal calendar and promotion crowding, three seeds | 0.734 (0.732 to 0.735) | 1.276 | 0.588 |
| Lift model, a price held four weeks becomes the regular price, three seeds | 0.674 (0.633 to 0.711) | 1.331 | 0.602 |
| Lift model + held price, calendar and crowding, three seeds | 0.919 (0.858 to 0.980) | 1.386 | 0.610 |
| Lift model trained for 24,000 steps instead of 6,000, three seeds | 1.473 (0.845 to 1.814) | 1.344 | 0.607 |

The last two rows are much worse overall while their item-store columns
barely move, so the damage is at the summed-up levels: a small bias in the
same direction on every item adds up at category and chain level. Longer
training and the calendar both cause it. The calendar also makes LightGBM
worse here.

All six runs of our model without the item-to-item part beat LightGBM
overall. The item-to-item part does not help here either. On single item-store weeks
LightGBM is still ahead, with or without a promotion code.

## Dominick's analgesics: training fixes and the world model (development windows, 3 seeds)

Lower is better. Reference: LightGBM 0.715, lift model 0.676, lift model trained for 24,000 steps 1.473.

| Run | Overall | Total | Item by store | Promo on | Promo off | Forecast / actual |
|---|---|---|---|---|---|---|
| Lift, stop on validation | 0.683 | 0.714 | 0.708 | 1.340 | 0.604 | 1.016 |
| Lift, totals in the loss | 0.819 | 1.042 | 0.701 | 1.291 | 0.601 | 0.909 |
| Lift, all fixes, long training | 0.716 | 0.751 | 0.712 | 1.341 | 0.605 | 0.971 |
| World model, fixes | 0.732 | 0.838 | 0.704 | 1.297 | 0.604 | 0.937 |
| World model, best recipe, fixes | 0.729 | 0.843 | 0.699 | 1.272 | 0.602 | 0.944 |

None beats the plain lift model (0.676). Stopping on validation keeps long
training safe (0.683 against 1.473 without it). Adding totals to the loss makes
the total worse, not better: it under-forecasts by 9%. The world model is level
with the lift model at item-by-store level and behind at the total, where it
under-forecasts by 6%.

## Combination with LightGBM, and negative binomial likelihood (development windows)

Lower is better. Frat 5 seeds, Dominick's analgesics 3 seeds. Combination weights are set on a validation window.

| Data | Run | Overall | Total | Item by store | Promo on | Display | Circular | Tag only | Promo off |
|---|---|---|---|---|---|---|---|---|---|
| Frat | LightGBM (reference) | 0.421 | | | | 0.817 | 0.822 | 0.508 | 0.405 |
| Frat | Lift model (reference) | 0.406 | | | | 0.840 | 0.863 | 0.537 | 0.406 |
| Frat | Lift + LightGBM combination | 0.388 | 0.307 | 0.485 | 0.701 | 0.800 | 0.814 | 0.511 | 0.396 |
| Frat | Lift, negative binomial | 0.410 | 0.320 | 0.506 | 0.739 | 0.842 | 0.864 | 0.546 | 0.408 |
| Dominick's | LightGBM (reference) | 0.715 | | | 1.259 | | | | 0.587 |
| Dominick's | Lift model (reference) | 0.676 | | | | | | | |
| Dominick's | Lift + LightGBM combination | 0.670 | 0.742 | 0.686 | 1.275 | | | | 0.586 |
| Dominick's | Lift, negative binomial | 0.665 | 0.701 | 0.701 | 1.289 | | | | 0.600 |

The combination is the best result on Frat (0.388) and beats both parts on
display and circular weeks. On Dominick's it is level with the lift model.
Negative binomial is level with Tweedie on both.

## New-item test, Breakfast at the Frat (development windows, 3 seeds)

One product in ten never seen in training. Item-by-store accuracy, lower is better.

| Run | Overall | New items | Known items |
|---|---|---|---|
| LightGBM | 0.627 | 1.510 | 0.509 |
| Lift model, no new-item handling (forecast zero) | 0.594 | 1.510 | 0.526 |
| Lift model, borrow from similar items | 0.461 | 0.890 | 0.526 |
| Lift model, borrow, item label hidden 10% in training | 0.459 | 0.879 | 0.525 |

LightGBM's new-item score equals the zero forecast (1.510): it returns no
usable forecast for an unseen series. Its row is a floor, not a fair
competitor, until it is given a cold-start route of its own.

## Unusual-plan test, first run

No test week had a lever mix under 2% of training rows, so the rare group was
empty. Promoted weeks with a usual mix: LightGBM 0.703, lift 0.733, world
model 0.735. The test needs a different definition of unusual before it says
anything.

## New items with a LightGBM route, and unusual plans judged item by item (Frat, 3 seeds)

| Run | Overall | New items | Known items |
|---|---|---|---|
| LightGBM, new item from labels, price and plan | 0.577 | 1.310 | 0.509 |
| Lift model, borrow from similar items (from above) | 0.461 | 0.890 | 0.526 |

Unusual = a lever mix the item had in under 2% of its own training weeks.

| Run | Unusual plans | Usual plans |
|---|---|---|
| LightGBM | 0.696 | 0.697 |
| Lift model | 0.743 | 0.725 |
| World model | 0.730 | 0.730 |

On new items the lift model with borrowing is well ahead of LightGBM (0.890
against 1.310). On unusual plans no model degrades much, and LightGBM stays
ahead on promoted weeks either way.

## World-model tricks from the literature, Breakfast at the Frat (development windows, 5 seeds)

Reference: world model 0.408, lift model 0.406 (display 0.840, circular 0.863, tag only 0.537), LightGBM 0.421.

| Run | Overall | Total | Item by store | Display | Circular | Tag only | Promo off | Forecast / actual |
|---|---|---|---|---|---|---|---|---|
| Plan split | 0.417 | 0.326 | 0.509 | 0.875 | 0.897 | 0.533 | 0.408 | 1.003 |
| Latent check, collapse penalty, near weeks weighted more | 0.427 | 0.355 | 0.517 | 0.872 | 0.936 | 0.544 | 0.409 | 1.034 |
| All of them | 0.424 | 0.342 | 0.516 | 0.869 | 0.907 | 0.547 | 0.411 | 1.030 |

None improves on the world model as it was. One untuned setting per trick.

## World model and tricks, Dominick's analgesics (development windows, 3 seeds)

Reference: LightGBM 0.715, lift model 0.676, earlier world model with the totals loss 0.732.

| Run | Overall | Total | Item by store | Promo on | Promo off | Forecast / actual |
|---|---|---|---|---|---|---|
| World model, validation stopping, no totals loss | 0.667 | 0.687 | 0.707 | 1.309 | 0.608 | 1.053 |
| + plan split | 0.704 | 0.793 | 0.693 | 1.276 | 0.594 | 1.031 |
| + latent check, collapse penalty, near weeks weighted more | 0.685 | 0.717 | 0.705 | 1.291 | 0.606 | 1.017 |
| + all of them | 0.730 | 0.852 | 0.692 | 1.282 | 0.592 | 1.017 |

Without the totals loss the world model is level with the lift model (0.667
against 0.676, three seeds, inside seed spread) and ahead of LightGBM. The
earlier gap (0.732) came from the totals loss, not from the world model. The
plan split helps at item-by-store level (0.693) and on promoted weeks (1.276)
but hurts the total (0.793).

## Plan A against plan B, Breakfast at the Frat (development windows, 3 seeds)

Same item and store, two weeks in the window with different lever mixes.

| Run | Picks the week that sold more | Miss on the size of the change (log scale, lower is better) |
|---|---|---|
| LightGBM | 78.2% | 0.448 |
| Lift model | 77.4% | 0.460 |
| World model | 77.5% | 0.459 |
| World model + plan split | 77.5% | 0.458 |

All four are level; LightGBM is marginally ahead on both scores.

## Overnight part 1: world model + LightGBM, five seeds on Dominick's, new items on Dominick's (development windows)

| Data | Run | Seeds | Overall | Total | Item by store | Promo on | Promo off |
|---|---|---|---|---|---|---|---|
| Frat | World model + LightGBM combination | 5 | 0.390 | 0.315 | 0.483 | 0.695 | 0.394 |
| Dominick's | World model + LightGBM combination | 3 | 0.668 | 0.738 | 0.685 | 1.267 | 0.588 |
| Dominick's | World model (no totals loss) | 5 | 0.666 | 0.688 | 0.706 | 1.303 | 0.606 |
| Dominick's | Lift model | 5 | 0.718 | 0.791 | 0.709 | 1.329 | 0.603 |
| Dominick's | LightGBM | 5 | 0.702 | 0.814 | 0.688 | 1.283 | 0.588 |

Frat combination: display weeks 0.799, circular weeks 0.809 (lift + LightGBM gave 0.388 overall, 0.800, 0.814).

The lift model's 0.676 on three seeds does not hold on five (0.718): seeds 4
and 5 were much worse, so it is unsteady on Dominick's. The world model holds
(0.667 on three seeds, 0.666 on five) and is ahead of LightGBM (0.702).

New items, Dominick's analgesics, 3 seeds (one product in ten never seen in training):

| Run | New items | Known items |
|---|---|---|
| LightGBM, from labels, price and plan | 0.923 | 0.685 |
| Lift model, borrow from similar items | 2.999 | 0.705 |

Borrowing fails here. The Dominick's setup gives the model only item and
store labels, so "most labels in common" has nothing to match on and the new
item borrows the sales level of arbitrary items in the store. The Frat result
(0.890 against 1.310) relied on category and manufacturer labels.

## All 28 Dominick's categories (development windows, one seed; analgesics is the five-seed mean)

Overall score and promoted-week score, lower is better. Same settings as analgesics, not tuned per category.

| Category | LightGBM | Lift | World model | Promo: LightGBM | Promo: Lift | Promo: World model |
|---|---|---|---|---|---|---|
| bat | 1.188 | 1.568 | 1.465 | 0.763 | 1.236 | 0.911 |
| oat | 1.084 | 1.026 | 1.060 | 3.394 | 3.394 | 3.405 |
| ptw | 0.486 | 0.445 | 0.485 | 2.591 | 2.542 | 2.530 |
| tti | 0.394 | 0.413 | 0.465 | 2.057 | 2.233 | 2.277 |
| ber | 0.454 | 0.460 | 0.470 | 0.998 | 1.048 | 1.017 |
| frd | 0.930 | 0.905 | 0.873 | 1.040 | 1.061 | 0.987 |
| soa | 0.517 | 0.521 | 0.516 | 0.900 | 1.034 | 1.022 |
| cra | 0.709 | 0.967 | 0.687 | 0.832 | 0.886 | 0.903 |
| did | 0.344 | 0.436 | 0.393 | 1.147 | 1.244 | 1.171 |
| tbr | 0.647 | 0.671 | 0.699 | 1.014 | 1.048 | 1.071 |
| fsf | 0.650 | 0.748 | 0.832 | 1.914 | 2.048 | 1.981 |
| tna | 1.163 | 1.030 | 0.959 | 1.440 | 1.418 | 1.533 |
| cig | 0.534 | 0.918 | 0.690 | n/a | n/a | n/a |
| frj | 0.314 | 0.360 | 0.376 | 0.486 | 0.566 | 0.591 |
| sna | 0.587 | 0.959 | 1.463 | 0.937 | 1.130 | 1.071 |
| tpa | 0.731 | 0.802 | 0.779 | 1.183 | 1.346 | 1.324 |
| lnd | 0.768 | 0.735 | 0.687 | 1.200 | 1.400 | 1.214 |
| fec | 0.414 | 0.358 | 0.397 | 1.063 | 1.363 | 1.529 |
| bjc | 0.992 | 1.248 | 1.229 | 1.926 | 2.714 | 3.070 |
| cer | 0.753 | 0.795 | 0.673 | 1.855 | 2.022 | 2.044 |
| cso | 0.613 | 0.532 | 0.583 | 1.226 | 1.258 | 1.272 |
| gro | 0.901 | 1.208 | 0.809 | 1.049 | 1.243 | 1.120 |
| sha | 0.994 | 0.938 | 0.906 | 0.919 | 1.084 | 1.069 |
| coo | 0.487 | 0.503 | 0.522 | 0.686 | 0.700 | 0.714 |
| che | 0.742 | 0.900 | 0.996 | 1.068 | 1.229 | 1.361 |
| fre | 0.417 | 0.374 | 0.382 | 0.732 | 0.770 | 0.735 |
| sdr | 0.747 | 1.406 | 0.864 | 0.721 | 1.425 | 0.878 |
| ana | 0.702 | 0.718 | 0.666 | 1.283 | 1.329 | 1.303 |
| Mean | 0.688 | 0.784 | 0.747 | 1.275 | 1.436 | 1.411 |
| Median | 0.676 | 0.772 | 0.689 | | | |

Categories won overall: LightGBM 14, lift model 5, world model 9 of 28. One of our two models beats LightGBM in 14. The world model beats the lift model in 15.
Categories won on promoted weeks: LightGBM 24, lift model 1, world model 2 of 27 (cigarettes has no promoted weeks in the windows).

LightGBM is the better model across Dominick's as a whole. Our models were set on analgesics and run unchanged; several categories go badly wrong (snacks 1.463, soft drinks lift 1.406, bath soap), which points to instability, not a small gap. One seed per category, so single-category differences are not reliable.

## M5 practice window: lift model and world model (one seed each)

Reference: LightGBM 0.715, plain model 0.699 (five-seed mean).

| Run | Overall | Total | Item by store | First day ahead | Last day ahead | Forecast / actual |
|---|---|---|---|---|---|---|
| Lift model | 0.683 | 0.563 | 0.832 | 0.597 | 0.765 | 0.954 |
| World model | 0.692 | 0.600 | 0.833 | 0.590 | 0.777 | 0.941 |

Both are ahead of LightGBM and of the plain model. One seed each, so the
difference between the two is not established. Both under-forecast by about 5%.

## Level fix on eight Dominick's categories (development windows, one seed)

The item's level taken from its latest 4 or 8 weeks, not the 26-week average. Overall score, lower is better.

| Category | LightGBM | Lift, as before | Lift, 4 weeks | Lift, 8 weeks | World model, as before | World model, 4 weeks | World model, 8 weeks |
|---|---|---|---|---|---|---|---|
| sna | 0.587 | 0.959 | 0.728 | 0.739 | 1.463 | 1.097 | 0.874 |
| sdr | 0.747 | 1.406 | 1.383 | 1.429 | 0.864 | 0.800 | 0.747 |
| bat | 1.188 | 1.568 | 1.528 | 2.203 | 1.465 | 1.217 | 1.195 |
| bjc | 0.992 | 1.248 | 1.173 | 1.123 | 1.229 | 1.131 | 1.100 |
| cig | 0.534 | 0.918 | 1.245 | 1.216 | 0.690 | 0.688 | 0.643 |
| che | 0.742 | 0.900 | 0.937 | 0.970 | 0.996 | 1.026 | 1.116 |
| ana (control) | 0.702 | 0.718 | 0.649 | 0.666 | 0.666 | 0.667 | 0.679 |
| fre (control) | 0.417 | 0.374 | 0.367 | 0.366 | 0.382 | 0.378 | 0.393 |

The fix helps where the diagnosis applied and does no harm to the controls,
but it does not close the gap. Snacks improves a lot and stays behind
LightGBM. The world model with an 8-week level reaches LightGBM on soft
drinks and bath soap. Cheese and cigarettes do not improve. The remaining
error is still at the category total: forecast over actual is far from 1 in
one window (cigarettes lift 1.58, soft drinks lift 1.33, cheese 0.84), and
item-by-store accuracy stays near LightGBM. One seed per run.

## World model with an 8-week level, all 28 Dominick's categories (development windows, one seed)

| Category | LightGBM | World model, 26-week level | World model, 8-week level |
|---|---|---|---|
| bat | 1.188 | 1.465 | 1.195 |
| oat | 1.084 | 1.060 | 1.036 |
| ptw | 0.486 | 0.485 | 0.425 |
| tti | 0.394 | 0.465 | 0.427 |
| ber | 0.454 | 0.470 | 0.505 |
| frd | 0.930 | 0.873 | 0.884 |
| soa | 0.517 | 0.516 | 0.518 |
| cra | 0.709 | 0.687 | 0.749 |
| did | 0.344 | 0.393 | 0.406 |
| tbr | 0.647 | 0.699 | 0.686 |
| fsf | 0.650 | 0.832 | 0.742 |
| tna | 1.163 | 0.959 | 1.004 |
| cig | 0.534 | 0.690 | 0.643 |
| frj | 0.314 | 0.376 | 0.342 |
| sna | 0.587 | 1.463 | 0.874 |
| tpa | 0.731 | 0.779 | 0.857 |
| lnd | 0.768 | 0.687 | 0.729 |
| fec | 0.414 | 0.397 | 0.417 |
| bjc | 0.992 | 1.229 | 1.100 |
| cer | 0.753 | 0.673 | 0.805 |
| cso | 0.613 | 0.583 | 0.500 |
| gro | 0.901 | 0.809 | 0.874 |
| sha | 0.994 | 0.906 | 0.921 |
| coo | 0.487 | 0.522 | 0.491 |
| che | 0.742 | 0.996 | 1.116 |
| fre | 0.417 | 0.382 | 0.393 |
| sdr | 0.747 | 0.864 | 0.747 |
| ana | 0.702 | 0.666 | 0.679 |
| Mean | 0.688 | 0.747 | 0.717 |
| Median | 0.676 | 0.689 | 0.736 |

The 8-week level beats LightGBM in 10 of 28 categories (the 26-week level did in 14) and improves on the 26-week level in 13. It removes the large failures (snacks 1.463 to 0.874, bath soap 1.465 to 1.195) and gives a little back in some categories that were fine. LightGBM is still ahead on the mean. One seed.

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

## Final test on the official M5 test period (23 May to 19 June 2016, three seeds)

Job `configs/jobs/final_m5.yaml`, commit 53fffe5 (clean), run 9 October 2026 on one Tesla T4. Settings fixed on the practice window before the run; reported as they came out, with no reruns.

| Model | Seed 1 | Seed 2 | Seed 3 | Mean | Range |
|---|---|---|---|---|---|
| LightGBM, overall | 0.548 | 0.547 | 0.547 | **0.547** | 0.547 to 0.548 |
| World model + normalisation, overall | 0.627 | 0.647 | 0.639 | 0.638 | 0.627 to 0.647 |
| LightGBM, total | 0.245 | 0.247 | 0.250 | 0.247 | |
| World model + normalisation, total | 0.363 | 0.387 | 0.388 | 0.379 | |
| LightGBM, item by store | 0.905 | 0.906 | 0.905 | 0.905 | |
| World model + normalisation, item by store | 0.896 | 0.898 | 0.896 | 0.897 | |
| LightGBM, forecast / actual | 0.979 | 0.980 | 0.978 | 0.979 | |
| World model + normalisation, forecast / actual | 1.023 | 1.027 | 1.029 | 1.026 | |

Published scores on the same period: winner 0.520, 10th-ranked 0.547, 50th-ranked 0.576, best statistical benchmark 0.671, seasonal naive 0.847. Our earlier LightGBM test run (seed 0) scored 0.552.

Findings.

- LightGBM is the better model on the official test period: 0.547 against 0.638. Its score equals the 10th-ranked entry.
- The world model sits between the best statistical benchmark (0.671) and the 50th-ranked entry (0.576).
- This reverses the practice window, where the world model led LightGBM (0.656 against 0.715).
- The gap is in the totals: 0.379 against 0.247. At item by store the world model is slightly better (0.897 against 0.905).
- The world model forecasts about 2.6% above actual sales in total. LightGBM is 2.1% below.
- Run times: world model about 101 minutes per seed (seed 1: 6,091 s wall, 6,017 s training, 3 s prediction); LightGBM about 78 minutes per seed.

| Run | Folder |
|---|---|
| World model + normalisation, seed 1 | `20261009T144236_m5_test_world_model_roll_norm_seed1_2d0e0f5e` |
| World model + normalisation, seed 2 | `20261009T162410_m5_test_world_model_roll_norm_seed2_be1ce9bf` |
| World model + normalisation, seed 3 | `20261009T180552_m5_test_world_model_roll_norm_seed3_a21d534c` |
| LightGBM, seed 1 | `20261009T194733_m5_test_lightgbm_seed1_df4acc88` |
| LightGBM, seed 2 | `20261009T210651_m5_test_lightgbm_seed2_3ca7ac16` |
| LightGBM, seed 3 | `20261009T222439_m5_test_lightgbm_seed3_885a8e3b` |

## Final test on the last 8 weeks of Dominick's (three seeds; frozen entrees world model two seeds so far)

Job `configs/jobs/final_dominicks.yaml`, commit 53fffe5, run 9 to 10 October 2026 on one Tesla T4. Settings fixed on the development windows before the run; reported as they came out. World model = brand switching, normalised roll-forward, average of three fits. Lower is better.

| Category | World model, overall (range) | LightGBM, overall (range) | World model, item by store | LightGBM, item by store | World model, forecast / actual | LightGBM, forecast / actual | Development (world model / LightGBM) |
|---|---|---|---|---|---|---|---|
| Cheese | 0.897 (0.881 to 0.928) | 0.708 (0.699 to 0.721) | 0.715 | 0.702 | 0.776 | 0.892 | 0.971 / 0.749 |
| Snack crackers | 0.448 (0.397 to 0.531) | 0.412 (0.405 to 0.419) | 0.608 | 0.596 | 0.929 | 0.917 | 1.092 / 0.586 |
| Soft drinks | 0.830 (0.805 to 0.868) | 0.989 (0.982 to 0.996) | 0.734 | 0.652 | 0.835 | 0.743 | 0.743 / 0.759 |
| Bottled juice | 0.987 (0.943 to 1.029) | 0.994 (0.989 to 0.998) | 0.816 | 0.750 | 0.839 | 0.787 | 1.048 / 0.987 |
| Analgesics | 0.867 (0.842 to 0.894) | 0.978 (0.971 to 0.987) | 0.713 | 0.714 | 0.925 | 0.904 | 0.622 / 0.701 |
| Frozen entrees | 0.489 (0.487 to 0.492) | 0.478 (0.475 to 0.483) | 0.561 | 0.529 | 0.888 | 0.795 | 0.391 / 0.419 |

Findings.

- The world model is better in two categories: soft drinks (0.830 against 0.989) and analgesics (0.867 against 0.978). Both were also its wins on the development windows. Every world-model seed beats every LightGBM seed in both.
- Bottled juice is level (0.987 against 0.994; the world model's seeds span LightGBM's).
- LightGBM is better in cheese (0.708 against 0.897, as on development), snack crackers (0.412 against 0.448) and frozen entrees (0.478 against 0.489 on two world-model seeds). Frozen entrees was a world-model win on development (0.391 against 0.419) and did not hold.
- The opposite pattern to M5: on Dominick's the world model's gains are at the totals and LightGBM is better or level item by store in every category. Both models forecast below actual sales in these 8 weeks, LightGBM more so (as low as 0.74 in soft drinks).
- The test weeks are harder than the development windows for both models in soft drinks, bottled juice and analgesics.
- Frozen entrees, world model seed 3: the runtime dropped during this run; the restart could not resume it (random states saved on the GPU were restored without moving them to the CPU, fixed after). It runs again from the start, settings unchanged (`configs/jobs/final_dominicks_fre_seed3.yaml`), and is added when in.

## Final test on the last 8 weeks of Breakfast at the Frat (three seeds)

Job `configs/jobs/final_frat.yaml`, commit 53fffe5, run 10 October 2026 on one Tesla T4. Settings fixed on the development windows before the run; reported as they came out. Lower is better.

| Model | Seed 1 | Seed 2 | Seed 3 | Mean | Range | Total (mean) | Item by store (mean) | Forecast / actual (mean) | Development |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM, all recorded levers | 0.561 | 0.551 | 0.535 | **0.549** | 0.535 to 0.561 | 0.503 | 0.608 | 0.937 | 0.421 |
| World model | 0.684 | 0.657 | 0.702 | 0.681 | 0.657 to 0.702 | 0.685 | 0.665 | 1.000 | 0.408 |
| Lift model, readout dropout and weight decay | 0.680 | 0.744 | 0.718 | 0.714 | 0.680 to 0.744 | 0.758 | 0.677 | 0.977 | 0.406 |

Findings.

- LightGBM is clearly better on the final 8 weeks: 0.549 against 0.681 for the world model and 0.714 for the lift model, and every LightGBM seed beats every seed of the other two. It is better at the total and item by store.
- On the development windows the three were level (0.406 to 0.421). The final weeks are harder for all three, much more so for ours.
- The world model is better than the lift model (0.681 against 0.714) and has no bias in total (forecast / actual 1.000); LightGBM is 6% under.

| Run | Folder |
|---|---|
| LightGBM, seeds 1 to 3 | `frat_final_lightgbm_seed1_0ae11c20`, `_seed2_af4d0223`, `_seed3_4ce9a34b` |
| Lift model, seeds 1 to 3 | `frat_final_lift_seed1_84375413`, `_seed2_276bab72`, `_seed3_c9b2c390` |
| World model, seeds 1 to 3 | `frat_final_world_model_seed1_b8d962b7`, `_seed2_625b5483`, `_seed3_c6d076ef` |
