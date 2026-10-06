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
