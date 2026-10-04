# Our model

Working name: retail state model. For each item in a store it holds a state.
Given what is planned for the coming periods, it forecasts sales from that state.

| Part | Job | Module |
|---|---|---|
| Item encoder | Turns an item's recent history into a state | `state_model.py` |
| Sales readout | Expected sales from the state and the period's plan | `state_model.py` |

Rules for code in this folder:

1. It is written for this repository, against public data.
2. Published components are imported from `rwm/prior_work/`, never copied in,
   so it stays clear what is ours and what is not.
3. Every component is kept only if removing it makes the measured result
   worse. Each such test is an experiment config and a recorded run.
4. The model implements the same `Forecaster` interface as prior work and is
   scored by the same code.

## Item-to-item part

Set `group_by` (which items can affect each other, for example `[store_id]`)
and `neighbours` to `similarity`, `attention` or `both`. With `group_by` alone
the model is trained on whole groups at a time but has no item-to-item part;
that is the like-for-like comparison. `neighbour_weights(future)` returns the
weight each item gave each neighbour for each forecast date. The versions are
described at the top of `state_model.py` and tested in
`tests/test_item_to_item.py`.

## Lever step

`regular_price`, `lift_readout` and `latent_weight` switch on its three parts;
`levers` names the columns that count as levers (by default every column in
`extra`). `breakdown(future)` splits each forecast into baseline and lift.
Described at the top of `state_model.py`, tested in `tests/test_lever_step.py`.
