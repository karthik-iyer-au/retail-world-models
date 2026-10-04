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
