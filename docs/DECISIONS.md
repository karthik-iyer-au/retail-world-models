# Decision log

One dated entry per design decision, newest last. Each entry says what was
decided and what it was based on. Decisions based on a result name the run.

## 2026-10-04

**Scope.** Public data only: M5, Dominick's Finer Foods, Breakfast at the
Frat. No employer or client code, data, weights or results.

**Dataset roles.** M5 is the accuracy check against published benchmarks.
Dominick's carries price, promotion and cost effects. Breakfast at the Frat
carries display and feature effects. Shelf space is out of scope because no
public dataset records it.

**The method is not chosen in advance.** The repository compares candidate
approaches (joint-embedding prediction, energy-based scoring, masked
reconstruction, pretrained time series models, gradient boosted trees) and
reports whichever the public-data results support.

**Every component must earn its place.** Kept only if removing it makes the
measured result worse.

**Energy-based component.** Its job is to flag forecasts for lever
combinations unlike anything in history. Test: forecast error should rise as
the energy score rises.

**Explanations are built into the model output.** Each lever's contribution
is the difference between the forecast with the lever on and off, not an
approximation fitted afterwards.

**Three separate checks.** Accuracy, lever effects, explanation faithfulness.
See `PROTOCOL.md`.

**Repository structure.** Data, prior work, our model, evaluation and
experiments are separate packages sharing one table shape and one model
interface. Every run records commit, config, data checksums, seed and
package versions.
