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

**Model design.** Agreed before any model code was written:

1. The state is per item and store, with category context added by a later
   item-to-item part. One state per category would lose the item detail.
2. The forecast loss is always part of training. Latent-space prediction is
   used for pretraining and as a supporting loss, never alone.
3. A lever a dataset does not record is passed to the model as "not known",
   not as zero, so one model runs on every dataset.
4. "Lever off" means regular price and no promotion. Regular price is the
   highest price in the last 12 weeks.
5. Build order: item encoder and sales readout on M5 first, then the
   item-to-item part and the lever step on Dominick's, then the lever
   breakdown and the familiarity score.

**Item-to-item part.** Conditional similarity, attention, and attention that
starts from the conditional similarity map are built as three versions of
the same model and compared on periods where levers changed.

**Benchmark fixed.** LightGBM, one model per store, with settings fixed
before the test-period run: 0.552 on the official M5 test period and 0.715
on the 28 days before it (runs in `results/promoted/`). Model development
uses the windows before the test period.

**Compute.** Tree models run on CPU. Deep models run on a Colab GPU through
`notebooks/colab_run.ipynb`, which calls the repository's commands and holds
no model or data code.
