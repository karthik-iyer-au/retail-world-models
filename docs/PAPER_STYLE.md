# Figure and table style for the papers

Reference: the author's MolSanity paper (public repo Kar488/molsanity, `paper/`) and the M-JEPA manuscript (JCIM revision 12). MolSanity sets the figure style; M-JEPA adds figure types. Style only is taken; no text or code is copied.

## Layout
- Two-column, 10pt. One overview diagram in the method section; most figures in Results; supplementary material separate.
- Every figure and table is produced by a script in the repo (figures and tables are generated, never hand-edited). Numbers quoted in the text come from generated macros.
- Typical figure: a wide row of 2 or 3 panels, 7.1 in wide, 2.3 to 3.3 in tall, bold panel letters (a), (b), (c). Single-column figures 3.3 in wide.

## Figure style (one shared style module)
- Serif font (Palatino / TeX Gyre Pagella), base 8 pt, titles 8.5, ticks 7.2, legend 7. Vector PDF, fonttype 42.
- Ink #1a1a1a, accent #264E78, muted #6b7280, grid #d9dde3, axis edge #9aa2ad at 0.6 lw.
- Categorical colours Okabe-Ito (#0072B2, #D55E00, #009E73, #CC79A7, #E69F00, #56B4E9, #000000), each model with a fixed colour and marker across every figure.
- Left and bottom spines only; light grid on one axis; frameless legends, often at figure level.
- The highlighted series in accent at lw 2.0; others muted at lw 1.0. Reference lines muted: dashed for chance or 1.0, dotted for zero.
- Error bars thin (lw 0.7, small caps); medians as short dark ticks; scatter points with white edges.

## Tables
- booktabs, small font, bold header, best value per group in bold (stated as emphasis only), three decimals, minus written as a proper minus.
- No plus-minus in tables: seed spread and intervals go to figures or supplementary tables.

## Captions
- Above the figure or table. Open with one bold sentence stating the finding, then what each panel shows, what colours and lines mean, and n.

## Overview diagram
- Drawn in matplotlib from the same style module: rounded boxes, thin arrows, two accent colours and a light grey panel, left-to-right flow, counts computed from results, one takeaway line.

## For this paper
- Fixed colours: LightGBM, world model, lift model, plain model, blends.
- Likely figures: overview of state, plan and roll-forward; accuracy by dataset and category (dot plot with seed ranges); promoted / after-promotion / ordinary forecast-over-actual; plan A vs plan B; new items; per-category wins.

## Chart ideas from example material

Chart types only, drawn in the MolSanity style above. No layout, data, names, numbers, branding or colours are taken from the examples.

Plan and promotion charts:
- dot plot of measured lift against the lift a model assumes, with the measured band, one row per deal type;
- diverging bars for gains and losses by item group, group size in brackets;
- waterfall from a baseline plan to a changed plan, one step per change;
- calendar strip: one row per item, one cell per week, cell shading for deal depth and display or circular;
- small multiples of weekly series with the scored weeks shaded.

Accuracy diagnostics:
- forecast against actual at each level of the hierarchy (item-store, item, store, category);
- over or under forecast by how fast the item sells (velocity bands);
- units missed by store and week (heatmap);
- category by week over the whole history, actual against the model, with the promotion plan by week underneath (items on deal, share of units on deal, average depth);
- one item in one store followed through;
- the items that miss most;
- every model on the same test, and where two models disagree.

## Figure types from the M-JEPA manuscript
- Overview in three columns: input, what the method learns, how it is used downstream.
- Staged pipeline with an explicit leakage barrier between selection and final test. Ours: development windows, then the locked final 8 weeks.
- Box plots with jittered points per run, faceted by condition, n under each box, shaded bands marking regimes.
- Faceted scatter of a setting against error, colour for one factor and marker for another.
- Small-multiple reliability diagrams, one panel per task, diagonal as perfect, shaded gap. Ours: forecast against actual by band, per category.
- Trade-off scatter, one point per task, quadrants labelled. Ours: accuracy gain against plan-response gain per category.

## Illustrations of the data (chart types only, redrawn on public data)
- The data cube: item x store x week, with one cell opened up to show the row (sales, price, promotion, display, feature, cost and the rest). In the paper this is the panel; for merchants, the data cube. Not "prism".
- Small pictures, one per data rule: values as of their week, every week present, why a week was low, items linked.
- One item in one store over the whole history: weekly sales coloured regular against promotion, discount below, display or feature below that.
- The item-week as a graph: one item linked to its stores, its offer, its substitutes, the season and its history, line width for units.
- Messages to state to sales: what each link says this week, the learned state, the forecast against actual. This is the world model's own picture.
