# Manuscript

Two-column preprint in the same format as the MolSanity paper: LaTeX, figures
from one matplotlib style module, booktabs tables and inline numbers generated
from the run records.

    make        # figures, tables, macros, then main.pdf (xelatex)

- `figs/rwdata.py`: every number used, read from `results/promoted/` or copied
  from the named section of `results/README.md`.
- `figs/style.py`: fonts, colours and one fixed colour and marker per model.
- `figs/make_figures.py`, `figs/make_tables.py`: write `figs/fig_*.pdf`,
  `generated/tab_*.tex` and `generated/macros.tex`.
- `main.tex` (preamble, title), `abstract.tex`, `body.tex`.
