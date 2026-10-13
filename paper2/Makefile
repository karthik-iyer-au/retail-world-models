# Rebuild the manuscript from the committed run records.
#
#   make            # figures + tables + PDF
#   make figures    # vector PDF figures only
#   make tables     # LaTeX tables + inline-number macros only
#   make clean      # remove build products (not the sources)
#
# Every number, table and figure comes from results/promoted/ and
# figs/rwdata.py through the scripts in figs/.

PY ?= python3
XELATEX ?= xelatex

.PHONY: all figures tables pdf clean

all: pdf

figures:
	cd figs && $(PY) make_figures.py

tables:
	cd figs && $(PY) make_tables.py

# Three passes: the second resolves labels and reflows floats, the third
# settles references that moved.
pdf: figures tables
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
