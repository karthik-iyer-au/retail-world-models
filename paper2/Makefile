# Rebuild the manuscript from the committed run records.
#
#   make            # figures + tables + PDF
#   make figures    # vector PDF figures only
#   make clean      # remove build products (not the sources)
#
# Every number, table and figure comes from results/paper2/ through the scripts in figs/.

PY ?= python3
XELATEX ?= xelatex

.PHONY: all figures pdf tmlr clean

all: pdf

figures:
	cd figs && $(PY) make_figures.py

# Three passes: the second resolves labels and reflows floats, the third
# settles references that moved.
pdf: figures
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex

# TMLR submission (anonymised, TMLR style file in tmlr/), pdflatex
tmlr: figures
	for i in 1 2 3; do pdflatex -interaction=nonstopmode main_tmlr.tex >/dev/null; done

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
