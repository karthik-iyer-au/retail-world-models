# Rebuild paper 3 from the committed run records (results/paper3).
#
#   make            # figures, tables, numbers and the PDF
#   make figures    # figures, tables and numbers only

PY ?= python3
XELATEX ?= xelatex

.PHONY: all figures pdf icwm clean

all: pdf

figures:
	cd figs && $(PY) make_figures.py

pdf: figures
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex

icwm: figures
	$(XELATEX) -interaction=nonstopmode main_icwm.tex
	$(XELATEX) -interaction=nonstopmode main_icwm.tex
	$(XELATEX) -interaction=nonstopmode main_icwm.tex

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
