# Rebuild the manuscript from the committed run records.
#
#   make            # figures + tables + PDF
#   make figures    # vector PDF figures only
#   make tables     # LaTeX tables + inline-number macros only
#   make clean      # remove build products (not the sources)
#
# Every number, table and figure comes from results/promoted/ and
# results/records/snapshot.json (read by figs/rwdata.py), and from
# results/paper1_revision/ (read by figs/make_revision*.py), through the scripts in figs/.
#
# Until results/records/snapshot.json has been built from the run folders
# (scripts/records_snapshot.py, on Colab), the build uses the hand-transcribed
# values in results/records/snapshot_transcribed.json and says so. Once the
# snapshot exists the transcribed file is never read.

PY ?= python3
XELATEX ?= xelatex

SNAPSHOT := $(CURDIR)/../results/records/snapshot.json
ifeq ($(wildcard $(SNAPSHOT)),)
export RWDATA_ALLOW_TRANSCRIBED := 1
$(warning results/records/snapshot.json not found: building from the hand-transcribed values (results/records/snapshot_transcribed.json))
else
unexport RWDATA_ALLOW_TRANSCRIBED
endif

.PHONY: all figures tables pdf clean

all: pdf

figures:
	cd figs && $(PY) make_figures.py
	cd figs && $(PY) make_revision_figures.py

tables:
	cd figs && $(PY) make_tables.py
	cd figs && $(PY) make_revision.py

# Three passes: the second resolves labels and reflows floats, the third
# settles references that moved.
pdf: figures tables
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex
	$(XELATEX) -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
