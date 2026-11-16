#!/usr/bin/env bash
# arXiv source archives for both papers (arXiv takes TeX source, not a TeX-made PDF). Each archive holds the .tex
# files, generated/ and the figure PDFs, and is checked to build with pdflatex (three passes, no errors) before packing.
#   scripts/make_arxiv_source.sh <out folder>
set -euo pipefail
OUT=$(realpath "${1:?out folder}"); ROOT=$(cd "$(dirname "$0")/.." && pwd); W=$(mktemp -d)
mkdir -p "$OUT"
# paper 1 (arXiv build: main.tex; the journal files are left out)
mkdir -p "$W/p1/generated" "$W/p1/figs"
cp "$ROOT"/paper/*.tex "$W/p1/"; rm -f "$W/p1/main_ijpe.tex" "$W/p1/supplement_ijpe.tex"
cp "$ROOT"/paper/generated/*.tex "$W/p1/generated/"
grep -oh 'includegraphics\(\[[^]]*\]\)\?{[^}]*}' "$ROOT"/paper/*.tex "$ROOT"/paper/generated/*.tex | sed 's/.*{//;s/}//' | sort -u |
  while read -r f; do cp "$ROOT/paper/$f" "$W/p1/figs/" 2>/dev/null || cp "$ROOT/paper/$f.pdf" "$W/p1/figs/"; done
# paper 2
mkdir -p "$W/p2/generated" "$W/p2/figs" "$W/p2/sections"
cp "$ROOT"/paper2/{main,abstract,body,references}.tex "$W/p2/"; cp "$ROOT"/paper2/sections/*.tex "$W/p2/sections/"
cp "$ROOT"/paper2/generated/*.tex "$W/p2/generated/"; cp "$ROOT"/paper2/figs/*.pdf "$W/p2/figs/"
for p in p1 p2; do
  (cd "$W/$p" && for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true; done
   if grep -q '^! ' main.log || [ ! -f main.pdf ]; then echo "$p: pdflatex errors"; exit 1; fi
   rm -f ./*.aux ./*.log ./*.out ./*.toc ./*.pdf)
done
tar -czf "$OUT/paper1_arxiv_source.tar.gz" -C "$W/p1" .
tar -czf "$OUT/paper2_arxiv_source.tar.gz" -C "$W/p2" .
rm -rf "${W:?}"
echo "written: $OUT/paper1_arxiv_source.tar.gz $OUT/paper2_arxiv_source.tar.gz"
