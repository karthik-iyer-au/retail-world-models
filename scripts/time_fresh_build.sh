#!/usr/bin/env bash
# Clone the repository afresh, run `make` in paper/, check the generated tables match the committed ones,
# and write the wall-clock time to paper/generated/build_values.tex (reported in Reproducibility and compute).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$(mktemp -d)"
git clone -q --depth 1 "file://$ROOT" "$D/repo"
cd "$D/repo/paper"
start=$(date +%s)
make > "$D/build.log" 2>&1
secs=$(( $(date +%s) - start ))
if [ -n "$(git status --short -- '*.tex')" ]; then
  echo "generated tables differ from the committed ones:"; git status --short -- '*.tex'; exit 1
fi
printf '\\newcommand{\\buildSeconds}{%s}\n\\newcommand{\\buildDate}{%s}\n' "$secs" "$(date -u '+%-d %B %Y')" \
  > "$ROOT/paper/generated/build_values.tex"
echo "fresh-clone build: $secs s, tables identical"
rm -rf "${D:?}"
