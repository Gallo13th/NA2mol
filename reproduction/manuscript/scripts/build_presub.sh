#!/bin/bash
# Build the presubmission supplement in a temporary directory.
# Usage: scripts/build_presub.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"
cp -R figures bst "$TMP"/
cp presubmission_supplement.tex presub_table1.tex presub_table2.tex \
   sn-bibliography.bib sn-jnl.cls "$TMP"/

cd "$TMP"
pdflatex -interaction=nonstopmode presubmission_supplement.tex >/dev/null
BIBINPUTS="$(pwd):" BSTINPUTS="$(pwd):$(pwd)/bst:" bibtex presubmission_supplement >/dev/null
pdflatex -interaction=nonstopmode presubmission_supplement.tex >/dev/null
pdflatex -interaction=nonstopmode presubmission_supplement.tex >/dev/null

if grep -qE "undefined (reference|citation)" presubmission_supplement.log; then
  echo "BUILD FAILED: undefined references/citations" >&2
  exit 1
fi
cp presubmission_supplement.pdf "$ROOT/presubmission_supplement.pdf"
echo "wrote $ROOT/presubmission_supplement.pdf"
