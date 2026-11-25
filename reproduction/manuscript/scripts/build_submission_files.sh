#!/bin/bash
# Build the Nature Biotechnology submission files in a temporary directory.
#
# Produces three separate PDFs, which is what NBT expects:
#   sn-article-revised.pdf   manuscript: main text + Methods + 4 figures + 2 tables
#   extended_data.pdf        Extended Data Fig. 1 + Extended Data Tables 1-9 (the 10-item cap)
#   supplementary.pdf        Supplementary Notes 1-6 + Supplementary Tables 1-5
#
# The item files (si_*.tex) are shared by the Extended Data and Supplementary masters and carry
# their own headings, so the split lives in extended_data.tex / supplementary.tex rather than in
# the item files. Regenerate the split with scripts/split_si_for_nbt.py only if the allocation or
# the numbering changes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"
cp -R figures bst "$TMP"/
cp sn-article-revised.tex extended_data.tex supplementary.tex \
   table_radar_values.tex table_overlap.tex \
   supplementary_notes_revised.tex ed_table1.tex si_*.tex \
   sn-bibliography.bib sn-jnl.cls "$TMP"/

cd "$TMP"

echo "== manuscript"
pdflatex -interaction=nonstopmode -halt-on-error sn-article-revised.tex >/dev/null
BIBINPUTS="$(pwd):" BSTINPUTS="$(pwd):$(pwd)/bst:" bibtex sn-article-revised >/dev/null
pdflatex -interaction=nonstopmode sn-article-revised.tex >/dev/null
pdflatex -interaction=nonstopmode sn-article-revised.tex >/dev/null
if grep -qE "undefined (reference|citation)" sn-article-revised.log; then
  echo "BUILD FAILED: undefined references/citations in the manuscript" >&2
  grep -E "undefined (reference|citation)" sn-article-revised.log >&2
  exit 1
fi
cp sn-article-revised.pdf "$ROOT/sn-article-revised.pdf"

for doc in extended_data supplementary; do
  echo "== $doc"
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" >/dev/null
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null
  if grep -q "LaTeX Warning: Reference" "$doc.log"; then
    echo "BUILD FAILED: unresolved references in $doc" >&2
    grep "LaTeX Warning: Reference" "$doc.log" >&2
    exit 1
  fi
  cp "$doc.pdf" "$ROOT/$doc.pdf"
done

# sn-jnl appends its footer number to the last line of a page that ends inside a float caption,
# which prints the number on top of the caption. Repair those runs before shipping.
FIXPN="$ROOT/scripts/fix_page_numbers.py"
if command -v python3 >/dev/null 2>&1 && [ -f "$FIXPN" ]; then
  for pdf in sn-article-revised.pdf extended_data.pdf supplementary.pdf; do
    python3 "$FIXPN" "$ROOT/$pdf" || echo "warning: page-number repair failed for $pdf" >&2
  done
fi

echo
printf '%-26s %s pages\n' sn-article-revised.pdf "$(pdfinfo sn-article-revised.pdf | awk '/^Pages/{print $2}')"
printf '%-26s %s pages\n' extended_data.pdf      "$(pdfinfo extended_data.pdf      | awk '/^Pages/{print $2}')"
printf '%-26s %s pages\n' supplementary.pdf      "$(pdfinfo supplementary.pdf      | awk '/^Pages/{print $2}')"
