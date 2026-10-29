#!/usr/bin/env bash
# Honest reproduction test for the numerical tables.
#
# The naive check (re-run the exporter, diff the output) is misleading here: the
# exporter guards several blocks with `if <cache>.exists():`, so when a cache is
# missing the block is skipped and the PRE-EXISTING file is left untouched — it
# then compares equal and looks reproducible.
#
# This script moves the committed tables aside first, regenerates from scratch, and
# reports what could and could not be produced. Anything absent afterwards is
# genuinely not reproducible from the inputs in this checkout.
#
#   tmux new-session -d -s vt 'bash reproduction/scripts/verify_tables.sh > /tmp/verify_tables.log 2>&1'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

TABLES="data/benchmark/output/tables"
SAVED="/tmp/tables_committed"

echo "=== baseline: the tables shipped with the checkout ==="
if [ ! -d "$TABLES" ]; then echo "MISSING $TABLES"; exit 1; fi
committed=$(cd "$TABLES" && find . -name '*.csv' | sort)
echo "$committed" | wc -l | xargs printf '  %s tables\n'

rm -rf "$SAVED"
cp -a "$TABLES" "$SAVED" || { echo "could not stage a backup"; exit 1; }

echo
echo "=== removing them and regenerating from scratch ==="
rm -rf "$TABLES"
start=$(date +%s)
python3 analysis/export_tables.py > /tmp/export_tables_run.log 2>&1
rc=$?
echo "  rc=$rc, $(( $(date +%s) - start ))s"
[ $rc -ne 0 ] && { echo "RESULT: export FAILED"; tail -20 /tmp/export_tables_run.log; exit 1; }

produced=$(cd "$TABLES" 2>/dev/null && find . -name '*.csv' | sort)
[ -z "$produced" ] && { echo "RESULT: nothing produced"; exit 1; }
echo "  $(echo "$produced" | wc -l) tables produced"

echo
echo "=== reproducible from scratch ==="
same=0
while read -r f; do
    [ -z "$f" ] && continue
    if [ ! -f "$SAVED/$f" ]; then
        echo "  NEW      $f"
    elif cmp -s "$SAVED/$f" "$TABLES/$f"; then
        echo "  IDENTICAL $f"; same=$((same+1))
    else
        echo "  DIFFERS   $f"
    fi
done <<< "$produced"

echo
echo "=== NOT produced (silently absent from this checkout) ==="
missing=0
while read -r f; do
    [ -z "$f" ] && continue
    if [ ! -f "$TABLES/$f" ]; then echo "  MISSING  $f"; missing=$((missing+1)); fi
done <<< "$committed"

echo
echo "═══════════════════════════════════════════════════════════"
echo "  byte-identical: $same"
echo "  not regenerated: $missing"
echo "═══════════════════════════════════════════════════════════"
[ "$missing" -eq 0 ] && echo "RESULT: every shipped table regenerates from the inputs present." \
                     || echo "RESULT: $missing table(s) cannot be regenerated — a required input is absent."
