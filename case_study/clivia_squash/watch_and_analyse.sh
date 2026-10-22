#!/usr/bin/env bash
# Wait for a generation run to finish, then analyse it.
#
#   bash watch_and_analyse.sh pilot_clivia
#
# Polls for the run's SDF, then runs analyse_generation.py and appends the report to
# reports/<prefix>.txt. Safe to leave running in tmux across an SSH drop.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1          # rna2smol/
HERE=case_study/clivia_squash
PY=/home/data/users/gaoletian/.conda/envs/rna2smol/bin/python
PREFIX="${1:?usage: watch_and_analyse.sh <output_prefix>}"
mkdir -p "$HERE/reports"
REPORT="$HERE/reports/$PREFIX.txt"

echo "watching for output/$PREFIX ..." | tee -a "$REPORT"
# genbypkt writes output/<prefix>/valid_mols.sdf; wait up to 6 h
for i in $(seq 1 2160); do
    # genbypkt merges to output/<prefix>/<basename>/<basename>.sdf
    SDF=$(find "output/$PREFIX" -mindepth 2 -maxdepth 2 -name '*.sdf' 2>/dev/null | head -1)
    if [ -s "$SDF" ]; then
        # a second sample 2 min later must be the same size -> write finished
        a=$(stat -c%s "$SDF"); sleep 120; b=$(stat -c%s "$SDF" 2>/dev/null || echo 0)
        if [ "$a" = "$b" ] && [ "$a" -gt 0 ]; then
            echo "[$(date +%H:%M:%S)] SDF stable at $a bytes; analysing" | tee -a "$REPORT"
            $PY "$HERE/analyse_generation.py" --sdf "$SDF" --label "$PREFIX" \
                --out "$HERE/reports/comparison_$PREFIX.csv" 2>&1 | tee -a "$REPORT"
            echo "[$(date +%H:%M:%S)] DONE" | tee -a "$REPORT"
            exit 0
        fi
    fi
    sleep 10
done
echo "timed out waiting for output/$PREFIX" | tee -a "$REPORT"
exit 1
