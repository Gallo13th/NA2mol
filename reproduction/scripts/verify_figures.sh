#!/usr/bin/env bash
# Run every statistical figure script and verify its outputs.
#
#   tmux new-session -d -s figs 'bash reproduction/scripts/verify_figures.sh > /tmp/verify_figures.log 2>&1'
#
# For each script: record runtime and exit code, then check that the expected
# artefacts were (re)written, are non-trivial in size, and are valid PDFs with at
# least one page. A script that "succeeds" without producing a figure is a failure.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

FIGDIR="reproduction/figures"
mkdir -p "$FIGDIR"
REPORT=/tmp/figure_report.txt
: > "$REPORT"

# script : expected output stem(s) in $FIGDIR
STEPS=(
  "analysis/fig2_cross_benchmark.py:cross_benchmark_transfer"
  "analysis/fig3_ablation.py:fig_fine_grained_ablation"
  "analysis/sfig1_chemical_space.py:chemical_space_audit"
  "analysis/sfig2_mapping_control.py:molcraft_mapping_control"
  "analysis/sfig3_property_landscape.py:property_landscape_audit"
  "case_study/kras_g4/polish_figure4.py:figure4"
  "case_study/kras_g4/assemble_extended_data_fig_kras.py:kras"
)

pass=0; fail=0; skip=0

for entry in "${STEPS[@]}"; do
    script="${entry%%:*}"
    stem="${entry##*:}"
    echo "───────────────────────────────────────────────────────────"
    echo ">>> $script"

    if [ ! -f "$script" ]; then
        echo "    SKIP: script not present"; skip=$((skip+1))
        echo "SKIP  $script (absent)" >> "$REPORT"; continue
    fi

    before=$(find "$FIGDIR" -newermt '1970-01-01' -name '*.pdf' -printf '%f\n' 2>/dev/null | sort)

    start=$(date +%s)
    timeout 3600 python3 "$script" > "/tmp/fig_$(basename "$script" .py).log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))

    # anything written or rewritten in this window
    produced=$(find "$FIGDIR" -name '*.pdf' -newermt "-${dur} seconds" -printf '%f\n' 2>/dev/null | sort)

    echo "    rc=$rc  ${dur}s"
    if [ $rc -ne 0 ]; then
        echo "    FAILED — last lines:"; tail -12 "/tmp/fig_$(basename "$script" .py).log" | sed 's/^/      /'
        echo "FAIL  $script (rc=$rc)" >> "$REPORT"; fail=$((fail+1)); continue
    fi

    if [ -z "$produced" ]; then
        echo "    NO PDF WRITTEN — treating as failure"
        echo "FAIL  $script (no output)" >> "$REPORT"; fail=$((fail+1)); continue
    fi

    bad=0
    while read -r f; do
        [ -z "$f" ] && continue
        path="$FIGDIR/$f"
        bytes=$(stat -c%s "$path" 2>/dev/null || echo 0)
        pages=$(pdfinfo "$path" 2>/dev/null | awk '/^Pages/{print $2}')
        if [ "${bytes:-0}" -lt 2000 ] || [ -z "$pages" ] || [ "${pages:-0}" -lt 1 ]; then
            echo "    BAD  $f (${bytes} bytes, pages=${pages:-none})"; bad=1
        else
            echo "    ok   $f (${bytes} bytes, ${pages} page(s))"
        fi
    done <<< "$produced"

    if [ $bad -eq 0 ]; then
        echo "PASS  $script -> $(echo "$produced" | tr '\n' ' ')" >> "$REPORT"; pass=$((pass+1))
    else
        echo "FAIL  $script (invalid output)" >> "$REPORT"; fail=$((fail+1))
    fi
done

echo
echo "═══════════════════════════════════════════════════════════"
cat "$REPORT"
echo "═══════════════════════════════════════════════════════════"
echo "pass=$pass fail=$fail skip=$skip"
[ $fail -eq 0 ] && echo "RESULT: every runnable figure script produced valid output." || echo "RESULT: $fail figure step(s) failed."
