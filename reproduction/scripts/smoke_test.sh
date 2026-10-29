#!/usr/bin/env bash
# Reproduction smoke test. Fails loudly when a prerequisite is absent; never
# reports success for an incomplete reproduction.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== 1. Prerequisite inputs =="
python3 reproduction/scripts/validate_inputs.py; v=$?

echo
echo "== 2. Figure-script data paths =="
python3 reproduction/scripts/check_figure_paths.py; p=$?

echo
echo "== 3. Script syntax =="
s=0
for f in analysis/*.py audit/scripts/*.py reproduction/scripts/*.py training/*.py case_study/kras_g4/*.py case_study/kras_g4/denovo2/*.py; do
    python3 -m py_compile "$f" 2>/dev/null || { echo "  SYNTAX FAIL: $f"; s=1; }
done
[ "$s" -eq 0 ] && echo "  all scripts compile"

echo
echo "=================================================="
echo "prerequisites rc=$v   figure-paths rc=$p   syntax rc=$s"
if [ "$v" -ne 0 ] || [ "$p" -ne 0 ] || [ "$s" -ne 0 ]; then
    echo "RESULT: reproduction inputs INCOMPLETE — see reproduction/manifests/READINESS.md"
    exit 1
fi
echo "RESULT: all checks passed"
