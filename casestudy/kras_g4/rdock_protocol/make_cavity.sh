#!/usr/bin/env bash
# make_cavity.sh <system.prm>
#
# Runs rbcavity -was: writes the docking site definition (<receptor>.as,
# cached and reused by rbdock) plus the grid files (.gs) required by the
# indexed scoring functions (RbtSolvIdxSF / RbtInterIdxSF).
set -euo pipefail

PRM=$(realpath "$1")
: "${RBT_ROOT:?RBT_ROOT must be exported}"

rbcavity -r "$PRM" -was
echo "[done] cavity + grids for $PRM"
