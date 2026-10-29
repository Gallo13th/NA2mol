#!/usr/bin/env bash
# prepare_receptor.sh <pocket.pdb> <out_dir> <ref_mol.sdf> [RADIUS] [RECEPTOR_FLEX] [NAME]
#
# rDock receptor preparation (User Guide compliant):
#   1. strip ALL hydrogens from the pocket PDB (avoids partial-H artifacts),
#      then re-add the complete set with obabel -h  -> all-atom MOL2
#      (polar H explicit; rDock deletes non-polar H itself)
#   2. fix_mol2.py: remove wrongly protonated phosphate H, rename atom
#      names (O5'->O5p apostrophe segfault guard, OP1/OP2->O1P/O2P) and
#      substructure names to the RbtIonicAtoms.prm convention
#      (DG4->G, ALA17->ALA, ...)
#   3. write the system .prm: RECEPTOR_FILE / RECEPTOR_FLEX are TOP-LEVEL
#      keys (never inside a SECTION); cavity via RbtLigandSiteMapper with
#      REF_MOL (co-crystal ligand SD, same coordinate frame) + RADIUS
#
# Outputs (in <out_dir>): <name>_receptor.mol2, <name>_system.prm
set -euo pipefail

POCKET=$(realpath "$1")
OUTDIR=$(realpath "$2")
REFMOL=$(realpath "$3")
RADIUS=${4:-10.0}
FLEX=${5:-3.0}
NAME=${6:-$(basename "$POCKET" .pdb | tr '[:upper:]' '[:lower:]')}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

command -v obabel >/dev/null || { echo "[FATAL] obabel not in PATH"; exit 1; }
: "${RBT_ROOT:?RBT_ROOT must be exported (conda env share/rdock-2013.1-1)}"

mkdir -p "$OUTDIR"
HEAVY="$OUTDIR/${NAME}_heavy.pdb"
RAW="$OUTDIR/${NAME}_raw.mol2"
MOL2="$OUTDIR/${NAME}_receptor.mol2"
PRM="$OUTDIR/${NAME}_system.prm"

# 1. strip H, re-add full set, convert to Tripos mol2
obabel "$POCKET" -O "$HEAVY" -d 2>/dev/null >/dev/null
obabel "$HEAVY" -O "$RAW" -h 2>/dev/null >/dev/null

# 2. rDock-oriented fixes (phosphate H, atom names, substructure names)
python "$SCRIPT_DIR/fix_mol2.py" "$RAW" "$MOL2"
rm -f "$RAW"

# 3. sanity report
echo "[receptor] substructure residues:"
awk '/@<TRIPOS>ATOM/{f=1;next}/@/{f=0}f&&NF{print $8}' "$MOL2" | sort | uniq -c | sed 's/^/   /'
echo "[receptor] mol2 atoms: $(awk '/@<TRIPOS>ATOM/{f=1;next}/@/{f=0}f&&NF' "$MOL2" | wc -l)"

# 4. system prm (all paths absolute; SECTION MAPPER syntax per User Guide 8.8)
cat > "$PRM" <<EOF
RBT_PARAMETER_FILE_V1.00
RECEPTOR_FILE $MOL2
RECEPTOR_FLEX $FLEX

SECTION MAPPER
    SITE_MAPPER RbtLigandSiteMapper
    REF_MOL $REFMOL
    RADIUS $RADIUS
    SMALL_SPHERE 1.0
    MIN_VOLUME 100
    MAX_CAVITIES 1
    VOL_INCR 0.0
    GRIDSTEP 0.5
END_SECTION
EOF

echo "[done] $MOL2"
echo "[done] $PRM  (REF_MOL=$REFMOL RADIUS=$RADIUS FLEX=$FLEX)"
