#!/usr/bin/env bash
# Verify and unpack the pretraining archive once the transfer has finished.
#
# The archive is fetched out-of-band (it lives on the authors' AutoDL storage and
# is far too large for git). This script only touches the local copy.
#
#   bash reproduction/scripts/finalize_pretrain.sh
#
# It is safe to re-run: extraction is skipped if the target already looks complete.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

ARCHIVE="data/pretrain.tar.gz"
TARGET="data/pretrain"
EXPECTED_BYTES=5916109016

# ── 1. presence and size ────────────────────────────────────────────────────
if [ ! -f "$ARCHIVE" ]; then
    echo "MISSING $ARCHIVE"
    echo "Fetch it from the authors' AutoDL storage before running this script."
    exit 1
fi

actual=$(stat -c%s "$ARCHIVE")
printf 'archive: %s\n' "$ARCHIVE"
printf 'size:    %s bytes (expected %s)\n' "$actual" "$EXPECTED_BYTES"
if [ "$actual" -ne "$EXPECTED_BYTES" ]; then
    echo "RESULT: transfer incomplete — size mismatch. Re-run the transfer (scp will resume)."
    exit 1
fi
echo "RESULT: size matches."

# ── 2. integrity: a full decompression pass, no extraction ──────────────────
echo "verifying archive integrity (this reads the whole file)..."
if ! gzip -t "$ARCHIVE"; then
    echo "RESULT: gzip integrity check FAILED — the archive is corrupt."
    exit 1
fi
echo "RESULT: gzip integrity OK."

# ── 3. what is inside ───────────────────────────────────────────────────────
echo
echo "top-level entries:"
tar tzf "$ARCHIVE" 2>/dev/null | awk -F/ '{print $1}' | sort -u | head -20

# ── 4. extract ──────────────────────────────────────────────────────────────
if [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
    echo
    echo "$TARGET already populated — skipping extraction. Remove it to force a fresh unpack."
else
    echo
    echo "extracting to $TARGET ..."
    mkdir -p "$TARGET"
    if tar xzf "$ARCHIVE" -C "$TARGET"; then
        echo "extraction OK"
    else
        echo "RESULT: extraction FAILED"
        exit 1
    fi
fi

# ── 5. report ───────────────────────────────────────────────────────────────
echo
echo "contents of $TARGET:"
du -sh "$TARGET"/* 2>/dev/null | sort -hr | head -15
echo
echo "RESULT: pretraining data ready under $TARGET."
echo "Record it in reproduction/manifests/data_files.json, then regenerate hashes:"
echo "  python3 reproduction/scripts/manifest_data_files.py"
