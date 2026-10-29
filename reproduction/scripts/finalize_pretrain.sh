#!/usr/bin/env bash
# Verify and unpack the pretraining archive.
#
# The archive lives on the authors' AutoDL storage and is far too large for git.
# It is fetched out-of-band into data/pretrain.tar.gz; this script only touches
# the local copy.
#
#   bash reproduction/scripts/finalize_pretrain.sh [--skip-verify]
#
# --skip-verify skips the full gzip pass (which reads all 5.5 GB). Use it only if
# integrity has already been confirmed; the size check always runs.
#
# Run long jobs under tmux so an SSH drop cannot kill the unpack:
#   tmux new-session -d -s pretrain 'bash reproduction/scripts/finalize_pretrain.sh > /tmp/finalize.log 2>&1'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

ARCHIVE="data/pretrain.tar.gz"
TARGET="data/pretrain"
EXPECTED_BYTES=5916109016
SKIP_VERIFY=0
[ "${1:-}" = "--skip-verify" ] && SKIP_VERIFY=1

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
    echo "RESULT: transfer incomplete — size mismatch. Re-run the transfer."
    exit 1
fi
echo "RESULT: size matches."

# ── 2. integrity ────────────────────────────────────────────────────────────
if [ "$SKIP_VERIFY" -eq 0 ]; then
    echo "verifying archive integrity (reads the whole file; takes ~20 min)..."
    if ! gzip -t "$ARCHIVE"; then
        echo "RESULT: gzip integrity check FAILED — the archive is corrupt."
        exit 1
    fi
    echo "RESULT: gzip integrity OK."
else
    echo "RESULT: integrity check skipped (--skip-verify)."
fi

# ── 3. listing is deliberately omitted ──────────────────────────────────────
# `tar tzf` must read every member's data to reach the next header, so even
# `| head -10` decompresses the whole archive when the first member is a
# multi-gigabyte LMDB. The post-extraction report below already shows the layout.
echo
echo "skipping the entry listing (it would decompress the archive); extracting next."

# ── 4. extract ──────────────────────────────────────────────────────────────
if [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
    echo
    echo "$TARGET already populated — skipping extraction."
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

# ── 5. flatten the AutoDL-internal prefix ───────────────────────────────────
# The archive stores the LMDB under autodl-tmp/merged/. Promote it so the
# repository layout is data/pretrain/merged/ rather than a nested autodl-tmp/.
if [ -d "$TARGET/autodl-tmp/merged" ] && [ ! -d "$TARGET/merged" ]; then
    mv "$TARGET/autodl-tmp/merged" "$TARGET/merged"
    rmdir "$TARGET/autodl-tmp" 2>/dev/null
    echo "flattened autodl-tmp/merged -> $TARGET/merged"
fi

# ── 6. report ───────────────────────────────────────────────────────────────
echo
echo "contents:"
du -sh "$TARGET"/* 2>/dev/null | sort -hr | head -15
echo
echo "RESULT: pretraining data ready under $TARGET."
echo "Next: python3 reproduction/scripts/manifest_data_files.py"
