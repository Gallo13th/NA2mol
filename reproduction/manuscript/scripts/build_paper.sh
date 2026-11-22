#!/bin/bash
# Backwards-compatible entry point: build the whole submission file set.
#
# The manuscript, Extended Data and Supplementary Information are now separate PDFs, so this
# wrapper just calls scripts/build_submission_files.sh. Use that script directly if you want the
# individual build steps.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/build_submission_files.sh" "$@"
