#!/bin/bash
# Refresh the standalone manuscript repository from overleaf/ and push it to GitHub.
#
# The Overleaf project is linked to a dedicated GitHub repository (not to the full analysis repository), because
# Overleaf's GitHub sync maps the repository root to the project root. This script keeps that mirror in step:
#   1. rsync overleaf/ (minus the zip) into overleaf-git/
#   2. commit any change with a message you pass as the first argument
#   3. push to origin
#
# Usage: scripts/sync_overleaf_repo.sh "manuscript: ..."
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/overleaf"
DST="$ROOT/overleaf-git"
MSG="${1:-manuscript: sync from the working copy}"

[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
[ -d "$DST/.git" ] || { echo "$DST is not a git repository" >&2; exit 1; }

rsync -a --delete --exclude 'RNA2smol_overleaf.zip' --exclude '.git' "$SRC"/ "$DST"/

cd "$DST"
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "no changes to commit"
else
  git add -A
  git commit -q -m "$MSG"
  echo "committed: $MSG"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -q origin main && echo "pushed to $(git remote get-url origin)"
else
  echo "no 'origin' remote yet; add it with:" >&2
  echo "  git -C $DST remote add origin git@github.com:Gallo13th/<repo>.git" >&2
  echo "  git -C $DST push -u origin main" >&2
fi
