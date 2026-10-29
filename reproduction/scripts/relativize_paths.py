#!/usr/bin/env python3
"""Rewrite machine-specific paths in analysis scripts to repository-relative paths.

Handles the patterns found in the checkout:
  github_official/DiffEDMol/<x>   -> ROOT / "<x>"
  github_official/RNALigBench/<x> -> ROOT / "data" / "benchmark" / "RNALigBench" / "<x>"
  /tmp/opencode/<x>                         -> ROOT / "reproduction" / "cache" / "<x>"
  ROOT / "RNA2smol_paper" / "figures"       -> ROOT / "reproduction" / "figures"

A `ROOT = Path(__file__).resolve().parents[N]` definition is inserted when the
file does not already define ROOT. `from pathlib import Path` is added if absent.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DiffEDMol = 'github_official/DiffEDMol'
RNALigBench = 'github_official/RNALigBench'
TmpOpen = '/tmp/opencode'

TARGETS = [
    *sorted((REPO / 'analysis').glob('*.py')),
    *sorted((REPO / 'audit/scripts').glob('*.py')),
    *sorted((REPO / 'g4tasks').rglob('*.py')),
]


def root_expr(rel: Path) -> str:
    depth = len(rel.parts) - 1
    return f'Path(__file__).resolve().parents[{depth}]'


def patch(path: Path, rel: Path) -> list[str]:
    src = original = path.read_text()
    notes: list[str] = []

    # 1. Figure output directories: the paper package is not in this repository.
    new = src.replace('ROOT / "RNA2smol_paper" / "figures"', 'ROOT / "reproduction" / "figures"')
    new = new.replace("ROOT / 'RNA2smol_paper'", "ROOT / 'reproduction'")
    if new != src:
        notes.append('figure output dir -> reproduction/figures')
    src = new

    # 2. Absolute repository data paths.
    if DiffEDMol in src or RNALigBench in src or TmpOpen in src:
        needs_root = not re.search(r'^ROOT\s*=', src, re.M)
        if needs_root:
            if 'from pathlib import Path' not in src:
                src = src.replace('\nimport ', '\nfrom pathlib import Path\nimport ', 1)
                notes.append('added pathlib import')
            # insert ROOT after the final top-level import line
            lines = src.splitlines()
            last = max((i for i, l in enumerate(lines)
                        if re.match(r'^(import |from )\S', l)), default=0)
            lines.insert(last + 1, f'ROOT = {root_expr(rel)}')
            src = '\n'.join(lines) + '\n'
            notes.append(f'inserted ROOT = parents[{len(rel.parts) - 1}]')

        src = src.replace(f"Path('{DiffEDMol}", f"(ROOT / '")
        src = re.sub(rf"Path\(\"{re.escape(DiffEDMol)}/([^\"]*)\"\)", r'(ROOT / "\1")', src)
        src = re.sub(rf"'{re.escape(DiffEDMol)}/([^']*)'", r"str(ROOT / '\1')", src)
        src = re.sub(rf'"{re.escape(DiffEDMol)}/([^"]*)"', r'str(ROOT / "\1")', src)
        src = src.replace(f"'{DiffEDMol}'", 'str(ROOT)')
        src = re.sub(rf"'{re.escape(RNALigBench)}/([^']*)'",
                     r"str(ROOT / 'data/benchmark/RNALigBench/\1')", src)
        src = re.sub(rf'"{re.escape(RNALigBench)}/([^"]*)"',
                     r'str(ROOT / "data/benchmark/RNALigBench/\1")', src)
        src = re.sub(rf"'{re.escape(TmpOpen)}/([^']*)'",
                     r"str(ROOT / 'reproduction/cache/\1')", src)
        src = re.sub(rf'Path\(\"{re.escape(TmpOpen)}/([^\"]*)\"\)',
                     r'(ROOT / "reproduction/cache/\1")', src)
        notes.append('absolute paths -> repository-relative')

    if src != original:
        path.write_text(src)
    return notes


def main() -> int:
    changed = 0
    for path in TARGETS:
        rel = path.relative_to(REPO)
        notes = patch(path, rel)
        if notes:
            changed += 1
            print(f'{rel}: {"; ".join(notes)}')
    print(f'\n{changed} file(s) rewritten')
    return 0


if __name__ == '__main__':
    sys.exit(main())
