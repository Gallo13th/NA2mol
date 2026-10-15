#!/usr/bin/env python3
"""Verify figure-script data paths: inputs must exist, outputs are created at runtime.

Parses `NAME = ROOT / "a" / "b"` assignments and classifies each as INPUT or
OUTPUT. A declaration is an output when its name ends in OUT/OUT_DIR or contains
PAPER/FIG/OUTPUT; everything else is an input that must already exist.
Exits non-zero if any INPUT is missing.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / 'analysis').glob('*.py')) + sorted((REPO / 'g4tasks').rglob('*.py'))

ASSIGN = re.compile(r'^([A-Z][A-Z0-9_]*)\s*=\s*(?:\(\s*)?ROOT\s*((?:/\s*"[^"]*"\s*)+)', re.M)
PART = re.compile(r'/\s*"([^"]*)"')
OUTPUT_HINT = re.compile(r'(OUT|OUTPUT|PAPER|FIG)')


def declarations(text):
    for m in ASSIGN.finditer(text):
        parts = PART.findall(m.group(2))
        if parts:
            yield m.group(1), parts


def main():
    missing_inputs = []
    for script in SCRIPTS:
        rel = script.relative_to(REPO)
        rows = list(declarations(script.read_text()))
        if not rows:
            continue
        print(f'\n{rel}')
        for name, parts in rows:
            target = REPO.joinpath(*parts)
            is_output = bool(OUTPUT_HINT.search(name))
            exists = target.exists()
            if exists:
                state = 'OK  '
            elif is_output:
                state = 'NEW '
            else:
                state = 'MISS'
                missing_inputs.append((str(rel), name, '/'.join(parts)))
            kind = 'dir' if target.is_dir() else ('file' if target.is_file() else '-')
            label = 'output' if is_output else 'input '
            print(f'  {state} {label} {name:24s} {".".join(parts)}')
    print('\n' + '=' * 68)
    if missing_inputs:
        print(f'{len(missing_inputs)} missing INPUT(S):')
        for script, name, path in missing_inputs:
            print(f'  {path}\n      declared in {script} as {name}')
    else:
        print('All declared inputs exist.')
    print('(NEW = output directory created at runtime)')
    return 1 if missing_inputs else 0


if __name__ == '__main__':
    sys.exit(main())
