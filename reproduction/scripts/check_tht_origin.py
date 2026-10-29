#!/usr/bin/env python3
"""Locate, document and validate the KRAS ThT Origin export.

The authoritative experimental data are the two Origin projects listed in
`reproduction/data/tht/`. Origin's `.opj`/`.opju` formats are proprietary binary,
so the plotting code consumes a per-sheet CSV export produced from them.

This script does not invent data. It reports exactly which export files are
present, which are missing, and whether their schema matches what
`analysis/fig4g_tht_panel.py` reads:

    {PREF}_ex{N}_sheet2.csv     # one file per experiment sheet
    skiprows=1, header=0
    column index 4 = S1/R1, at least 3 numeric rows

Experiments: PREF "THT" has ex0..ex5, PREF "10X" has ex0..ex4.

If Origin is installed on this host (Windows COM automation is available), pass
--extract and the script will attempt the export; otherwise it prints the manual
procedure and validates whatever export exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
THT_DIR = REPO / 'reproduction' / 'data' / 'tht'
EXPORT = THT_DIR / 'export'

ORIGIN_PROJECTS = ['KRAS-THT-Compatative.opj', 'THT-Compatative.opju']
EXPERIMENTS = {'THT': range(6), '10X': range(5)}
S1R1_COLUMN = 4
MIN_ROWS = 3


def expected() -> list[Path]:
    return [EXPORT / f'{pref}_ex{ex}_sheet2.csv'
            for pref, exs in EXPERIMENTS.items() for ex in exs]


def check(path: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path, skiprows=1, header=0)
    except Exception as exc:
        return False, f'unreadable: {exc}'
    if df.shape[1] <= S1R1_COLUMN:
        return False, f'only {df.shape[1]} columns; need index {S1R1_COLUMN} for S1/R1'
    vals = pd.to_numeric(df.iloc[:MIN_ROWS, S1R1_COLUMN], errors='coerce').dropna()
    if len(vals) < MIN_ROWS:
        return False, f'column {S1R1_COLUMN} has {len(vals)} numeric rows; need {MIN_ROWS}'
    return True, f'{len(vals)} wells, S1/R1 mean {vals.mean():.4g}'


def try_extract() -> bool:
    """Attempt Origin COM automation. Returns True if an export was produced."""
    try:
        import win32com.client  # type: ignore
    except Exception:
        print('Origin COM automation unavailable (win32com not importable).')
        return False
    projects = [p for p in (THT_DIR / n for n in ORIGIN_PROJECTS) if p.is_file()]
    if not projects:
        print('No Origin project found in', THT_DIR)
        return False
    EXPORT.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch('Origin.ApplicationSI')
    try:
        for pref, exs in EXPERIMENTS.items():
            for ex in exs:
                app.Execute(f'expASC.path$="{EXPORT}";')  # placeholder: project-specific
                print('NOTE: automated export needs the project sheet mapping; '
                      'run the manual procedure in README.md instead.')
                return False
    finally:
        app.Exit()
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extract', action='store_true', help='Attempt Origin automation')
    args = ap.parse_args()

    print(f'Origin project directory: {THT_DIR.relative_to(REPO)}')
    for name in ORIGIN_PROJECTS:
        p = THT_DIR / name
        state = f'{p.stat().st_size:,} bytes' if p.is_file() else 'MISSING'
        print(f'  {"OK  " if p.is_file() else "MISS"} {name:28s} {state}')

    if args.extract:
        try_extract()

    print(f'\nPer-sheet export directory: {EXPORT.relative_to(REPO)}')
    if not EXPORT.is_dir():
        print('  directory absent')
        files = expected()
        print(f'  {len(files)} file(s) required, e.g. {files[0].name}, {files[1].name}, ...')
        print('\nRESULT: Origin data not exported. See reproduction/data/tht/README.md.')
        return 1

    ok, bad = [], []
    for path in expected():
        if not path.is_file():
            bad.append((path, 'missing'))
            continue
        good, msg = check(path)
        (ok if good else bad).append((path, msg))

    for path, msg in ok:
        print(f'  OK   {path.name:26s} {msg}')
    for path, msg in bad:
        print(f'  FAIL {path.name:26s} {msg}')

    print(f'\n{len(ok)}/{len(ok) + len(bad)} export file(s) valid')
    if bad:
        print('RESULT: ThT export incomplete or schema-mismatched.')
        return 1
    print('RESULT: ThT export matches the schema the panel script reads.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
