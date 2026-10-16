#!/usr/bin/env python3
"""Build the bond-JS caches that `export_tables.py` consumes for the radar tables.

`analysis/prepare_bond_js.py` and `analysis/prepare_bond_js_dude.py` are the legacy
per-method visualisation scripts, but they also contain the only implementation of
the bond-length/angle Jensen-Shannon metrics. `export_tables.py` reads those metrics
from `cache_bond_js.pkl` / `cache_dude_bond_js.pkl` to build the radar profile
tables (t7a-t7c, t8, d8a-d8c) used by Main Fig. 2.

Without the caches, `export_tables.py` silently degrades: `d8b`/`d8c` are not written
at all, so `analysis/fig2_cross_benchmark.py` fails with FileNotFoundError.

This driver imports the modules and calls only the cache function, skipping their
(superseded) figure rendering.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
OUT = ROOT / 'data' / 'benchmark' / 'output'


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick(mod, *names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f'none of {names} found in {mod.__name__}')


def build(script: str, cache_name: str) -> int:
    path = ROOT / 'analysis' / script
    print(f'\n=== {script} -> {cache_name} ===')
    mod = load_module(path)
    df = pick(mod, 'df', 'DF')
    methods = list(pick(mod, 'MAIN_METHODS')) + list(pick(mod, 'ABLATION_METHODS'))
    fn = pick(mod, '_compute_bond_js_for_methods')
    cache = OUT / cache_name
    print(f'  methods: {len(methods)}')
    results = fn(df, methods, cache)
    got = sum(1 for v in results.values() if not (v.get('bond_len_js') != v.get('bond_len_js')))
    print(f'  wrote {cache.relative_to(ROOT)}  ({got}/{len(results)} methods with finite JS)')
    return 0


def main() -> int:
    rc = build('prepare_bond_js.py', 'cache_bond_js.pkl')
    rc |= build('prepare_bond_js_dude.py', 'cache_dude_bond_js.pkl')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
