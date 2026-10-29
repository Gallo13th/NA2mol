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
(superseded) figure rendering. Slow — RDKit 3D embedding, roughly 5-10 min per cache.

    python3 reproduction/scripts/build_bond_js_caches.py [--only smrt|dude|both]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
OUT = ROOT / 'data' / 'benchmark' / 'output'

# The two scripts were written independently and name the same things differently.
SPECS = {
    'smrt': ('prepare_bond_js.py', 'cache_bond_js.pkl',
             ('_compute_bond_js_for_methods', '_compute_bond_js'),
             (('MAIN_METHODS', 'ABLATION_METHODS'), ('METHODS',))),
    'dude': ('prepare_bond_js_dude.py', 'cache_dude_bond_js.pkl',
             ('_compute_bond_js', '_compute_bond_js_for_methods'),
             (('METHODS',), ('MAIN_METHODS', 'ABLATION_METHODS'))),
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f'none of {names} found in {mod.__name__}')


def method_list(mod, alternatives):
    for combo in alternatives:
        if all(hasattr(mod, n) for n in combo):
            out = []
            for n in combo:
                out.extend(list(getattr(mod, n)))
            return out
    raise AttributeError(f'no method list found in {mod.__name__}')


def build(key: str) -> int:
    script, cache_name, fn_names, method_alts = SPECS[key]
    print(f'\n=== {script} -> {cache_name} ===', flush=True)
    mod = load_module(ROOT / 'analysis' / script)
    df = first_attr(mod, ('df', 'DF'))
    methods = method_list(mod, method_alts)
    fn = first_attr(mod, fn_names)
    cache = OUT / cache_name
    print(f'  data rows: {len(df):,}; method groups: {len(methods)}', flush=True)
    results = fn(df, methods, cache)
    finite = sum(1 for v in results.values()
                 if v.get('bond_len_js') == v.get('bond_len_js'))
    rel = cache.relative_to(ROOT)
    print(f'  wrote {rel}  ({finite}/{len(results)} groups finite)', flush=True)
    if finite == 0:
        print('  FAIL: no finite values', flush=True)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', choices=['smrt', 'dude', 'both'], default='both')
    a = ap.parse_args()
    keys = ['smrt', 'dude'] if a.only == 'both' else [a.only]
    rc = 0
    for k in keys:
        rc |= build(k)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
