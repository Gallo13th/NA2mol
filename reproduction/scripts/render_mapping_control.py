#!/usr/bin/env python3
"""Render Supplementary Fig. 2 (MolCRAFT random-mapping control) from staged tables.

Matplotlib's font-cache build is very slow on the shared /home mount, so this
entry point pins the cache to local fast storage and limits BLAS threads.
"""
import os

# Must precede any matplotlib import inside the wrapped module.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('NA2SMOL_MPLCONFIGDIR', '/tmp/na2smol_mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true', help='Skip rendering')
    args = parser.parse_args()

    subprocess.run([sys.executable, str(ROOT / 'reproduction/scripts/export_dude_tables.py')],
                   check=True, cwd=ROOT)

    if args.tables_only:
        return

    sys.path.insert(0, str(ROOT / 'data' / 'benchmark'))
    script = ROOT / 'analysis' / 'sfig2_mapping_control.py'
    spec = importlib.util.spec_from_file_location('mapping_control', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.TABLE_DIR = ROOT / 'reproduction/results/benchmark_tables/dude'
    module.RAW_DATA = ROOT / 'data/benchmark/dude_stat_all.csv'
    module.OUT_DIR = ROOT / 'reproduction/results/molcraft_mapping_control'
    module.PAPER_DIR = ROOT / 'reproduction/figures'

    original_save_all = module.save_all

    def save_all(fig, stem, *a, **kw):
        return original_save_all(fig, stem, include_tiff=False, fast=True)

    module.save_all = save_all
    module.main()


if __name__ == '__main__':
    main()
