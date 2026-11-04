"""Recovery after removing targets whose reference actives are chemically close to training data.

The published sensitivity analysis covers only the four original generators. The three SBDD
baselines added to the benchmark have per-target maximum ECFP4-Tanimoto values but no filtered
recovery, so this extends the same analysis to them on exactly the same target exclusions.

A target is excluded when any of its reference actives reaches the similarity threshold against
the training source. Recovery per target is clip((max_sim - 0.25) / 0.25, 0, 1), the area under
the per-target retention curve over tau in [0.25, 0.50], and the reported value is its mean over
the surviving targets.

The exclusion logic is calibrated first: on the two benchmarks it must reproduce the eligible-target
counts used in the published table (RNA 18 and 14; DUD-E 84 and 81).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/data/users/gaoletian/rna2smol')
SB = Path('/home/data/users/gaoletian/sbdd_baselines')
AUDIT = ROOT / 'audit' / 'tables'

# benchmark label in each source -> the label used in the audit tables
BENCH = {'dude': 'DUD-E', 'smrtbench': 'SMRT-Net'}
RECOVERY = {'DiffSBDD': 'diffsbdd', 'GraphBP': 'graphbp_a1500', 'FLAG': 'flag'}
KNOWN_COUNTS = {('SMRT-Net', 'ft'): 18, ('SMRT-Net', 'pretrain'): 14,
                ('DUD-E', 'ft'): 84, ('DUD-E', 'pretrain'): 81}


def excluded_targets() -> dict[tuple[str, str], set[str]]:
    """Targets to drop, keyed by (benchmark, source)."""
    ref = pd.read_csv(AUDIT / 'reference_active_overlap.csv', low_memory=False)
    ref['target_id'] = ref['target_id'].astype(str)
    out = {}
    for bench in ('SMRT-Net', 'DUD-E'):
        sub = ref[ref['benchmark'] == bench]
        for src, col in (('ft', 'maxts_ft_train'), ('pretrain', 'maxts_pretrain')):
            hit = sub[pd.to_numeric(sub[col], errors='coerce') >= 0.5]
            out[(bench, src)] = set(hit['target_id'].unique())
    return out


def per_target(bench_key: str, stem: str) -> pd.DataFrame:
    d = pd.read_csv(SB / 'smoke_out' / f'ecfp4_recovery_{stem}.csv')
    d = d[d['bench'] == bench_key].copy()
    d['target_id'] = d['target'].astype(str).str.split('_').str[0]
    d['max_sim'] = pd.to_numeric(d['max_sim'], errors='coerce')
    d['recovery'] = np.clip((d['max_sim'] - 0.25) / 0.25, 0.0, 1.0)
    return d.groupby('target_id', as_index=False)['recovery'].max()


def main() -> int:
    excl = excluded_targets()
    universe = {}
    ref = pd.read_csv(AUDIT / 'reference_active_overlap.csv', low_memory=False)
    ref['target_id'] = ref['target_id'].astype(str)
    for bench in ('SMRT-Net', 'DUD-E'):
        universe[bench] = set(ref[ref['benchmark'] == bench]['target_id'].unique())

    # calibrate against the published eligible-target counts before trusting the extension
    for (bench, src), want in KNOWN_COUNTS.items():
        got = len(universe[bench] - excl[(bench, src)])
        status = 'ok' if got == want else 'MISMATCH'
        print(f'  {bench:9s} {src:9s} eligible {got:3d} (published {want:3d}) {status}')
        if got != want:
            raise SystemExit('exclusion logic does not reproduce the published counts')

    rows = []
    for method, stem in RECOVERY.items():
        for bench in ('SMRT-Net', 'DUD-E'):
            bench_key = next(k for k, v in BENCH.items() if v == bench)
            pt = per_target(bench_key, stem).set_index('target_id')['recovery']
            pt = pt.reindex(sorted(universe[bench])).fillna(0.0)
            for src in ('all', 'ft', 'pretrain'):
                keep = list(universe[bench] - excl[(bench, src)]) if src != 'all' else list(universe[bench])
                vals = [pt[t] for t in keep if t in pt.index]
                rows.append(dict(method=method, benchmark=bench, source=src,
                                 n_eligible=len(vals),
                                 recovery=float(np.mean(vals)) if vals else float('nan')))

    out = pd.DataFrame(rows)
    dest = AUDIT / 'recovery_filtered_new_baselines.csv'
    out.to_csv(dest, index=False)
    print(f'\nwrote {dest}')
    print(out.pivot_table(index='method', columns=['benchmark', 'source'],
                          values='recovery').round(3).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
