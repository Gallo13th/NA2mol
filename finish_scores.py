"""Finish the radar score tables: raw ECFP4, target specificity, and the new baselines.

Three changes to data/benchmark/output/tables/radar/radar_scores_{bench}.csv:

1. ``ecfp4`` becomes the raw recovery area instead of the null-normalised score. Recovery is
   already the area under the per-target retention curve and lies in [0, 1], so it needs no
   cross-method rescaling; rescaling it against each method's own null was what mixed
   method-specific information into a figure meant to compare methods.

2. ``ecfp4_specificity`` = observed - permutation null, with ``s_ecfp4_specificity`` the same
   value mapped onto [0, 1] over radar.ECFP4_SPEC_RANGE for drawing.

3. The three new SBDD baselines get full rows: property axes scored from their exported
   molecules, ECFP4 axes from their own permutation run (radar_export/ecfp4_null.csv).

Run after recomputing any of the inputs, then analysis/export_radar_table.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/data/users/gaoletian/rna2smol')
SB = Path('/home/data/users/gaoletian/sbdd_baselines')
sys.path.insert(0, str(ROOT / 'analysis'))
import radar  # noqa: E402

TABLES = ROOT / 'data' / 'benchmark' / 'output' / 'tables' / 'radar'
NULLS = pd.concat([pd.read_csv('/tmp/ecfp4_null_diffsbdd.csv'),
                   pd.read_csv('/tmp/ecfp4_null_gf.csv')], ignore_index=True)
MOLECULES = SB / 'radar_export' / 'new_baselines_molecules.csv'

BENCH = {'dude': 'dude', 'smrtbench': 'smrtbench'}
NEW = ('DiffSBDD', 'GraphBP', 'FLAG')


def property_axes(which: str) -> dict[str, dict]:
    df = pd.read_csv(MOLECULES, low_memory=False)
    df = df[df['bench'] == which]
    out = {}
    for mode, sub in df.groupby('mode'):
        sub = sub.copy()
        for c in ('affinity', 'rmsd', 'qed', 'sa_score', 'tpsa', 'strain'):
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
        out[mode] = radar.property_scores(sub)
        out[mode]['n_molecules'] = len(sub)
        out[mode]['n_targets'] = sub['pdb_id'].nunique()
    return out


def main() -> int:
    for which in BENCH:
        path = TABLES / f'radar_scores_{which}.csv'
        base = pd.read_csv(path)

        # 1 and 2: existing rows
        base['ecfp4'] = base['recovery_observed'].clip(0.0, 1.0)
        base['ecfp4_specificity'] = base['recovery_observed'] - base['recovery_random']
        base['s_ecfp4_specificity'] = base['ecfp4_specificity'].map(
            radar.ecfp4_specificity_score)

        # 3: the new baselines
        axes = property_axes(which)
        nulls = NULLS[NULLS['bench'] == which].set_index('method')
        rows = []
        for mode in NEW:
            if mode not in axes or mode not in nulls.index:
                continue
            r = dict(method=mode, benchmark=which)
            r.update(axes[mode])
            obs = float(nulls.loc[mode, 'recovery_observed'])
            rnd = float(nulls.loc[mode, 'recovery_random'])
            r['recovery_observed'] = obs
            r['recovery_random'] = rnd
            r['recovery_random_sd'] = float(nulls.loc[mode, 'recovery_random_sd'])
            r['ecfp4'] = np.clip(obs, 0.0, 1.0)
            r['ecfp4_specificity'] = obs - rnd
            r['s_ecfp4_specificity'] = radar.ecfp4_specificity_score(obs - rnd)
            r['n_targets_universe'] = int(nulls.loc[mode, 'n_targets'])
            rows.append(r)

        out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
        # drop rows whose ECFP4 axes predate this change, then rewrite
        out.to_csv(path, index=False)
        print(f'{which}: {len(out)} rows, methods: {sorted(out["method"].unique())}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
