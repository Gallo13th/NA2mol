"""Recovery curves for all seven benchmarked methods on each reference-set subset.

The four original benchmark runs carry the per-molecule ECFP4-TS in the combined table; the three
later generators (DiffSBDD, GraphBP, FLAG) only ship SMILES, so their best similarity to each
target's reference actives is computed here and written next to the existing values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reproduction' / 'figures' / 'fig3_recovery_curves_by_def.csv'
COMBINED = ROOT / 'data/benchmark/output/smrtbench_combined_results.csv'
ACTIVES = ROOT / 'audit/tables/reference_active_overlap.csv'
NEWB = Path('/home/data/users/gaoletian/sbdd_baselines/radar_export/new_baselines_molecules.csv')
TAUS = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
MODES = {'RNA2smol (Only General Feat)': 'NA2smol', 'ED2Mol': 'ED2Mol',
         'MolCRAFT': 'MolCRAFT', 'Pocket2Mol': 'Pocket2Mol'}
METHODS = ['NA2smol', 'ED2Mol', 'MolCRAFT', 'Pocket2Mol', 'DiffSBDD', 'FLAG', 'GraphBP']


def fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def main() -> int:
    act = pd.read_csv(ACTIVES, low_memory=False)
    act = act[(act['benchmark'] == 'SMRT-Net') & (act['parse_ok'] == True)]  # noqa: E712
    targets = sorted(set(act['target_id']))
    act_fps = {t: [f for f in (fp(s) for s in g['canonical'].dropna()) if f is not None]
               for t, g in act.groupby('target_id')}
    print(f'{len(targets)} targets, actives per target '
          f'{min(len(v) for v in act_fps.values())}-{max(len(v) for v in act_fps.values())}')

    # per-target best ECFP4-TS per method
    best: dict[tuple[str, str], float] = {}
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df['mode'].isin(MODES)]
    for mode, sub in df.groupby('mode'):
        m = MODES[mode]
        for tid, g in sub.groupby('pdb_id'):
            v = pd.to_numeric(g['max_ecfp4_tanimoto'], errors='coerce').max()
            if np.isfinite(v):
                best[(m, tid)] = float(v)
    print(f'combined table: {len(best)} (method, target) pairs')

    extra = pd.read_csv(NEWB, low_memory=False)
    extra = extra[(extra['bench'] == 'smrtbench') & extra['mode'].isin(['DiffSBDD', 'FLAG', 'GraphBP'])]
    for mode, sub in extra.groupby('mode'):
        for tid, g in sub.groupby('pdb_id'):
            ref = act_fps.get(tid, [])
            if not ref:
                continue
            vals = []
            for smi in g['SMILES'].dropna().unique():
                f = fp(smi)
                if f is not None:
                    vals.append(max(DataStructs.BulkTanimotoSimilarity(f, ref)))
            if vals:
                best[(mode, tid)] = float(max(vals))
        print(f'{mode}: {sum(1 for (m, _) in best if m == mode)} targets scored')

    sets = {
        'All actives': targets,
        'FT-similar removed': sorted(set(act.loc[act['maxts_ft_train'] < 0.5, 'target_id'])),
        'PT-similar removed': sorted(set(act.loc[act['maxts_pretrain'] < 0.5, 'target_id'])),
    }
    rows = []
    for name, keep in sets.items():
        for m in METHODS:
            vals = np.array([best.get((m, t), np.nan) for t in keep], dtype=float)
            for tau in TAUS:
                rows.append({'method': m, 'definition': name, 'n_targets': len(keep),
                             'tau': tau, 'frac_targets': float(np.nanmean(vals >= tau))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(res[res['tau'] == 0.25].pivot_table(index='method', columns='definition',
                                              values='frac_targets').round(2).to_string())
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
