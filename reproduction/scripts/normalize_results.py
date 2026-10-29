#!/usr/bin/env python3
"""Add canonical manuscript labels to staged result tables without discarding source values.

Mapping is lossless: original columns are preserved and canonical names are added
as new columns. External baselines keep their published names; only case variants
and abbreviation variants are unified.
"""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'reproduction/results'

BENCHMARK = {'SMRT-Net': 'RNA test set', 'SMRTnet': 'RNA test set',
             'SMRTNet': 'RNA test set', 'DUD-E': 'DUD-E'}
MODEL = {
    'RNA2smol': 'NA2smol',
    'RNA2smol (Only General Feat)': 'NA2smol (full)',
    'RNA2smol (Baseline)': 'NA2smol (baseline)',
    'ablation:RNA2smol (Baseline)': 'NA2smol (baseline)',
    'ablation:RNA2smol (Only General Feat)': 'NA2smol (full)',
    'ablation:Pretrain Only': 'NA2smol (Pretrain Only)',
    'ablation:Pretrain+Constrain': 'NA2smol (Pretrain+Constrain)',
    'MolCRAFT': 'MolCRAFT',
    'MolCRAFT (Original)': 'MolCRAFT (original residue mapping)',
    'Pocket2Mol': 'Pocket2Mol',
    'ED2Mol': 'ED2Mol',
    'ed2mol': 'ED2Mol',
}
SETTING = {'ft_exact': 'fine-tuning exact match', 'ft_T0.5': 'fine-tuning Tanimoto >= 0.5',
           'ft_T0.6': 'fine-tuning Tanimoto >= 0.6', 'ft_T0.7': 'fine-tuning Tanimoto >= 0.7',
           'pretrain_exact': 'pretraining exact match',
           'pretrain_T0.5': 'pretraining Tanimoto >= 0.5'}

TARGETS = {
    'leakage/recovery_manuscript_authoritative.csv': {
        'benchmark': BENCHMARK, 'method': MODEL, 'filter': SETTING},
    'benchmark_tables/dude/d2_ecfp4_target_success_by_threshold.csv': {'method': MODEL},
    'benchmark_tables/dude/d7_virtual_screening_funnel.csv': {'method': MODEL},
}

def main():
    applied, unresolved = [], []
    for rel, spec in TARGETS.items():
        path = RESULTS / rel
        if not path.is_file():
            print('SKIP (absent):', rel)
            continue
        df = pd.read_csv(path)
        for column, mapping in spec.items():
            if column not in df.columns:
                continue
            canonical = f'{column}_canonical'
            df[canonical] = df[column].map(lambda v: mapping.get(str(v), v))
            unmapped = sorted(set(df[column].astype(str)) - set(mapping))
            if unmapped:
                unresolved.append({'file': rel, 'column': column, 'values': unmapped})
            applied.append({'file': rel, 'column': column, 'canonical_column': canonical,
                            'distinct_values': len(set(df[column].astype(str)))})
            print(f'{rel}: {column} -> {canonical}; unmapped={unmapped or "none"}')
        df.to_csv(path, index=False)
    out = ROOT / 'reproduction/manifests/naming_normalization.json'
    out.write_text(json.dumps({'benchmark_map': BENCHMARK, 'model_map': MODEL,
                               'setting_map': SETTING, 'applied': applied,
                               'unresolved_values': unresolved}, indent=2) + '\n')
    print(f'\n{len(applied)} columns normalized; originals preserved.')
    if unresolved:
        print('NOTE: unresolved values are retained verbatim and listed in the manifest.')

if __name__ == '__main__':
    main()
