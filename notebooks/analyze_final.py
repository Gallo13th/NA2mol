#!/usr/bin/env python
"""Final property analysis: only molecules actually fixed (SMILES changed)."""
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = (ROOT / 'notebooks' / 'eval_fixed_rings')
datasets = {'smrtnet': EVAL_DIR/'smrtnet_onlygeneral_0504', 'dude': EVAL_DIR/'dude_onlygeneral_0506'}

all_rows = []
for ds_name, ds_dir in datasets.items():
    for csv_path in sorted(ds_dir.glob('*/*_comparison.csv')):
        df = pd.read_csv(csv_path)
        df['dataset'] = ds_name
        all_rows.append(df)

full = pd.concat(all_rows, ignore_index=True)

# Filter: keep only molecules with negative affinity both before AND after fix
full = full[(full['orig_affinity'] < 0) & (full['fixed_affinity'] < 0)].copy()
changed = full[full['smi_changed'] == True].copy()

changed['dqed'] = changed['fixed_qed'] - changed['orig_qed']
changed['dsa'] = changed['fixed_sa'] - changed['orig_sa']
changed['dlogp'] = changed['fixed_logp'] - changed['orig_logp']
changed['dmw'] = changed['fixed_mw'] - changed['orig_mw']
changed['drmsd'] = changed['fixed_rmsd'] - changed['orig_rmsd']

n_total = len(full)
n_changed = len(changed)

print("=" * 70)
print("PROPERTY CHANGES: ONLY MOLECULES WITH RINGS FIXED (SMILES changed)")
print("=" * 70)
print(f"  Total molecules: {n_total:,}")
print(f"  Actually fixed:  {n_changed:,} ({n_changed/n_total*100:.1f}%)")

for ds_name in ['smrtnet', 'dude']:
    d = changed[changed['dataset'] == ds_name]
    ds_total = len(full[full['dataset'] == ds_name])
    print(f"\n--- {ds_name}: {len(d):,} fixed / {ds_total:,} total ({len(d)/ds_total*100:.1f}%) ---")
    for label, col in [('dAff', 'delta_affinity'), ('dQED', 'dqed'), ('dSA', 'dsa'),
                        ('dlogP', 'dlogp'), ('dMW', 'dmw'), ('dRMSD', 'drmsd')]:
        v = d[col].dropna()
        print(f"  {label:8s}: mean={v.mean():+.4f}  median={v.median():+.4f}  std={v.std():.4f}  [{v.min():+.3f}, {v.max():+.3f}]")
    aff = d['delta_affinity'].dropna()
    qed = d['dqed'].dropna()
    sa = d['dsa'].dropna()
    print(f"  BetterAff: {(aff<0).mean()*100:.1f}%  BetterQED: {(qed>0).mean()*100:.1f}%  BetterSA: {(sa<0).mean()*100:.1f}%")

print(f"\n--- OVERALL: {n_changed:,} fixed molecules ---")
for label, col in [('dAff', 'delta_affinity'), ('dQED', 'dqed'), ('dSA', 'dsa'),
                    ('dlogP', 'dlogp'), ('dMW', 'dmw'), ('dRMSD', 'drmsd')]:
    v = changed[col].dropna()
    print(f"  {label:8s}: mean={v.mean():+.4f}  median={v.median():+.4f}  std={v.std():.4f}  [{v.min():+.3f}, {v.max():+.3f}]")
aff = changed['delta_affinity'].dropna()
qed = changed['dqed'].dropna()
sa = changed['dsa'].dropna()
print(f"  BetterAff: {(aff<0).mean()*100:.1f}%  BetterQED: {(qed>0).mean()*100:.1f}%  BetterSA: {(sa<0).mean()*100:.1f}%")
