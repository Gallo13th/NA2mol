#!/usr/bin/env python
"""Analyze real vs cosmetic SMILES changes after planar ring fix."""
import pandas as pd, numpy as np
from pathlib import Path
from rdkit import Chem
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
changed = full[full['smi_changed'] == True].copy()

# Classify: real = molecular formula changed
def mol_formula(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        atoms = {}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            atoms[sym] = atoms.get(sym, 0) + 1
        return ''.join(f"{k}{v}" for k, v in sorted(atoms.items()))
    except:
        return None

sample = changed.sample(min(2000, len(changed)), random_state=42)
sample['orig_formula'] = sample['orig_SMILES'].apply(mol_formula)
sample['fixed_formula'] = sample['fixed_SMILES'].apply(mol_formula)
sample['formula_changed'] = (sample['orig_formula'] != sample['fixed_formula']) & \
                             sample['orig_formula'].notna() & sample['fixed_formula'].notna()

n_real = sample['formula_changed'].sum()
n_cosmetic = len(sample) - n_real
print(f"Sample of {len(sample)} SMILES-changed molecules:")
print(f"  Molecular formula ACTUALLY changed: {n_real} ({n_real/len(sample)*100:.1f}%)")
print(f"  Same formula, diff SMILES (cosmetic):  {n_cosmetic} ({n_cosmetic/len(sample)*100:.1f}%)")

# Show examples
cosmetic = sample[~sample['formula_changed']].head(3)
print("\n--- Cosmetic SMILES changes (same formula, re-minimization noise) ---")
for _, row in cosmetic.iterrows():
    print(f"  orig:  {row['orig_SMILES'][:90]}")
    print(f"  fixed: {row['fixed_SMILES'][:90]}")
    print(f"  MW: {row['orig_mw']:.1f} -> {row['fixed_mw']:.1f}  dAff: {row['delta_affinity']:.2f}")
    print()

real = sample[sample['formula_changed']].head(3)
print("--- Real formula changes (ring was actually fixed) ---")
for _, row in real.iterrows():
    print(f"  orig:  {row['orig_SMILES'][:90]}")
    print(f"  fixed: {row['fixed_SMILES'][:90]}")
    print(f"  formula: {row['orig_formula']} -> {row['fixed_formula']}")
    print(f"  MW: {row['orig_mw']:.1f} -> {row['fixed_mw']:.1f}  dAff: {row['delta_affinity']:.2f}")
    print()

# How many molecules have |MW change| > 1 Da?
full['mw_delta'] = full['fixed_mw'] - full['orig_mw']
real_all = full[full['mw_delta'].abs() > 1.0]
print(f"--- All molecules with |ΔMW| > 1 Da (real change) ---")
print(f"  Total: {len(real_all)} / {len(full)} ({len(real_all)/len(full)*100:.1f}%)")
for ds in ['smrtnet', 'dude']:
    r = real_all[real_all['dataset'] == ds]
    print(f"  {ds}: {len(r)} ({len(r)/len(full[full['dataset']==ds])*100:.1f}%)")

# Now property analysis for REAL changes
real_all['dqed'] = real_all['fixed_qed'] - real_all['orig_qed']
real_all['dsa'] = real_all['fixed_sa'] - real_all['orig_sa']
real_all['dlogp'] = real_all['fixed_logp'] - real_all['orig_logp']
real_all['drmsd'] = real_all['fixed_rmsd'] - real_all['orig_rmsd']

print(f"\n{'='*60}")
print(f"PROPERTY DELTAS: ONLY MOLECULES WITH REAL STRUCTURAL CHANGES (|ΔMW| > 1 Da)")
print(f"{'='*60}")
for ds_name in ['smrtnet', 'dude']:
    d = real_all[real_all['dataset'] == ds_name]
    print(f"\n--- {ds_name} ({len(d)} molecules) ---")
    for label, col in [('dAff', 'delta_affinity'), ('dQED', 'dqed'), ('dSA', 'dsa'),
                        ('dlogP', 'dlogp'), ('dMW', 'mw_delta'), ('dRMSD', 'drmsd')]:
        v = d[col].dropna()
        print(f"  {label:8s}: mean={v.mean():+.4f}  median={v.median():+.4f}  std={v.std():.4f}  [{v.min():+.3f}, {v.max():+.3f}]")
    aff = d['delta_affinity'].dropna()
    qed = d['dqed'].dropna()
    sa = d['dsa'].dropna()
    print(f"  BetterAff: {(aff<0).mean()*100:.1f}%  BetterQED: {(qed>0).mean()*100:.1f}%  BetterSA: {(sa<0).mean()*100:.1f}%")

print(f"\n--- OVERALL ({len(real_all)} molecules) ---")
for label, col in [('dAff', 'delta_affinity'), ('dQED', 'dqed'), ('dSA', 'dsa'),
                    ('dlogP', 'dlogp'), ('dMW', 'mw_delta'), ('dRMSD', 'drmsd')]:
    v = real_all[col].dropna()
    print(f"  {label:8s}: mean={v.mean():+.4f}  median={v.median():+.4f}  std={v.std():.4f}  [{v.min():+.3f}, {v.max():+.3f}]")
aff = real_all['delta_affinity'].dropna()
qed = real_all['dqed'].dropna()
sa = real_all['dsa'].dropna()
print(f"  BetterAff: {(aff<0).mean()*100:.1f}%  BetterQED: {(qed>0).mean()*100:.1f}%  BetterSA: {(sa<0).mean()*100:.1f}%")
