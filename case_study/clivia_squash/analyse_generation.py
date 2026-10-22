#!/usr/bin/env python3
"""Compare free-generated molecules against the Clivia/Squash quinoline series.

For each generated molecule in the SDF, report:
  - best ECFP4 Tanimoto to any of mol1-mol5 (the five fluorescent-active quinolines)
  - whether the 2,4-diaryl-quinoline scaffold is present
  - the closest target molecule

This is the retrospective question: can unconstrained generation from the aptamer
pocket reproduce the series?

Usage:
  python analyse_generation.py --sdf output/<prefix>/valid_mols.sdf --label clivia_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

WORK = Path(__file__).resolve().parent

# The five compounds read off the mol1-mol5 images, formulas verified with RDKit
# against the printed molecular weights (mol1 357.41 and mol2 356.43 match exactly).
TARGETS = {
    'mol1': dict(formula='C23H19NO3', mw=357.41, smi='COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1'),
    'mol2': dict(formula='C23H20N2O2', mw=356.43, smi='COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1'),
    'mol3': dict(formula='C25H24N2O2', mw=384.48, smi='COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1'),
    'mol4': dict(formula='C22H18N2O2', mw=342.40, smi='COc1ccc(-c2cc(-c3ccncc3)nc3ccc(OC)cc23)cc1'),
    'mol5': dict(formula='C23H19NO4', mw=373.41, smi='COc1ccc2c(c1)C(-c1ccc(O)cc1)C=C(-c1cccc(C(=O)O)c1)N2'),
}
# 2,4-diaryl quinoline: quinoline core with aryl at C2 and C4
QUINOLINE_CORE = Chem.MolFromSmarts('c1ccc2ncccc2c1')
DIARYL_QUINOLINE = Chem.MolFromSmarts('c1ccc2nc(-c3ccccc3)cc(-c3ccccc3)c2c1')


def fps(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sdf', required=True)
    ap.add_argument('--label', default='run')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    sdf = Path(a.sdf)
    if not sdf.is_absolute():
        sdf = Path.cwd() / sdf
    if not sdf.is_file():
        raise SystemExit(f'missing {sdf}')

    ref = {k: fps(v['smi']) for k, v in TARGETS.items()}
    supp = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
    rows = []
    for i, mol in enumerate(supp):
        if mol is None:
            continue
        fp = GEN.GetFingerprint(mol)
        sims = {k: DataStructs.TanimotoSimilarity(fp, v) for k, v in ref.items()}
        best = max(sims, key=sims.get)
        rows.append(dict(
            idx=i,
            smiles=Chem.MolToSmiles(mol),
            heavy=mol.GetNumHeavyAtoms(),
            mw=round(Descriptors_MW(mol), 2),
            best_target=best,
            best_ecfp4=round(sims[best], 4),
            **{f'sim_{k}': round(v, 4) for k, v in sims.items()},
            has_quinoline=bool(mol.HasSubstructMatch(QUINOLINE_CORE)),
            has_diaryl_quinoline=bool(mol.HasSubstructMatch(DIARYL_QUINOLINE)),
        ))

    if not rows:
        raise SystemExit(f'no readable molecules in {sdf}')

    df = pd.DataFrame(rows).sort_values('best_ecfp4', ascending=False)
    out = Path(a.out) if a.out else WORK / f'comparison_{a.label}.csv'
    df.to_csv(out, index=False)

    print(f'=== {a.label}: {len(df)} generated molecules vs mol1-mol5 ===')
    print(f'ECFP4 to the closest target compound:')
    for q in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        n = int((df.best_ecfp4 >= q).sum())
        print(f'  >= {q:.2f}: {n:6d}  ({100*n/len(df):5.2f}%)')
    print(f'\nquinoline core present     : {int(df.has_quinoline.sum()):6d} '
          f'({100*df.has_quinoline.mean():.2f}%)')
    print(f'2,4-diaryl quinoline       : {int(df.has_diaryl_quinoline.sum()):6d} '
          f'({100*df.has_diaryl_quinoline.mean():.2f}%)')
    print(f'\nclosest target distribution:')
    print(df.best_target.value_counts().to_string())
    print(f'\ntop 10 by ECFP4:')
    cols = ['smiles', 'heavy', 'mw', 'best_target', 'best_ecfp4']
    print(df[cols].head(10).to_string(index=False))
    print(f'\nwrote {out}')
    return 0


def Descriptors_MW(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(mol)


if __name__ == '__main__':
    raise SystemExit(main())
