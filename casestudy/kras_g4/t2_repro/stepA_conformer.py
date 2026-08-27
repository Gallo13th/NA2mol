#!/usr/bin/env python3
"""t2 Phase1 step A: ETKDG conformer + prep rDock input (19 heavy atoms, polar-H only -> none)."""
import os, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
T2 = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'
OUT = os.path.dirname(os.path.abspath(__file__))

m = Chem.MolFromSmiles(T2)
m = Chem.AddHs(m)
ps = AllChem.ETKDGv3()
ps.randomSeed = 42
ps.useSmallRingTorsions = True
cid = AllChem.EmbedMolecule(m, ps)
assert cid == 0
conv = AllChem.MMFFOptimizeMolecule(m, mmffVariant='MMFF94s', maxIters=2000)
print('MMFF convergence (0=ok):', conv)
m = Chem.RemoveHs(m)   # t2 has no polar H (all H on C)
assert m.GetNumAtoms() == 19
# keep SMILES atom order: build from SMILES order (AddHs/RemoveHs preserves heavy order)
canon_ref = Chem.MolToSmiles(Chem.MolFromSmiles(T2))
canon_now = Chem.MolToSmiles(m)
assert canon_now == canon_ref, (canon_now, canon_ref)
path = os.path.join(OUT, 't2_dock_input.sdf')
w = Chem.SDWriter(path)
m.SetProp('_Name', 't2_benzothiazolyl_ch2_benzothiazole')
w.write(m)
w.close()
print('wrote', path, 'atoms:', m.GetNumAtoms())
# quick geometry sanity: S-C bond lengths etc.
conf = m.GetConformer()
import numpy as np
pos = conf.GetPositions()
for b in m.GetBonds():
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    if m.GetAtomWithIdx(i).GetSymbol() == 'S' or m.GetAtomWithIdx(j).GetSymbol() == 'S':
        d = np.linalg.norm(pos[i] - pos[j])
        print(f'  bond {i}-{j} ({m.GetAtomWithIdx(i).GetSymbol()}-{m.GetAtomWithIdx(j).GetSymbol()}): {d:.3f} A')
