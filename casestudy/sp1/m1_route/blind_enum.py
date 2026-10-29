#!/usr/bin/env python3
"""Blind enumeration of scaffold bond-order x substituent space on the
cold-start assembled seed (17-atom skeleton).

Key semantics:
  bond-order axis (3): {saturated, C4=O, C4=O + C2=C3}
  substituent axis (32): C3-OH, A5-OH, A7-OH, B2'-OH, B3'-OH (any subset)
=> 96 enumerated molecules (sanitize guards; duplicates removed).

Each is scored by rDock (-n 5) + RNAsmol 6-model ensemble; the ranking is
purely score-driven, no target structure used as an oracle during
construction (the targets m1/m2/m3 enter ONLY in the reporting step).
"""
import os, sys, json, itertools, subprocess
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

G = 'casestudy/sp1/m1_route'
OUT = f'{G}/blind_enum'
os.makedirs(OUT, exist_ok=True)
seed = next(Chem.SDMolSupplier(f'{G}/global_scan/seed_g0.sdf', removeHs=True))
P0 = np.array(seed.GetConformer().GetPositions())

def is_sp3(a):
    return a.GetSymbol() == 'C' and not any(
        b.GetBondType() == Chem.BondType.AROMATIC for b in a.GetBonds())

# roles
O_ring = next(a.GetIdx() for a in seed.GetAtoms() if a.GetSymbol() == 'O')
ringO = seed.GetAtomWithIdx(O_ring)
ringO_nbrs = [x.GetIdx() for x in ringO.GetNeighbors()]
C2 = next(nb.GetIdx() for nb in ringO.GetNeighbors()
          if is_sp3(nb) and any(x.GetIsAromatic() and x.GetIdx() not in ringO_nbrs
                                for x in nb.GetNeighbors()))
C3 = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C2).GetNeighbors()
          if is_sp3(seed.GetAtomWithIdx(nb.GetIdx())))
C4 = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C3).GetNeighbors()
          if is_sp3(seed.GetAtomWithIdx(nb.GetIdx())) and nb.GetIdx() != C2)
C4a = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C4).GetNeighbors()
           if seed.GetAtomWithIdx(nb.GetIdx()).GetIsAromatic())
Aring = next(r for r in seed.GetRingInfo().AtomRings()
             if C4a in r and all(seed.GetAtomWithIdx(i).GetIsAromatic() for i in r)
             and C4 not in r and O_ring not in r and len(r) == 6)
aord = [C4a]
while len(aord) < 6:
    for cc in Aring:
        if cc not in aord and seed.GetBondBetweenAtoms(cc, aord[-1]):
            aord.append(cc); break
c5, c7 = aord[1], aord[3]
BrC = [a.GetIdx() for a in seed.GetAtoms()
       if a.GetIsAromatic() and a.GetIdx() not in Aring and a.GetSymbol() == 'C']
link = next(n.GetIdx() for n in seed.GetAtomWithIdx(C2).GetNeighbors()
            if n.GetIsAromatic() and n.GetIdx() not in Aring)
bring_ring = next(r for r in seed.GetRingInfo().AtomRings()
                  if link in r and set(r) <= set(BrC) | {link})
bord = [link]
while len(bord) < 6:
    for cc in bring_ring:
        if cc not in bord and seed.GetBondBetweenAtoms(cc, bord[-1]):
            bord.append(cc); break
pB2, pB3, pB4 = bord[1], bord[2], bord[3]
ohB_have = next((x for x in bring_ring
                 if any(nb.GetSymbol() == 'O' and nb.GetIdx() != O_ring
                        for nb in seed.GetAtomWithIdx(x).GetNeighbors())), None)
print(f"roles: C2={C2} C3={C3} C4={C4} c5={c5} c7={c7} B(2'={pB2},3'={pB3},4'={pB4}) 已有OH {ohB_have}")

def add_OH(mol, P, c_idx, ring_atoms):
    Pr = np.array([P[i] for i in ring_atoms])
    n = np.linalg.svd(Pr - Pr.mean(0))[2][2]
    v = P[c_idx] - Pr.mean(0)
    v = v - (v @ n) * n
    if np.linalg.norm(v) < 1e-9:
        v = np.cross(n, P[c_idx])
    v /= np.linalg.norm(v)
    rw = Chem.RWMol(mol)
    oi = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c_idx, oi, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    a = m2.GetAtomWithIdx(oi)
    a.SetNumExplicitHs(1); a.SetNoImplicit(True); a.SetFormalCharge(0)
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None, None
    return m2, np.vstack([P, P[c_idx] + v * 1.37])

def oxidize(mol, P, c):
    rw = Chem.RWMol(mol)
    oi = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c, oi, Chem.BondType.DOUBLE)
    rw.GetAtomWithIdx(c).SetNumExplicitHs(0); rw.GetAtomWithIdx(c).SetNoImplicit(True)
    o = rw.GetAtomWithIdx(oi)
    o.SetNumExplicitHs(0); o.SetNoImplicit(True); o.SetFormalCharge(0)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None, None
    v = P[c] - P[np.where(np.linalg.norm(P - P[c], axis=-1) > 0.01)[0]].mean(0)
    if np.linalg.norm(v) < 1e-9:
        v = np.array([1., 0., 0.])
    v /= np.linalg.norm(v)
    return m2, np.vstack([P, P[c] + v * 1.22])

def dehydrogenate(mol, i, j):
    rw = Chem.RWMol(mol)
    rw.RemoveBond(i, j); rw.AddBond(i, j, Chem.BondType.DOUBLE)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
        Chem.SetAromaticity(m2)
    except Exception:
        return None
    return m2

def envelope(mol, P):
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(k, Point3D(*map(float, P[k])))
    mol.RemoveAllConformers(); mol.AddConformer(conf, assignId=True)
    return mol

# ---- enumerate -> SDFs ----
variants = []
for ox in [0, 1]:
    for dh in [0, 1]:
        for oh_subset in itertools.product([0, 1], repeat=5):
            # subset: (C3, A5, A7, B2', B3')
            m, P = Chem.Mol(seed), P0.copy()
            if ox:
                m, P = oxidize(m, P, C4)
                if m is None: continue
            if dh:
                m = dehydrogenate(m, C2, C3)
                if m is None: continue
            for on, (cidx, ring) in zip(oh_subset, [
                    (C3, aord), (c5, aord), (c7, aord), (pB2, bord), (pB3, bord)]):
                if on and cidx is not None:
                    m, P = add_OH(m, P, cidx, ring if ring != aord else list(Aring))
                if m is None: break
            if m is None: continue
            smi = Chem.MolToSmiles(m)
            if '.' in smi: continue
            variants.append((smi, m, P))
# dedupe
seen = set(); uniq = []
for smi, m, P in variants:
    if smi in seen: continue
    seen.add(smi); uniq.append((smi, m, P))
print(f"variants: {len(variants)} -> unique {len(uniq)} (sanitized)")

writer = Chem.SDWriter(f'{OUT}/enumerated.sdf')
for i, (smi, m, P) in enumerate(uniq):
    m.SetProp('_Name', f'v{i:03d}')
    m.SetProp('_SMILES', smi)
    writer.write(envelope(m, P))
writer.close()
json.dump([s for s, _, _ in uniq], open(f'{OUT}/smiles_list.json', 'w'), indent=2)
print('[saved] enumerated.sdf + smiles_list.json')
