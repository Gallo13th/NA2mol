#!/usr/bin/env python3
"""Step 2: mutation set on the cold-start assembled seed -> verify m1/m2/m3
reachable; write mutant SDFs for scoring.

Mutations (elements/substituents only; scaffold coordinates untouched):
  - dehydrogenate C2=C3 (chromanone -> chromone)
  - add phenol OH at defined positions (new O placed at 1.37 A outward,
    in the ring plane - standard sp2 C-O geometry)
"""
import sys, os
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

OUTD = 'casestudy/sp1/m1_route'
seed = next(Chem.SDMolSupplier(f'{OUTD}/m1_route_seed.sdf', removeHs=True))
P0 = np.array(seed.GetConformer().GetPositions())
print(f"seed: {Chem.MolToSmiles(seed)} ({seed.GetNumAtoms()} atoms)")

CHROMANONE = Chem.MolFromSmarts('O=C1CCOc2ccccc21')
mt = seed.GetSubstructMatches(CHROMANONE)[0]
Os = [i for i in mt if seed.GetAtomWithIdx(i).GetSymbol() == 'O']
O_ring = next(i for i in Os if sum(1 for n in seed.GetAtomWithIdx(i).GetNeighbors()) == 2)
O_carb = next(i for i in Os if i != O_ring)
C4 = next(n.GetIdx() for n in seed.GetAtomWithIdx(O_carb).GetNeighbors())
mtset = set(mt)
C3 = next(n.GetIdx() for n in seed.GetAtomWithIdx(C4).GetNeighbors()
          if n.GetIdx() in mtset and not n.GetIsAromatic() and n.GetIdx() != O_carb)
C2 = next(n.GetIdx() for n in seed.GetAtomWithIdx(O_ring).GetNeighbors()
          if n.GetIdx() in mtset and not n.GetIsAromatic())
if not any(n.GetIdx() == C2 for n in seed.GetAtomWithIdx(C3).GetNeighbors()):
    C2, C3 = C3, C2
BringC = [a.GetIdx() for a in seed.GetAtoms() if a.GetIdx() not in mtset and a.GetSymbol() == 'C' and a.GetIsAromatic()]
BringC = sorted(BringC)
print(f"roles: C2={C2} C3={C3} O_ring={O_ring} B-ring C={BringC}")

def add_OH(mol, P, c_idx):
    """attach an OH oxygen at ring carbon c_idx: 1.37 A outward in ring plane."""
    ri = mol.GetRingInfo()
    ring = next(r for r in ri.AtomRings() if c_idx in r)
    Pring = np.array([P[i] for i in ring])
    n = np.linalg.svd(Pring - Pring.mean(0))[2][2]
    v = P[c_idx] - Pring.mean(0)
    v = v - (v @ n) * n
    v /= np.linalg.norm(v)
    opos = P[c_idx] + v * 1.37
    rw = Chem.RWMol(mol)
    oidx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c_idx, oidx, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2)
    P2 = np.vstack([P, opos])
    return m2, P2

def dehydrogenate(mol, i, j):
    rw = Chem.RWMol(mol)
    b = rw.GetBondBetweenAtoms(i, j)
    rw.RemoveBond(i, j)
    rw.AddBond(i, j, Chem.BondType.DOUBLE)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2, catchErrors=True)
    try:
        Chem.SetAromaticity(m2)
    except Exception:
        pass
    return m2

def write(m, P, name):
    conf = Chem.Conformer(m.GetNumAtoms())
    for k in range(m.GetNumAtoms()):
        conf.SetAtomPosition(k, Point3D(*map(float, P[k])))
    m.RemoveAllConformers(); m.AddConformer(conf, assignId=True)
    m.SetProp('_Name', name)
    w = Chem.SDWriter(f'{OUTD}/{name}.sdf'); w.write(m); w.close()
    return Chem.MolToSmiles(m)

# B-ring mapping: ring atoms in order; find positions relative to the link (C1')
# link = the B-ring C bonded to C2
link = next(n.GetIdx() for n in seed.GetAtomWithIdx(C2).GetNeighbors() if n.GetIdx() in BringC)
ri = seed.GetRingInfo()
bring_ring = next(r for r in ri.AtomRings() if link in r)
bring_ring = list(bring_ring)
# order the ring: BFS from link
ordered = [link]
while len(ordered) < 6:
    for c in bring_ring:
        if c not in ordered and seed.GetBondBetweenAtoms(c, ordered[-1]):
            ordered.append(c); break
# positions: 2',3',4',5',6' = ordered[1..5]
pos = {2: ordered[1], 3: ordered[2], 4: ordered[3], 5: ordered[4], 6: ordered[5]}
# existing 4'-OH already on the B-ring? find ring C with O neighbor
have_OH = [c for c in bring_ring
           if any(nb.GetSymbol() == 'O' and nb.GetIdx() != O_ring and nb.GetIdx() != O_carb
                  for nb in seed.GetAtomWithIdx(c).GetNeighbors())]
print(f"B-ring link={link} ordered={ordered} 已有OH位: {[k for k,v in pos.items() if v in have_OH]}")

# A-ring (benzene of chromanone) positions 5,6,7,8: aromatic C in mt
AringC = [i for i in mt if seed.GetAtomWithIdx(i).GetIsAromatic()]
aring = next(r for r in ri.AtomRings() if set(r) == set(AringC))
# order A-ring starting from C4a (bonded to C4)
C4a = next(n.GetIdx() for n in seed.GetAtomWithIdx(C4).GetNeighbors() if n.GetIdx() in AringC)
C8a = next(n.GetIdx() for n in seed.GetAtomWithIdx(O_ring).GetNeighbors() if n.GetIdx() in AringC)
aord = [C4a]
while len(aord) < 6:
    for c in aring:
        if c not in aord and seed.GetBondBetweenAtoms(c, aord[-1]):
            aord.append(c); break
# C5 adjacent to C4a on the side away from C8a; C8 adjacent to C8a
# canonical order along ring: C4a(0) C5(1) C6(2) C7(3) C8(4) C8a(5)
aord = [C4a]
while len(aord) < 6:
    for c in aring:
        if c not in aord and seed.GetBondBetweenAtoms(c, aord[-1]):
            aord.append(c); break
if aord.index(C8a) == 1:            # walked the wrong way; reverse
    aord = [aord[0]] + aord[:0:-1]
c5, c6, c7, c8 = aord[1], aord[2], aord[3], aord[4]
print(f"A-ring order: {aord} -> C5={c5} C6={c6} C7={c7} C8={c8}")

# ── mutation ladder ────────────────────────────────────────────────────────
M1 = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
M2 = 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'
M3 = 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
targets = {'m1': M1, 'm2': M2, 'm3': M3}

# ladder: (name, dehyd?, C3-OH?, A-ring OH set, B-ring extra OH set)
ladder = [
    ('mut_seed_sat',  False, set(), set(),     set()),   # chromanone as-is
    ('mut_dehyd',     True,  set(), set(),     set()),   # chromone core
    ('mut_dehyd_3OH', True,  {3},  set(),     set()),   # +C3-OH
    ('mut_m1',        True,  {3},  {5, 7},    set()),   # == m1 (A-ring 5,7-diOH)
    ('mut_m2',        True,  {3},  {5, 7},    {2}),     # m1 + 2'-OH
    ('mut_m3',        True,  {3},  {5, 7},    {3}),     # m1 + 3'-OH
]
results = {}
for name, dehyd, oh3, aOH, bOH in ladder:
    m = Chem.Mol(seed); P = P0.copy()
    if dehyd:
        m = dehydrogenate(m, C2, C3)
    if 3 in oh3:
        m, P = add_OH(m, P, C3)
    for p_ in sorted(aOH):
        m, P = add_OH(m, P, {'5': c5, '6': c6, '7': c7}[str(p_)])
    for p_ in sorted(bOH):
        m, P = add_OH(m, P, pos.get(p_, pos[3]))
    smi = write(m, P, name)
    hit = [t for t, s in targets.items() if s == smi]
    results[name] = smi
    print(f"{name:18s}: {smi[:85]}{'  ←== ' + hit[0] if hit else ''}")
json_out = {k: v for k, v in results.items()}
import json
json.dump(json_out, open(f'{OUTD}/mutation_smiles.json', 'w'), indent=2)
print('[saved] mutants + mutation_smiles.json')
