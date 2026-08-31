#!/usr/bin/env python3
"""Step 2 (chroman edition): mutation ladder on the clean cold-start seed.

Seed = 2-phenylchroman (independent chroman core from gen[3192] + independent
phenol B-ring from gen[9945], both native pocket coordinates).

Mutations (scaffold coordinates untouched; new atoms placed at standard
bond lengths along outward ring directions):
  M0 seed_sat  : as-is (2-phenylchroman, saturated)
  M1 +ox       : C4 oxidized CH2 -> C=O (chroman-4-one)
  M2 +dehyd    : C2=C3 dehydrogenation (chromone)
  M3 +C3-OH
  M4 +A5,A7-OH  == m1
  M5 +B2'-OH    == m2
  M6 +B3'-OH    == m3
"""
import sys, os, json
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

# roles on the chroman seed
O_ring = next(a.GetIdx() for a in seed.GetAtoms() if a.GetSymbol() == 'O')
ringO = seed.GetAtomWithIdx(O_ring)
# C2 = sp3 C bonded to O_ring AND to an aromatic C outside the chroman (the B-ring link)
C2 = None
for nb in ringO.GetNeighbors():
    if not nb.GetIsAromatic():
        nbs = [x for x in nb.GetNeighbors() if x.GetIdx() != O_ring]
        if any(x.GetIsAromatic() and not x.GetIdx() in
               [y.GetIdx() for y in seed.GetAtomWithIdx(O_ring).GetNeighbors()] for x in nbs) \
           and any(not x.GetIsAromatic() for x in nbs):
            C2 = nb.GetIdx(); break
assert C2 is not None
C3 = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C2).GetNeighbors()
          if not nb.GetIsAromatic() and nb.GetIdx() != O_ring)
C4 = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C3).GetNeighbors()
          if not nb.GetIsAromatic() and nb.GetIdx() != C2)
print(f"roles: O_ring={O_ring} C2={C2} C3={C3} C4={C4}")

# A-ring = the benzene fused to the pyran (contains C4a = aromatic neighbor of C4)
C4a = next(n.GetIdx() for n in seed.GetAtomWithIdx(C4).GetNeighbors() if n.GetIsAromatic())
Aring = next(r for r in seed.GetRingInfo().AtomRings() if C4a in r)
C8a = next(nb.GetIdx() for nb in ringO.GetNeighbors() if nb.GetIsAromatic())
aord = [C4a]
while len(aord) < 6:
    for c in Aring:
        if c not in aord and seed.GetBondBetweenAtoms(c, aord[-1]):
            aord.append(c); break
if aord.index(C8a) == 1:
    aord = [aord[0]] + aord[:0:-1]
c5, c6, c7, c8 = aord[1], aord[2], aord[3], aord[4]
print(f"A-ring: C5={c5} C6={c6} C7={c7} C8={c8}")

# B-ring
BrC = [a.GetIdx() for a in seed.GetAtoms()
       if a.GetIsAromatic() and a.GetIdx() not in Aring and a.GetSymbol() == 'C']
link = next(n.GetIdx() for n in seed.GetAtomWithIdx(C2).GetNeighbors()
            if n.GetIsAromatic() and n.GetIdx() not in Aring)
bring_ring = next(r for r in seed.GetRingInfo().AtomRings() if link in r and set(r) <= set(BrC) | {link})
bord = [link]
while len(bord) < 6:
    for c in bring_ring:
        if c not in bord and seed.GetBondBetweenAtoms(c, bord[-1]):
            bord.append(c); break
pos = {2: bord[1], 3: bord[2], 4: bord[3], 5: bord[4], 6: bord[5]}
have_OH = [k for k, v in pos.items()
           if any(nb.GetSymbol() == 'O' and nb.GetIdx() != O_ring for nb in seed.GetAtomWithIdx(v).GetNeighbors())]
print(f"B-ring: link={link}, 已有OH位 {have_OH}")

def place_dir(P, ring_atoms, c):
    Pr = np.array([P[i] for i in ring_atoms])
    n = np.linalg.svd(Pr - Pr.mean(0))[2][2]
    v = P[c] - Pr.mean(0)
    v = v - (v @ n) * n
    return v / np.linalg.norm(v)

def add_OH(mol, P, c_idx, ring_atoms):
    v = place_dir(P, ring_atoms, c_idx)
    rw = Chem.RWMol(mol)
    oidx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c_idx, oidx, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    a = m2.GetAtomWithIdx(oidx)
    a.SetNumExplicitHs(1); a.SetNoImplicit(True); a.SetFormalCharge(0)
    Chem.SanitizeMol(m2)
    return m2, np.vstack([P, P[c_idx] + v * 1.37])

def oxidize_C4(mol, P):
    """C4 CH2 -> C=O: add carbonyl O along the outward sp2 direction."""
    # direction: away from ring centroid of the chroman heavy atoms
    chrom = list(range(10))  # first 10 atoms are the chroman core in the seed
    v = P[C4] - P[chrom].mean(0)
    # project into the C3-C4-C4a plane is overkill; outward radial is fine
    v /= np.linalg.norm(v)
    rw = Chem.RWMol(mol)
    oidx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(C4, oidx, Chem.BondType.DOUBLE)
    a4 = rw.GetAtomWithIdx(C4)
    a4.SetNumExplicitHs(0); a4.SetNoImplicit(True)
    o = rw.GetAtomWithIdx(oidx)
    o.SetNumExplicitHs(0); o.SetNoImplicit(True); o.SetFormalCharge(0)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2)
    return m2, np.vstack([P, P[C4] + v * 1.22])

def dehydrogenate(mol, i, j):
    rw = Chem.RWMol(mol)
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

M1S = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
M2S = 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'
M3S = 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
targets = {'m1': M1S, 'm2': M2S, 'm3': M3S}

# ladder build
ladder = []
m, P = seed, P0
smi = write(m, P, 'mut_seed_sat')
ladder.append(('mut_seed_sat', smi))
# + C4 oxidation
m, P = oxidize_C4(m, P)
ladder.append(('mut_ox', write(m, P, 'mut_ox')))
# + C2=C3 dehydrogenation
m = dehydrogenate(m, C2, C3)
ladder.append(('mut_ox_dehyd', write(m, P, 'mut_ox_dehyd')))
# + C3-OH
m, P = add_OH(m, P, C3, list(range(10)))
ladder.append(('mut_ox_dehyd_3OH', write(m, P, 'mut_ox_dehyd_3OH')))
# + A5, A7 OH  (== m1)
mA, PA = add_OH(m, P, c5, list(Aring))
mA, PA = add_OH(mA, PA, c7, list(Aring))
ladder.append(('mut_m1', write(mA, PA, 'mut_m1')))
# m2: m1 + B2'-OH
mB, PB = add_OH(mA, PA, pos[2], list(bring_ring))
ladder.append(('mut_m2', write(mB, PB, 'mut_m2')))
# m3: m1 + B3'-OH
mC, PC = add_OH(mA, PA, pos[3], list(bring_ring))
ladder.append(('mut_m3', write(mC, PC, 'mut_m3')))

print()
for name, smi in ladder:
    hit = [t for t, s in targets.items() if s == smi]
    print(f"{name:20s}: {smi[:82]}{'  ←== ' + hit[0] if hit else ''}")
json.dump(dict(ladder), open(f'{OUTD}/mutation_smiles.json', 'w'), indent=2)
print('[saved] mutants')
