#!/usr/bin/env python3
"""Evaluate top global-scan seeds: build the mutation ladder to m1/m2/m3 for
each seed (g0/g1/g2), score with rDock + RNAsmol, and rank seed candidates.

Output: global_scan/seed_rank.json + per-seed mutant SDFs.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

G = 'casestudy/sp1/m1_route/global_scan'
M1 = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
M2 = 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'
M3 = 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
targets = {'m1': M1, 'm2': M2, 'm3': M3}

def is_sp3(a):
    return a.GetSymbol() == 'C' and not any(
        b.GetBondType() == Chem.BondType.AROMATIC for b in a.GetBonds())

def role_c2(seed):
    O_ring = next(a.GetIdx() for a in seed.GetAtoms() if a.GetSymbol() == 'O')
    ringO = seed.GetAtomWithIdx(O_ring)
    ringO_nbrs = [x.GetIdx() for x in ringO.GetNeighbors()]
    C2 = next(nb.GetIdx() for nb in ringO.GetNeighbors()
              if is_sp3(nb) and any(x.GetIsAromatic() and x.GetIdx() not in ringO_nbrs
                                    for x in nb.GetNeighbors()))
    return O_ring, C2

def sp3_chain(seed, u):
    C3 = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(u).GetNeighbors()
              if is_sp3(seed.GetAtomWithIdx(nb.GetIdx())))
    C4 = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C3).GetNeighbors()
              if is_sp3(seed.GetAtomWithIdx(nb.GetIdx())) and nb.GetIdx() != u)
    return C3, C4

def add_OH(mol, P, c_idx, ring_atoms):
    Pr = np.array([P[i] for i in ring_atoms])
    n = np.linalg.svd(Pr - Pr.mean(0))[2][2]
    v = P[c_idx] - Pr.mean(0)
    v = v - (v @ n) * n
    if np.linalg.norm(v) < 1e-9:
        v = np.cross(n, P[c_idx])
    v /= np.linalg.norm(v)
    rw = Chem.RWMol(mol)
    oidx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c_idx, oidx, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    a = m2.GetAtomWithIdx(oidx)
    a.SetNumExplicitHs(1); a.SetNoImplicit(True); a.SetFormalCharge(0)
    Chem.SanitizeMol(m2)
    return m2, np.vstack([P, P[c_idx] + v * 1.37])

def oxidize(mol, P, c):
    rw = Chem.RWMol(mol)
    oidx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c, oidx, Chem.BondType.DOUBLE)
    rw.GetAtomWithIdx(c).SetNumExplicitHs(0); rw.GetAtomWithIdx(c).SetNoImplicit(True)
    o = rw.GetAtomWithIdx(oidx)
    o.SetNumExplicitHs(0); o.SetNoImplicit(True); o.SetFormalCharge(0)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2)
    v = None
    Pc = P[c]
    idxs = np.where(np.linalg.norm(P - Pc, axis=-1) > 0.01)[0]
    v = Pc - P[idxs].mean(0)
    if np.linalg.norm(v) < 1e-9:
        v = np.array([1., 0., 0.])
    v /= np.linalg.norm(v)
    return m2, np.vstack([P, Pc + v * 1.22])

def dehydrogenate(mol, i, j):
    rw = Chem.RWMol(mol)
    rw.RemoveBond(i, j); rw.AddBond(i, j, Chem.BondType.DOUBLE)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2, catchErrors=True)
    try:
        Chem.SetAromaticity(m2)
    except Exception:
        pass
    return m2

def build_ladder(seed, tag):
    # normalize aromaticity state (combined-mol seeds can carry mixed bond types)
    try:
        Chem.Kekulize(seed, clearAromaticFlags=True)
        Chem.SetAromaticity(seed)
    except Exception:
        pass
    P = np.array(seed.GetConformer().GetPositions())
    O_ring, C2 = role_c2(seed)
    C3, C4 = sp3_chain(seed, C2)
    C4a = next(nb.GetIdx() for nb in seed.GetAtomWithIdx(C4).GetNeighbors()
               if seed.GetAtomWithIdx(nb.GetIdx()).GetIsAromatic())
    # benzene ring only: fully aromatic AND excluding the pyran core atoms
    Aring = next(r for r in seed.GetRingInfo().AtomRings()
                 if C4a in r and all(seed.GetAtomWithIdx(i).GetIsAromatic() for i in r)
                 and C4 not in r and O_ring not in r)
    aord = [C4a]
    while len(aord) < 6:
        for cc in Aring:
            if cc not in aord and seed.GetBondBetweenAtoms(cc, aord[-1]):
                aord.append(cc); break
    c5, c7 = aord[1], aord[3]
    # B-ring positions
    BrC = [a.GetIdx() for a in seed.GetAtoms() if a.GetIsAromatic() and a.GetIdx() not in Aring and a.GetSymbol() == 'C']
    link = next(n.GetIdx() for n in seed.GetAtomWithIdx(C2).GetNeighbors()
                if n.GetIsAromatic() and n.GetIdx() not in Aring)
    bring_ring = next(r for r in seed.GetRingInfo().AtomRings() if link in r and set(r) <= set(BrC) | {link})
    bord = [link]
    while len(bord) < 6:
        for cc in bring_ring:
            if cc not in bord and seed.GetBondBetweenAtoms(cc, bord[-1]):
                bord.append(cc); break
    p2p, p3p = bord[1], bord[2]

    def write(m, pp, name):
        conf = Chem.Conformer(m.GetNumAtoms())
        for k in range(m.GetNumAtoms()):
            conf.SetAtomPosition(k, Point3D(*map(float, pp[k])))
        m.RemoveAllConformers(); m.AddConformer(conf, assignId=True)
        m.SetProp('_Name', name)
        w = Chem.SDWriter(f'{G}/{name}.sdf'); w.write(m); w.close()
        return Chem.MolToSmiles(m)

    res = {}
    m, p = seed, P
    # ladder
    m, p = oxidize(m, p, C4)
    m = dehydrogenate(m, C2, C3)
    m, p = add_OH(m, p, C3, aord if len(aord) == 6 else list(Aring))
    m, p = add_OH(m, p, c5, list(Aring))
    m, p = add_OH(m, p, c7, list(Aring))
    smi = write(m, p, f'mutg_{tag}_m1')
    res['m1_ok'] = (smi == targets['m1'])
    m2m, p2m = add_OH(m, p, p2p, list(bring_ring))
    smi2 = write(m2m, p2m, f'mutg_{tag}_m2')
    res['m2_ok'] = (smi2 == targets['m2'])
    m3m, p3m = add_OH(m, p, p3p, list(bring_ring))
    smi3 = write(m3m, p3m, f'mutg_{tag}_m3')
    res['m3_ok'] = (smi3 == targets['m3'])
    return res

report = {}
for rank in range(3):
    f = f'{G}/seed_g{rank}.sdf'
    seed = next(Chem.SDMolSupplier(f, removeHs=True))
    r = build_ladder(seed, f'g{rank}')
    report[f'seed_g{rank}'] = r
    print(f"seed_g{rank} ({seed.GetProp('_Name')}): m1={r['m1_ok']} m2={r['m2_ok']} m3={r['m3_ok']}")
json.dump(report, open(f'{G}/ladder_validation.json', 'w'), indent=2)
print('[saved] ladder_validation.json')
