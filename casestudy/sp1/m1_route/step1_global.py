#!/usr/bin/env python3
"""Global assembly scan: all independent chroman cores (89) x all independent
phenol B-rings (4452) from the 18641-molecule cold-start library.

For each pair: rigid biphenyl assembly (C2-para 1.48 A, ~40 deg twist
sweep), keep clash-free pairs (fragment dmin >= 1.55 A, pocket dmin >= 1.0 A),
rank by fragment dmin; report the best combinations. Outputs the top
candidates' seeds for the mutation-ladder evaluation.
"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')
from util.ligand_code_util import parse_pdb_file

LIB = 'casestudy/sp1/library/cold_start_18641.sdf'
OUTD = 'casestudy/sp1/m1_route'
os.makedirs(OUTD, exist_ok=True)
os.makedirs(f'{OUTD}/global_scan', exist_ok=True)
PP = np.array(parse_pdb_file('casestudy/sp1/sp1_recp_pocket_pocket.pdb')['pos'])

lib = Chem.SDMolSupplier(LIB, removeHs=True)
CHROMAN = Chem.MolFromSmarts('C1COc2ccccc2C1')
PHENOL = Chem.MolFromSmarts('c1ccccc1[OX2H0,OX2H1]')

def independence(m, frag_atoms):
    frag = set(frag_atoms)
    ri = m.GetRingInfo()
    fused = sum(1 for r in ri.AtomRings() if set(r) & frag and not set(r) <= frag)
    ext = sum(1 for i in frag for nb in m.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in frag)
    return fused, ext

def extract(m, keep_idx):
    keep = sorted(keep_idx)
    rw = Chem.RWMol(m)
    for i in sorted(set(range(m.GetNumAtoms())) - set(keep), reverse=True):
        rw.RemoveAtom(i)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    P = np.array(m.GetConformer().GetPositions())[keep]
    conf = Chem.Conformer(out.GetNumAtoms())
    for k in range(out.GetNumAtoms()):
        conf.SetAtomPosition(k, Point3D(*map(float, P[k])))
    out.RemoveAllConformers(); out.AddConformer(conf, assignId=True)
    return out, P

def roles_chroman(core, P):
    O_ring = next(a.GetIdx() for a in core.GetAtoms() if a.GetSymbol() == 'O')
    ringO = core.GetAtomWithIdx(O_ring)
    C2 = next(nb.GetIdx() for nb in ringO.GetNeighbors() if not nb.GetIsAromatic())
    C3 = next(nb.GetIdx() for nb in core.GetAtomWithIdx(C2).GetNeighbors()
              if not core.GetAtomWithIdx(nb.GetIdx()).GetIsAromatic() and nb.GetIdx() != O_ring)
    C4 = next(nb.GetIdx() for nb in core.GetAtomWithIdx(C3).GetNeighbors()
              if not core.GetAtomWithIdx(nb.GetIdx()).GetIsAromatic() and nb.GetIdx() != C2)
    C4a = next(nb.GetIdx() for nb in core.GetAtomWithIdx(C4).GetNeighbors()
               if core.GetAtomWithIdx(nb.GetIdx()).GetIsAromatic())
    return dict(O_ring=O_ring, C2=C2, C3=C3, C4=C4, C4a=C4a,
                Aring=[i for i in range(core.GetNumAtoms()) if core.GetAtomWithIdx(i).GetIsAromatic()])

def rot_between(a, b):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-9: return np.eye(3) if c > 0 else None
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K@K*(1/(1+c))

def rot_axis(P, axis, th):
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
    return (R @ P.T).T

print('[1/3] extracting independent chroman cores...', flush=True)
cores = []
for idx, m in enumerate(lib):
    if m is None: continue
    if sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0: continue
    for mt in m.GetSubstructMatches(CHROMAN, uniquify=True):
        fused, ext = independence(m, mt)
        if fused == 0:
            core, P = extract(m, mt)
            r = roles_chroman(core, P)
            cores.append(dict(lib_idx=idx, n_atom=core.GetNumAtoms(),
                              core=core, P=P, ext=ext, **r))
            break
print(f'    cores: {len(cores)}')

print('[2/3] extracting independent phenol B-rings...', flush=True)
brings = []
for idx, m in enumerate(lib):
    if m is None: continue
    if sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0: continue
    if len(m.GetRingInfo().AtomRings()) != 1: continue
    mts = m.GetSubstructMatches(PHENOL, uniquify=True)
    if not mts: continue
    oI = mts[0][6]
    if m.GetAtomWithIdx(oI).GetDegree() != 1: continue
    frag, P = extract(m, sorted(mts[0][:6]) + [oI])
    ohC = next(c for c in range(6) if any(nb.GetIdx() == 6 for nb in frag.GetAtomWithIdx(c).GetNeighbors()))
    para = (ohC + 3) % 6
    brings.append(dict(lib_idx=idx, mol=frag, P=P, ohC=ohC, para=para))
print(f'    B-rings: {len(brings)}')

print('[3/3] global assembly scan...', flush=True)
best_pairs = []
for ci, c in enumerate(cores):
    Pc = c['P']; C2 = c['C2']
    n_c = np.linalg.svd(Pc - Pc.mean(0))[2][2]
    v = Pc[C2] - Pc.mean(0); v -= (v @ n_c)*n_c; v /= np.linalg.norm(v)
    for bi, b in enumerate(brings):
        Pb = b['P']; para = b['para']
        n_b = np.linalg.svd(Pb[:6] - Pb[:6].mean(0))[2][2]
        vout = Pb[para] - Pb[:6].mean(0)
        vout -= (vout @ n_b)*n_b
        if np.linalg.norm(vout) < 1e-9: continue
        vout /= np.linalg.norm(vout)
        R = rot_between(vout, -v)
        if R is None: continue
        anchor = Pc[C2] + v*1.48
        Pb1 = (R @ (Pb - Pb[para]).T).T + anchor
        best = None
        for th in np.linspace(0, 2*np.pi, 13):
            Pb2 = rot_axis(Pb1 - anchor, v, th) + anchor
            D = np.linalg.norm(Pc[:,None,:]-Pb2[None,:,:], axis=-1)
            D[C2, para] = 9e9
            dm = D.min()
            if best is None or dm > best[0]:
                best = (dm, Pb2)
        dmin, Pb2 = best
        if dmin < 1.55: continue
        Pall = np.vstack([Pc, Pb2])
        Dpk = np.linalg.norm(Pall[:,None,:]-PP[None,:,:], axis=-1).min()
        if Dpk < 1.0: continue
        best_pairs.append((round(float(dmin),2), round(float(Dpk),2), ci, bi,
                           c['lib_idx'], b['lib_idx'], Pall))
best_pairs.sort(key=lambda x: (-x[0], -x[1]))
print(f'    clash-free pairs: {len(best_pairs)}; top:')
for d, dk, ci, bi, lia, lib, _ in best_pairs[:15]:
    print(f"      dmin={d:.2f} pocket={dk:.2f}  core gen[{lia}] x bring gen[{lib}]")

json.dump([{'dmin': d, 'pocket': dk, 'ci': ci, 'bi': bi, 'core_lib': lia,
            'bring_lib': lib} for d, dk, ci, bi, lia, lib, _ in best_pairs[:10]],
          open(f'{OUTD}/global_scan/ranked_pairs.json', 'w'), indent=2)

# save the top 3 assembled seeds
for rank, (d, dk, ci, bi, lia, lib, Pall) in enumerate(best_pairs[:3]):
    c, b = cores[ci], brings[bi]
    seed = Chem.CombineMols(c['core'], b['mol'])
    rw = Chem.RWMol(seed)
    nA = c['core'].GetNumAtoms()
    bo = next(a.GetIdx() for a in rw.GetAtoms()
              if a.GetIdx() >= nA and a.GetSymbol() == 'O' and a.GetDegree() == 1)
    ohC = next(nb.GetIdx() for nb in rw.GetAtomWithIdx(bo).GetNeighbors())
    bringc = [a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() >= nA and a.GetIsAromatic()]
    bord = [ohC]
    while len(bord) < 6:
        for x in bringc:
            if x not in bord and rw.GetBondBetweenAtoms(x, bord[-1]):
                bord.append(x); break
    rw.AddBond(c['C2'], bord[3], Chem.BondType.SINGLE)
    s = rw.GetMol()
    for a in s.GetAtoms():
        if a.GetSymbol() == 'O' and a.GetDegree() == 1:
            a.SetNumExplicitHs(1); a.SetNoImplicit(True); a.SetFormalCharge(0)
    s = rw.GetMol()
    conf = Chem.Conformer(s.GetNumAtoms())
    for k, p in enumerate(Pall):
        conf.SetAtomPosition(k, Point3D(*map(float, p)))
    s.RemoveAllConformers(); s.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(s)
    fn = f'{OUTD}/global_scan/seed_g{rank}.sdf'
    s.SetProp('_Name', f'gseed{rank}_core{lia}xbring{lib}')
    s.SetProp('_srcCore', f'sp1 cold-start gen[{lia}] independent chroman')
    s.SetProp('_srcBring', f'sp1 cold-start gen[{lib}] independent phenol')
    w = Chem.SDWriter(fn); w.write(s); w.close()
    print(f'[seed] {fn}: {Chem.MolToSmiles(s)}')
print('[done] global scan complete')
