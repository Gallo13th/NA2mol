#!/usr/bin/env python3
"""m1 compliant route: cold-start chromanone core + cold-start phenol B-ring.

Steps
  1. extract the chroman-4-one core (11 atoms, native pocket coordinates)
     from each library molecule that contains it;
  2. extract a phenol 6-ring (B-ring fragment) from a DIFFERENT library
     molecule (native coordinates);
  3. rigid biphenyl assembly: B-ring attached at the core C2 position,
     C2-C1' = 1.48 A, ring planes ~40 deg (biphenyl norm), clash-checked
     against itself and the pocket;
  4. mutation set (elements/substituents applied on the cold-start scaffold,
     evaluated separately): dehydrogenate C2=C3 -> chromone; add hydroxyls
     at A-ring 5,6,7 and B-ring 4' (m1), +3' (m2), or the 3,4' pattern (m3).
Outputs -> casestudy/sp1/m1_route/
"""
import sys, os, json
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

LIB = '/tmp/opencode/sp1_cold_result_keep.sdf'
OUTD = 'casestudy/sp1/m1_route'
os.makedirs(OUTD, exist_ok=True)
POCKET = 'casestudy/sp1/sp1_recp_pocket_pocket.pdb'
sys.path.insert(0, '.')
from util.ligand_code_util import parse_pdb_file
PP = np.array(parse_pdb_file(POCKET)['pos'])

lib = Chem.SDMolSupplier(LIB, removeHs=True)
CHROMANONE = Chem.MolFromSmarts('C1COc2ccccc2C1')  # chroman (saturated; C4 oxidized later as mutation)
PHENOL = Chem.MolFromSmarts('c1ccccc1[OX2H0,OX2H1]')  # 6 ring C + phenol/ether O (charge-free, deg<=2)

def extract(m, keep_idx):
    keep = set(keep_idx)
    rw = Chem.RWMol(m)
    for i in sorted(set(range(m.GetNumAtoms())) - keep, reverse=True):
        rw.RemoveAtom(i)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    order = sorted(keep)
    P = np.array(m.GetConformer().GetPositions())[order]
    conf = Chem.Conformer(out.GetNumAtoms())
    for k in range(out.GetNumAtoms()):
        conf.SetAtomPosition(k, Point3D(*map(float, P[k])))
    out.RemoveAllConformers(); out.AddConformer(conf, assignId=True)
    return out, P

def rot_between(a, b):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-9: return np.eye(3) if c > 0 else None
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K@K*(1/(1+c))

# ── 1. cores ────────────────────────────────────────────────────────────────
def core_roles(mol, mt):
    """programmatic role assignment on the extracted chromanone core."""
    Os = [i for i in mt if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
    O_ring = next(i for i in Os if sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in set(mt)) == 2)
    O_carb = next(i for i in Os if i != O_ring)
    C4 = next(n.GetIdx() for n in mol.GetAtomWithIdx(O_carb).GetNeighbors())
    # C3 = saturated C bonded to C4 (in mt); C2 = saturated C bonded to O_ring & C3
    mtset = set(mt)
    C3 = next(n.GetIdx() for n in mol.GetAtomWithIdx(C4).GetNeighbors()
              if n.GetIdx() in mtset and n.GetIdx() != O_carb and not n.GetIsAromatic())
    C2 = next(n.GetIdx() for n in mol.GetAtomWithIdx(O_ring).GetNeighbors()
              if n.GetIdx() in mtset and not n.GetIsAromatic() and n.GetIdx() != C4 and n.GetIdx() != C3)
    if not any(n.GetIdx() == C2 for n in mol.GetAtomWithIdx(C3).GetNeighbors()):
        C2, C3 = C3, C2   # ensure C2-O_ring, C3-C4 adjacency
    return dict(O1=O_ring, O4=O_carb, C4=C4, C3=C3, C2=C2)

def independence(m, frag_atoms):
    frag = set(frag_atoms)
    ri = m.GetRingInfo()
    fused = sum(1 for r in ri.AtomRings() if set(r) & frag and not set(r) <= frag)
    ext = sum(1 for i in frag for nb in m.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in frag)
    return fused, ext

allcores = []
for idx, m in enumerate(lib):
    if m is None: continue
    for mt in m.GetSubstructMatches(CHROMANONE, uniquify=True):
        fused, ext = independence(m, mt)
        if fused == 0:
            allcores.append((ext, idx, sorted(mt)))
allcores.sort()
cores = []
for ext, idx, mt in allcores[:4]:
    m = lib[idx]
    core, P = extract(m, mt)
    # chroman has no carbonyl; roles: O_ring, C2/C3 (sp3 CH2 adjacent chain)
    O_ring = next(i for i in range(core.GetNumAtoms()) if core.GetAtomWithIdx(i).GetSymbol()=='O')
    mtsat = [a.GetIdx() for a in core.GetAtoms() if not a.GetIsAromatic()]
    # sp3 C adjacent to O_ring = C2; the other two sp3 = C3, C4
    C2 = next(n.GetIdx() for n in core.GetAtomWithIdx(O_ring).GetNeighbors() if not core.GetAtomWithIdx(n.GetIdx()).GetIsAromatic())
    C3 = next(n.GetIdx() for n in core.GetAtomWithIdx(C2).GetNeighbors() if not core.GetAtomWithIdx(n.GetIdx()).GetIsAromatic() and n.GetIdx() != O_ring)
    C4 = next(n.GetIdx() for n in core.GetAtomWithIdx(C3).GetNeighbors() if not core.GetAtomWithIdx(n.GetIdx()).GetIsAromatic() and n.GetIdx() != C2)
    cores.append(dict(lib_idx=idx, mol=core, P=P, O1=O_ring, C2=C2, C3=C3, C4=C4))
    print(f"[core] gen[{idx}] ext={ext} {Chem.MolToSmiles(core)} C2@{C2} C3@{C3} C4@{C4}")

# ── 2. B-ring phenols (from molecules WITHOUT chromanone, to stay independent)
core_libs = {c['lib_idx'] for c in cores}
brings = []
for idx, m in enumerate(lib):
    if m is None or idx in core_libs: continue
    if sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0: continue  # neutral only
    ri = m.GetRingInfo()
    if len(ri.AtomRings()) != 1: continue          # single-ring molecules only
    mts = m.GetSubstructMatches(PHENOL, uniquify=True)
    if not mts: continue
    oI = mts[0][6]
    oa = m.GetAtomWithIdx(oI)
    if oa.GetDegree() != 1: continue               # true phenol OH (terminal O)
    mt = sorted(mts[0][:6]) + [oI]
    frag, P = extract(m, mt)
    brings.append(dict(lib_idx=idx, mol=frag, P=P))
    if len(brings) >= 20: break
print(f"[B-ring] {len(brings)} phenol fragments")

# ── 3. assembly ────────────────────────────────────────────────────────────
def assemble(core, bring, tol_clash=1.55):
    Pc, Pb = core['P'], bring['P']
    C2 = Pc[core['C2']]; O1 = Pc[core['O1']]
    # out-direction at C2: away from ring centroid, in ring plane
    n_core = np.linalg.svd(Pc - Pc.mean(0))[2][2]
    v = C2 - Pc.mean(0); v -= (v @ n_core)*n_core; v /= np.linalg.norm(v)
    # B-ring: attach carbon = para to its OH (4' position)
    # find para C in the 6-ring
    ringC = list(range(6))
    oh = 6
    para = None
    ring = Chem.RWMol(bring['mol']).GetMol()
    for c in ringC:
        # BFS distance 3 in ring = para
        path3 = (c + 3) % 6
        para = path3
    # simply: para = (ohC + 3) % 6 where ohC is the ring C bearing OH
    ohC = next(c for c in ringC
               if any(nb.GetIdx() == oh for nb in ring.GetAtomWithIdx(c).GetNeighbors()))
    para = (ohC + 3) % 6
    C1p = Pb[para]
    n_b = np.linalg.svd(Pb[:6] - Pb[:6].mean(0))[2][2]
    vout = C1p - Pb[:6].mean(0); vout -= (vout @ n_b)*n_b; vout /= np.linalg.norm(vout)
    # place: C1p at C2 + v*1.48; B-ring plane rotated 40 deg about bond axis
    best = None
    for th in np.linspace(0, 2*np.pi, 49):
        R = rot_between(vout, -v)
        if R is None: continue
        Pb1 = (R @ (Pb - C1p).T).T + C2 + v*1.48
        # dihedral twist about the bond axis
        axis = v
        Pb2 = rot_axis_simple(Pb1 - (C2 + v*1.48), axis, th) + C2 + v*1.48
        D = np.linalg.norm(Pc[:,None,:]-Pb2[None,:,:], axis=-1)
        D[core['C2'], para] = 9e9
        if best is None or D.min() > best[0]:
            best = (D.min(), Pb2)
    dmin, Pb2 = best
    Pall = np.vstack([Pc, Pb2])
    Dpk = np.linalg.norm(Pall[:,None,:]-PP[None,:,:], axis=-1).min()
    return dmin, Dpk, Pall

def rot_axis_simple(P, axis, th):
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
    return (R @ P.T).T

best_asm = None
for ci, core in enumerate(cores):
    for bi, bring in enumerate(brings):
        try:
            dmin, Dpk, Pall = assemble(core, bring)
        except Exception:
            continue
        if dmin > 1.55 and Dpk > 1.0:
            score = min(dmin, 3.0)
            if best_asm is None or score > best_asm[0]:
                best_asm = (score, ci, bi, dmin, Dpk, Pall)
                print(f"[asm] core{ci}(gen{core['lib_idx']}) × bring{bi}(gen{bring['lib_idx']}): dmin={dmin:.2f} pocket={Dpk:.2f}")
if best_asm is None:
    print("!! 无 clash-free 组合, 放宽门槛重扫")
    for ci, core in enumerate(cores):
        for bi, bring in enumerate(brings):
            try:
                dmin, Dpk, Pall = assemble(core, bring)
            except Exception:
                continue
            if dmin > 1.2 and Dpk > 0.8:
                print(f"[asm-loose] core{ci}×bring{bi}: dmin={dmin:.2f} pocket={Dpk:.2f}")
                best_asm = (min(dmin,3.0), ci, bi, dmin, Dpk, Pall)
                break
        if best_asm: break

score, ci, bi, dmin, Dpk, Pall = best_asm
core, bring = cores[ci], brings[bi]
print(f"\n[选定] core gen[{core['lib_idx']}] × B-ring gen[{bring['lib_idx']}], dmin={dmin:.2f}, pocket={Dpk:.2f}")

# 合并分子(核11 + 苯酚环7 = 18原子), C2-C1' 成键
combined = Chem.CombineMols(core['mol'], bring['mol'])
rw = Chem.RWMol(combined)
nA = core['mol'].GetNumAtoms()
# locate the B-ring phenol O (terminal, neutral) AFTER atom reordering by extract
bo = next(a.GetIdx() for a in rw.GetAtoms()
          if a.GetIdx() >= nA and a.GetSymbol() == 'O' and a.GetDegree() == 1)
ohC = next(nb.GetIdx() for nb in rw.GetAtomWithIdx(bo).GetNeighbors())
bring_c = [a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() >= nA and a.GetIsAromatic()]
bord = [ohC]
while len(bord) < 6:
    for c in bring_c:
        if c not in bord and rw.GetBondBetweenAtoms(c, bord[-1]):
            bord.append(c); break
para = bord[3]                      # position opposite the 4'-OH
rw.AddBond(core['C2'], para, Chem.BondType.SINGLE)
print(f"[bond] C2({core['C2']}) - B-ring para({para}); 4'-OH at {ohC}")
seed = rw.GetMol()
# normalize the B-ring phenol oxygen (extraction can leave odd H states)
for a in seed.GetAtoms():
    if a.GetSymbol() == 'O' and a.GetDegree() == 1:
        a.SetNumExplicitHs(1)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
seed = rw.GetMol()
conf = Chem.Conformer(seed.GetNumAtoms())
for k, p in enumerate(Pall):
    conf.SetAtomPosition(k, Point3D(*map(float, p)))
seed.RemoveAllConformers(); seed.AddConformer(conf, assignId=True)
Chem.SanitizeMol(seed)
seed.SetProp('_Name', f"m1route_seed_core{core['lib_idx']}xring{bring['lib_idx']}")
seed.SetProp('_srcCore', f"sp1 cold-start gen[{core['lib_idx']}] chromanone core")
seed.SetProp('_srcBring', f"sp1 cold-start gen[{bring['lib_idx']}] phenol ring")
w = Chem.SDWriter(f'{OUTD}/m1_route_seed.sdf'); w.write(seed); w.close()
print(f"[seed] {Chem.MolToSmiles(seed)}")
json.dump({'core_lib': core['lib_idx'], 'bring_lib': bring['lib_idx'],
           'dmin': round(float(dmin),2), 'pocket_dmin': round(float(Dpk),2)},
          open(f'{OUTD}/assembly_manifest.json','w'), indent=2)
print("[saved] m1_route_seed.sdf + assembly_manifest.json")
