#!/usr/bin/env python3
"""t2 Round-4 seeds (W): topology-correct fragments with conformationally-reasonable
INDEPENDENT coordinates.

W1 (16 atom, free 3) = MeBT + Ph : ETKDG conformers of BT-CH2-Ph (connected analog),
    cut the CH2-Ph bond, keep coordinates (relative geometry = realistic torsions).
    Select conformers with diverse C2<->Ph-edge distances.
W0 (15 atom, free 4) = BT + Ph : ETKDG conformers of BT-CH2-CH2-Ph analog, cut the chain.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r4')
os.makedirs(OUT, exist_ok=True)

BT = Chem.MolFromSmiles('c1nc2ccccc2s1')
ANALOG16 = Chem.MolFromSmiles('c1ccc2sc(Cc3ccccc3)nc2c1')   # BT-CH2-Ph
ANALOG17 = Chem.MolFromSmiles('c1ccc2sc(CCc3ccccc3)nc2c1')   # BT-CH2-CH2-Ph
assert ANALOG16.GetNumAtoms() == 16 and ANALOG17.GetNumAtoms() == 17

ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)
rng = np.random.default_rng(2029)


def c2_idx(m):
    for a in m.GetAtoms():
        if a.GetIsAromatic() and a.GetIsAromatic() and a.IsInRingSize(5) and a.GetSymbol() == 'C' \
           and a.GetDegree() == 3:
            return a.GetIdx()
    return None


def embed_n(analog, n, seed0):
    """return list of (mol) ETKDG conformers"""
    out = []
    for k in range(n):
        mh = Chem.AddHs(Chem.Mol(analog))
        ps = AllChem.ETKDGv3(); ps.randomSeed = seed0 + k
        ps.useSmallRingTorsions = True
        cid = AllChem.EmbedMolecule(mh, ps)
        if cid != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
        out.append(Chem.RemoveHs(mh))
    return out


def cut_and_place(m, keep_bond_cut_atoms, drop_atoms, rot=True):
    """build seed = atoms of m minus drop_atoms, minus the bond between cut pair; translate to BER."""
    drop = set(drop_atoms)
    keep = [i for i in range(m.GetNumAtoms()) if i not in drop]
    cut_bond = None
    for b in m.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in drop and e not in drop:
            cut_bond = e
        elif e in drop and s not in drop:
            cut_bond = s
    old2new = {old: new for new, old in enumerate(keep)}
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(keep))
    pos = m.GetConformer().GetPositions()
    for old in keep:
        a = Chem.Atom(m.GetAtomWithIdx(old).GetSymbol())
        if m.GetAtomWithIdx(old).GetIsAromatic():
            a.SetIsAromatic(True)
        rw.AddAtom(a)
        conf.SetAtomPosition(old2new[old], Point3D(*pos[old]))
    for b in m.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in old2new and e in old2new:
            rw.AddBond(old2new[s], old2new[e], b.GetBondType())
    seed = rw.GetMol()
    seed.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(seed)
    # rigid transform to BER
    p = seed.GetConformer().GetPositions()
    if rot:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        p = p @ q.T
    p = p - p.mean(0) + BER
    conf = seed.GetConformer()
    for i in range(seed.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*p[i]))
    return seed


def cut_bond_keep_atoms(m, bond_atoms, rot=True):
    """seed = ALL atoms of m, minus the bond between the two given atoms; translate to BER."""
    a0, a1 = bond_atoms
    rw = Chem.RWMol(Chem.Mol(m))
    b = rw.GetBondBetweenAtoms(a0, a1)
    assert b is not None
    rw.RemoveBond(a0, a1)
    seed = rw.GetMol()
    Chem.SanitizeMol(seed)
    p = seed.GetConformer().GetPositions()
    if rot:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        p = p @ q.T
    p = p - p.mean(0) + BER
    conf = seed.GetConformer()
    for i in range(seed.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*p[i]))
    return seed


def gap_c2_ph(seed_m):
    """min dist BT-C2 <-> Ph carbon (Ph = 6-ring aromatic carbons OUTSIDE the BT fragment)."""
    pos = seed_m.GetConformer().GetPositions()
    c2 = None
    for a in seed_m.GetAtoms():
        if a.GetIsAromatic() and a.IsInRingSize(5) and a.GetSymbol() == 'C' and a.GetDegree() == 3:
            c2 = a.GetIdx()
            break
    if c2 is None:
        return None
    frags = Chem.GetMolFrags(seed_m)
    c2_frag = next(f for f in frags if c2 in f)
    ph = [a.GetIdx() for a in seed_m.GetAtoms()
          if a.GetIsAromatic() and a.IsInRingSize(6) and a.GetIdx() not in c2_frag]
    if not ph:
        return None
    return float(min(np.linalg.norm(pos[c2] - pos[j]) for j in ph))


# ---- W1: MeBT + Ph = ETKDG(BT-CH2-Ph) with CH2-Ph bond cut (all 16 atoms kept) ----
m16 = ANALOG16
ch2 = None
for a in m16.GetAtoms():
    if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetTotalNumHs() == 2 and a.GetDegree() == 2:
        nbrs = set(n.GetIdx() for n in a.GetNeighbors())
        rings5 = set(i for i in nbrs if m16.GetAtomWithIdx(i).IsInRingSize(5))
        rings6 = set(i for i in nbrs if m16.GetAtomWithIdx(i).IsInRingSize(6) and not m16.GetAtomWithIdx(i).IsInRingSize(5))
        if rings5 and rings6:
            ph_c1 = list(rings6)[0]
            ch2 = a.GetIdx()
print('ANALOG16 bridge CH2 idx:', ch2)
confs16 = embed_n(m16, 24, 9000)
selected = []
for m in confs16:
    seed = cut_bond_keep_atoms(m, (ch2, ph_c1), rot=True)
    assert seed.GetNumAtoms() == 16
    smi = Chem.MolToSmiles(seed)
    assert smi.count('.') == 1, smi
    g = gap_c2_ph(seed)
    selected.append((g, seed))
selected.sort(key=lambda t: t[0])
print('W1 conformer gaps:', [round(g, 2) for g, _ in selected])
# pick diverse: min, q1, med, q3 (ideal t2 span ~2.8-3.6 for CH2->C2->b1->S->Ph)
picks = [selected[0], selected[len(selected) // 4], selected[len(selected) // 2], selected[3 * len(selected) // 4]]
for k, (g, s) in enumerate(picks):
    name = f'seed_w1_mebtph{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); s.SetProp('_Name', name[:-4]); w.write(s); w.close()
    print(f'  [W1] {name} gap={g:.2f} smi={Chem.MolToSmiles(s)}')

# ---- W0: BT + Ph from ETKDG(BT-CH2-Ph) with the CH2 atom dropped ----
m17 = ANALOG16
chain = [ch2]
print('W0 drop atoms:', chain)
confs17 = embed_n(m17, 24, 9500)
sel0 = []
for m in confs17:
    seed = cut_and_place(m, None, chain, rot=True)
    assert seed.GetNumAtoms() == 15
    smi = Chem.MolToSmiles(seed)
    assert smi.count('.') == 1, smi
    g = gap_c2_ph(seed)
    sel0.append((g, seed))
sel0.sort(key=lambda t: t[0])
print('W0 conformer gaps:', [round(g, 2) for g, _ in sel0])
picks0 = [sel0[len(sel0) // 4], sel0[len(sel0) // 2], sel0[3 * len(sel0) // 4], sel0[min(len(sel0) - 1, 7 * len(sel0) // 8)]]
for k, (g, s) in enumerate(picks0):
    name = f'seed_w0_btph{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); s.SetProp('_Name', name[:-4]); w.write(s); w.close()
    print(f'  [W0] {name} gap={g:.2f} smi={Chem.MolToSmiles(s)}')

# featurization check
from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
for f in sorted(os.listdir(OUT)):
    mt = mol_to_tensor(parse_sdf_file(path=os.path.join(OUT, f)), mode='molcraft')
    cls = sorted(MITA[int(k)] for k in mt['atom_feature'].long())
    exp16 = sorted([(6, True)] * 13 + [(7, True), (16, True), (6, False)])
    exp15 = sorted([(6, True)] * 13 + [(7, True), (16, True)])
    tag = 'W1-16' if cls == exp16 else ('W0-15' if cls == exp15 else 'UNEXPECTED')
    print(f'  {f}: n={len(cls)} {tag}')
