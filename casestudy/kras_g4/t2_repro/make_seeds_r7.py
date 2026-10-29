#!/usr/bin/env python3
"""Round-7 seeds (Y): TWO benzothiazoles, C2...C2 ~2.5 A apart, free = 1 (CH2) -> t2.

Geometry chain (fully independent of t2's own conformation):
  (a) evo : model output BT-CH2-Ph (run B, model-native pose): delete the bridge CH2,
            align a second benzothiazole onto the Ph so its C2 lands at the old ipso
            position (ring superposition, the sanctioned 'align' move), drop the Ph.
  (b) etkdg: fresh ETKDG(BT-CH2-Ph) -> same construction.
Result: seed = BT + BT2 (18 atoms, 2 fragments), BT1-C2...BT2-C2 ~= old C2...ipso ~= 2.5 A.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r7')
os.makedirs(OUT, exist_ok=True)

BT = Chem.MolFromSmiles('c1ccc2scnc2c1')       # benzothiazole (RDKit canonical form)


def kabsch(P, Q):
    """rotation+translation mapping P onto Q (Nx3)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - P.mean(0) @ R.T
    return R, t


def get_bt_atom_map(m):
    """c2 = 5-ring carbon bonded to BOTH heteroatoms (S and N); fusions bond one each."""
    ring5 = None
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) == 5 and sum(1 for i in ring if m.GetAtomWithIdx(i).GetSymbol() in ('N', 'S')) == 2:
            ring5 = set(ring)
    c2 = None
    fus = []
    for i in ring5:
        a = m.GetAtomWithIdx(i)
        if a.GetSymbol() != 'C':
            continue
        het_nbrs = [n for n in a.GetNeighbors() if n.GetIdx() in ring5 and n.GetSymbol() in ('N', 'S')]
        if len(het_nbrs) == 2:
            c2 = i
        elif len(het_nbrs) == 1:
            fus.append(i)
    ring6 = None
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) == 6 and set(fus) & set(ring):
            ring6 = set(ring)
    return c2, fus, ring6


def build_btbt(source16, tag):
    """source16 = BT-CH2-Ph (16 atoms, with conformer). Build BT+BT2 seed:
    BT1 = source BT core; BT2 = fresh benzothiazole with its C2 pinned at the old Ph-ipso,
    ring center along -(ipso->CH2) direction, plane normal aligned to BT1's ring normal."""
    c2a, fusa, ring6a = get_bt_atom_map(source16)
    pos = source16.GetConformer().GetPositions()
    ch2 = None
    for a in source16.GetAtoms():
        if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetDegree() == 2 and a.GetTotalNumHs() == 2:
            nbr = set(n.GetIdx() for n in a.GetNeighbors())
            if any(source16.GetAtomWithIdx(i).IsInRingSize(5) for i in nbr):
                ch2 = a.GetIdx()
    ipso = None
    for n in source16.GetAtomWithIdx(ch2).GetNeighbors():
        if n.GetIdx() != c2a and n.GetIsAromatic():
            ipso = n.GetIdx()
    bt_match = source16.GetSubstructMatch(Chem.MolFromSmarts('c1ccc2scnc2c1')) or \
        source16.GetSubstructMatch(Chem.MolFromSmarts('c1ccc2nc(C)sc2c1'))
    assert bt_match and len(bt_match) == 9
    keep1 = list(bt_match)
    assert c2a in keep1

    # BT1 plane normal (SVD)
    p1 = pos[list(keep1)]
    _, _, vt1 = np.linalg.svd(p1 - p1.mean(0), full_matrices=False)
    n1 = vt1[2]
    w = pos[ch2] - pos[ipso]
    w_hat = w / np.linalg.norm(w)          # C2->CH2 bond direction for the new BT2

    m2 = Chem.AddHs(Chem.Mol(BT))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 77
    AllChem.EmbedMolecule(m2, ps)
    m2 = Chem.RemoveHs(m2)
    c2b, fusb, ring6b = get_bt_atom_map(m2)
    p2 = m2.GetConformer().GetPositions()
    c2_pos = p2[c2b]
    center2 = p2.mean(0)
    out_dir = center2 - c2_pos
    out_hat = out_dir / np.linalg.norm(out_dir)     # C2 -> ring center
    _, _, vt2 = np.linalg.svd(p2 - center2, full_matrices=False)
    n2 = vt2[2]

    # rotation R: out_hat -> -w_hat ; n2 -> n1 (then Gram-Schmidt)
    def rot_between(a, b):
        a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
        v = np.cross(a, b); c = float(np.dot(a, b))
        if c > 1 - 1e-9:
            return np.eye(3)
        if c < -1 + 1e-9:
            ax = np.cross(a, [1, 0, 0])
            if np.linalg.norm(ax) < 1e-6:
                ax = np.cross(a, [0, 1, 0])
            ax /= np.linalg.norm(ax)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            return np.eye(3) + 2 * K @ K
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + K + K @ K * (1 / (1 + c))

    R1 = rot_between(out_hat, -w_hat)
    n2r = n2 @ R1.T
    # rotate about w_hat to align n2r with n1
    axis = w_hat
    n_perp = n2r - np.dot(n2r, axis) * axis
    n1_perp = n1 - np.dot(n1, axis) * axis
    R2 = rot_between(n_perp, n1_perp)
    # apply R2 about origin then R1
    p2r = (R2 @ p2.T).T
    p2r = (R1 @ p2r.T).T
    # pin C2 at ipso
    p2r = p2r + (pos[ipso] - p2r[c2b])

    # assemble seed
    rw = Chem.RWMol()
    conf = Chem.Conformer(18)
    for o in keep1:
        a = Chem.Atom(source16.GetAtomWithIdx(o).GetSymbol())
        if source16.GetAtomWithIdx(o).GetIsAromatic():
            a.SetIsAromatic(True)
        rw.AddAtom(a)
        conf.SetAtomPosition(len(rw.GetAtoms()) - 1, Point3D(*pos[o]))
    o2n1 = {o: n for n, o in enumerate(keep1)}
    for b in source16.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in o2n1 and e in o2n1:
            rw.AddBond(o2n1[s], o2n1[e], b.GetBondType())
    for i2 in range(9):
        a = Chem.Atom(m2.GetAtomWithIdx(i2).GetSymbol())
        if m2.GetAtomWithIdx(i2).GetIsAromatic():
            a.SetIsAromatic(True)
        rw.AddAtom(a)
        conf.SetAtomPosition(9 + i2, Point3D(*p2r[i2]))
    for b in m2.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rw.AddBond(9 + s, 9 + e, b.GetBondType())
    seed = rw.GetMol()
    seed.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(seed)
    assert seed.GetNumAtoms() == 18
    smi = Chem.MolToSmiles(seed)
    assert smi.count('.') == 1 and smi.count('scnc') == 2, smi
    d = np.linalg.norm(pos[c2a] - p2r[c2b])
    clash = min(np.linalg.norm(pos[list(keep1)][:, None] - p2r[None], axis=-1).min(), 99)
    print(f'  [{tag}] C2...C2 = {d:.2f} A  min-clash = {clash:.2f} A  smi ok')
    return seed, d


seeds = []
# (a) evo sources: model outputs BT-CH2-Ph
for smi_src in ['COCc1ccc2nc(Cc3ccccc3)sc2c1', 'COc1ccc2sc(Cc3ccccc3)nc2c1',
                'COc1cc(Cl)c2sc(Cc3ccccc3)nc2c1']:
    m = Chem.MolFromSmiles(smi_src)
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = 21
    AllChem.EmbedMolecule(mh, ps)
    AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
    m3 = Chem.RemoveHs(mh)
    s, d = build_btbt(m3, 'etkdg-of-model-output')
    seeds.append((f'seed_y_evo{len(seeds)+1}.sdf', s, d))

# (b) pure ETKDG of BT-CH2-Ph with different seeds
for k in range(4):
    m = Chem.MolFromSmiles('c1ccc2sc(Cc3ccccc3)nc2c1')
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = 31 + k
    ps.useSmallRingTorsions = True
    AllChem.EmbedMolecule(mh, ps)
    AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
    m3 = Chem.RemoveHs(mh)
    s, d = build_btbt(m3, f'etkdg{k}')
    seeds.append((f'seed_y_etkdg{"" if k == 0 else f"_{k+1}"}.sdf', s, d))

# place at BER centroid (rigid)
ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)
rng = np.random.default_rng(2032)
for name, s, d in seeds:
    p = s.GetConformer().GetPositions()
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    p = p @ q.T
    p = p - p.mean(0) + BER
    conf = s.GetConformer()
    for i in range(18):
        conf.SetAtomPosition(i, Point3D(*p[i]))
    w = Chem.SDWriter(os.path.join(OUT, name)); s.SetProp('_Name', name[:-4]); w.write(s); w.close()

from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
for f in sorted(os.listdir(OUT)):
    mt = mol_to_tensor(parse_sdf_file(path=os.path.join(OUT, f)), mode='molcraft')
    cls = sorted(MITA[int(k)] for k in mt['atom_feature'].long())
    exp = sorted([(6, True)] * 14 + [(7, True), (7, True), (16, True), (16, True)])
    print(f'  {f}: n={len(cls)} {"OK" if cls == exp else "UNEXPECTED"}')
