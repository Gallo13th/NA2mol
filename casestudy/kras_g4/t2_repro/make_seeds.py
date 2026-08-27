#!/usr/bin/env python3
"""t2 Phase2: build inpainting seeds (9-atom benzothiazole S-b / 10-atom 2-methylbenzothiazole S-a).

Seed legality (user-approved):
  (a) etkdg  : SMILES topology + independent ETKDG coords, random rotation, translated
               to the BER102 pocket centroid (no target-pose information used)
  (b) libsub : benzimidazole fused core extracted from BER102 library molecules at their
               model-native pocket coordinates, then N->S element replacement (isoelectronic
               replacement path, Phase2A)
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
SEED_DIR = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds')
os.makedirs(SEED_DIR, exist_ok=True)

ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)
print('BER102 centroid:', np.round(BER, 2))

BENZOTHIAZOLE = Chem.MolFromSmiles('c1ccc2scnc2c1')      # 9 atoms
MEBT = Chem.MolFromSmiles('Cc1nc2ccccc2s1')               # 2-methylbenzothiazole, 10 atoms
assert BENZOTHIAZOLE.GetNumAtoms() == 9 and MEBT.GetNumAtoms() == 10
print('benzothiazole:', Chem.MolToSmiles(BENZOTHIAZOLE), '| 2-methyl:', Chem.MolToSmiles(MEBT))


def embed_rotate_translate(mol, seed, center, n_rots=1, rng=None):
    """ETKDG+MMFF embed, random rigid rotations, translate to center. Returns list of mols."""
    out = []
    for k in range(n_rots):
        mh = Chem.AddHs(Chem.Mol(mol))
        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed + k
        ps.useSmallRingTorsions = True
        cid = AllChem.EmbedMolecule(mh, ps)
        assert cid == 0, cid
        AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
        m2 = Chem.RemoveHs(mh)
        pos = m2.GetConformer().GetPositions()
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        pos = pos @ q.T
        pos = pos - pos.mean(0) + center
        conf = m2.GetConformer()
        for i in range(m2.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(*pos[i]))
        out.append(m2)
    return out


def write(mol, name):
    p = os.path.join(SEED_DIR, name)
    w = Chem.SDWriter(p); mol.SetProp('_Name', name[:-4]); w.write(mol); w.close()
    print(f'  {name}: {Chem.MolToSmiles(mol)} centroid {np.round(mol.GetConformer().GetPositions().mean(0), 1)}')
    return p


rng = np.random.default_rng(2026)

print('\n=== (a) ETKDG independent-coordinate seeds ===')
n = 0
for m, base, seed0 in [(BENZOTHIAZOLE, 'seed_bt_etkdg', 3000), (MEBT, 'seed_mebt_etkdg', 4000)]:
    for k, mm in enumerate(embed_rotate_translate(m, seed0, BER, n_rots=4, rng=rng)):
        write(mm, f'{base}{"" if k == 0 else f"_{k+1}"}.sdf')
        n += 1

print('\n=== (b) library benzimidazole-core -> N->S seeds (BER102 native coords) ===')
BIM_CORE = Chem.MolFromSmarts('c1ccc2nc[nX2]c2c1')
BIM_CORE2 = Chem.MolFromSmarts('c1ccc2[nX2]cnc2c1')


def extract_core(mol, match, nh_old):
    """extract fused core with coords; nh_old = parent atom idx of the pyrrolic [nH]."""
    rw = Chem.RWMol()
    old2new = {}
    conf = Chem.Conformer(len(match))
    src = mol.GetConformer()
    for new, old in enumerate(match):
        a = Chem.Atom(mol.GetAtomWithIdx(old).GetSymbol())
        if old == nh_old:
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(1)
            a.SetNoImplicit(False)
        rw.AddAtom(a)
        old2new[old] = new
        p = src.GetAtomPosition(old)
        conf.SetAtomPosition(new, Point3D(p.x, p.y, p.z))
    for b in mol.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in old2new and e in old2new:
            rw.AddBond(old2new[s], old2new[e], b.GetBondType())
    out = rw.GetMol()
    out.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(out)
    return out


def nh_to_s(mol):
    """replace the pyrrolic [nH] (S-analog site: between fusion-C and C2) by aromatic S"""
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetTotalNumHs() == 1:
            a.SetAtomicNum(16)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
            return True
    return False


supp = Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_gen2k/valid_mols.sdf'), removeHs=True)
made = 0
want = ['O=C1CCN(c2ccc3[nH]c(-c4ccc(O)cc4)nc3c2)C1',        # lib#473 d=0.8
        'N[C@H](C(=O)O)c1ccc2nc(-c3ccc(Br)cc3)[nH]c2c1',    # lib#43  d=1.2
        'COc1ccc(-c2nc3cc(C(=O)O)ccc3[nH]2)c(C)c1']          # lib#111 d=2.0
for m in supp:
    if m is None:
        continue
    smi = Chem.MolToSmiles(m)
    if smi not in want:
        continue
    ok = False
    for patt in (BIM_CORE, BIM_CORE2):
        for mt in m.GetSubstructMatches(patt):
            # locate pyrrolic [nH] among the match (has 1 H, 2 ring nbrs)
            nh_old = None
            for old in mt:
                a = m.GetAtomWithIdx(old)
                if a.GetSymbol() == 'N' and a.GetTotalNumHs() == 1:
                    ring_nbrs = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in set(mt)]
                    if len(ring_nbrs) == 2:
                        nh_old = old
                        break
            if nh_old is None:
                continue
            core = extract_core(m, mt, nh_old)
            if nh_to_s(core):
                Chem.SanitizeMol(core)
                s = Chem.MolToSmiles(core)
                assert s == 'c1ccc2scnc2c1', s
                write(core, f'seed_bt_lib{made + 1}.sdf')
                made += 1
                ok = True
                break
        if ok:
            break
    if made >= 3:
        break
print(f'lib N->S seeds: {made}')

# ---- featurization sanity: every seed parses to 7*C* + N* + S* classes ----
from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
print('\n=== featurization check (model classes) ===')
for f in sorted(os.listdir(SEED_DIR)):
    p = os.path.join(SEED_DIR, f)
    mt = mol_to_tensor(parse_sdf_file(path=p), mode='molcraft')
    cls = [MITA[int(k)] for k in mt['atom_feature'].long()]
    syms = sorted((z, a) for z, a in cls)
    exp_bt = sorted([(6, True)] * 7 + [(7, True), (16, True)])
    exp_mebt = sorted([(6, True)] * 7 + [(7, True), (16, True), (6, False)])
    tag = 'BT' if syms == exp_bt else ('MeBT' if syms == exp_mebt else 'UNEXPECTED')
    print(f'  {f}: n={len(cls)} classes={tag} {syms if tag == "UNEXPECTED" else ""}')
print('\nSEEDS READY:', len(os.listdir(SEED_DIR)))
