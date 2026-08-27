#!/usr/bin/env python3
"""t2 Round-2 seeds: 16-atom benzothiazole-CH2-benzene core.

(a) evo  : strip the CH2OCH3 decoration from Run-B model output
           COCc1ccc2nc(Cc3ccccc3)sc2c1 -> Cc1... core, keep model-generated coords
(b) etkdg: BT-CH2-Ph built from SMILES, ETKDG+MMFF, random rotations, translated to
           BER102 centroid (independent-coordinate chain)
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r2')
os.makedirs(OUT, exist_ok=True)

CORE = Chem.MolFromSmiles('c1ccc2sc(Cc3ccccc3)nc2c1')
assert CORE.GetNumAtoms() == 16
print('16-atom core:', Chem.MolToSmiles(CORE))

ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)

# ---- (a) evolution seed: strip decoration from model output ----
src = None
supp = Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/gen_runB/valid_mols.sdf'), removeHs=True)
for m in supp:
    if m is None:
        continue
    if Chem.MolToSmiles(m) == 'COCc1ccc2nc(Cc3ccccc3)sc2c1':
        src = m
        break
assert src is not None
core = CORE
match = src.GetSubstructMatch(core)   # graph isomorphism (same skeleton)
assert len(match) == 16
rw = Chem.RWMol(core)
conf = Chem.Conformer(16)
sp = src.GetConformer().GetPositions()
for new, old in enumerate(match):
    p = sp[old]
    conf.SetAtomPosition(new, Point3D(*p))
core2 = rw.GetMol()
core2.AddConformer(conf, assignId=True)
Chem.SanitizeMol(core2)
w = Chem.SDWriter(os.path.join(OUT, 'seed_btch2ph_evo.sdf'))
core2.SetProp('_Name', 'btch2ph_evo_from_runB')
w.write(core2); w.close()
c = core2.GetConformer().GetPositions()
print(f'evo seed centroid {np.round(c.mean(0), 2)} (from Run-B model output coords)')

# ---- (b) ETKDG independent seeds ----
rng = np.random.default_rng(2027)
for k in range(6):
    mh = Chem.AddHs(Chem.Mol(CORE))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 5000 + k
    ps.useSmallRingTorsions = True
    cid = AllChem.EmbedMolecule(mh, ps)
    assert cid == 0
    AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
    m2 = Chem.RemoveHs(mh)
    pos = m2.GetConformer().GetPositions()
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    pos = pos @ q.T
    pos = pos - pos.mean(0) + BER
    conf = m2.GetConformer()
    for i in range(m2.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*pos[i]))
    name = 'seed_btch2ph_etkdg.sdf' if k == 0 else f'seed_btch2ph_etkdg_{k+1}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name))
    m2.SetProp('_Name', name[:-4]); w.write(m2); w.close()
    print(f'  {name} centroid {np.round(pos.mean(0), 1)}')

# ---- featurization check ----
from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
for f in sorted(os.listdir(OUT)):
    mt = mol_to_tensor(parse_sdf_file(path=os.path.join(OUT, f)), mode='molcraft')
    cls = sorted(MITA[int(k)] for k in mt['atom_feature'].long())
    exp = sorted([(6, True)] * 13 + [(7, True), (16, True), (6, False)])
    print(f'  {f}: n={len(cls)} {"OK" if cls == exp else cls}')
