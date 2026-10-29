#!/usr/bin/env python3
"""Round-6 seeds: X18/X17 variants cut from the rDock-docked analog pose (BER102 R6).

The analog (BT-CH2-NH-C6H4-SH, 18 heavy + 2 polar H) is docked independently by rDock;
the best SCORE.INTER pose (polar H stripped positionally) gives a pocket-compatible
fragment arrangement. Cut the CH2-N bond (+ sep variants) -> inpainting seeds.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r6')
os.makedirs(OUT, exist_ok=True)

poses = []
for m in Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/x18analog_dock.sd'),
                            removeHs=False, sanitize=False):
    if m is None:
        continue
    poses.append((float(m.GetProp('SCORE.INTER')), m))
poses.sort(key=lambda t: t[0])
print('best 5 poses INTER:', [round(s, 2) for s, _ in poses[:5]])
best_s, best = poses[0]
# strip the 2 polar H positionally (they are the last 2 atoms)
assert best.GetNumAtoms() == 20
heavy = Chem.RWMol(best)
conf = best.GetConformer().GetPositions()
ids = [a.GetIdx() for a in best.GetAtoms() if a.GetSymbol() == 'H']
for i in sorted(ids, reverse=True):
    heavy.RemoveAtom(i)
analog = heavy.GetMol()
Chem.SanitizeMol(analog)
assert analog.GetNumAtoms() == 18
assert Chem.MolToSmiles(analog) == 'Sc1ccccc1NCc1nc2ccccc2s1'
print(f'docked analog: INTER={best_s:.2f} smiles={Chem.MolToSmiles(analog)} centroid={np.round(conf[:18].mean(0), 2)}')

# atom indices in the docked heavy mol
n_idx = ch2_idx = sh_idx = None
for a in analog.GetAtoms():
    if a.GetSymbol() == 'N' and not a.GetIsAromatic() and a.GetDegree() == 2:
        n_idx = a.GetIdx()
    if a.GetSymbol() == 'S' and not a.GetIsAromatic():
        sh_idx = a.GetIdx()
    if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetTotalNumHs() == 2 and a.GetDegree() == 2:
        ch2_idx = a.GetIdx()
print(f'N={n_idx} CH2={ch2_idx} SH={sh_idx}')


def build(drop_atoms, sep):
    pos = np.array(analog.GetConformer().GetPositions())
    adj = {i: set() for i in range(18)}
    for b in analog.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if {s, e} == {ch2_idx, n_idx}:
            continue
        adj[s].add(e); adj[e].add(s)
    frag = set()
    stack = [n_idx]
    while stack:
        c = stack.pop()
        if c in frag:
            continue
        frag.add(c)
        stack.extend(adj[c] - frag)
    axis = pos[n_idx] - pos[ch2_idx]
    axis /= np.linalg.norm(axis)
    drop = set(drop_atoms)
    keep = [i for i in range(18) if i not in drop]
    old2new = {o: n for n, o in enumerate(keep)}
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(keep))
    for o in keep:
        p = pos[o] + (axis * sep if o in frag else 0.0)
        a = Chem.Atom(analog.GetAtomWithIdx(o).GetSymbol())
        if analog.GetAtomWithIdx(o).GetIsAromatic():
            a.SetIsAromatic(True)
        rw.AddAtom(a)
        conf.SetAtomPosition(old2new[o], Point3D(*p))
    for b in analog.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in old2new and e in old2new and {s, e} != {ch2_idx, n_idx}:
            rw.AddBond(old2new[s], old2new[e], b.GetBondType())
    seed = rw.GetMol()
    seed.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(seed)
    return seed


for k, sep in enumerate([0.9, 1.0, 1.1, 1.2]):
    x18 = build([], sep)
    assert x18.GetNumAtoms() == 18 and Chem.MolToSmiles(x18).count('.') == 1
    name = f'seed_d18{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); x18.SetProp('_Name', name[:-4]); w.write(x18); w.close()
    print(f'[D18] {name} sep={sep} smi={Chem.MolToSmiles(x18)}')
x17a = build([n_idx], 1.0)
x17b = build([sh_idx], 1.0)
for seed, name in [(x17a, 'seed_d17a.sdf'), (x17b, 'seed_d17b.sdf')]:
    assert seed.GetNumAtoms() == 17
    w = Chem.SDWriter(os.path.join(OUT, name)); seed.SetProp('_Name', name[:-4]); w.write(seed); w.close()
    print(f'[D17] {name} smi={Chem.MolToSmiles(seed)}')

from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
for f in sorted(os.listdir(OUT)):
    mt = mol_to_tensor(parse_sdf_file(path=os.path.join(OUT, f)), mode='molcraft')
    cls = sorted(MITA[int(k)] for k in mt['atom_feature'].long())
    print(f'  {f}: n={len(cls)} {sorted((z, a) for z, a in set(cls))}')
