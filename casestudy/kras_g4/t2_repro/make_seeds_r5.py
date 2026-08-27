#!/usr/bin/env python3
"""t2 Round-5 seeds (X): near-complete t2 subgraphs with ETKDG-analog-cut geometry.

Analog = BT-CH2-NH-C6H4(o-SH)  (connected, real molecule) -> ETKDG conformers ->
cut the CH2-N bond -> fragments {MeBT, 2-aminothiophenol} with realistic relative geometry.
Variants (atom deletions from the same conformer, coordinates kept):
  X18 : MeBT + 2-aminothiophenol   (18 atoms, free 1 = b1 aromatic C)
  X17a: MeBT + thiophenol          (17 atoms, free 2 = b1 + N)
  X17b: MeBT + aniline             (17 atoms, free 2 = b1 + S)
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r5')
os.makedirs(OUT, exist_ok=True)

ANALOG = Chem.MolFromSmiles('Sc1ccccc1NCc1nc2ccccc2s1')   # BT-CH2-NH-Ph(o-SH), 17 atoms
assert ANALOG.GetNumAtoms() == 18
print('analog:', Chem.MolToSmiles(ANALOG), ANALOG.GetNumAtoms(), 'atoms')

ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)
rng = np.random.default_rng(2030)

# identify atoms: N (bridge NH), CH2, S(H)
n_idx = ch2_idx = sh_idx = None
for a in ANALOG.GetAtoms():
    if a.GetSymbol() == 'N' and not a.GetIsAromatic() and a.GetDegree() == 2:
        n_idx = a.GetIdx()
    if a.GetSymbol() == 'S' and not a.GetIsAromatic():
        sh_idx = a.GetIdx()
    if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetTotalNumHs() == 2 and a.GetDegree() == 2:
        ch2_idx = a.GetIdx()
print(f'N={n_idx} CH2={ch2_idx} SH={sh_idx}')


def build_seed(conf_m, drop_atoms, cut_pair, sep=0.0, rot=True):
    """cut + optional rigid translation of the fragment containing 'n_idx' along CH2->N axis by sep (A)."""
    drop = set(drop_atoms)
    keep = [i for i in range(conf_m.GetNumAtoms()) if i not in drop]
    pos = np.array(conf_m.GetConformer().GetPositions())
    # which kept atoms belong to the moving fragment (connected to n_idx side after cut)?
    import networkx as nx  # not guaranteed; do BFS manually
    adj = {i: set() for i in keep}
    for b in conf_m.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in adj and e in adj:
            if cut_pair and {s, e} == set(cut_pair):
                continue
            adj[s].add(e); adj[e].add(s)
    # fragment containing n_idx (if kept) else containing sh_idx side
    anchor = n_idx if n_idx in adj else None
    if anchor is None:
        for i in adj:
            if conf_m.GetAtomWithIdx(i).GetSymbol() == 'N':
                anchor = i
                break
    frag = set()
    if anchor is not None and sep != 0.0:
        stack = [anchor]
        while stack:
            cur = stack.pop()
            if cur in frag:
                continue
            frag.add(cur)
            stack.extend(adj[cur] - frag)
        axis = pos[n_idx] - pos[ch2_idx]
        axis = axis / np.linalg.norm(axis)
        for i in frag:
            pos[i] = pos[i] + axis * sep
    old2new = {o: n for n, o in enumerate(keep)}
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(keep))
    for o in keep:
        a = Chem.Atom(conf_m.GetAtomWithIdx(o).GetSymbol())
        if conf_m.GetAtomWithIdx(o).GetIsAromatic():
            a.SetIsAromatic(True)
        rw.AddAtom(a)
        conf.SetAtomPosition(old2new[o], Point3D(*pos[o]))
    for b in conf_m.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in old2new and e in old2new:
            if cut_pair and {s, e} == set(cut_pair):
                continue
            rw.AddBond(old2new[s], old2new[e], b.GetBondType())
    seed = rw.GetMol()
    seed.AddConformer(conf, assignId=True)
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


def score_geometry(conf_m):
    """closability: dist(CH2, N-bridge) small (b1 bonds both), S reachable.
    b1 sits between CH2 & N; needs S within ~2.9 of that midpoint."""
    pos = conf_m.GetConformer().GetPositions()
    d_ch2_n = np.linalg.norm(pos[ch2_idx] - pos[n_idx])
    mid = (pos[ch2_idx] + pos[n_idx]) / 2
    d_s_mid = np.linalg.norm(pos[sh_idx] - mid)
    return d_ch2_n, d_s_mid


confs = []
for k in range(40):
    mh = Chem.AddHs(Chem.Mol(ANALOG))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 11000 + k
    ps.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mh, ps) != 0:
        continue
    AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
    confs.append(Chem.RemoveHs(mh))
print(f'{len(confs)} conformers embedded')

scored = []
for m in confs:
    d1, d2 = score_geometry(m)
    scored.append((abs(d1 - 2.6) + abs(d2 - 1.6), d1, d2, m))   # ideal: ch2-n ~2.6 via b1, S ~1.6 from mid
scored.sort(key=lambda t: t[0])
print('top conformer metrics (ch2-n, S-mid):')
for s, d1, d2, m in scored[:8]:
    print(f'  score={s:.2f} ch2_n={d1:.2f} S_mid={d2:.2f}')

# separation search: translate the N-side fragment along CH2->N by sep
# target geometry: CH2..N ~2.5 (b1 bonds both), S within 1.7-2.1 of b1*
def sep_score(conf_m, sep):
    pos = np.array(conf_m.GetConformer().GetPositions())
    axis = pos[n_idx] - pos[ch2_idx]
    axis = axis / np.linalg.norm(axis)
    npos = pos[n_idx] + axis * sep
    spos = pos[sh_idx] + axis * sep
    d_ch2_n = np.linalg.norm(pos[ch2_idx] - npos)
    b1star = pos[ch2_idx] + 1.45 * (npos - pos[ch2_idx]) / np.linalg.norm(npos - pos[ch2_idx])
    d_s_b1 = np.linalg.norm(spos - b1star)
    return abs(d_ch2_n - 2.5) + abs(d_s_b1 - 1.85)

cands = []
for m in confs:
    for sep in np.arange(0.8, 1.5, 0.1):
        cands.append((sep_score(m, sep), m, round(float(sep), 2)))
cands.sort(key=lambda t: t[0])
print('best (conformer, sep):')
for s, m, sep in cands[:8]:
    pos = np.array(m.GetConformer().GetPositions())
    print(f'  score={s:.2f} sep={sep}')

best = []
for s, m, sep in cands:
    if any(m is mm for _, mm, _ss in best):
        continue
    best.append((s, m, sep))
    if len(best) >= 4:
        break

for k, (s, m, sep) in enumerate(best):
    # X18: cut CH2-N bond, keep all atoms, separate by sep
    x18 = build_seed(m, [], (ch2_idx, n_idx), sep=sep)
    assert x18.GetNumAtoms() == 18, x18.GetNumAtoms()
    smi = Chem.MolToSmiles(x18)
    assert smi.count('.') == 1, smi
    name = f'seed_x18{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); x18.SetProp('_Name', name[:-4]); w.write(x18); w.close()
    print(f'  [X18] {name} sep={sep} score={s:.2f} smi={smi}')
    # X17a: cut CH2-N, drop N, separate by sep (thiophenol side moves)
    x17a = build_seed(m, [n_idx], (ch2_idx, n_idx), sep=sep)
    assert x17a.GetNumAtoms() == 17
    name = f'seed_x17a{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); x17a.SetProp('_Name', name[:-4]); w.write(x17a); w.close()
    print(f'  [X17a] {name} sep={sep} smi={Chem.MolToSmiles(x17a)}')
    # X17b: cut CH2-N, drop S, separate by sep (aniline side moves)
    x17b = build_seed(m, [sh_idx], (ch2_idx, n_idx), sep=sep)
    assert x17b.GetNumAtoms() == 17
    name = f'seed_x17b{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); x17b.SetProp('_Name', name[:-4]); w.write(x17b); w.close()
    print(f'  [X17b] {name} sep={sep} smi={Chem.MolToSmiles(x17b)}')

from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
for f in sorted(os.listdir(OUT)):
    mt = mol_to_tensor(parse_sdf_file(path=os.path.join(OUT, f)), mode='molcraft')
    cls = sorted(MITA[int(k)] for k in mt['atom_feature'].long())
    n = len(cls)
    nC = sum(1 for z, a in cls if z == 6 and a)
    nN = sum(1 for z, a in cls if z == 7)
    nS = sum(1 for z, a in cls if z == 16)
    ok = (n, nC, nN, nS) in [(18, 13, 2, 2), (17, 13, 1, 2), (17, 13, 2, 1)]
    print(f'  {f}: n={n} C*={nC} N={nN} S={nS} {"OK" if ok else "UNEXPECTED " + str(cls)}')
