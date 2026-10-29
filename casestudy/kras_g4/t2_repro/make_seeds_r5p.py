#!/usr/bin/env python3
"""X18-P seeds: conformer selection where S sits BETWEEN the CH3 and N anchors
(blocks the model's preferred chain-insertion spot, favors the 3-anchor thiazole closure)."""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r5p')
os.makedirs(OUT, exist_ok=True)
ANALOG = Chem.MolFromSmiles('Sc1ccccc1NCc1nc2ccccc2s1')
ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)
rng = np.random.default_rng(2031)

n_idx = ch2_idx = sh_idx = None
for a in ANALOG.GetAtoms():
    if a.GetSymbol() == 'N' and not a.GetIsAromatic() and a.GetDegree() == 2:
        n_idx = a.GetIdx()
    if a.GetSymbol() == 'S' and not a.GetIsAromatic():
        sh_idx = a.GetIdx()
    if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetTotalNumHs() == 2 and a.GetDegree() == 2:
        ch2_idx = a.GetIdx()

def geom(m, sep):
    pos = np.array(m.GetConformer().GetPositions())
    axis = pos[n_idx] - pos[ch2_idx]; axis /= np.linalg.norm(axis)
    npos = pos[n_idx] + axis * sep
    spos = pos[sh_idx] + axis * sep
    # b1 ideal spot: centroid of triangle (CH2, N', S) scaled to ~1.5 from each
    tri = np.array([pos[ch2_idx], npos, spos])
    b1star = tri.mean(0)
    d = [np.linalg.norm(b1star - p) for p in tri]
    # S projection onto the CH2->N' axis (0=at CH2, 1=at N')
    t = np.dot(spos - pos[ch2_idx], npos - pos[ch2_idx]) / np.linalg.norm(npos - pos[ch2_idx])**2
    return d, t, pos, spos, npos

confs = []
for k in range(60):
    mh = Chem.AddHs(Chem.Mol(ANALOG))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 12000 + k
    ps.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mh, ps) != 0:
        continue
    AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
    confs.append(Chem.RemoveHs(mh))

cands = []
for m in confs:
    for sep in np.arange(0.8, 1.5, 0.05):
        d, t, *_ = geom(m, sep)
        # criteria: all three anchors ~equidistant 1.4-2.0 from centroid; S between (t in 0.3-0.7)
        score = sum(abs(x - 1.55) for x in d)
        if 0.25 <= t <= 0.75:
            score -= 1.0   # bonus: S on the axis
        cands.append((score, round(float(sep), 2), round(float(t), 2), m))
cands.sort(key=lambda x: x[0])
print('best (score, sep, S-axis-frac):')
for s, sep, t, m in cands[:10]:
    print(f'  {s:.2f} sep={sep} t={t}')

used = []
for s, sep, t, m in cands:
    if any(m is mm for mm in used):
        continue
    used.append(m)
    if len(used) > 3:
        break

def build(m, sep):
    pos = np.array(m.GetConformer().GetPositions())
    axis = pos[n_idx] - pos[ch2_idx]; axis /= np.linalg.norm(axis)
    npos = pos[n_idx] + axis * sep
    pos[n_idx] = npos
    pos[sh_idx] += axis * sep
    # move ALL atoms on the N-side fragment (BFS from n_idx, cut CH2-N bond)
    adj = {i: set() for i in range(m.GetNumAtoms())}
    for b in m.GetBonds():
        s2, e2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if {s2, e2} == {ch2_idx, n_idx}:
            continue
        adj[s2].add(e2); adj[e2].add(s2)
    frag = set()
    stack = [n_idx]
    while stack:
        c = stack.pop()
        if c in frag: continue
        frag.add(c); stack.extend(adj[c] - frag)
    base = np.array(m.GetConformer().GetPositions())
    for i in frag:
        pos[i] = base[i] + axis * sep
    rw = Chem.RWMol()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i in range(m.GetNumAtoms()):
        a = Chem.Atom(m.GetAtomWithIdx(i).GetSymbol())
        if m.GetAtomWithIdx(i).GetIsAromatic():
            a.SetIsAromatic(True)
        rw.AddAtom(a)
        conf.SetAtomPosition(i, Point3D(*pos[i]))
    for b in m.GetBonds():
        s2, e2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if {s2, e2} == {ch2_idx, n_idx}:
            continue
        rw.AddBond(s2, e2, b.GetBondType())
    seed = rw.GetMol()
    seed.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(seed)
    p = seed.GetConformer().GetPositions()
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    p = p @ q.T
    p = p - p.mean(0) + BER
    conf = seed.GetConformer()
    for i in range(seed.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*p[i]))
    return seed

for k, m in enumerate(used):
    # recover its best sep
    best = min((c for c in cands if c[3] is m), key=lambda x: x[0])
    seed = build(m, best[1])
    smi = Chem.MolToSmiles(seed)
    assert seed.GetNumAtoms() == 18 and smi.count('.') == 1, smi
    name = f'seed_x18p{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); seed.SetProp('_Name', name[:-4]); w.write(seed); w.close()
    print(f'[X18p] {name} sep={best[1]} t={best[2]} score={best[0]:.2f} smi={smi}')
