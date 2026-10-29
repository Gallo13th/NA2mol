#!/usr/bin/env python3
"""t2 Round-3 seeds: TOPOLOGY-CORRECT (t2 subgraph) seeds.

t2 = BT-CH2-BT'  (CH2 bonds the C2 of BOTH benzothiazoles; C2 is exocyclic to both benzenes).
Correct subgraph seeds:
  V1 (15 atom, free 4): BT + Ph disconnected  (delete the bridge CH2 from model outputs
      containing BT-CH2-Ph; keeps model-native BT<->Ph relative geometry)
  V2 (16 atom, free 3): BT + Ph + C(H4)  (from BT-CH(CH3)-Ph outputs: strip CH3 + bridge bonds,
      keep the bridge-C coords as an isolated CH4 seed atom)
  V0 (ETKDG, free 4): BT embedded at BER centroid + Ph placed at 3.0-4.5 A from BT-C2
      in random directions (fully independent coords)
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

PROJ = '/home/data/users/gaoletian/rna2smol'
OUT = os.path.join(PROJ, 'casestudy/kras_g4/t2_repro/inpaint_seeds_r3')
os.makedirs(OUT, exist_ok=True)

BT = Chem.MolFromSmiles('c1nc2ccccc2s1')      # benzothiazole, 9
PH = Chem.MolFromSmiles('c1ccccc1')           # benzene, 6
T2 = Chem.MolFromSmiles('c1ccc2sc(Cc3nc4ccccc4s3)nc2c1')
# sanity: BT+Ph is a subgraph of t2? (t2 minus CH2,N,S,C2' = BT + benzene)
from rdkit.Chem import rdFMCS
mcs = rdFMCS.FindMCS([T2, Chem.MolFromSmiles('c1nc2ccccc2s1.c1ccccc1')])
print('MCS(BT.Ph, t2):', mcs.numAtoms, 'atoms:', mcs.smartsString[:80])

ref = next(Chem.SDMolSupplier(os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'), removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)

BT_SMARTS = Chem.MolFromSmarts('c1nc2ccccc2s1')
PH_SMARTS = Chem.MolFromSmarts('c1ccccc1')
BRIDGE_CH2 = Chem.MolFromSmarts('[s;r5]~[c;r5](~[n;r5])~[#6H2]~[c;r6]1ccccc1')   # BT-C2-CH2-Ph
BRIDGE_CH = Chem.MolFromSmarts('[s;r5]~[c;r5](~[n;r5])~[#6H1](~[#6H3])~[c;r6]1ccccc1')  # BT-C2-CH(CH3)-Ph


def extract_pair(mol, bt_match, ph_match):
    keep = list(bt_match) + list(ph_match)
    keep_set = set(keep)
    rw = Chem.RWMol()
    old2new = {}
    conf = Chem.Conformer(len(keep))
    src = mol.GetConformer().GetPositions()
    for new, old in enumerate(keep):
        a = Chem.Atom(mol.GetAtomWithIdx(old).GetSymbol())
        rw.AddAtom(a)
        old2new[old] = new
        conf.SetAtomPosition(new, Point3D(*src[old]))
    for b in mol.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in keep_set and e in keep_set and (s in set(bt_match)) == (e in set(bt_match)):
            rw.AddBond(old2new[s], old2new[e], b.GetBondType())
    out = rw.GetMol()
    out.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(out)
    return out, old2new


def c2_to_ph_gap(mol, bt_match, ph_match):
    """min distance from BT C2 (the CH2-bonded ring carbon) to any phenyl carbon."""
    pos = mol.GetConformer().GetPositions()
    # find C2: aromatic C in bt_match bonded to an atom outside bt_match via CH2... simpler:
    # C2 = the bt atom whose distance to phenyl centroid is smallest among 5-ring atoms
    ph_c = pos[list(ph_match)].mean(0)
    best = None
    for i in bt_match:
        a = mol.GetAtomWithIdx(i)
        if a.GetIsAromatic() and a.IsInRingSize(5):
            d = min(np.linalg.norm(pos[i] - pos[j]) for j in ph_match)
            if best is None or d < best[1]:
                best = (i, d)
    return best


# ---------- scan outputs ----------
BRIDGE_CH2 = Chem.MolFromSmarts('[s;r5]~[c;r5](~[n;r5])~[#6H2]~[c;r6]1ccccc1')          # BT-C2-CH2-Ph
BRIDGE_CH = Chem.MolFromSmarts('[s;r5]~[c;r5](~[n;r5])~[#6H1](~[#6H3])~[c;r6]1ccccc1')   # BT-C2-CH(CH3)-Ph
v1_src, v2_src = [], []
for run in ['A', 'B', 'C', 'D', 'E', 'S']:
    f = f'{PROJ}/casestudy/kras_g4/t2_repro/gen_run{run}/valid_mols.sdf'
    if not os.path.exists(f):
        continue
    for m in Chem.SDMolSupplier(f, removeHs=True):
        if m is None or m.GetNumAtoms() > 23:
            continue
        for mt in m.GetSubstructMatches(BRIDGE_CH2):
            bts = m.GetSubstructMatches(BT_SMARTS)
            phs = m.GetSubstructMatches(PH_SMARTS)
            if not bts or not phs:
                break
            ph_first = mt[4]
            for bt in bts:
                if mt[1] not in bt:
                    continue
                for ph in phs:
                    if ph_first in ph and not (set(bt) & set(ph)):
                        i2, gap = c2_to_ph_gap(m, bt, ph)
                        if 2.3 <= gap <= 4.6 and len(v1_src) < 4:
                            v1_src.append((gap, run, m, bt, ph))
                        break
                break
            break
        for mt in m.GetSubstructMatches(BRIDGE_CH):
            bts = m.GetSubstructMatches(BT_SMARTS)
            phs = m.GetSubstructMatches(PH_SMARTS)
            if not bts or not phs:
                break
            ph_first = mt[5]
            for bt in bts:
                if mt[1] not in bt:
                    continue
                for ph in phs:
                    if ph_first in ph and not (set(bt) & set(ph)):
                        i2, gap = c2_to_ph_gap(m, bt, ph)
                        if 2.4 <= gap <= 3.4 and len(v2_src) < 4:
                            v2_src.append((gap, run, m, bt, ph))
                        break
                break
            break
        if len(v1_src) >= 4 and len(v2_src) >= 4:
            break
    if len(v1_src) >= 4 and len(v2_src) >= 4:
        break

print(f'\nV1 sources (BT+Ph linked, gap 3.0-4.6): {len(v1_src)}')
for gap, run, m, bt, ph in v1_src:
    print(f'  gap={gap:.2f} [{run}] {Chem.MolToSmiles(m)[:75]}')
print(f'V2 sources (BT-CHx-Ph, gap 2.4-3.2): {len(v2_src)}')
for gap, run, m, bt, ph in v2_src:
    print(f'  gap={gap:.2f} [{run}] {Chem.MolToSmiles(m)[:75]}')

rng = np.random.default_rng(2028)
n_written = 0

# ---------- V1 seeds: BT + Ph (15 atoms, disconnected) ----------
for k, (gap, run, m, bt, ph) in enumerate(v1_src[:4]):
    pair, _ = extract_pair(m, list(bt), list(ph))
    assert pair.GetNumAtoms() == 15
    smi = Chem.MolToSmiles(pair)
    assert smi in ('c1ccccc1.c1ccc2scnc2c1', 'c1ccc2scnc2c1.c1ccccc1'), smi
    name = f'seed_v1_btph{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); pair.SetProp('_Name', name[:-4]); w.write(pair); w.close()
    n_written += 1
    print(f'  [V1] {name} gap={gap:.2f} from [{run}]')

# ---------- V2 seeds: BT + Ph + C(H4) at old bridge position (16 atoms, 3 fragments) ----------
for k, (gap, run, m, bt, ph) in enumerate(v2_src[:4]):
    # bridge C = the sp3 C linking BT-C2 and Ph
    bt_set, ph_set = set(bt), set(ph)
    bridge = None
    for b in m.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (s in bt_set and e in ph_set) or (e in bt_set and s in ph_set):
            a_s = m.GetAtomWithIdx(s); a_e = m.GetAtomWithIdx(e)
            if not a_s.GetIsAromatic() and not a_e.GetIsAromatic():
                pass  # direct BT-Ph bond, no bridge atom; skip this source
    # find sp3 CH between: BT-C2 -> CHx -> Ph
    for a in m.GetAtoms():
        if a.GetSymbol() != 'C' or a.GetIsAromatic():
            continue
        nbrs = set(n.GetIdx() for n in a.GetNeighbors())
        if nbrs & bt_set and nbrs & ph_set:
            bridge = a.GetIdx()
            break
    if bridge is None:
        continue
    keep = list(bt) + list(ph) + [bridge]
    pair, _ = extract_pair(m, list(bt) + [bridge], list(ph))
    # remove all bonds of the bridge C (it becomes isolated CH4 seed)
    rw = Chem.RWMol(pair)
    bidx = len(bt)
    for b in list(rw.GetBonds()):
        if b.GetBeginAtomIdx() == bidx or b.GetEndAtomIdx() == bidx:
            rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    seed_m = rw.GetMol()
    Chem.SanitizeMol(seed_m)
    assert seed_m.GetNumAtoms() == 16
    smi = Chem.MolToSmiles(seed_m)
    assert smi.count('.') == 2 and smi.startswith('C.') and 'c1ccccc1' in smi and 'scnc' in smi, smi
    name = f'seed_v2_btphc{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); seed_m.SetProp('_Name', name[:-4]); w.write(seed_m); w.close()
    n_written += 1
    print(f'  [V2] {name} gap={gap:.2f} from [{run}] smi={smi}')

# ---------- V0 seeds: ETKDG BT at BER + Ph at 3.2-4.2 A random directions ----------
for k in range(4):
    mh = Chem.AddHs(Chem.Mol(BT))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 7000 + k
    AllChem.EmbedMolecule(mh, ps)
    AllChem.MMFFOptimizeMolecule(mh)
    bt3 = Chem.RemoveHs(mh)
    pos = bt3.GetConformer().GetPositions()
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    pos = pos @ q.T
    pos = pos - pos.mean(0) + BER
    conf = bt3.GetConformer()
    for i in range(bt3.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*pos[i]))
    # benzene at d from C2 (atom idx 0 of 'c1nc2ccccc2s1' is C2)
    ph_mh = Chem.AddHs(Chem.Mol(PH))
    ps2 = AllChem.ETKDGv3(); ps2.randomSeed = 8000 + k
    AllChem.EmbedMolecule(ph_mh, ps2)
    ph3 = Chem.RemoveHs(ph_mh)
    ppos = ph3.GetConformer().GetPositions()
    q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q2) < 0:
        q2[:, 0] *= -1
    ppos = ppos @ q2.T
    d = rng.uniform(3.2, 4.2)
    dirv = rng.normal(size=3)
    dirv /= np.linalg.norm(dirv)
    ppos = ppos - ppos.mean(0) + pos[0] + dirv * d
    pconf = ph3.GetConformer()
    for i in range(ph3.GetNumAtoms()):
        pconf.SetAtomPosition(i, Point3D(*ppos[i]))
    # combine into one SDF (2 fragments)
    rw = Chem.RWMol(Chem.CombineMols(bt3, ph3))
    seed_m = rw.GetMol()
    Chem.SanitizeMol(seed_m)
    assert seed_m.GetNumAtoms() == 15
    name = f'seed_v0_btph_etkdg{"" if k == 0 else f"_{k+1}"}.sdf'
    w = Chem.SDWriter(os.path.join(OUT, name)); seed_m.SetProp('_Name', name[:-4]); w.write(seed_m); w.close()
    n_written += 1
    print(f'  [V0] {name} d={d:.2f}')

print(f'\nwritten {n_written} seeds to {OUT}')

# ---------- featurization check ----------
from util.ligand_code_util import parse_sdf_file
from dataloader.featurizer import mol_to_tensor
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
for f in sorted(os.listdir(OUT)):
    mt = mol_to_tensor(parse_sdf_file(path=os.path.join(OUT, f)), mode='molcraft')
    cls = sorted(MITA[int(k)] for k in mt['atom_feature'].long())
    exp15 = sorted([(6, True)] * 12 + [(7, True), (16, True)])           # 6+6 arom C + N + S = 14? no:
    # BT = 7 arom C + N + S; Ph = 6 arom C -> 13 C* + N* + S*
    exp15 = sorted([(6, True)] * 13 + [(7, True), (16, True)])
    exp16 = exp15 + [(6, False)]
    tag = 'V1/V0-15' if cls == exp15 else ('V2-16' if cls == exp16 else 'UNEXPECTED')
    print(f'  {f}: n={len(cls)} {tag if tag != "UNEXPECTED" else cls}')
