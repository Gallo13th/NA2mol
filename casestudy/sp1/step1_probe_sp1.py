#!/usr/bin/env python3
"""Step 1: element probe at the sp1 pocket using the flavone-skeleton geometry.

Protocol (hyperparameters & formulas mirror step3_probe_ber102.py exactly):
  - model: onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_t111429 ckpt
  - pocket: sp1_recp_pocket.pdb (654 atoms, general feat)
  - skeleton: 21-atom flavone skeleton (C-smeared, aromatic flags preserved),
    coordinates = MMFF-optimized 3D conformer aligned so the chromone core sits
    at the pocket center (centered on dummy ligand center, 67.73 -16.56 -47.84)
  - masked probe: per-position uniform theta, all else locked to target class
  - two experiments:
      exp1  arom_base: aromatic-correct skeleton, mask each position -> 21x13 probs
      exp3  full-element lock (all 21 = target elements), mask each -> P[true]

Targets (flavones):
  m1 O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12   21 atoms
  m2 ...c2ccc(O)cc2O)/                         22 atoms (B-ring 3'-OH)
  m3 ...c2ccc(O)c(O)c2)/                       22 atoms (B-ring 3,4-diOH)
Atom order (m1): see FLAVONE_SMILES below = RDKit SMILES order.
"""
import json
import os
import sys
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from util.ligand_code_util import parse_sdf_file, parse_pdb_file
from dataloader.featurizer import pocket_to_tensor, mol_to_tensor, ProteinLigandDataLoader
from util.mol_utils import MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/_archive_ber101_era'))
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJECT_ROOT)

IDX_C = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(6, False)]
IDX_Carom = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(6, True)]
IDX_N = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(7, False)]
IDX_Narom = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(7, True)]
IDX_O = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(8, False)]
IDX_Oarom = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(8, True)]
IDX_S = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(16, False)]
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.items()}

FLAVONE_SMILES = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'   # m1, canonical target
NATOMS = 21

POCKET_PDB = os.path.join(SCRIPT_DIR, 'sp1_recp_pocket.pdb')
OUT_SDF = os.path.join(SCRIPT_DIR, 'skeleton_flavone_sp1.sdf')
OUT_JSON = os.path.join(SCRIPT_DIR, 'probe_sp1_results.json')

# Which positions are O or C=O in target (tgt idx): used to label report
TGT_ATOMS = [(6, True), (8, False), (6, False), (6, False), (6, True),  # 0-4
             (6, False), (6, True), (6, True), (6, True), (6, True),    # 5-9
             (6, True), (6, True), (6, True), (6, True), (6, True),     # 10-14
             (6, True), (8, False), (6, True), (6, True), (6, True), (6, True)]  # 15-20
# NOTE: exact mapping derived from m1 SMILES below in code (never hardcode Z*2K)
O_SITES = set()
for idx, (z, ar) in enumerate(TGT_ATOMS):
    if z == 8:
        O_SITES.add(idx)
print(f'[setup] O sites: {sorted(O_SITES)}')


def build_skeleton():
    """m1 skeleton: all elements smeared to C (aromatic flags keep), coords from
    MMFF-optimized flavone conformer, translated to pocket center."""
    tgt = Chem.MolFromSmiles(FLAVONE_SMILES)
    assert tgt.GetNumAtoms() == NATOMS
    tgt_arom = {a.GetIdx(): a.GetIsAromatic() for a in tgt.GetAtoms()}
    o_idx = {a.GetIdx() for a in tgt.GetAtoms() if a.GetSymbol() == 'O'}
    print(f'[setup] O atom indices (m1 SMILES order): {sorted(o_idx)}')

    # Kekulize first (explicit bond orders), then smear O -> C preserving
    # bond types and clearing aromatic flags (valid C valence => can kekulize).
    k = Chem.Mol(tgt)
    Chem.Kekulize(k)
    rw = Chem.RWMol(k)
    for a in rw.GetAtoms():
        a.SetAtomicNum(6)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
    skel = rw.GetMol()
    res = Chem.SanitizeMol(skel, catchErrors=True)
    if res != Chem.SanitizeFlags.SANITIZE_NONE:
        raise RuntimeError(f'skeleton sanitize failed: {res}')
    print(f'[skel  ] C-skeleton built (smear O->C, kekulized bond orders)')

    # MMFF-optimized 3D coords from the FULL flavone (hydroxyls included) then
    # keep skeleton (C-smeared) but copy ALL atom coords (same element count)
    m3d = Chem.AddHs(tgt)
    AllChem.EmbedMolecule(m3d, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(m3d, maxIters=500)
    except Exception:
        pass
    m3d = Chem.RemoveHs(m3d)
    assert m3d.GetNumAtoms() == NATOMS
    conf = m3d.GetConformer()
    coords = np.array(conf.GetPositions())

    # align: chromone centroid -> pocket center (from dummy ligand)
    dummy = Chem.SDMolSupplier(os.path.join(SCRIPT_DIR, 'dummy_ligand.sdf'))[0]
    center = np.array(dummy.GetConformer().GetPositions()[0])
    # chromone core centroid = ring atoms {0,1,2,3,8,6} (approx by all heavy - planar)
    # simpler: use all-atom centroid
    skel_conf = Chem.Conformer(NATOMS)
    for i in range(NATOMS):
        p = coords[i] - coords.mean(axis=0) + center
        skel_conf.SetAtomPosition(i, [float(x) for x in p])
    skel.RemoveAllConformers()
    skel.AddConformer(skel_conf, assignId=True)
    w = Chem.SDWriter(OUT_SDF)
    w.write(skel)
    w.close()
    print(f'[skel  ] {NATOMS}-atom C-skeleton at pocket center {center}')
    global O_SITES
    O_SITES = o_idx
    return o_idx


def main():
    o_idx = build_skeleton()
    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    print(f'[probe ] sp1 pocket atoms: {pocket_tensor["pos"].shape[0]}')
    base = mol_to_tensor(parse_sdf_file(path=OUT_SDF), mode='molcraft')
    assert int(base['pos'].shape[0]) == NATOMS
    arom_ktrue = base['atom_feature'].long().tolist()

    # target classes for each position (from m1 element map)
    target_tcls = {}
    for idx in range(NATOMS):
        if idx in o_idx:
            target_tcls[idx] = IDX_O  # all O in m1 are (8,False) phenolic/carbonyl
        else:
            atom = None
        # derive from aromatic flag of skeleton
    tgt_cls = np.zeros(NATOMS, dtype=int)
    for i in range(NATOMS):
        if i in o_idx:
            tgt_cls[i] = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(8, False)]
        else:
            # C position, aromatic flag from skeleton
            tgt_cls[i] = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(6, arom_ktrue[i] == IDX_Carom)]
    print(f'[probe ] target classes: {[(i, CLS_NAME[int(t)] , arom_ktrue[i]) for i, t in enumerate(tgt_cls)]}')

    # exp1: arom-base skeleton, no locks, mask each position
    # exp3: full target lock (C/O), mask each position
    conds = [
        ('arom_base', {}, None),
        ('exp3_full', {i: int(tgt_cls[i]) for i in range(NATOMS)}, None),
    ]
    graphs, graph_meta = [], []
    for key, locks, _ in conds:
        mt = dict(base)  # shallow; mutate atom_feature per item below
        for k, v in mt.items():
            if torch.is_tensor(v):
                mt[k] = v.clone()
        for i, cls in locks.items():
            mt['atom_feature'][i] = cls
        masks = [None] + [(i,) for i in range(NATOMS)]
        for m in masks:
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((key, m))
    print(f'[probe ] {len(graphs)} graphs')

    model = load_model('cuda' if torch.cuda.is_available() else 'cpu')
    loader = ProteinLigandDataLoader(graphs, batch_size=60, shuffle=False)

    @torch.no_grad()
    def run_forward():
        device = next(model.parameters()).device
        gamma = float(1 - 0.03 ** 2)
        results = {}
        gstart = 0
        for batch in loader:
            batch = batch.to(device)
            bl = batch.mol_element_batch
            N = bl.shape[0]
            gids = bl.cpu().numpy()
            ktrue = batch.mol_atom_feature.long()
            logits = torch.zeros(N, K, device=device)
            logits[torch.arange(N, device=device), ktrue] = model.beta1
            theta = torch.softmax(logits, dim=-1)
            for g_local in range(batch.num_graphs):
                _, mask = graph_meta[gstart + g_local]
                if mask:
                    for pi in mask:
                        gi = np.where(gids == g_local)[0][pi]
                        theta[gi, :] = 1.0 / K
            t = torch.ones(N, 1, device=device)
            mu = batch.mol_pos * gamma / 2.0
            coord_pred, p0_h_pred, k_hat, _, _ = model.interdependency_modeling(
                time=t,
                pocket_pos=batch.pocket_pos / 2,
                pocket_v=batch.pocket_atom_feature.float(),
                batch_pocket=batch.pocket_element_batch,
                ed_pos=torch.empty((0, 3), device=device),
                ed_v=torch.empty((0, 4), device=device),
                batch_ed=batch.ed_ed_code_batch,
                batch_ligand=bl,
                theta_h_t=theta,
                mu_pos_t=mu,
                gamma_coord=torch.full((N, 1), gamma, device=device),
                pocket_res=None, pocket_ele=None,
            )
            probs = torch.softmax(torch.log(p0_h_pred.clamp(min=1e-12)), dim=-1)
            for g_local in range(batch.num_graphs):
                cond_key, mask = graph_meta[gstart + g_local]
                rows = np.where(gids == g_local)[0]
                results.setdefault(cond_key, {})[mask] = probs[rows].float().cpu().numpy()
            gstart += batch.num_graphs
        return results

    results = run_forward()

    def pn(p): return float(p[IDX_N] + p[IDX_Narom])
    def po(p): return float(p[IDX_O] + p[IDX_Oarom])

    ab = results['arom_base']
    ab_full = ab[None]
    ab_masked = np.stack([ab[(i,)][i] for i in range(NATOMS)])
    e3 = results['exp3_full']
    e3_full = e3[None]
    e3_masked = np.stack([e3[(i,)][i] for i in range(NATOMS)])

    echo_ab = int((ab_full.argmax(1) == np.array(arom_ktrue)).sum())
    echo_e3 = int((e3_full.argmax(1) == tgt_cls).sum())
    print(f'\n[sanity] echo full-info argmax==class: arom_base {echo_ab}/{NATOMS}, exp3 {echo_e3}/{NATOMS}')

    report = {'flavone': FLAVONE_SMILES, 'n_atoms': NATOMS,
              'o_sites': {str(i): True for i in o_idx},
              'arom_ktrue': arom_ktrue, 'sanity_echo': {'arom_base': echo_ab, 'exp3_full': echo_e3},
              'exp1': {}, 'exp3': {}, 'exp3_summary': {}}
    print('\n===== Exp1: masked probe per position (arom-base skeleton) =====')
    print(f"{'atom':>4} {'kls':>4} {'tgt':>4} | {'P[C]':>7} {'P[N]':>7} {'P[O]':>7} | top1")
    for i in range(NATOMS):
        mk = ab_masked[i]
        tgt = 'O' if i in o_idx else ('C*' if arom_ktrue[i] == IDX_Carom else 'C')
        print(f'{i:>4} {CLS_NAME[arom_ktrue[i]]:>4} {tgt:>4} | {summary_line(mk)}')
        report['exp1'][i] = {'cls': CLS_NAME[arom_ktrue[i]], 'tgt': tgt,
                             'PC': round(1 - pn(mk) - po(mk), 4), 'PN': round(pn(mk), 4),
                             'PO': round(po(mk), 4),
                             'top1': CLS_NAME[int(mk.argmax())],
                             'probs': [round(float(x), 5) for x in mk]}
    print('\n===== Exp3: full target lock, per-position P[true] =====')
    for i in range(NATOMS):
        mk = e3_masked[i]
        p_true = float(mk[tgt_cls[i]])
        print(f'{i:>4} [{CLS_NAME[int(tgt_cls[i])]}] P[true]={p_true*100:7.2f}%  top1={CLS_NAME[int(mk.argmax())]}')
        report['exp3'][i] = {'tgt': CLS_NAME[int(tgt_cls[i])], 'P_true': round(p_true, 4),
                             'top1': CLS_NAME[int(mk.argmax())],
                             'probs': [round(float(x), 5) for x in mk]}
    ptrue = np.array([e3_masked[i][tgt_cls[i]] for i in range(NATOMS)])
    report['exp3_summary'] = {'mean': round(float(ptrue.mean()), 4), 'min': round(float(ptrue.min()), 4),
                              'min_atom': int(ptrue.argmin()),
                              'gt50': int((ptrue > 0.5).sum()),
                              'o_sites_mean': round(float(ptrue[[i for i in range(NATOMS) if i in o_idx]].mean()), 4)}
    print(f'  summary: mean={ptrue.mean()*100:.1f}% min={ptrue.min()*100:.1f}% ({int(ptrue.argmin())}) '
          f'>50%: {(ptrue > 0.5).sum()}/{NATOMS} | O位点均值={ptrue[[i for i in range(NATOMS) if i in o_idx]].mean()*100:.1f}%')

    np.save(OUT_JSON.replace('.json', '_exp1_masked.npy'), ab_masked)
    np.save(OUT_JSON.replace('.json', '_exp3_masked.npy'), e3_masked)
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f'\n[saved] {OUT_JSON}')


def summary_line(mk):
    pc = float(mk[IDX_C] + mk[IDX_Carom])
    pn2 = float(mk[IDX_N] + mk[IDX_Narom])
    po2 = float(mk[IDX_O] + mk[IDX_Oarom])
    return f'{pc*100:6.1f}% {pn2*100:6.1f}% {po2*100:6.1f}% | {CLS_NAME[int(mk.argmax())]}({mk.max()*100:.0f}%)'


if __name__ == '__main__':
    main()
