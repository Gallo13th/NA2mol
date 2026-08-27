#!/usr/bin/env python3
"""Step3: aromatic-aware masked element probe at the KPT102-docked cyanine pose.

Exact replica of ber102_rdock/step3_probe_ber102.py (same model/ckpt/featurize/
t=1/beta1/gamma, aromatic-aware beliefs), only:
  - pocket = KPT102 14 A pocket (kpt102_pocket.pdb, 7X8O MODEL 1)
  - skeleton coordinates = best rDock pose of the cyanine in the KPT102
    cavity (Step 2 output), atom order = target-SMILES order (0..24)
  - N sites kept in tgt indexing: {2: quinoline N (7,False),
    12: naphthyridinium [n+] (7,True), 18: naphthyridine n (7,True)}

Exp1  arom_base: aromatic-correct beliefs, per-position mask -> 25x13 probs
Exp3  full target element lock (3N + 22C) -> per-position P[true class]

Head-to-head comparison is against BER102 (probe_ber102_results.json) —
same pocket type, different co-crystal ligand/cofactor context.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from util.ligand_code_util import parse_sdf_file, parse_pdb_file
from dataloader.featurizer import pocket_to_tensor, mol_to_tensor, ProteinLigandDataLoader
from util.mol_utils import MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB

IDX_C = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(6, False)]     # 1
IDX_Carom = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(6, True)]  # 2
IDX_N = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(7, False)]     # 3
IDX_Narom = MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(7, True)]  # 4
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.items()}

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/_archive_ber101_era'))
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJECT_ROOT)

TARGET_SMILES = 'CCN1C=C/C(=C/C=C/c2cc[n+]3ccccc3n2)c2ccccc21'
DOCKED_SDF = os.path.join(SCRIPT_DIR, 'cyanine_best_pose.sdf')
POCKET_PDB = os.path.join(SCRIPT_DIR, 'kpt102_pocket.pdb')
OUT_SDF = os.path.join(SCRIPT_DIR, 'skeleton_aromatic_kpt102.sdf')
OUT_JSON = os.path.join(SCRIPT_DIR, 'probe_kpt102_results.json')
BER_JSON = os.path.join(os.path.dirname(SCRIPT_DIR), 'ber102_rdock', 'probe_ber102_results.json')

NATOMS = 25
N_SITES = {2: (7, False), 12: (7, True), 18: (7, True)}   # tgt idx -> target class


def make_aromatic_skeleton():
    """smear target elements to C (keep bond orders/aromaticity), attach the
    docked-pose coordinates (same tgt atom order)."""
    tgt = Chem.MolFromSmiles(TARGET_SMILES)
    assert tgt.GetNumAtoms() == NATOMS
    tgt_arom = {a.GetIdx(): a.GetIsAromatic() for a in tgt.GetAtoms()}

    rw = Chem.RWMol(tgt)
    for a in rw.GetAtoms():
        a.SetAtomicNum(6)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
    skel = rw.GetMol()
    Chem.SanitizeMol(skel)
    arom2 = {a.GetIdx(): a.GetIsAromatic() for a in skel.GetAtoms()}
    assert all(arom2[i] == tgt_arom[i] for i in range(NATOMS)), 'aromaticity drift'
    n_arom = sum(arom2.values())
    assert n_arom == 16, f'expect 16 aromatic C, got {n_arom}'

    pose = next(Chem.SDMolSupplier(DOCKED_SDF, removeHs=True, sanitize=False))
    assert pose is not None and pose.GetNumAtoms() == NATOMS
    syms = [a.GetSymbol() for a in pose.GetAtoms()]
    tgt_syms = [a.GetSymbol() for a in tgt.GetAtoms()]
    assert syms == tgt_syms, 'docked pose atom order != SMILES order'
    Chem.SanitizeMol(pose)
    coords = pose.GetConformer().GetPositions()

    conf = Chem.Conformer(NATOMS)
    for i in range(NATOMS):
        conf.SetAtomPosition(i, [float(x) for x in coords[i]])
    skel.RemoveAllConformers()
    skel.AddConformer(conf, assignId=True)
    w = Chem.SDWriter(OUT_SDF)
    w.write(skel)
    w.close()
    print(f'[skel  ] aromatic skeleton saved: 16x(6,True) + 9x(6,False), '
          f'coords = best KPT102 redock pose')
    return tgt_arom


def clone_mt(base):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}


def apply_locks(base, locks):
    mt = clone_mt(base)
    for i, cls in locks.items():
        mt['atom_feature'][i] = cls
    return mt


@torch.no_grad()
def run_forward_multi(model, loader, graph_meta, beta1):
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
        logits[torch.arange(N, device=device), ktrue] = beta1
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


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[probe ] device={device}')

    tgt_arom = make_aromatic_skeleton()

    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    print(f"[probe ] KPT102 pocket atoms: {pocket_tensor['pos'].shape[0]}")
    base = mol_to_tensor(parse_sdf_file(path=OUT_SDF), mode='molcraft')
    arom_ktrue = base['atom_feature'].long().tolist()
    assert int(base['pos'].shape[0]) == NATOMS
    assert sum(1 for k in arom_ktrue if k == IDX_Carom) == 16

    full_target_locks = {i: MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[c] for i, c in N_SITES.items()}
    conds = [
        ('arom_base', {}, [None] + [(i,) for i in range(NATOMS)]),
        ('exp3_full', full_target_locks, [None] + [(i,) for i in range(NATOMS)]),
    ]
    graphs, graph_meta = [], []
    for key, locks, masks in conds:
        mt = apply_locks(base, locks)
        for m in masks:
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((key, m))
    print(f'[probe ] {len(graphs)} graphs')

    model = load_model(device)
    print('[probe ] model loaded')
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def pn(p): return float(p[IDX_N] + p[IDX_Narom])
    def pc(p): return float(p[IDX_C] + p[IDX_Carom])

    ab = results['arom_base']
    ab_full = ab[None]
    ab_masked = np.stack([ab[(i,)][i] for i in range(NATOMS)])
    e3 = results['exp3_full']
    e3_full = e3[None]
    e3_masked = np.stack([e3[(i,)][i] for i in range(NATOMS)])

    tgt_cls = np.array([full_target_locks.get(i, arom_ktrue[i]) for i in range(NATOMS)])
    echo_ab = int((ab_full.argmax(1) == np.array(arom_ktrue)).sum())
    echo_e3 = int((e3_full.argmax(1) == tgt_cls).sum())
    print(f'\n[sanity] echo full-info argmax==locked: arom_base {echo_ab}/25, '
          f'exp3_full {echo_e3}/25')

    ber = json.load(open(BER_JSON))

    report = {'arom_ktrue': arom_ktrue,
              'n_sites': {str(k): list(v) for k, v in N_SITES.items()},
              'sanity_echo': {'arom_base': echo_ab, 'exp3_full': echo_e3},
              'exp1': {}, 'exp3': {}, 'exp3_summary': {},
              'ber102_comparison': {}}

    print('\n===== Exp1: masked probs per position (KPT102 pose vs BER102 pose) =====')
    print(f"{'atom':>4} {'cls':>5} {'tgt':>4} | {'PN_KPT':>7} {'PN_BER':>7} {'d':>7} | {'P[N*]':>6} {'P[N]':>6} {'top1':>5} {'p%':>5} | BERtop1")
    for i in range(NATOMS):
        mk = ab_masked[i]
        b = ber['exp1'][str(i)]
        d = pn(mk) - b['PN']
        tgt = 'N' if i in N_SITES else 'C'
        print(f'{i:>4} {CLS_NAME[arom_ktrue[i]]:>5} {tgt:>4} | {pn(mk)*100:6.2f}% {b["PN"]*100:6.2f}% {d*100:+6.2f}% | '
              f'{mk[IDX_Narom]*100:5.2f}% {mk[IDX_N]*100:5.2f}% {CLS_NAME[int(mk.argmax())]:>5} {mk.max()*100:4.0f}% | '
              f'{b["top1"]}({b["probs"][int(np.argmax(b["probs"]))]*100:.0f}%)')
        report['exp1'][i] = {'cls': CLS_NAME[arom_ktrue[i]], 'tgt': tgt,
                             'PN': round(pn(mk), 4), 'PN_ber102': b['PN'],
                             'PNarom_ch': round(float(mk[IDX_Narom]), 4),
                             'PNaliph_ch': round(float(mk[IDX_N]), 4),
                             'PC': round(pc(mk), 4),
                             'top1': CLS_NAME[int(mk.argmax())],
                             'p_top1': round(float(mk.max()), 4),
                             'probs': [round(float(x), 5) for x in mk]}

    print('\n----- Exp1 core: 3 N sites -----')
    for i, (z, ar) in N_SITES.items():
        mk = ab_masked[i]
        b = ber['exp1'][str(i)]
        print(f'  tgt{i} tgt=({z},{ar}): P[N] KPT102 {pn(mk)*100:.2f}%  '
              f'(BER102 {b["PN"]*100:.2f}%)  '
              f'[N*={mk[IDX_Narom]*100:.2f}% N={mk[IDX_N]*100:.2f}%]  '
              f'P[C]={pc(mk)*100:.2f}%  top1 {CLS_NAME[int(mk.argmax())]}({mk.max()*100:.0f}%)')
        report['ber102_comparison'][str(i)] = {
            'PN_kpt102': round(pn(mk), 4), 'PN_ber102': b['PN'],
            'delta': round(pn(mk) - b['PN'], 4)}

    print('\n===== Exp3: full target element lock, per-position P[true] =====')
    for i in range(NATOMS):
        mk = e3_masked[i]
        p_true = float(mk[tgt_cls[i]])
        b3 = ber['exp3'][str(i)]
        print(f'{i:>4} {CLS_NAME[int(tgt_cls[i])]:>5} | P[true]={p_true*100:7.2f}% (BER102 {b3["P_true"]*100:6.2f}%)  '
              f'top1 {CLS_NAME[int(mk.argmax())]}')
        report['exp3'][i] = {'tgt': CLS_NAME[int(tgt_cls[i])], 'P_true': round(p_true, 4),
                             'P_true_ber102': b3['P_true'],
                             'top1': CLS_NAME[int(mk.argmax())],
                             'probs': [round(float(x), 5) for x in mk]}
    ptrue_all = np.array([e3_masked[i][tgt_cls[i]] for i in range(NATOMS)])
    report['exp3_summary'] = {'mean_P_true': round(float(ptrue_all.mean()), 4),
                              'min_P_true': round(float(ptrue_all.min()), 4),
                              'min_atom': int(ptrue_all.argmin()),
                              'n_P_true_gt50': int((ptrue_all > 0.5).sum()),
                              'ber102_mean_P_true': ber['exp3_summary']['mean_P_true'],
                              'n_sites': {str(i): {'P_true': round(float(e3_masked[i][tgt_cls[i]]), 4),
                                                   'P_true_allN': round(pn(e3_masked[i]), 4),
                                                   'P_true_ber102': ber['exp3_summary']['n_sites'][str(i)]['P_true']} for i in N_SITES}}
    print(f'  summary: mean P[true]={ptrue_all.mean()*100:.2f}% (BER102 {ber["exp3_summary"]["mean_P_true"]*100:.2f}%)  '
          f'min={ptrue_all.min()*100:.2f}% (atom {int(ptrue_all.argmin())})  '
          f'>50%: {(ptrue_all > 0.5).sum()}/25')
    for i in N_SITES:
        print(f'  tgt{i}: P[true {CLS_NAME[int(tgt_cls[i])]}]={e3_masked[i][tgt_cls[i]]*100:.2f}%  '
              f'P[N total]={pn(e3_masked[i])*100:.2f}%  '
              f'(BER102 P[true]={ber["exp3_summary"]["n_sites"][str(i)]["P_true"]*100:.2f}%)')

    np.save(OUT_JSON.replace('.json', '_base_masked.npy'), ab_masked)
    np.save(OUT_JSON.replace('.json', '_base_full.npy'), ab_full)
    np.save(OUT_JSON.replace('.json', '_exp3_masked.npy'), e3_masked)
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n[saved] {OUT_JSON}')


if __name__ == '__main__':
    main()
