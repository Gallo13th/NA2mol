#!/usr/bin/env python3
"""t2-DENOVO Phase E: bridge-position P[C] probe on the NH-bridged bis-benzothiazole
D1 product (c1ccc2sc(Nc3nc4ccccc4s3)nc2c1, Tc 0.517 vs t2).

The pure de novo chain reached the bis-benzothiazole skeleton with an NH bridge
(the model's own choice at the single free bridge atom of the Phase-BC seed).
Question: at that geometry, what does the model believe the bridge atom should
be? P[C] on the CH2 channel (6, False) is the endorsement for an N->C bridge
replacement (-> t2). t2 is used ONLY as the reporting criterion; the probed
molecule and its conformation are 100% model products.

Protocol identical to phaseA_probe.py / phaseA3_context_probe.py (same ckpt /
pocket / t=1 / beta1 / gamma, masked belief = uniform 1/13, coords/2).

Steps:
  E1 extract all NH-bridge conformers from gen_runD1/valid_mols.sdf (SMILES match)
  E2 masked probe at bridge N (original generated pose)
  E3 redock each (BER102 r6 validated protocol: polar-H SD, rbdock -n 10 -s 42,
     rank by SCORE.INTER) -> best pose -> re-probe (pose dependence is known)

Gate: P[C] (CH2 channel (6,False)) > 10% at a redock pose -> Phase-E N->C replacement.

Usage: CUDA_VISIBLE_DEVICES=3 python casestudy/kras_g4/t2_denovo/phaseE_bridge_probe.py
"""
import os, sys, json, shutil, subprocess

PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, 'casestudy/kras_g4/_archive_ber101_era'))

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

from util.ligand_code_util import parse_sdf_file, parse_pdb_file
from dataloader.featurizer import pocket_to_tensor, mol_to_tensor, ProteinLigandDataLoader
from dataloader.featurizer import MAP_ATOM_TYPE_AROMATIC_TO_INDEX as MAP_ATI
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MAP_ITA

import util.mol_utils  # noqa: F401
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJ)

IDX_C = MAP_ATI[(6, False)]        # CH2 channel
IDX_Carom = MAP_ATI[(6, True)]
IDX_N = MAP_ATI[(7, False)]
IDX_Narom = MAP_ATI[(7, True)]
IDX_O = MAP_ATI[(8, False)]
IDX_S = MAP_ATI[(16, False)]
IDX_Sarom = MAP_ATI[(16, True)]
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_ITA.items()}

NH_SMILES = 'c1ccc2sc(Nc3nc4ccccc4s3)nc2c1'
POCKET_PDB = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
VALID_SDF = 'casestudy/kras_g4/t2_denovo/gen_runD1/valid_mols.sdf'
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseE_bridge'
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, 'phaseE_probe_results.json')

RBT_ROOT = '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1'
SYSTEM_PRM = os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_full_r6_system.prm')
PROTOCOL = os.path.join(RBT_ROOT, 'data/scripts/dock_solv.prm')


def bridge_n_idx(m):
    """exocyclic NH bridging two aromatic rings: N, not in ring, degree 2,
    both neighbors aromatic C, has 1 H."""
    for a in m.GetAtoms():
        if a.GetSymbol() != 'N' or a.IsInRing() or a.GetDegree() != 2:
            continue
        if a.GetTotalNumHs() != 1:
            continue
        nb = a.GetNeighbors()
        if all(n.GetSymbol() == 'C' and n.GetIsAromatic() for n in nb):
            return a.GetIdx()
    return None


def polar_h_only(m):
    m2 = Chem.AddHs(Chem.Mol(m), addCoords=True)
    rw = Chem.RWMol(m2)
    for a in list(rw.GetAtoms()):
        if a.GetSymbol() == 'H':
            heavy = [n for n in a.GetNeighbors() if n.GetSymbol() != 'H']
            if heavy and heavy[0].GetSymbol() == 'C':
                rw.RemoveAtom(a.GetIdx())
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


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
            time=t, pocket_pos=batch.pocket_pos / 2, pocket_v=batch.pocket_atom_feature.float(),
            batch_pocket=batch.pocket_element_batch,
            ed_pos=torch.empty((0, 3), device=device), ed_v=torch.empty((0, 4), device=device),
            batch_ed=batch.ed_ed_code_batch, batch_ligand=bl,
            theta_h_t=theta, mu_pos_t=mu,
            gamma_coord=torch.full((N, 1), gamma, device=device),
            pocket_res=None, pocket_ele=None)
        probs = torch.softmax(torch.log(p0_h_pred.clamp(min=1e-12)), dim=-1)
        for g_local in range(batch.num_graphs):
            cond_key, mask = graph_meta[gstart + g_local]
            rows = np.where(gids == g_local)[0]
            results.setdefault(cond_key, {})[mask] = probs[rows].float().cpu().numpy()
        gstart += batch.num_graphs
    return results


def redock_best(mol, key):
    """BER102 r6 protocol redock; returns (best SCORE.INTER, heavy-atom mol w/ pose)."""
    inp = os.path.join(OUTDIR, f'{key}_dockin.sdf')
    mh = polar_h_only(mol)
    mh.SetProp('_Name', key)
    w = Chem.SDWriter(inp); w.write(mh); w.close()
    out_base = os.path.join(OUTDIR, f'{key}_rdock')
    cmd = ['rbdock', '-i', inp, '-o', out_base, '-r', SYSTEM_PRM, '-p', PROTOCOL,
           '-n', '10', '-s', '42']
    subprocess.run(cmd, check=True, env={**os.environ, 'RBT_ROOT': RBT_ROOT},
                   stdout=subprocess.DEVNULL)
    recs = []
    for pm in Chem.SDMolSupplier(out_base + '.sd', removeHs=False, sanitize=False):
        if pm is None:
            continue
        p = pm.GetPropsAsDict()

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float('nan')
        recs.append((num(p.get('SCORE.INTER')), pm))
    assert recs, f'{key}: no rbdock records'
    recs.sort(key=lambda t: t[0])
    best_score, pm = recs[0]
    heavy_idx = [j for j, a in enumerate(pm.GetAtoms()) if a.GetSymbol() != 'H']
    syms = [pm.GetAtomWithIdx(j).GetSymbol() for j in heavy_idx]
    ref_syms = [a.GetSymbol() for a in mol.GetAtoms()]
    assert syms == ref_syms, f'{key}: atom order mismatch'
    xyz = pm.GetConformer().GetPositions()[heavy_idx]
    bm = Chem.Mol(mol)
    conf = Chem.Conformer(bm.GetNumAtoms())
    for j in range(bm.GetNumAtoms()):
        conf.SetAtomPosition(j, Point3D(*xyz[j]))
    bm.RemoveAllConformers()
    bm.AddConformer(conf, assignId=True)
    return best_score, bm, len(recs)


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[E] device={device}')

    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    print(f'[E] BER102 pocket atoms: {pocket_tensor["pos"].shape[0]}')

    # E1: extract NH-bridge conformers (snapshot valid_mols.sdf: D1 still running)
    snap = os.path.join(OUTDIR, 'valid_mols_snapshot.sdf')
    shutil.copy(VALID_SDF, snap)
    nh_ref = Chem.MolFromSmiles(NH_SMILES)
    mols = []
    for i, m in enumerate(Chem.SDMolSupplier(snap, removeHs=True)):
        if m is None:
            continue
        if m.HasSubstructMatch(nh_ref) and m.GetNumAtoms() == nh_ref.GetNumAtoms():
            ni = bridge_n_idx(m)
            assert ni is not None
            mols.append((m.GetProp('_Name'), m, ni))
    print(f'[E] NH-bridge conformers found: {len(mols)}')
    assert mols, 'no NH-bridge molecules in valid_mols.sdf'

    # E3 pre-pass: redock each (CPU, before model load)
    redock = {}
    for name, m, ni in mols:
        key = name
        sc, bm, nposes = redock_best(m, key)
        redock[name] = (sc, bm, nposes)
        print(f'[E] redock {name}: best SCORE.INTER = {sc:.2f} ({nposes} poses)')

    # build graphs: orig + redock poses, masked at bridge N
    graphs, graph_meta = [], []
    tensors = {}
    for name, m, ni in mols:
        poses = {}
        for tag, mol in [('orig', m), ('redock', redock[name][1])]:
            p = os.path.join(OUTDIR, f'{name}_{tag}.sdf')
            w = Chem.SDWriter(p); w.write(mol); w.close()
            mt = mol_to_tensor(parse_sdf_file(path=p), mode='molcraft')
            poses[tag] = mt
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((f'{name}|{tag}', None))
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((f'{name}|{tag}', (ni,)))
        tensors[name] = poses
    print(f'[E] {len(graphs)} graphs')

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def vec(p):
        return {
            'P_C_ch2': round(float(p[IDX_C]), 4),
            'P_C_arom': round(float(p[IDX_Carom]), 4),
            'P_C_total': round(float(p[IDX_C] + p[IDX_Carom]), 4),
            'P_N_total': round(float(p[IDX_N] + p[IDX_Narom]), 4),
            'P_N_sp2': round(float(p[IDX_N]), 4),
            'P_O': round(float(p[IDX_O]), 4),
            'P_S': round(float(p[IDX_S] + p[IDX_Sarom]), 4),
            'top1': CLS_NAME[int(p.argmax())],
            'top3': [(CLS_NAME[j], round(float(p[j]), 4)) for j in np.argsort(p)[::-1][:3]],
        }

    report = {'nh_smiles': NH_SMILES, 'per_mol': {}}
    print('\n===== Phase E bridge probe: masked belief at bridge N =====')
    print(f"{'conformer':>28} {'pose':>7} {'P[C]ch2':>8} {'P[C]tot':>8} {'P[N]':>7} "
          f"{'P[O]':>6} {'P[S]':>6}  top3")
    gate = []
    for name, m, ni in mols:
        e = {'bridge_N_idx': ni, 'redock_SCORE_INTER': round(redock[name][0], 2),
             'n_poses': redock[name][2], 'poses': {}}
        for tag in ('orig', 'redock'):
            mk = results[f'{name}|{tag}'][(ni,)][ni]
            fk = results[f'{name}|{tag}'][None][ni]
            e['poses'][tag] = {'masked': vec(mk), 'fullinfo': vec(fk)}
            if tag == 'redock':
                gate.append((float(mk[IDX_C]), name))
            v = e['poses'][tag]['masked']
            print(f"{name:>28} {tag:>7} {v['P_C_ch2']*100:7.2f}% {v['P_C_total']*100:7.2f}% "
                  f"{v['P_N_total']*100:6.2f}% {v['P_O']*100:5.2f}% {v['P_S']*100:5.2f}%  "
                  f"{v['top3']}")
        report['per_mol'][name] = e

    gate.sort(reverse=True)
    best_pc, best_name = gate[0]
    report['_gate'] = {
        'criterion': 'P[C] CH2 channel (6,False) at bridge, redock pose',
        'threshold': 0.10,
        'best': {'name': best_name, 'P_C_ch2_redock': round(best_pc, 4)},
        'all_redock_PC': {n: round(p, 4) for p, n in gate},
        'pass': bool(best_pc > 0.10),
    }
    print(f"\n[E GATE] best redock-pose P[C](CH2) = {best_pc*100:.2f}% ({best_name}) "
          f"-> {'PASS' if best_pc > 0.10 else 'FAIL'} (>10%)")

    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[E] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
