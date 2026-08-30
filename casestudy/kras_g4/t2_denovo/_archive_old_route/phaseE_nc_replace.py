#!/usr/bin/env python3
"""t2-DENOVO Phase E2: probability-endorsed N->C bridge replacement.

Gate from phaseE_bridge_probe.py: at the redock pose of the D1 product
denovoD1_r12_i043_s3054 (NH-bridged bis-benzothiazole), masked bridge belief
P[C] CH2 channel = 22.1% > 10% (P[C] total 49.4%). The replacement target and
its conformation are 100% model products (D1 generation + rDock pose); the
endorsement comes from the model's own masked distribution. t2 SMILES is used
ONLY to verify/report the identity of the replacement product.

Steps:
  E2a N->C at the bridge (SetAtomicNum(6) + 2 explicit H), sanitize
      -> canonical SMILES check vs t2 (report criterion)
  E2b re-probe at the new CH2 site (self-consistency: P[C] should be high)
  E2c rdock full scoring (BER102 r6 protocol, -n 10 -s 42)

Usage: CUDA_VISIBLE_DEVICES=3 python casestudy/kras_g4/t2_denovo/phaseE_nc_replace.py
"""
import os, sys, json, subprocess

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

IDX_C = MAP_ATI[(6, False)]
IDX_Carom = MAP_ATI[(6, True)]
IDX_N = MAP_ATI[(7, False)]
IDX_Narom = MAP_ATI[(7, True)]
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_ITA.items()}

T2_TARGET = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'   # verification criterion ONLY
SRC_NAME = 'denovoD1_r12_i043_s3054'
SRC_SDF = 'casestudy/kras_g4/t2_denovo/phaseE_bridge/denovoD1_r12_i043_s3054_redock.sdf'
POCKET_PDB = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseE_bridge'
OUT_JSON = os.path.join(OUTDIR, 'phaseE_replace_results.json')

RBT_ROOT = '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1'
SYSTEM_PRM = os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_full_r6_system.prm')
PROTOCOL = os.path.join(RBT_ROOT, 'data/scripts/dock_solv.prm')


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


def rdock_score(mol, key):
    inp = os.path.join(OUTDIR, f'{key}_dockin.sdf')
    mh = polar_h_only(mol)
    mh.SetProp('_Name', key)
    w = Chem.SDWriter(inp); w.write(mh); w.close()
    out_base = os.path.join(OUTDIR, f'{key}_rdock')
    cmd = ['rbdock', '-i', inp, '-o', out_base, '-r', SYSTEM_PRM, '-p', PROTOCOL,
           '-n', '10', '-s', '42']
    subprocess.run(cmd, check=True, env={**os.environ, 'RBT_ROOT': RBT_ROOT},
                   stdout=subprocess.DEVNULL)
    scores = []
    for pm in Chem.SDMolSupplier(out_base + '.sd', removeHs=False, sanitize=False):
        if pm is None:
            continue
        try:
            scores.append(float(pm.GetProp('SCORE.INTER')))
        except (TypeError, ValueError):
            continue
    assert scores, f'{key}: no rbdock records'
    return min(scores), scores


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    src = next(Chem.SDMolSupplier(SRC_SDF, removeHs=True))
    smi_src = Chem.MolToSmiles(src)
    print(f'[E2] source: {SRC_NAME} redock pose  {smi_src}')

    # ---- E2a: N -> C(CH2) at the bridge ----
    ni = None
    for a in src.GetAtoms():
        if a.GetSymbol() == 'N' and not a.IsInRing() and a.GetDegree() == 2 \
                and a.GetTotalNumHs() == 1 \
                and all(n.GetSymbol() == 'C' and n.GetIsAromatic() for n in a.GetNeighbors()):
            ni = a.GetIdx()
    assert ni is not None
    rw = Chem.RWMol(Chem.Mol(src))
    a = rw.GetAtomWithIdx(ni)
    a.SetAtomicNum(6)
    a.SetNoImplicit(True)
    a.SetNumExplicitHs(2)
    a.SetFormalCharge(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    Chem.SetAromaticity(out, Chem.AromaticityModel.AROMATICITY_DEFAULT)
    smi_out = Chem.MolToSmiles(out)
    t2_canon = Chem.MolToSmiles(Chem.MolFromSmiles(T2_TARGET))
    is_t2 = (smi_out == t2_canon)
    print(f'[E2a] N->C(CH2) -> {smi_out}')
    print(f'[E2a] canonical SMILES == t2 ? {is_t2}  (t2: {t2_canon})')

    out.SetProp('_Name', 't2_denovo_replaced')
    outp = os.path.join(OUTDIR, 't2_replaced.sdf')
    w = Chem.SDWriter(outp); w.write(out); w.close()
    print(f'[E2a] saved {outp}')

    # ---- E2b: re-probe at the new CH2 site (self-consistency) ----
    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    mt = mol_to_tensor(parse_sdf_file(path=outp), mode='molcraft')
    cls_true = int(mt['atom_feature'][ni].item())
    print(f'[E2b] new bridge atom class: {CLS_NAME[cls_true]} (expected 6)')
    assert CLS_NAME[cls_true] == '6'
    graphs = [build_graphs(pocket_tensor, mt), build_graphs(pocket_tensor, mt)]
    graph_meta = [('replaced', None), ('replaced', (ni,))]
    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))
    mk = results['replaced'][(ni,)][ni]
    fk = results['replaced'][None][ni]
    probe = {
        'masked': {
            'P_C_ch2': round(float(mk[IDX_C]), 4),
            'P_C_arom': round(float(mk[IDX_Carom]), 4),
            'P_C_total': round(float(mk[IDX_C] + mk[IDX_Carom]), 4),
            'P_N_total': round(float(mk[IDX_N] + mk[IDX_Narom]), 4),
            'top1': CLS_NAME[int(mk.argmax())],
            'top3': [(CLS_NAME[j], round(float(mk[j]), 4)) for j in np.argsort(mk)[::-1][:3]],
        },
        'fullinfo': {'P_C_ch2': round(float(fk[IDX_C]), 4), 'top1': CLS_NAME[int(fk.argmax())]},
    }
    print(f"[E2b] re-probe at CH2: masked P[C]ch2={probe['masked']['P_C_ch2']*100:.2f}% "
          f"P[C]tot={probe['masked']['P_C_total']*100:.2f}% P[N]={probe['masked']['P_N_total']*100:.2f}% "
          f"top1={probe['masked']['top1']}")

    # ---- E2c: rdock scoring (BER102 standard protocol) ----
    best_sc, all_sc = rdock_score(out, 't2_replaced')
    print(f'[E2c] rdock t2_replaced: best SCORE.INTER = {best_sc:.2f} ({len(all_sc)} poses)')

    report = {
        'source': {'name': SRC_NAME, 'pose': 'redock_best_SCORE.INTER_-41.11',
                   'smiles': smi_src},
        'gate_e1': json.load(open(os.path.join(OUTDIR, 'phaseE_probe_results.json')))['_gate'],
        'replacement': {'site': ni, 'op': 'N->C(CH2) + 2H', 'smiles_out': smi_out,
                        'is_t2': bool(is_t2)},
        'reprobe': probe,
        'rdock': {'best_SCORE_INTER': round(best_sc, 2),
                  'all_SCORE_INTER': [round(s, 2) for s in sorted(all_sc)]},
        'product_sdf': outp,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[E2] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
