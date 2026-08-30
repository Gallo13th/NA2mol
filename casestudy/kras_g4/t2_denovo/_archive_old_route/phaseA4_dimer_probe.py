#!/usr/bin/env python3
"""t2-DENOVO Phase A4 probe: masked P[S] at both pyrrolic-N sites of the redocked
bis-benzimidazole-CH2 dimers (F4 full / F4b CH2-deleted seed shape).

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseA4_dimer_probe.py
"""
import os, sys, json

PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, 'casestudy/kras_g4/_archive_ber101_era'))

import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from util.ligand_code_util import parse_sdf_file, parse_pdb_file
from dataloader.featurizer import pocket_to_tensor, mol_to_tensor, ProteinLigandDataLoader
from dataloader.featurizer import MAP_ATOM_TYPE_AROMATIC_TO_INDEX as MAP_ATI
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MAP_ITA

import util.mol_utils  # noqa: F401
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJ)

IDX_S = MAP_ATI[(16, False)]
IDX_Sarom = MAP_ATI[(16, True)]
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_ITA.items()}

POCKET_PDB = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
MANIFEST = 'casestudy/kras_g4/t2_denovo/phaseA4_dimer/manifest.json'
OUT_JSON = 'casestudy/kras_g4/t2_denovo/phaseA4_probe_results.json'


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


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')

    manifest = json.load(open(MANIFEST))
    graphs, graph_meta, ctx = [], [], []
    for e in manifest:
        for fam, path in [('F4', e['F4']), ('F4b', e['F4b'])]:
            m = next(Chem.SDMolSupplier(path, removeHs=True))
            if m is None:
                print('[skip]', path)
                continue
            mt = mol_to_tensor(parse_sdf_file(path=path), mode='molcraft')
            name = f"{e['name']}_p{e['pose']}_{fam}"
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((name, None))
            for si in (e['pyr1'], e['pyr2']):
                graphs.append(build_graphs(pocket_tensor, mt))
                graph_meta.append((name, (si,)))
            ctx.append((name, fam, e))
    print(f'[A4probe] {len(graphs)} graphs, {len(ctx)} contexts')

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p): return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    report, rows = {}, []
    for name, fam, e in ctx:
        r = results[name]
        sites = {}
        for si in (e['pyr1'], e['pyr2']):
            mk = r[(si,)][si]
            sites[si] = {'PS': round(ps(mk), 4), 'PSarom': round(float(mk[IDX_Sarom]), 4),
                         'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                         'top1': CLS_NAME[int(mk.argmax())]}
            rows.append((ps(mk), name, fam, si, e['score']))
        report[name] = {'family': fam, 'score': e['score'], 'sites': sites}

    rows.sort(key=lambda t: -t[0])
    print('\n===== A4 dimer probe: P[S] at pyrrolic-N (gate >10%) =====')
    for p, name, fam, si, sc in rows:
        flag = '  <<GATE' if p > 0.10 else ''
        print(f'{p*100:7.2f}%  {fam}  {name:>44} site{si} score={sc}{flag}')
    allps = np.array([r[0] for r in rows])
    f4 = np.array([r[0] for r in rows if r[1].endswith('_F4')])
    f4b = np.array([r[0] for r in rows if r[1].endswith('_F4b')])
    print(f'\n[A4] all: max={allps.max()*100:.2f}% >10%: {(allps>0.10).sum()}/{len(allps)}')
    if len(f4):
        print(f'     F4 (full dimer): max={f4.max()*100:.2f}% median={np.median(f4)*100:.2f}%')
    if len(f4b):
        print(f'     F4b (seed shape): max={f4b.max()*100:.2f}% median={np.median(f4b)*100:.2f}%')
    report['_summary'] = {'max': round(float(allps.max()), 4),
                          'n_gt10': int((allps > 0.10).sum()), 'n': len(allps),
                          'F4_max': round(float(f4.max()), 4) if len(f4) else None,
                          'F4b_max': round(float(f4b.max()), 4) if len(f4b) else None,
                          'top_ctx': rows[0][1] if rows else None}
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[A4probe] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
