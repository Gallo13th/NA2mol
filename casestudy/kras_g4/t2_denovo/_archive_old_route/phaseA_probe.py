#!/usr/bin/env python3
"""t2-DENOVO Phase A2: aromatic-aware masked element probe on redocked isostere poses.

THE DECISION GATE of the pure de novo path.

For each Phase-A1 candidate (library isostere, rDock best pose in BER102):
  masked probe (belief=uniform 1/13) at the diaza sites -> P[S] (ch10+ch11),
  the model's endorsement of an N->S (benzothiazole-forming) replacement at that
  geometry. Original in-pocket library pose probed as baseline comparison.

Protocol identical to t2_repro/stepB_probe_t2.py (same ckpt / pocket / t=1 /
beta1 / gamma; model 13-class table of dataloader/featurizer.py:
S aliphatic=10, S aromatic=11).

Gate: any candidate with P[S] > 10% at a benzimidazole pyrrolic-N -> Phase B.

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseA_probe.py
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

import util.mol_utils  # noqa: F401  (fragment index preload, cwd=PROJ)
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJ)

IDX_S = MAP_ATI[(16, False)]      # 10
IDX_Sarom = MAP_ATI[(16, True)]   # 11
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_ITA.items()}

POCKET_PDB = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
SUMMARY = 'casestudy/kras_g4/t2_denovo/phaseA_redock/summary.json'
OUT_JSON = 'casestudy/kras_g4/t2_denovo/phaseA_probe_results.json'


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
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[A2] device={device}')

    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    print(f'[A2] BER102 pocket atoms: {pocket_tensor["pos"].shape[0]}')

    summary = json.load(open(SUMMARY))
    tmpdir = '/tmp/opencode/t2_denovo_A'
    os.makedirs(tmpdir, exist_ok=True)

    # original library mols (for baseline poses)
    orig = {}
    for src, path in [('gen2k', 'casestudy/kras_g4/ber102_gen2k/valid_mols.sdf'),
                      ('lib10k', 'casestudy/kras_g4/library_10k.sdf')]:
        supp = Chem.SDMolSupplier(path, removeHs=True)
        for e in summary:
            if e['src'] == src:
                orig[e['key']] = supp[e['idx']]

    graphs, graph_meta = [], []
    tensors = {}
    for e in summary:
        key = e['key']
        poses = {}
        for tag, mol in [('redock', next(Chem.SDMolSupplier(e['best_pose_sdf'], removeHs=True))),
                         ('orig', orig[key])]:
            p = os.path.join(tmpdir, f'{key}_{tag}.sdf')
            w = Chem.SDWriter(p); w.write(mol); w.close()
            mt = mol_to_tensor(parse_sdf_file(path=p), mode='molcraft')
            poses[tag] = mt
        tensors[key] = poses
        graphs.append(build_graphs(pocket_tensor, poses['redock']))
        graph_meta.append((f'{key}|redock', None))
        graphs.append(build_graphs(pocket_tensor, poses['orig']))
        graph_meta.append((f'{key}|orig', None))
        for label, aidx in e['sites'].items():
            graphs.append(build_graphs(pocket_tensor, poses['redock']))
            graph_meta.append((f'{key}|redock', (aidx,)))
            graphs.append(build_graphs(pocket_tensor, poses['orig']))
            graph_meta.append((f'{key}|orig', (aidx,)))
    print(f'[A2] {len(graphs)} graphs')

    model = load_model(device)
    print('[A2] model loaded')
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def psar(p): return float(p[IDX_Sarom])
    def pn(p):
        return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p):
        return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    report = {}
    rows = []
    # sanity: full-info echo
    n_ok, n_tot = 0, 0
    for e in summary:
        kt = tensors[e['key']]['redock']['atom_feature'].long()
        full = results[f"{e['key']}|redock"][None]
        n_ok += int((full.argmax(1) == kt.numpy()).sum()); n_tot += len(kt)
    print(f'[A2 sanity] full-info echo argmax==true: {n_ok}/{n_tot}')

    for e in summary:
        key = e['key']
        entry = {'smiles': e['smiles'], 'cls': e['cls'], 'natoms': e['natoms'],
                 'best_SCORE_INTER': e['best_SCORE_INTER'], 'sites': {}}
        for label, aidx in e['sites'].items():
            s = {}
            for tag in ('redock', 'orig'):
                mk = results[f'{key}|{tag}'][(aidx,)][aidx]
                s[tag] = {'PS': round(ps(mk), 4), 'PSarom': round(psar(mk), 4),
                          'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                          'top1': CLS_NAME[int(mk.argmax())]}
            entry['sites'][label] = s
            rows.append((s['redock']['PS'], s['orig']['PS'], key, label,
                         e['cls'], e['best_SCORE_INTER']))
        report[key] = entry

    print('\n===== Phase A2 decision table: P[S] at diaza sites (redock vs orig pose) =====')
    print(f"{'P[S]redock':>10} {'P[S]orig':>9}  candidate            site           class")
    rows.sort(reverse=True)
    for pr, po, key, label, cls, sc in rows:
        flag = '  <<GATE' if pr > 0.10 else ''
        print(f'{pr*100:9.2f}% {po*100:8.2f}%  {key:>18} {label:>12} {cls:>16}{flag}')

    ps_redock = np.array([r[0] for r in rows])
    gate = [r for r in rows if r[0] > 0.10]
    print(f'\n[A2] P[S]@redock: max={ps_redock.max()*100:.2f}% median={np.median(ps_redock)*100:.2f}% '
          f'>5%: {(ps_redock>0.05).sum()}/{len(ps_redock)}  >10%: {len(gate)}/{len(ps_redock)}')
    report['_gate'] = {'max_PS_redock': round(float(ps_redock.max()), 4),
                       'n_sites': len(ps_redock),
                       'n_gt5': int((ps_redock > 0.05).sum()),
                       'n_gt10': len(gate),
                       'gate_pass_keys': [f'{r[2]}:{r[3]}' for r in gate]}

    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[A2] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
