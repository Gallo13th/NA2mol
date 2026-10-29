#!/usr/bin/env python3
"""Phase1 supplement 2: masked probe at the [nH] site (benzothiazole-S analog) of
REAL library benzimidazole molecules in their generated BER102-pocket poses."""
import json, os, sys

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
from dataloader.featurizer import MAP_ATOM_TYPE_AROMATIC_TO_INDEX as MAP_ATI

IDX_N = MAP_ATI[(7, False)]
IDX_Narom = MAP_ATI[(7, True)]
IDX_S = MAP_ATI[(16, False)]
IDX_Sarom = MAP_ATI[(16, True)]
K = 13

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/_archive_ber101_era'))
import util.mol_utils  # noqa: F401  pre-load fragment index (cwd guard)
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJECT_ROOT)

POCKET_PDB = os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb')
BENZIMID = Chem.MolFromSmarts('c1ccc2nc[nX2]c2c1')
BENZIMID2 = Chem.MolFromSmarts('c1ccc2[nX2]cnc2c1')


def nh_site(mol, match):
    """return the [nH] atom idx of the matched benzimidazole (S-analog site) + the other N idx"""
    s = set(match)
    ns = [i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
    if len(ns) != 2:
        return None, None
    # [nH]: has explicit/implicit H and degree 2 in ring (pyrrolic)
    nh = [i for i in ns if mol.GetAtomWithIdx(i).GetTotalNumHs() == 1 and
          len([n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in s]) == 2]
    if len(nh) != 1:
        return None, None
    other = [i for i in ns if i != nh[0]]
    return nh[0], other[0]


def clone_mt(base):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}


@torch.no_grad()
def run_forward(model, loader, graph_meta, beta1):
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
            _, probe_idx = graph_meta[gstart + g_local]
            if probe_idx is not None:
                gi = np.where(gids == g_local)[0][probe_idx]
                theta[gi, :] = 1.0 / K
        t = torch.ones(N, 1, device=device)
        mu = batch.mol_pos * gamma / 2.0
        coord_pred, p0_h_pred, k_hat, _, _ = model.interdependency_modeling(
            time=t, pocket_pos=batch.pocket_pos / 2,
            pocket_v=batch.pocket_atom_feature.float(),
            batch_pocket=batch.pocket_element_batch,
            ed_pos=torch.empty((0, 3), device=device),
            ed_v=torch.empty((0, 4), device=device),
            batch_ed=batch.ed_ed_code_batch,
            batch_ligand=bl, theta_h_t=theta, mu_pos_t=mu,
            gamma_coord=torch.full((N, 1), gamma, device=device),
            pocket_res=None, pocket_ele=None)
        probs = torch.softmax(torch.log(p0_h_pred.clamp(min=1e-12)), dim=-1)
        for g_local in range(batch.num_graphs):
            key, probe_idx = graph_meta[gstart + g_local]
            rows = np.where(gids == g_local)[0]
            results.setdefault(key, {})[probe_idx] = probs[rows].float().cpu().numpy()
        gstart += batch.num_graphs
    return results


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')

    supp = Chem.SDMolSupplier('casestudy/kras_g4/ber102_gen2k/valid_mols.sdf', removeHs=True)
    cands = []
    for i, m in enumerate(supp):
        if m is None or not m.GetNumConformers() or m.GetNumAtoms() >= 100:
            continue
        for patt in (BENZIMID, BENZIMID2):
            for mt in m.GetSubstructMatches(patt):
                nh, other = nh_site(m, mt)
                if nh is not None:
                    cands.append((i, m, nh, other))
                    break
            if cands and cands[-1][0] == i:
                break
    print(f'ber102 benzimidazole cands with resolvable [nH] site: {len(cands)}')

    tmpdir = '/tmp/opencode/t2_libprobe'
    os.makedirs(tmpdir, exist_ok=True)
    graphs, graph_meta, recs = [], [], {}
    for k, (libi, m, nh, other) in enumerate(cands):
        p = f'{tmpdir}/m{libi}.sdf'
        w = Chem.SDWriter(p); w.write(m); w.close()
        try:
            mt = mol_to_tensor(parse_sdf_file(path=p), mode='molcraft')
        except Exception as e:
            print(f'  skip #{libi}: {e}')
            continue
        key = f'lib{libi}'
        graphs.append(build_graphs(pocket_tensor, mt)); graph_meta.append((key, None))
        for site in (nh, other):
            graphs.append(build_graphs(pocket_tensor, mt)); graph_meta.append((key, site))
        recs[key] = {'lib_idx': libi, 'nh': nh, 'other_n': other,
                     'smiles': Chem.MolToSmiles(m), 'natoms': int(mt['pos'].shape[0])}

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[IDX_N] + p[IDX_Narom])

    report = {}
    print(f"\n{'lib#':>6} {'P[S]@nH':>9} {'P[N]@nH':>9} {'top1@nH':>8} | {'P[S]@otherN':>10} | smiles")
    ps_list = []
    for key, rec in recs.items():
        r = results[key]
        nhp = r[rec['nh']][rec['nh']]
        otp = r[rec['other_n']][rec['other_n']]
        top1 = int(nhp.argmax())
        name = {v: k for k, v in MAP_ATI.items()}
        t1 = f"{name[top1][0]}{'*' if name[top1][1] else ''}"
        ps_list.append(ps(nhp))
        print(f"{rec['lib_idx']:>6} {ps(nhp)*100:8.2f}% {pn(nhp)*100:8.2f}% {t1:>8} | {ps(otp)*100:9.2f}% | {rec['smiles'][:70]}")
        report[key] = {'lib_idx': rec['lib_idx'], 'PS_at_nH': round(ps(nhp), 4),
                       'PSarom_at_nH': round(float(nhp[IDX_Sarom]), 4),
                       'PN_at_nH': round(pn(nhp), 4), 'top1': t1,
                       'PS_at_otherN': round(ps(otp), 4),
                       'probs_nH': [round(float(x), 5) for x in nhp]}
    ps_arr = np.array(ps_list)
    print(f'\nsummary P[S]@[nH] over {len(ps_arr)} library benzimidazoles: '
          f'min={ps_arr.min()*100:.2f}% med={np.median(ps_arr)*100:.2f}% max={ps_arr.max()*100:.2f}% '
          f'n>10%: {(ps_arr > 0.10).sum()}')
    with open(os.path.join(SCRIPT_DIR, 'probe_lib_benzimidazole.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('[saved] probe_lib_benzimidazole.json')


if __name__ == '__main__':
    main()
