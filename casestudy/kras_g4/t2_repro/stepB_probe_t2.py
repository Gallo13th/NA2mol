#!/usr/bin/env python3
"""t2 Phase1 core probe: aromatic-aware masked element probe at the BER102-docked t2 pose.

Protocol identical to ber102_rdock/step3_probe_ber102.py (same ckpt/pocket/t=1/beta1/gamma).

Experiments:
  exp1 arom_base : all-C aromatic skeleton (18 arom C + 1 sp3 C), per-position mask
                   -> at thiazole-S sites (idx 4,15): P[S*] ch10 / P[S] ch9 ?  (core decision)
  exp2 full_t2   : true t2 elements (2x S* ch10, 2x N* ch4, 15x arom C*, 1x sp3 C ch1?),
                   per-position mask -> P[true]; S-site P[S] = endorsement of S given t2 context
  exp3 allN_iso  : S->NH isoelectronic lock (benzimidazole analog), mask ex-S sites
                   -> P[S] = replacement backing from model-preferred all-N context
  exp4 pose_robust: exp1/exp2 S-sites on top-5 docked poses (P[S] stability)
"""
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
from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MAP_ITA

# NOTE: the model's 13-class table is dataloader/featurizer.py's (NOT util/mol_utils.py):
#   ... (15,True)=9, (16,False)=10, (16,True)=11, (17,False)=12
IDX_C = MAP_ATI[(6, False)]      # 1
IDX_Carom = MAP_ATI[(6, True)]   # 2
IDX_N = MAP_ATI[(7, False)]      # 3
IDX_Narom = MAP_ATI[(7, True)]   # 4
IDX_S = MAP_ATI[(16, False)]     # 10
IDX_Sarom = MAP_ATI[(16, True)]  # 11
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_ITA.items()}

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/_archive_ber101_era'))
import util.mol_utils  # noqa: F401  (pre-load fragment index while cwd is still PROJECT_ROOT;
#                      the archived probe script hijacks cwd to casestudy/ at import time)
from probe_element_prob import load_model, build_graphs  # noqa: E402
os.chdir(PROJECT_ROOT)

T2_SMILES = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'
POCKET_PDB = os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb')
BEST_SDF = os.path.join(SCRIPT_DIR, 't2_best_pose.sdf')
TOP5_SDF = os.path.join(SCRIPT_DIR, 't2_top5_poses.sdf')
OUT_JSON = os.path.join(SCRIPT_DIR, 'probe_t2_results.json')
NATOMS = 19
S_SITES = [4, 15]
N_SITES = [8, 16]
BRIDGE = 6


def make_skeleton(pose_coords):
    """half-blind base A (benzothiophene-type): S-sites kept as aromatic S,
    N-sites -> aromatic CH. Chemically valid (benzothiophene kekulizes)."""
    tgt = Chem.MolFromSmiles(T2_SMILES)
    assert tgt.GetNumAtoms() == NATOMS
    tgt_arom = {a.GetIdx(): a.GetIsAromatic() for a in tgt.GetAtoms()}
    rw = Chem.RWMol(tgt)
    for a in rw.GetAtoms():
        if a.GetIdx() in N_SITES:
            a.SetAtomicNum(6)
            a.SetFormalCharge(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(1)
            a.SetNoImplicit(False)
    skel = rw.GetMol()
    Chem.SanitizeMol(skel)
    arom2 = {a.GetIdx(): a.GetIsAromatic() for a in skel.GetAtoms()}
    assert all(arom2[i] == tgt_arom[i] for i in range(NATOMS)), 'aromaticity drift'
    assert sum(arom2.values()) == 18
    conf = Chem.Conformer(NATOMS)
    for i in range(NATOMS):
        conf.SetAtomPosition(i, [float(x) for x in pose_coords[i]])
    skel.RemoveAllConformers()
    skel.AddConformer(conf, assignId=True)
    return skel


def make_full_t2(pose_coords):
    """true t2 molecule with pose coords (SMILES atom order)."""
    tgt = Chem.MolFromSmiles(T2_SMILES)
    m = Chem.Mol(tgt)
    conf = Chem.Conformer(NATOMS)
    for i in range(NATOMS):
        conf.SetAtomPosition(i, [float(x) for x in pose_coords[i]])
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    return m


def make_allN_iso(pose_coords):
    """S->NH isoelectronic lock (benzimidazole-CH2-benzothiazole... no: both S->NH)."""
    tgt = Chem.MolFromSmiles(T2_SMILES)
    rw = Chem.RWMol(tgt)
    for i in S_SITES:
        a = rw.GetAtomWithIdx(i)
        assert a.GetSymbol() == 'S'
        a.SetAtomicNum(7)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1)
        a.SetNoImplicit(False)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    assert Chem.MolToSmiles(m) == Chem.MolToSmiles(Chem.MolFromSmiles('c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1')), \
        Chem.MolToSmiles(m)
    conf = Chem.Conformer(NATOMS)
    for i in range(NATOMS):
        conf.SetAtomPosition(i, [float(x) for x in pose_coords[i]])
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    return m


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


def write_tmp(mol, path):
    w = Chem.SDWriter(path)
    w.write(mol)
    w.close()
    return path


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[probe] device={device}')

    pose = next(Chem.SDMolSupplier(BEST_SDF, removeHs=True, sanitize=False))
    Chem.SanitizeMol(pose)
    assert [a.GetSymbol() for a in pose.GetAtoms()] == [a.GetSymbol() for a in Chem.MolFromSmiles(T2_SMILES).GetAtoms()]
    coords = pose.GetConformer().GetPositions()

    skel = make_skeleton(coords)
    full2 = make_full_t2(coords)
    alln = make_allN_iso(coords)
    write_tmp(skel, os.path.join(SCRIPT_DIR, 'skeleton_aromatic_t2.sdf'))
    write_tmp(full2, os.path.join(SCRIPT_DIR, 't2_pose_chem.sdf'))
    write_tmp(alln, os.path.join(SCRIPT_DIR, 't2_allN_iso.sdf'))

    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    print(f"[probe] BER102 pocket atoms: {pocket_tensor['pos'].shape[0]}")

    tmpdir = '/tmp/opencode/t2_probe'
    os.makedirs(tmpdir, exist_ok=True)
    base_skel = mol_to_tensor(parse_sdf_file(path=write_tmp(skel, f'{tmpdir}/skel.sdf')), mode='molcraft')
    base_full = mol_to_tensor(parse_sdf_file(path=write_tmp(full2, f'{tmpdir}/full.sdf')), mode='molcraft')
    base_alln = mol_to_tensor(parse_sdf_file(path=write_tmp(alln, f'{tmpdir}/alln.sdf')), mode='molcraft')

    # true classes of t2
    full_ktrue = base_full['atom_feature'].long().tolist()
    alln_ktrue = base_alln['atom_feature'].long().tolist()
    print('[probe] t2 true classes:', [CLS_NAME[k] for k in full_ktrue])
    print('[probe] allN iso classes:', [CLS_NAME[k] for k in alln_ktrue])
    assert full_ktrue[S_SITES[0]] == IDX_Sarom and full_ktrue[S_SITES[1]] == IDX_Sarom
    assert alln_ktrue[S_SITES[0]] == IDX_Narom and alln_ktrue[S_SITES[1]] == IDX_Narom

    graphs, graph_meta = [], []
    # exp1: arom_base (all-C skeleton), mask each position
    for m in [None] + [(i,) for i in range(NATOMS)]:
        graphs.append(build_graphs(pocket_tensor, base_skel))
        graph_meta.append(('exp1_arom_base', m))
    # exp2: full t2 elements, mask each position
    for m in [None] + [(i,) for i in range(NATOMS)]:
        graphs.append(build_graphs(pocket_tensor, base_full))
        graph_meta.append(('exp2_full_t2', m))
    # exp3: all-N iso, mask ex-S sites + N sites + bridge
    for m in [None] + [(i,) for i in S_SITES + N_SITES + [BRIDGE]]:
        graphs.append(build_graphs(pocket_tensor, base_alln))
        graph_meta.append(('exp3_allN_iso', m))
    print(f'[probe] {len(graphs)} graphs')

    # exp4: pose robustness on top-5 poses (exp1-style skeleton, S-sites + exp2-style S-lock)
    top5 = [m for m in Chem.SDMolSupplier(TOP5_SDF, removeHs=True, sanitize=False)]
    e4_meta = []
    for pi, pm in enumerate(top5):
        Chem.SanitizeMol(pm)
        co = pm.GetConformer().GetPositions()
        sk = make_skeleton(co)
        fu = make_full_t2(co)
        bsk = mol_to_tensor(parse_sdf_file(path=write_tmp(sk, f'{tmpdir}/sk_p{pi}.sdf')), mode='molcraft')
        bfu = mol_to_tensor(parse_sdf_file(path=write_tmp(fu, f'{tmpdir}/fu_p{pi}.sdf')), mode='molcraft')
        for i in S_SITES:
            graphs.append(build_graphs(pocket_tensor, bsk)); graph_meta.append((f'exp4_pose{pi}_arom', (i,)))
            graphs.append(build_graphs(pocket_tensor, bfu)); graph_meta.append((f'exp4_pose{pi}_full', (i,)))
        e4_meta.append(f'exp4_pose{pi}')

    model = load_model(device)
    print('[probe] model loaded')
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[IDX_N] + p[IDX_Narom])
    def pc(p): return float(p[IDX_C] + p[IDX_Carom])

    report = {}

    # ---- exp1 ----
    e1 = results['exp1_arom_base']
    e1_full = e1[None]
    e1_masked = np.stack([e1[(i,)][i] for i in range(NATOMS)])
    echo = int((e1_full.argmax(1) == np.array(base_skel['atom_feature'].long())).sum())
    print(f'\n[sanity] exp1 echo full-info argmax==C class: {echo}/19')
    print('\n===== Exp1: benzothiophene base (S kept, N->CH), masked probs =====')
    print(f"{'atom':>4} {'true(t2)':>8} | {'P[S]':>6} {'P[S*]':>6} {'P[N]':>6} {'P[C]':>6} {'top1':>5}")
    report['exp1'] = {}
    for i in range(NATOMS):
        mk = e1_masked[i]
        sym = ['S', 'N', 'C', 'C(H2)'][0] if i in S_SITES else ('N' if i in N_SITES else ('C(H2)' if i == BRIDGE else 'C'))
        print(f'{i:>4} {sym:>8} | {ps(mk)*100:5.2f}% {mk[IDX_Sarom]*100:5.2f}% {pn(mk)*100:5.2f}% {pc(mk)*100:5.2f}% {CLS_NAME[int(mk.argmax())]:>5}')
        report['exp1'][i] = {'tgt': sym, 'PS': round(ps(mk), 4), 'PSarom': round(float(mk[IDX_Sarom]), 4),
                             'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                             'top1': CLS_NAME[int(mk.argmax())],
                             'probs': [round(float(x), 5) for x in mk]}
    print('\n----- Exp1 CORE: S sites (4,15) -----')
    for i in S_SITES:
        mk = e1_masked[i]
        print(f'  S{i}: P[S]={ps(mk)*100:.2f}% [S*={mk[IDX_Sarom]*100:.2f}% S={mk[IDX_S]*100:.2f}%] '
              f'P[N]={pn(mk)*100:.2f}% P[C]={pc(mk)*100:.2f}% top1 {CLS_NAME[int(mk.argmax())]}({mk.max()*100:.1f}%)')

    # ---- exp2 ----
    e2 = results['exp2_full_t2']
    e2_full = e2[None]
    e2_masked = np.stack([e2[(i,)][i] for i in range(NATOMS)])
    tgt_cls = np.array(full_ktrue)
    echo2 = int((e2_full.argmax(1) == tgt_cls).sum())
    print(f'\n[sanity] exp2 echo: {echo2}/19')
    print('\n===== Exp2: full t2 element lock, per-position P[true] =====')
    report['exp2'] = {}
    ptrue_all = []
    for i in range(NATOMS):
        mk = e2_masked[i]
        pt = float(mk[tgt_cls[i]])
        ptrue_all.append(pt)
        print(f'{i:>4} {CLS_NAME[int(tgt_cls[i])]:>4} | P[true]={pt*100:6.2f}%  P[S]={ps(mk)*100:5.2f}% '
              f'P[N]={pn(mk)*100:5.2f}% P[C]={pc(mk)*100:5.2f}%  top1 {CLS_NAME[int(mk.argmax())]}')
        report['exp2'][i] = {'tgt': CLS_NAME[int(tgt_cls[i])], 'P_true': round(pt, 4),
                             'PS': round(ps(mk), 4), 'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                             'top1': CLS_NAME[int(mk.argmax())],
                             'probs': [round(float(x), 5) for x in mk]}
    ptrue_all = np.array(ptrue_all)
    print(f'  summary: mean P[true]={ptrue_all.mean()*100:.2f}%  min={ptrue_all.min()*100:.2f}% (atom {int(ptrue_all.argmin())})  >50%: {(ptrue_all > 0.5).sum()}/19')
    for i in S_SITES:
        print(f'  S{i}: P[S]={ps(e2_masked[i])*100:.2f}%  (element-replacement endorsement)')
    report['exp2_summary'] = {'mean_P_true': round(float(ptrue_all.mean()), 4),
                              'min_P_true': round(float(ptrue_all.min()), 4),
                              'n_gt50': int((ptrue_all > 0.5).sum())}

    # ---- exp3 ----
    e3 = results['exp3_allN_iso']
    print('\n===== Exp3: all-N isoelectronic lock (S->NH), masked at key sites =====')
    report['exp3'] = {}
    for i in S_SITES + N_SITES + [BRIDGE]:
        mk = e3[(i,)][i]
        lbl = 'ex-S' if i in S_SITES else ('N' if i in N_SITES else 'bridge')
        print(f'  {i:>2} ({lbl:>6}, now {CLS_NAME[alln_ktrue[i]]}): P[S]={ps(mk)*100:6.2f}% [S*={mk[IDX_Sarom]*100:.2f}%] '
              f'P[N]={pn(mk)*100:6.2f}% P[C]={pc(mk)*100:6.2f}% top1 {CLS_NAME[int(mk.argmax())]}')
        report['exp3'][i] = {'label': lbl, 'now': CLS_NAME[alln_ktrue[i]],
                             'PS': round(ps(mk), 4), 'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                             'top1': CLS_NAME[int(mk.argmax())],
                             'probs': [round(float(x), 5) for x in mk]}

    # ---- exp4 ----
    print('\n===== Exp4: S-site P[S] across top-5 docked poses =====')
    report['exp4'] = {}
    for pi in range(5):
        row = {}
        for i in S_SITES:
            a = results[f'exp4_pose{pi}_arom'][(i,)][i]
            f = results[f'exp4_pose{pi}_full'][(i,)][i]
            row[i] = {'arom_PS': round(ps(a), 4), 'full_PS': round(ps(f), 4)}
            print(f'  pose{pi} S{i}: arom-base P[S]={ps(a)*100:5.2f}%   full-t2 P[S]={ps(f)*100:5.2f}%')
        report['exp4'][f'pose{pi}'] = row

    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(SCRIPT_DIR, 'probe_t2_exp1_masked.npy'), e1_masked)
    np.save(os.path.join(SCRIPT_DIR, 'probe_t2_exp2_masked.npy'), e2_masked)
    print(f'\n[saved] {OUT_JSON}')


if __name__ == '__main__':
    main()
