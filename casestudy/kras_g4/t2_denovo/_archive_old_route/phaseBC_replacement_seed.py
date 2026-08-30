#!/usr/bin/env python3
"""t2-DENOVO Phase B+C: probability-endorsed N->S replacement on the 18-atom
seed shape (two separated benzimidazole fragments) -> two-benzothiazole anchor
seed for Phase-D inpainting.

Inputs (all t2-free):
  - generic bis-BM dimer conformer (ETKDG seed 11) at the A6 gate-passing
    placements (P[S] > 10% in-pocket endorsement; intrinsic out-of-pocket
    endorsement 50-63%) + generic centroid placements (R7 convention).
  - the CH2 of the dimer is DELETED before replacement: the seed is the
    18-atom 2-fragment shape; the bridge atom is left for the MODEL to
    generate (free = 1). The full t2 molecule is never constructed here.

Steps:
  B1 re-verify endorsement at the chosen placements (masked P[S] at both pyr sites)
  B2 N->S element replacement on both fragments (aromatic S), sanitize per fragment
     (benzimidazole -> benzothiazole must be chemically legal)
  B3 re-probe the replaced seed at both S sites (endorsement retention)
  C  seed set = 3 endorsed placements + 4 generic centroid placements;
     record C2...C2 distances (must be 2.4-2.7 A insertion range) and clash stats

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseBC_replacement_seed.py
"""
import os, sys, json

PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, 'casestudy/kras_g4/_archive_ber101_era'))

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
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
REF_SDF = 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseBC_seeds'
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, 'phaseBC_report.json')

DIMER_SMILES = 'c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1'
GATE_PLACEMENTS = [  # from A6 scan, P[S] > 10%
    ('gp1', 30, 1, -1.5, -0.75, 0.0, 0.1246),
    ('gp2', 0, 1, 1.5, 1.5, 0.0, 0.1124),
    ('gp3', 0, -1, 1.5, 1.5, 0.0, 0.1124),
]


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


def axes(P):
    c = P.mean(0)
    _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0], vt[2]


def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def rot_axis(ax, th):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def make_dimer_placement():
    """deterministic rebuild of the A6 dimer + gate placements (same code path)."""
    m0 = Chem.MolFromSmiles(DIMER_SMILES)
    mh = Chem.AddHs(Chem.Mol(m0))
    ps = AllChem.ETKDGv3(); ps.randomSeed = 11; ps.useSmallRingTorsions = True
    AllChem.EmbedMolecule(mh, ps)
    AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
    mol = Chem.RemoveHs(mh)
    pyr, ch2 = [], None
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetIsAromatic() and a.GetTotalNumHs() > 0:
            pyr.append(a.GetIdx())
        if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetDegree() == 2 \
                and a.GetTotalNumHs() == 2:
            ch2 = a.GetIdx()
    assert len(pyr) == 2 and ch2 is not None
    P0 = mol.GetConformer().GetPositions()
    c0, l0, n0 = axes(P0)

    ref = next(Chem.SDMolSupplier(REF_SDF, removeHs=True))
    R = ref.GetConformer().GetPositions()
    ref_cent, ref_long, ref_norm = axes(R)
    e1 = ref_long / np.linalg.norm(ref_long)
    en = ref_norm / np.linalg.norm(ref_norm)
    e2 = np.cross(en, e1); e2 /= np.linalg.norm(e2)

    Ra = rot_between(n0, en)
    l0r = l0 @ Ra.T
    Rb = rot_between(l0r, e1)
    Ralign = Rb @ Ra
    P_al = (Ralign @ P0.T).T + (ref_cent - (Ralign @ P0.T).T.mean(0))

    placements = {}
    for tag, th_deg, flip, dx, dy, dz, _ps in GATE_PLACEMENTS:
        Rth = rot_axis(en * flip, np.deg2rad(th_deg))
        P = (Rth @ (P_al - ref_cent).T).T + ref_cent
        P = P + dx * e1 + dy * e2 + dz * en
        placements[tag] = P
    BER = ref_cent
    return mol, pyr, ch2, placements, BER, P_al


def n2s_seed(seed_mol_bm, pyr_seed):
    """N->S (aromatic) on both fragments of the 18-atom seed. Returns new mol."""
    rw = Chem.RWMol(seed_mol_bm)
    for si in pyr_seed:
        a = rw.GetAtomWithIdx(si)
        assert a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0
        a.SetAtomicNum(16)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    Chem.SetAromaticity(out, Chem.AromaticityModel.AROMATICITY_DEFAULT)
    return out


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    pxyz = pocket_tensor['pos'].numpy()

    mol19, pyr, ch2, placements, BER, P_al = make_dimer_placement()

    # 18-atom seed-shape template (delete CH2)
    rw = Chem.RWMol(mol19)
    rw.RemoveAtom(ch2)
    seed_bm = rw.GetMol()
    Chem.SanitizeMol(seed_bm)
    sites_seed = [si if si < ch2 else si - 1 for si in pyr]
    smi_bm = Chem.MolToSmiles(seed_bm)
    assert smi_bm.count('.') == 1, smi_bm
    print(f'[BC] seed shape (pre-replacement): {smi_bm}  sites {sites_seed}')

    # C2...C2 distance in seed shape (C2s were the CH2 neighbors)
    ch2_nbrs = [n.GetIdx() for n in mol19.GetAtomWithIdx(ch2).GetNeighbors()]
    P19 = mol19.GetConformer().GetPositions()
    d_c2c2 = np.linalg.norm(P19[ch2_nbrs[0]] - P19[ch2_nbrs[1]])
    print(f'[BC] generic C2...C2 (ETKDG dimer): {d_c2c2:.2f} A')

    # B2: replacement on template (geometry-independent check)
    seed_bt = n2s_seed(seed_bm, sites_seed)
    smi_bt = Chem.MolToSmiles(seed_bt)
    frag_canon = Chem.MolToSmiles(Chem.MolFromSmiles('c1ccc2scnc2c1'))
    frags = sorted(smi_bt.split('.'))
    assert frags == sorted([frag_canon, frag_canon]), smi_bt
    print(f'[B2] N->S replacement legal: {smi_bt}')

    # build placement coordinate sets (3 endorsed + 4 generic centroid rotations)
    rng = np.random.default_rng(2032)
    pose_sets = []
    for tag, P in placements.items():
        pose_sets.append((f'seedBC_{tag}_endorsed', np.delete(P, ch2, axis=0)))
    for k in range(4):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        P0 = np.delete(P_al, ch2, axis=0)
        P = (q @ P0.T).T
        P = P - P.mean(0) + BER
        pose_sets.append((f'seedBC_centroid{k}_generic', P))

    graphs, graph_meta, ctx = [], [], []
    seed_files = []
    report = {'dimer_smiles': DIMER_SMILES, 'seed_bm_smiles': smi_bm,
              'seed_bt_smiles': smi_bt, 'c2c2_dimer': round(float(d_c2c2), 3),
              'B1_pre': {}, 'B3_post': {}, 'seeds': {}}

    for name, Ps in pose_sets:
        dmin = min(np.linalg.norm(Ps[:, None] - pxyz[None], axis=-1).min(), 99)
        d_c2 = np.linalg.norm(Ps[ch2_nbrs[0] if ch2_nbrs[0] < ch2 else ch2_nbrs[0] - 1] -
                              Ps[ch2_nbrs[1] if ch2_nbrs[1] < ch2 else ch2_nbrs[1] - 1])
        # pre-replacement context (B1 re-verify)
        m_pre = Chem.Mol(seed_bm)
        conf = Chem.Conformer(18)
        for i in range(18):
            conf.SetAtomPosition(i, Point3D(*Ps[i]))
        m_pre.RemoveAllConformers(); m_pre.AddConformer(conf, assignId=True)
        p_pre = os.path.join(OUTDIR, f'{name}_pre.sdf')
        w = Chem.SDWriter(p_pre); w.write(m_pre); w.close()
        mt = mol_to_tensor(parse_sdf_file(path=p_pre), mode='molcraft')
        graphs.append(build_graphs(pocket_tensor, mt)); graph_meta.append((f'{name}|pre', None))
        for si in sites_seed:
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((f'{name}|pre', (si,)))
        # post-replacement seed (B3 + seed output)
        m_post = Chem.Mol(seed_bt)
        conf = Chem.Conformer(18)
        for i in range(18):
            conf.SetAtomPosition(i, Point3D(*Ps[i]))
        m_post.RemoveAllConformers(); m_post.AddConformer(conf, assignId=True)
        p_post = os.path.join(OUTDIR, f'{name}.sdf')
        m_post.SetProp('_Name', name)
        w = Chem.SDWriter(p_post); w.write(m_post); w.close()
        seed_files.append(p_post)
        mt2 = mol_to_tensor(parse_sdf_file(path=p_post), mode='molcraft')
        graphs.append(build_graphs(pocket_tensor, mt2)); graph_meta.append((f'{name}|post', None))
        for si in sites_seed:
            graphs.append(build_graphs(pocket_tensor, mt2))
            graph_meta.append((f'{name}|post', (si,)))
        cls = sorted(MAP_ITA[int(k)] for k in mt2['atom_feature'].long())
        exp = sorted([(6, True)] * 14 + [(7, True), (7, True), (16, True), (16, True)])
        ok_cls = (cls == exp)
        report['seeds'][name] = {'dmin_pocket': round(float(dmin), 2),
                                 'c2c2': round(float(d_c2), 3), 'classes_ok': ok_cls}
        print(f'[BC] {name}: dmin={dmin:.2f} C2...C2={d_c2:.2f} classes={"OK" if ok_cls else "BAD"}')

    print(f'[BC] {len(graphs)} graphs (pre/post probes)')
    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p): return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    print('\n===== B1 (pre-replacement P[S]) vs B3 (post-replacement P[S] at S sites) =====')
    for name, _ in pose_sets:
        r_pre = results[f'{name}|pre']
        r_post = results[f'{name}|post']
        pre = [round(ps(r_pre[(si,)][si]), 4) for si in sites_seed]
        post = [round(ps(r_post[(si,)][si]), 4) for si in sites_seed]
        pn_post = [round(pn(r_post[(si,)][si]), 4) for si in sites_seed]
        top_post = [CLS_NAME[int(r_post[(si,)][si].argmax())] for si in sites_seed]
        report['B1_pre'][name] = pre
        report['B3_post'][name] = {'PS': post, 'PN': pn_post, 'top1': top_post}
        print(f'  {name:>34}: pre P[S]={[f"{x*100:.1f}%" for x in pre]}  '
              f'post P[S]={post} top1={top_post}')
    report['seed_files'] = seed_files
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[BC] seeds -> {OUTDIR} ({len(seed_files)} files), report -> {OUT_JSON}')


if __name__ == '__main__':
    main()
