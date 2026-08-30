#!/usr/bin/env python3
"""t2-DENOVO Phase A6b: two-site endorsement scan for the bis-BT seed pose.

Discovery from run D1: sample_by_seed's final classes come from the model's own
belief p0_h_final (the fixed-atom type enforcement in the final output is
commented out in the model code), so a seed S site whose masked P[S] is low
gets RESAMPLED to N during generation (run-D1 products are C15N3S1 — site4 S
flipped to N exactly as the B3 post-replacement probe predicted: 0.2-8%).

=> the Phase-C seed pose must have BOTH S sites endorsed. This scan probes the
18-atom bis-BT seed (the exact generation-time context, S beliefs visible at
both sites) over a generic rigid pose grid:
  in-plane rotation {0,30,...,150} x flip x translations (in-plane +-3 A @1 A,
  normal +-1 A @0.5 A) x inter-ring torsion {0,45,-45,90} deg about C2..C2
Clash filter vs pocket, then masked probe at BOTH S sites.
Gate: min(P[S]_site_a, P[S]_site_b) > 10% (both sites endorsed).

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseA6b_two_site_scan.py
"""
import os, sys, json

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

IDX_S = MAP_ATI[(16, False)]
IDX_Sarom = MAP_ATI[(16, True)]
K = 13
CLS_NAME = {i: f"{z}{'*' if a else ''}" for i, (z, a) in MAP_ITA.items()}

POCKET_PDB = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
REF_SDF = 'casestudy/kras_g4/ber102_rdock/ber102_ref.sdf'
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseA6b_twosite'
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, 'scan_results.json')
CLASH_CUT = 2.0
MAX_KEEP = 240


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
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * Kx @ Kx
    Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Kx + Kx @ Kx * (1 / (1 + c))


def rot_axis(ax, th):
    ax = ax / np.linalg.norm(ax)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    pxyz = pocket_tensor['pos'].numpy()

    ref = next(Chem.SDMolSupplier(REF_SDF, removeHs=True))
    R = ref.GetConformer().GetPositions()
    ref_cent, ref_long, ref_norm = axes(R)
    e1 = ref_long / np.linalg.norm(ref_long)
    en = ref_norm / np.linalg.norm(ref_norm)
    e2 = np.cross(en, e1); e2 /= np.linalg.norm(e2)

    # bis-BT seed template: take from Phase BC seed (internal geometry canonical)
    seed0 = next(Chem.SDMolSupplier(
        'casestudy/kras_g4/t2_denovo/phaseBC_seeds/seedBC_gp1_endorsed.sdf', removeHs=True))
    P_seed = seed0.GetConformer().GetPositions()
    # S site indices (aromatic S) and C2 indices (carbons between S and N in 5-ring)
    s_idx = [a.GetIdx() for a in seed0.GetAtoms() if a.GetSymbol() == 'S']
    assert len(s_idx) == 2
    frags = [set(f) for f in Chem.GetMolFrags(seed0)]
    assert len(frags) == 2
    c2_idx = []
    for frag_atoms in frags:
        ring5 = None
        for ring in seed0.GetRingInfo().AtomRings():
            if len(ring) == 5 and set(ring) <= frag_atoms:
                ring5 = set(ring)
        for i in ring5:
            a = seed0.GetAtomWithIdx(i)
            if a.GetSymbol() == 'C':
                het = [n for n in a.GetNeighbors() if n.GetIdx() in ring5
                       and seed0.GetAtomWithIdx(n.GetIdx()).GetSymbol() in ('N', 'S')]
                if len(het) == 2:
                    c2_idx.append(i)
    assert len(c2_idx) == 2, c2_idx
    frag_of_c2b = [f for f in frags if c2_idx[1] in f][0]
    print(f'[A6b] S sites {s_idx}, C2 sites {c2_idx}')
    d_c2 = np.linalg.norm(P_seed[c2_idx[0]] - P_seed[c2_idx[1]])
    print(f'[A6b] template C2...C2 = {d_c2:.2f} A')

    c0, l0, n0 = axes(P_seed)
    Ralign = rot_between(n0, en) if True else None
    l0r = l0 @ Ralign.T
    Ralign = rot_between(l0r, e1) @ Ralign
    P_al = (Ralign @ P_seed.T).T
    P_al = P_al + (ref_cent - P_al.mean(0))
    c2a, c2b = c2_idx
    axis_dir = P_al[c2b] - P_al[c2a]
    axis_dir /= np.linalg.norm(axis_dir)

    def apply_torsion(P, deg):
        """rotate the c2b fragment about the C2a->C2b axis through C2b by deg."""
        R = rot_axis(axis_dir, np.deg2rad(deg))
        out = P.copy()
        for i in sorted(frag_of_c2b):
            out[i] = (R @ (P[i] - P[c2b])) + P[c2b]
        return out

    placements = []
    for tor in (0, 45, -45, 90):
        Pt = apply_torsion(P_al, tor) if tor else P_al
        for flip in (1, -1):
            for th in (0, 30, 60, 90, 120, 150):
                Rth = rot_axis(en * flip, np.deg2rad(th))
                Pb = (Rth @ (Pt - ref_cent).T).T + ref_cent
                for dx in (-3, -2, -1, 0, 1, 2, 3):
                    for dy in (-3, -2, -1, 0, 1, 2, 3):
                        for dz in (-1, -0.5, 0, 0.5, 1):
                            Pp = Pb + dx * e1 + dy * e2 + dz * en
                            placements.append((f'tor{tor}_f{flip}_th{th}_{dx}_{dy}_{dz}', Pp))
    print(f'[A6b] {len(placements)} raw placements')

    kept = []
    for name, P in placements:
        dmin = min(np.linalg.norm(P[:, None] - pxyz[None], axis=-1).min(), 99)
        if dmin >= CLASH_CUT:
            kept.append((name, P, dmin))
    print(f'[A6b] {len(kept)} clash-free')
    if len(kept) > MAX_KEEP:
        # diversity: keep spread by hashing strides
        step = len(kept) / MAX_KEEP
        kept = [kept[int(i * step)] for i in range(MAX_KEEP)]
    print(f'[A6b] probing {len(kept)}')

    graphs, graph_meta, ctx = [], [], []
    for name, P, dmin in kept:
        mm = Chem.Mol(seed0)
        conf = Chem.Conformer(18)
        for i in range(18):
            conf.SetAtomPosition(i, Point3D(*P[i]))
        mm.RemoveAllConformers()
        mm.AddConformer(conf, assignId=True)
        path = os.path.join(OUTDIR, f'{name}.sdf')
        w = Chem.SDWriter(path); w.write(mm); w.close()
        mt = mol_to_tensor(parse_sdf_file(path=path), mode='molcraft')
        graphs.append(build_graphs(pocket_tensor, mt))
        graph_meta.append((name, None))
        for si in s_idx:
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((name, (si,)))
        ctx.append((name, dmin))
    print(f'[A6b] {len(graphs)} graphs')

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=40, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p): return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    report, rows = {}, []
    for name, dmin in ctx:
        r = results[name]
        s0 = ps(r[(s_idx[0],)][s_idx[0]])
        s1 = ps(r[(s_idx[1],)][s_idx[1]])
        top0 = CLS_NAME[int(r[(s_idx[0],)][s_idx[0]].argmax())]
        top1 = CLS_NAME[int(r[(s_idx[1],)][s_idx[1]].argmax())]
        both = min(s0, s1)
        rows.append((both, s0, s1, name, dmin, top0, top1))
        report[name] = {'PS_a': round(s0, 4), 'PS_b': round(s1, 4),
                        'minPS': round(both, 4), 'dmin': round(dmin, 2),
                        'top_a': top0, 'top_b': top1}
    rows.sort(key=lambda t: -t[0])
    print('\n===== A6b two-site scan (top 20 by min P[S]) =====')
    for both, s0, s1, name, dmin, t0, t1 in rows[:20]:
        flag = '  <<BOTH-GATE' if both > 0.10 else ''
        print(f'min={both*100:6.2f}% (a={s0*100:6.2f}% b={s1*100:6.2f}%) dmin={dmin:4.1f} '
              f'top=({t0},{t1}) {name}{flag}')
    mins = np.array([r[0] for r in rows])
    n_both = int((mins > 0.10).sum())
    n_both5 = int((mins > 0.05).sum())
    print(f'\n[A6b] n={len(mins)} minPS: max={mins.max()*100:.2f}% median={np.median(mins)*100:.2f}% '
          f'>10%: {n_both}  >5%: {n_both5}')
    report['_summary'] = {'n': len(mins), 'max_minPS': round(float(mins.max()), 4),
                          'n_minPS_gt10': n_both, 'n_minPS_gt5': n_both5,
                          'top10': [r[3] for r in rows[:10]]}
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[A6b] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
