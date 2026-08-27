#!/usr/bin/env python3
"""t2-DENOVO Phase A6: probe-guided rigid pose scan of the generic bis-BM dimer.

The model's P[S] is pose-localized (same topology: 84% at the deep-stacking pose,
9% at near poses, ~1% elsewhere). Instead of guessing one pose, scan a generic
rigid grid anchored at the pocket's natural reference (berberine crystal pose —
experimental co-crystal ligand, NOT t2) and let the model's own probabilities
find the S-endorsing locus:

  bases    : berberine-axis placement (long-axis/normal aligned, in-plane rot scan)
             + local perturbations of the best A4 dimer pose (P[S]=9% context)
  scan     : in-plane rot {0,30,60,90,120,150} x flip {+-} x translation grid
             (in-plane +-1.5 A @0.75, normal +-0.75 A @0.75)
  filter   : clash >= 2.0 A vs probe-pocket atoms; max 220 survivors (spread)
  probe    : F4b seed shape (18 atoms), both pyrrolic-N sites

Gate: P[S] > 10% -> Phase B on the best placement.

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseA6_posescan.py
"""
import os, sys, json, glob

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
A4_BEST = 'casestudy/kras_g4/t2_denovo/phaseA4_dimer/dimer2_gen2k1271xgen2k473_p0_F4.sdf'
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseA6_posescan'
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, 'probe_results.json')

DIMER_SMILES = 'c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1'
CLASH_CUT = 2.0
MAX_KEEP = 220


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
    e2 = np.cross(ref_norm, e1); e2 /= np.linalg.norm(e2)
    en = ref_norm / np.linalg.norm(ref_norm)

    # dimer conformer (ETKDG) + pyr sites + ch2
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
    # align l0->e1, n0->en once
    Ra = rot_between(n0, en)
    l0r = l0 @ Ra.T
    Rb = rot_between(l0r, e1)
    Ralign = Rb @ Ra
    P_al = (Ralign @ P0.T).T + (ref_cent - (Ralign @ P0.T).T.mean(0))

    # A4 best pose local base
    a4 = next(Chem.SDMolSupplier(A4_BEST, removeHs=True))
    P_a4 = a4.GetConformer().GetPositions()

    def build_seed_shape(P):
        """18-atom 2-fragment seed shape coords from 19-atom dimer coords."""
        return np.delete(P, ch2, axis=0)

    seed_mol = Chem.Mol(mol)
    rw = Chem.RWMol(seed_mol)
    rw.RemoveAtom(ch2)
    seed_mol = rw.GetMol()
    Chem.SanitizeMol(seed_mol)
    sites_seed = [si if si < ch2 else si - 1 for si in pyr]

    # ---- generate placements ----
    placements = []
    # (a) berberine-referenced grid
    for flip in (1, -1):
        for th_deg in (0, 30, 60, 90, 120, 150):
            Rth = rot_axis(en * flip, np.deg2rad(th_deg))
            Pb = (Rth @ (P_al - ref_cent).T).T + ref_cent
            for dx in (-1.5, -0.75, 0, 0.75, 1.5):
                for dy in (-1.5, -0.75, 0, 0.75, 1.5):
                    for dz in (-0.75, 0, 0.75):
                        Pp = Pb + dx * e1 + dy * e2 + dz * en
                        placements.append((f'ber_t{th_deg}_f{flip}_{dx}_{dy}_{dz}', Pp))
    # (b) A4-best local perturbations
    for rx in (-15, 0, 15):
        for rz in (-15, 0, 15):
            _, lx, nx = axes(P_a4)
            Rr = rot_axis(lx, np.deg2rad(rx)) @ rot_axis(nx, np.deg2rad(rz))
            Pr = (Rr @ (P_a4 - P_a4.mean(0)).T).T + P_a4.mean(0)
            for dx in (-0.75, 0, 0.75):
                for dz in (-0.75, 0, 0.75):
                    e1l, _, enl = axes(P_a4)
                    e2l = np.cross(enl, e1l); e2l /= np.linalg.norm(e2l)
                    Pp = Pr + dx * e1l + dz * enl
                    placements.append((f'a4_r{rx}_{rz}_{dx}_{dz}', Pp))

    # clash filter
    kept = []
    for name, P in placements:
        dmin = min(np.linalg.norm(P[:, None] - pxyz[None], axis=-1).min(), 99)
        if dmin >= CLASH_CUT:
            kept.append((name, P, dmin))
    print(f'[A6] {len(placements)} placements -> {len(kept)} clash-free (>= {CLASH_CUT} A)')
    # spread-cap: sort by dmin desc, keep max
    if len(kept) > MAX_KEEP:
        kept.sort(key=lambda t: -t[2])
        kept = kept[:MAX_KEEP]
    print(f'[A6] probing {len(kept)} placements')

    graphs, graph_meta, ctx = [], [], []
    conf_proto = seed_mol.GetConformer()
    for name, P, dmin in kept:
        Ps = build_seed_shape(P)
        mm = Chem.Mol(seed_mol)
        conf = Chem.Conformer(18)
        for i in range(18):
            conf.SetAtomPosition(i, Point3D(*Ps[i]))
        mm.RemoveAllConformers()
        mm.AddConformer(conf, assignId=True)
        path = os.path.join(OUTDIR, f'{name}.sdf')
        w = Chem.SDWriter(path); w.write(mm); w.close()
        mt = mol_to_tensor(parse_sdf_file(path=path), mode='molcraft')
        graphs.append(build_graphs(pocket_tensor, mt))
        graph_meta.append((name, None))
        for si in sites_seed:
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((name, (si,)))
        ctx.append({'name': name, 'dmin': round(float(dmin), 2)})
    print(f'[A6] {len(graphs)} graphs')

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=40, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p): return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    report, rows = {}, []
    for e in ctx:
        r = results[e['name']]
        sites = {}
        for si in sites_seed:
            mk = r[(si,)][si]
            sites[si] = {'PS': round(ps(mk), 4), 'PSarom': round(float(mk[IDX_Sarom]), 4),
                         'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                         'top1': CLS_NAME[int(mk.argmax())]}
            rows.append((ps(mk), e['name'], si, e['dmin']))
        report[e['name']] = {**e, 'sites': sites}

    rows.sort(key=lambda t: -t[0])
    print('\n===== A6 pose-scan probe (top 25) =====')
    for p, name, si, dmin in rows[:25]:
        flag = '  <<GATE' if p > 0.10 else ''
        print(f'{p*100:7.2f}%  site{si}  dmin={dmin:4.1f}  {name}{flag}')
    allps = np.array([r[0] for r in rows])
    print(f'\n[A6] n={len(allps)} max={allps.max()*100:.2f}% median={np.median(allps)*100:.2f}% '
          f'>5%: {(allps>0.05).sum()}  >10%: {(allps>0.10).sum()}')
    report['_summary'] = {'n': len(allps), 'max': round(float(allps.max()), 4),
                          'median': round(float(np.median(allps)), 4),
                          'n_gt5': int((allps > 0.05).sum()),
                          'n_gt10': int((allps > 0.10).sum()),
                          'best': rows[0][1] if rows else None,
                          'best_PS': round(rows[0][0], 4) if rows else None}
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[A6] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
