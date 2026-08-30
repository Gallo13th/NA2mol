#!/usr/bin/env python3
"""t2-DENOVO Phase A3 (candidate-pool expansion after A2 gate failure).

A2 result: single-isostere redock poses give P[S] <= 1.25% (0/48 > 10%) — the
redock-only bet did NOT flip the probe. Known contrast: at t2 geometry the
all-N isostere context (bis-fused-diaza + CH2 bridge, exp3) gives P[S]=83.7%
while single benzimidazoles give ~0.1%. => context (2nd fused ring / bridge),
not just pose, drives P[S].

This script tests CONTEXT hypotheses using only model-product coordinates
(gen2k redock poses) + generic chemistry (no t2 topology/conformation):

  F1 trimmed  : benzimidazole 9-atom core alone at its redock-pose coords
                (does the substituent decoration suppress P[S]?)
  F2 pair     : core1 (redock pose) + core2 placed by generic bridge geometry
                (C2...C2 in {2.45,2.55,2.65} A, in-plane, coplanar normals,
                clash-checked vs pocket) — NO bridge atom
  F3 pair+CH2 : same + generic sp3 CH2 at the bisector (C-CH2 1.51 A,
                angle 112 deg, in-plane)

Probe: masked P[S] (ch10+ch11) at the pyrrolic-N site of each core.
Gate: any context with P[S] > 10% at a pyrrolic site.

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseA3_context_probe.py
"""
import os, sys, json, itertools

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
SUMMARY = 'casestudy/kras_g4/t2_denovo/phaseA_redock/summary.json'
A2_JSON = 'casestudy/kras_g4/t2_denovo/phaseA_probe_results.json'
OUT_JSON = 'casestudy/kras_g4/t2_denovo/phaseA3_results.json'
OUT_SDF_DIR = 'casestudy/kras_g4/t2_denovo/phaseA3_contexts'
os.makedirs(OUT_SDF_DIR, exist_ok=True)

BIM_T = [Chem.MolFromSmiles('c1ccc2nc[nH]c2c1'), Chem.MolFromSmiles('c1ccc2[nH]cnc2c1')]


def get_bm_core(m):
    """template-copy extraction: benzimidazole SMILES template + coords mapped via
    substructure match (guarantees a sanitizable aromatic core)."""
    for tmpl in BIM_T:
        for mt in m.GetSubstructMatches(tmpl):
            core = Chem.Mol(tmpl)
            conf = Chem.Conformer(9)
            pos = m.GetConformer().GetPositions()
            for i, o in enumerate(mt):
                conf.SetAtomPosition(i, Point3D(*pos[o]))
            core.RemoveAllConformers()
            core.AddConformer(conf, assignId=True)
            pyr, pyd, c2 = None, None, None
            ring5 = None
            for ring in core.GetRingInfo().AtomRings():
                if len(ring) == 5:
                    ring5 = set(ring)
            if ring5 is None:
                continue
            for i in ring5:
                a = core.GetAtomWithIdx(i)
                if a.GetSymbol() == 'N':
                    if a.GetTotalNumHs() > 0:
                        pyr = i
                    else:
                        pyd = i
                elif a.GetSymbol() == 'C':
                    het = [n for n in a.GetNeighbors() if n.GetIdx() in ring5
                           and n.GetSymbol() == 'N']
                    if len(het) == 2:
                        c2 = i
            if pyr is not None and c2 is not None:
                return core, pyr, pyd, c2
    return None


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


def plane_normal(P):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[2]


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


def build_pair(core1, core2, c2_1, c2_2, v, d, with_ch2=False):
    """core1 stays; core2 rotated so its C2->ringcenter dir = -v, normal aligned to
    core1's normal, C2 pinned at core1.C2 + d*v. Optionally add generic sp3 CH2.
    Built via CombineMols (aromatic templates stay sanitizable)."""
    P1 = core1.GetConformer().GetPositions()
    P2 = core2.GetConformer().GetPositions()
    n1 = plane_normal(P1)
    out2 = P2.mean(0) - P2[c2_2]
    out2 /= np.linalg.norm(out2)
    n2 = plane_normal(P2)
    R1 = rot_between(out2, -v)
    n2r = n2 @ R1.T
    axis = v
    n_perp = n2r - np.dot(n2r, axis) * axis
    n1_perp = n1 - np.dot(n1, axis) * axis
    if np.linalg.norm(n_perp) < 1e-6 or np.linalg.norm(n1_perp) < 1e-6:
        R2 = np.eye(3)
    else:
        R2 = rot_between(n_perp, n1_perp)
    P2r = (R1 @ (R2 @ P2.T)).T
    target = P1[c2_1] + d * v
    P2r = P2r + (target - P2r[c2_2])

    core2r = Chem.Mol(core2)
    conf2 = Chem.Conformer(9)
    for i in range(9):
        conf2.SetAtomPosition(i, Point3D(*P2r[i]))
    core2r.RemoveAllConformers()
    core2r.AddConformer(conf2, assignId=True)

    mol = Chem.CombineMols(core1, core2r)
    if not with_ch2:
        return mol, None

    mid = 0.5 * (P1[c2_1] + target)
    h = d / 2.0
    r = 1.51
    off = np.sqrt(max(r * r - h * h, 0.0))
    e = np.cross(v, n1)
    e /= np.linalg.norm(e)
    ch2_pos = mid + off * e
    rw = Chem.RWMol(mol)
    ch = Chem.Atom(6)
    ch.SetNoImplicit(True)
    ch.SetNumExplicitHs(2)
    rw.AddAtom(ch)
    rw.AddBond(c2_1, 18, Chem.BondType.SINGLE)
    rw.AddBond(18, 9 + c2_2, Chem.BondType.SINGLE)
    for ai in (c2_1, 9 + c2_2):
        a = rw.GetAtomWithIdx(ai)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    out = rw.GetMol()
    conf = Chem.Conformer(19)
    P = out.GetConformer().GetPositions()
    for i in range(18):
        conf.SetAtomPosition(i, Point3D(*P[i]))
    conf.SetAtomPosition(18, Point3D(*ch2_pos))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(out)
    return out, 18


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[A3] device={device}')

    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    pxyz = pocket_tensor['pos'].numpy()

    summary = json.load(open(SUMMARY))
    a2 = json.load(open(A2_JSON))
    # rank benzimidazole candidates by redock P[S] at pyrrolic site
    bims = []
    for e in summary:
        if e['cls'] != 'benzimidazole':
            continue
        ps = a2[e['key']]['sites']['pyrrolicN']['redock']['PS']
        bims.append((ps, e))
    bims.sort(key=lambda t: -t[0])
    print('[A3] BM candidates by P[S]@redock(pyrrolic):',
          [(e['key'], p) for p, e in bims])

    cores = {}
    for ps, e in bims:
        m = next(Chem.SDMolSupplier(e['best_pose_sdf'], removeHs=True))
        r = get_bm_core(m)
        if r is None:
            print(f'  [skip] {e["key"]}: core extraction failed')
            continue
        core, pyr, pyd, c2 = r
        cores[e['key']] = dict(core=core, pyr=pyr, pyd=pyd, c2=c2)
        print(f'  [core] {e["key"]}: pyr={pyr} pyd={pyd} c2={c2} smi={Chem.MolToSmiles(core)}')
    keys = list(cores)
    if len(keys) < 2:
        print('[A3] need >=2 cores'); return

    tmpdir = '/tmp/opencode/t2_denovo_A3'
    os.makedirs(tmpdir, exist_ok=True)
    graphs, graph_meta, ctx_info = [], [], {}

    def add_ctx(name, mol, mask_sites, info):
        p = os.path.join(OUT_SDF_DIR, name + '.sdf')
        w = Chem.SDWriter(p); w.write(mol); w.close()
        mt = mol_to_tensor(parse_sdf_file(path=p), mode='molcraft')
        ctx_info[name] = info
        graphs.append(build_graphs(pocket_tensor, mt))
        graph_meta.append((name, None))
        for si in mask_sites:
            graphs.append(build_graphs(pocket_tensor, mt))
            graph_meta.append((name, (si,)))

    # F1: trimmed cores
    for k in keys:
        c = cores[k]
        add_ctx(f'F1_{k}', c['core'], [c['pyr']] + ([c['pyd']] if c['pyd'] is not None else []),
                {'family': 'F1_trimmed', 'members': [k]})
    print(f'[A3] F1: {len(keys)} trimmed cores')

    # F2/F3: pairs with generic placement
    pairs = [(keys[0], k) for k in keys[1:4]] + [(keys[1], keys[4]) if len(keys) > 4 else ()]
    pairs = [p for p in pairs if p]
    DISTS = [2.45, 2.55, 2.65]
    n_placed = 0
    for (k1, k2) in pairs:
        c1, c2 = cores[k1], cores[k2]
        P1 = c1['core'].GetConformer().GetPositions()
        c2pos = P1[c1['c2']]
        n1 = plane_normal(P1)
        # in-plane basis at C2
        e1 = np.cross(n1, [0, 0, 1.0])
        if np.linalg.norm(e1) < 1e-3:
            e1 = np.cross(n1, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n1, e1)
        kept = 0
        for deg in range(0, 360, 30):
            th = np.deg2rad(deg)
            v = np.cos(th) * e1 + np.sin(th) * e2
            for d in DISTS:
                mol2, _ = build_pair(c1['core'], c2['core'], c1['c2'], c2['c2'], v, d, False)
                P2 = mol2.GetConformer().GetPositions()[9:]
                if np.linalg.norm(P2.min(0) - pxyz.max(0), ) > 0:  # noop guard
                    pass
                dmin_pocket = min(np.linalg.norm(P2[:, None] - pxyz[None], axis=-1).min(), 99)
                dmin_pair = min(np.linalg.norm(P2[:, None] - P1[None], axis=-1).min(), 99)
                if dmin_pocket < 2.2 or dmin_pair < 2.3:
                    continue
                nm = f'F2_{k1}__{k2}_d{int(d*100)}_a{deg}'
                add_ctx(nm, mol2, [c1['pyr'], 9 + c2['pyr']],
                        {'family': 'F2_pair', 'members': [k1, k2], 'd': d, 'deg': deg,
                         'dmin_pocket': round(float(dmin_pocket), 2),
                         'dmin_pair': round(float(dmin_pair), 2)})
                mol3, chidx = build_pair(c1['core'], c2['core'], c1['c2'], c2['c2'], v, d, True)
                nm3 = nm.replace('F2_', 'F3_')
                add_ctx(nm3, mol3, [c1['pyr'], 9 + c2['pyr']],
                        {'family': 'F3_pair_ch2', 'members': [k1, k2], 'd': d, 'deg': deg})
                kept += 1
                n_placed += 1
                if kept >= 4:
                    break
            if kept >= 4:
                break
        print(f'[A3] pair {k1}+{k2}: {kept} clash-free placements')
    print(f'[A3] total graphs: {len(graphs)} ({n_placed} pair placements x2 families)')

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p): return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    report = {}
    rows = []
    for name, info in ctx_info.items():
        r = results[name]
        sites = {}
        for si in sorted(r.keys(), key=lambda x: (x is None, x)):
            if si is None:
                continue
            mk = r[si][si]
            sites[str(si)] = {'PS': round(ps(mk), 4), 'PN': round(pn(mk), 4),
                              'PC': round(pc(mk), 4), 'top1': CLS_NAME[int(mk.argmax())]}
            rows.append((ps(mk), name, si, info['family']))
        report[name] = {**info, 'sites': sites}

    rows.sort(reverse=True)
    print('\n===== A3 context probe: P[S] at pyrrolic sites =====')
    for p, name, si, fam in rows[:40]:
        flag = '  <<GATE>10%' if p > 0.10 else ''
        print(f'{p*100:7.2f}%  {fam:>11}  {name[:64]:>64} site{si}{flag}')
    by_fam = {}
    for p, name, si, fam in rows:
        by_fam.setdefault(fam, []).append(p)
    print('\n family summary:')
    for fam, ps_list in by_fam.items():
        a = np.array(ps_list)
        print(f'  {fam:>11}: n={len(a)} max={a.max()*100:.2f}% median={np.median(a)*100:.2f}% '
              f'>10%: {(a > 0.10).sum()}')
    report['_summary'] = {fam: {'n': len(v), 'max': round(float(np.max(v)), 4),
                                'median': round(float(np.median(v)), 4),
                                'n_gt10': int((np.array(v) > 0.10).sum())}
                          for fam, v in by_fam.items()}
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[A3] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
