#!/usr/bin/env python3
"""t2-DENOVO Phase A5: berberine-referenced template placement of the generic
bis-benzimidazole-CH2 dimer -> masked probe at pyrrolic-N sites.

Evidence so far: P[S] tracks pose LOCATION (dimer poses within ~2 A of the
berberine crystal centroid give 9%; poses that wander give ~1%). The berberine
co-crystal pose is the natural geometric reference of this pocket (the rDock
cavity itself is defined by REF_MOL=berberine). Berberine is NOT t2 — using its
crystal pose as a template is pocket data, not target data.

Placement (all generic):
  dimer conformer = ETKDG+MMFF from generic SMILES (multiple seeds)
  superposition    = centroid->centroid, plane normal->normal, long axis->long axis
  scan             = in-plane rotation {0,45,90,135} x normal sign {+,-} (8 poses/conf)
  clash filter     = min dist to probe-pocket atoms >= 2.3 A

Probe: F4 (19-atom dimer) and F4b (18-atom seed shape) at both pyrrolic-N sites.
Gate: P[S] > 10%.

Usage: CUDA_VISIBLE_DEVICES=1 python casestudy/kras_g4/t2_denovo/phaseA5_berref_probe.py
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
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseA5_berref'
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, 'probe_results.json')

DIMER_SMILES = 'c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1'   # generic isostere dimer


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
    return c, vt[0], vt[2]  # centroid, long axis, normal


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


def place(mol_coords, ref_long, ref_norm, ref_cent, theta_deg, flip):
    """align mol long axis->ref_long, normal->(+-)ref_norm, then rotate theta about
    the normal, translate centroid->ref_cent."""
    c, long_ax, n = axes(mol_coords)
    n = n if np.dot(n, np.cross(long_ax, np.cross(long_ax, n))) != 0 else n
    # build right-handed frame from long axis + normal
    side = np.cross(n, long_ax)
    side /= np.linalg.norm(side)
    long_ax = np.cross(side, n)
    tgt_n = ref_norm * (1.0 if not flip else -1.0)
    R1 = rot_between(n, tgt_n)
    la = long_ax @ R1.T
    R2 = rot_between(la, ref_long)
    R = R2 @ R1
    P = (R @ mol_coords.T).T
    # in-plane rotation theta about ref normal
    ax = tgt_n / np.linalg.norm(tgt_n)
    th = np.deg2rad(theta_deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R3 = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    P = (R3 @ P.T).T
    P = P + (ref_cent - P.mean(0))
    return P


def main():
    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pocket = parse_pdb_file(path=POCKET_PDB)
    pocket_tensor = pocket_to_tensor(pocket, mode='general')
    pxyz = pocket_tensor['pos'].numpy()

    ref = next(Chem.SDMolSupplier(REF_SDF, removeHs=True))
    R = ref.GetConformer().GetPositions()
    ref_cent, ref_long, ref_norm = axes(R)
    print(f'[A5] berberine axes: centroid={np.round(ref_cent,2)} normal={np.round(ref_norm,2)}')

    # conformers of the generic dimer
    m0 = Chem.MolFromSmiles(DIMER_SMILES)
    confs = []
    for seed in (11, 23, 37, 51):
        mh = Chem.AddHs(Chem.Mol(m0))
        ps = AllChem.ETKDGv3(); ps.randomSeed = seed
        ps.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(mh, ps) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mh, mmffVariant='MMFF94s', maxIters=2000)
        confs.append((seed, Chem.RemoveHs(mh)))
    print(f'[A5] {len(confs)} conformers')

    # locate pyrrolic N + bridge CH2 idx
    m0c = confs[0][1]
    pyr, ch2 = [], None
    for a in m0c.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0 and a.GetIsAromatic():
            pyr.append(a.GetIdx())
        if a.GetSymbol() == 'C' and not a.GetIsAromatic() and a.GetDegree() == 2 \
                and a.GetTotalNumHs() == 2:
            ch2 = a.GetIdx()
    assert len(pyr) == 2 and ch2 is not None, (pyr, ch2)
    print(f'[A5] pyrrolic sites {pyr}, bridge CH2 idx {ch2}')

    graphs, graph_meta, ctx = [], [], []
    kept = 0
    for seed, mol in confs:
        P0 = mol.GetConformer().GetPositions()
        for flip in (False, True):
            for theta in (0, 45, 90, 135):
                P = place(P0, ref_long, ref_norm, ref_cent, theta, flip)
                dmin = min(np.linalg.norm(P[:, None] - pxyz[None], axis=-1).min(), 99)
                if dmin < 2.3:
                    continue
                kept += 1
                for fam in ('F4', 'F4b'):
                    if fam == 'F4':
                        mm = Chem.Mol(mol)
                        n_at = 19
                        sites_probe = pyr
                    else:
                        mm = Chem.Mol(mol)
                        n_at = 18
                        sites_probe = [si if si < ch2 else si - 1 for si in pyr]
                    conf = Chem.Conformer(n_at)
                    j = 0
                    for i in range(19):
                        if fam == 'F4b' and i == ch2:
                            continue
                        conf.SetAtomPosition(j, Point3D(*P[i]))
                        j += 1
                    if fam == 'F4b':
                        rw = Chem.RWMol(mm)
                        rw.RemoveAtom(ch2)
                        mm = rw.GetMol()
                        Chem.SanitizeMol(mm)
                    mm.RemoveAllConformers()
                    mm.AddConformer(conf, assignId=True)
                    name = f'c{seed}_f{int(flip)}_t{theta}_{fam}'
                    path = os.path.join(OUTDIR, name + '.sdf')
                    w = Chem.SDWriter(path); w.write(mm); w.close()
                    mt = mol_to_tensor(parse_sdf_file(path=path), mode='molcraft')
                    graphs.append(build_graphs(pocket_tensor, mt))
                    graph_meta.append((name, None))
                    for si in sites_probe:
                        graphs.append(build_graphs(pocket_tensor, mt))
                        graph_meta.append((name, (si,)))
                    ctx.append({'name': name, 'seed': seed, 'flip': flip, 'theta': theta,
                                'fam': fam, 'dmin': round(float(dmin), 2),
                                'sites': sites_probe})
    print(f'[A5] kept {kept} placements, {len(graphs)} graphs')

    model = load_model(device)
    loader = ProteinLigandDataLoader(graphs, batch_size=30, shuffle=False)
    results = run_forward_multi(model, loader, graph_meta, float(model.beta1))

    def ps(p): return float(p[IDX_S] + p[IDX_Sarom])
    def pn(p): return float(p[MAP_ATI[(7, False)]] + p[MAP_ATI[(7, True)]])
    def pc(p): return float(p[MAP_ATI[(6, False)]] + p[MAP_ATI[(6, True)]])

    report, rows = {}, []
    for e in ctx:
        r = results[e['name']]
        sites = {}
        for si in e['sites']:
            mk = r[(si,)][si]
            sites[si] = {'PS': round(ps(mk), 4), 'PSarom': round(float(mk[IDX_Sarom]), 4),
                         'PN': round(pn(mk), 4), 'PC': round(pc(mk), 4),
                         'top1': CLS_NAME[int(mk.argmax())]}
            rows.append((ps(mk), e['name'], e['fam'], si, e['seed'], e['theta'], e['flip']))
        report[e['name']] = {**e, 'sites': sites}

    rows.sort(key=lambda t: -t[0])
    print('\n===== A5 berberine-referenced placement probe =====')
    for p, name, fam, si, seed, theta, flip in rows[:24]:
        flag = '  <<GATE' if p > 0.10 else ''
        print(f'{p*100:7.2f}%  {fam}  {name:>22} site{si}{flag}')
    allps = np.array([r[0] for r in rows])
    print(f'\n[A5] max={allps.max()*100:.2f}% median={np.median(allps)*100:.2f}% '
          f'>10%: {(allps>0.10).sum()}/{len(allps)}  >5%: {(allps>0.05).sum()}/{len(allps)}')
    report['_summary'] = {'max': round(float(allps.max()), 4),
                          'n_gt10': int((allps > 0.10).sum()),
                          'n_gt5': int((allps > 0.05).sum()), 'n': len(allps),
                          'best_ctx': rows[0][1] if rows else None}
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[A5] saved {OUT_JSON}')


if __name__ == '__main__':
    main()
