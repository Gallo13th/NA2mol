#!/usr/bin/env python3
"""Diagnostic probe: D18 seed + ghost C at the thiazole-closure spot.
If P[C*] high + coord_pred points to the closure spot -> sampling-limited (sample more).
If P[C*] low -> this seed geometry is element-field-dead."""
import json, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')
from util.ligand_code_util import parse_sdf_file, parse_pdb_file
from dataloader.featurizer import pocket_to_tensor, mol_to_tensor, ProteinLigandDataLoader
from dataloader.featurizer import MAP_ATOM_TYPE_AROMATIC_TO_INDEX as MAP_ATI
K = 13
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/_archive_ber101_era'))
import util.mol_utils  # noqa
from probe_element_prob import load_model, build_graphs  # noqa
os.chdir(PROJECT_ROOT)

POCKET = os.path.join(PROJECT_ROOT, 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb')
IDX_Carom = MAP_ATI[(6, True)]

seed_path = os.path.join(SCRIPT_DIR, 'inpaint_seeds_r6/seed_d18_2.sdf')
seed = next(Chem.SDMolSupplier(seed_path, removeHs=True))
assert seed.GetNumAtoms() == 18
pos = seed.GetConformer().GetPositions()
frags = Chem.GetMolFrags(seed)
mebt = next(f for f in frags if len(f) == 10)
phns = next(f for f in frags if len(f) == 8)
# anchors: CH3 C (sp3 in mebt), N (sp3-ish NH2 in phns), S (sh in phns)
ch3 = next(i for i in mebt if seed.GetAtomWithIdx(i).GetSymbol() == 'C' and not seed.GetAtomWithIdx(i).GetIsAromatic())
nH = next(i for i in phns if seed.GetAtomWithIdx(i).GetSymbol() == 'N')
sh = next(i for i in phns if seed.GetAtomWithIdx(i).GetSymbol() == 'S')
print('anchors:', ch3, nH, sh)
# ghost positions to test
tri = np.array([pos[ch3], pos[nH], pos[sh]])
center = tri.mean(0)
ghosts = {'closure_center': center,
          'chain_mid(N-CH3)': (pos[ch3] + pos[nH]) / 2,
          'chain_mid(S-CH3)': (pos[ch3] + pos[sh]) / 2}

pocket = parse_pdb_file(path=POCKET)
pocket_tensor = pocket_to_tensor(pocket, mode='general')
tmpdir = '/tmp/opencode/t2_ghost'
os.makedirs(tmpdir, exist_ok=True)

graphs, graph_meta = [], []
records = {}
for gname, gpos in ghosts.items():
    rw = Chem.RWMol(Chem.Mol(seed))
    a = Chem.Atom(6)
    a.SetNoImplicit(True); a.SetNumExplicitHs(0); a.SetNoImplicit(False)
    rw.AddAtom(a)
    conf = Chem.Conformer(19)
    for i in range(18):
        conf.SetAtomPosition(i, Point3D(*pos[i]))
    conf.SetAtomPosition(18, Point3D(*gpos))
    m19 = rw.GetMol()
    m19.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(m19)   # ghost C isolated (no bonds) -> mol + disconnected C
    p = os.path.join(tmpdir, f'g_{gname}.sdf')
    w = Chem.SDWriter(p); w.write(m19); w.close()
    mt = mol_to_tensor(parse_sdf_file(path=p), mode='molcraft')
    # full-info echo + masked ghost
    graphs.append(build_graphs(pocket_tensor, mt)); graph_meta.append((gname, None))
    graphs.append(build_graphs(pocket_tensor, mt)); graph_meta.append((gname, 18))
    records[gname] = {'natoms': int(mt['pos'].shape[0])}
    print(gname, 'ghost class:', int(mt['atom_feature'][18]))

model = load_model('cuda:0' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
gamma = float(1 - 0.03 ** 2)
results = {}
with torch.no_grad():
    loader = ProteinLigandDataLoader(graphs, batch_size=8, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        bl = batch.mol_element_batch
        N = bl.shape[0]
        gids = bl.cpu().numpy()
        ktrue = batch.mol_atom_feature.long()
        logits = torch.zeros(N, K, device=device)
        logits[torch.arange(N, device=device), ktrue] = float(model.beta1)
        theta = torch.softmax(logits, dim=-1)
        gstart = 0
        for g_local in range(batch.num_graphs):
            key, mask = graph_meta[gstart + g_local]
            if mask:
                gi = np.where(gids == g_local)[0][mask if isinstance(mask, int) else mask[0]]
                theta[gi, :] = 1.0 / K
        t = torch.ones(N, 1, device=device)
        mu = batch.mol_pos * gamma / 2.0
        coord_pred, p0_h_pred, k_hat, _, _ = model.interdependency_modeling(
            time=t, pocket_pos=batch.pocket_pos / 2,
            pocket_v=batch.pocket_atom_feature.float(), batch_pocket=batch.pocket_element_batch,
            ed_pos=torch.empty((0, 3), device=device), ed_v=torch.empty((0, 4), device=device),
            batch_ed=batch.ed_ed_code_batch, batch_ligand=bl, theta_h_t=theta, mu_pos_t=mu,
            gamma_coord=torch.full((N, 1), gamma, device=device),
            pocket_res=None, pocket_ele=None)
        probs = torch.softmax(torch.log(p0_h_pred.clamp(min=1e-12)), dim=-1)
        gstart = 0
        for g_local in range(batch.num_graphs):
            key, mask = graph_meta[gstart + g_local]
            rows = np.where(gids == g_local)[0]
            results.setdefault(key, {})[mask] = (probs[rows].float().cpu().numpy(),
                                                 coord_pred[rows].float().cpu().numpy())
        gstart += batch.num_graphs

print('\n===== ghost-C probe (D18 seed geometry) =====')
for gname in ghosts:
    pm, cm = results[gname][18]
    pfull, cfull = results[gname][None]
    pc_ghost = float(pm[18][IDX_Carom])
    top1 = int(pm[18].argmax())
    from dataloader.featurizer import MAP_INDEX_TO_ATOM_TYPE_AROMATIC as MITA
    name = {v: k for k, v in MAP_ATI.items()}
    # coord_pred for ghost: distance to anchors
    cp = cm[18]
    d_ch3 = np.linalg.norm(cp - pos[ch3]); d_n = np.linalg.norm(cp - pos[nH]); d_s = np.linalg.norm(cp - pos[sh])
    print(f'{gname:>18}: P[C*]={pc_ghost*100:6.2f}%  top1={name[top1]}({pm[18].max()*100:.1f}%)  '
          f'coord_pred dists: CH3={d_ch3:.2f} N={d_n:.2f} S={d_s:.2f}')
