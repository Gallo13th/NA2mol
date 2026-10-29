#!/usr/bin/env python3
"""Phase F2: batch ring-N probe over ALL X1 BMZ-CH2-BMZ product poses.

The model generated 568 poses of the bis-benzimidazole-CH2 scaffold; each pose
is a model-native pocket attitude. Probe both ring-N sites of every pose
(aromatic-aware, uniform belief, t=1) and rank by P[S] to find poses where the
N->S ring conversion is endorsed (gate > 10%).
"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from util.ligand_code_util import parse_pdb_file
from dataloader.featurizer import pocket_to_tensor, mol_to_tensor, ProteinLigandDataLoader, PocketEDMol
from util.mol_utils import MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX as T2I, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB as I2T
sys.path.insert(0, 'casestudy/kras_g4/_archive_ber101_era')
from probe_element_prob import build_graphs  # noqa: E402 (hijacks cwd)
os.chdir('/home/data/users/gaoletian/rna2smol')
from accelerate import Accelerator
from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from rdkit.Geometry import Point3D

K = 13
IDX_S = T2I[(16, True)]
TARGET_SMI = 'c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1'

poses = []
for m in Chem.SDMolSupplier('casestudy/kras_g4/t2_denovo/gen_runX1/valid_mols.sdf', removeHs=True):
    if m is None or not m.HasProp('_Name'):
        continue
    if Chem.MolToSmiles(m) == TARGET_SMI:
        poses.append(m)
print(f'[F2] poses: {len(poses)}', flush=True)

args_dict = {'config_file': 'config.yaml', 'exp_name': 't', 'revision': 'r',
             'debug': True, 'resume': False, 'wandb_resume_id': None,
             'empty_folder': False, 'test_only': True, 'seed': 42, 'no_wandb': True,
             'logging_level': 'warning', 'random_rot': False, 'pos_noise_std': 0,
             'pos_normalizer': 2.0, 'batch_size': 1, 'epochs': 1, 'v_loss_weight': 1,
             'lr': 1e-4, 'scheduler': 'plateau', 'weight_decay': 0, 'max_grad_norm': 'Q',
             'sigma1_coord': 0.03, 'beta1': 1.5, 't_min': 1e-4, 'use_discrete_t': True,
             'discrete_steps': 1000, 'destination_prediction': True,
             'sampling_strategy': 'end_back_pmf', 'time_emb_mode': 'simple',
             'time_emb_dim': 1, 'pos_init_mode': 'zero', 'ckpt_path': 'best',
             'num_samples': 1, 'sample_steps': 100, 'sample_num_atoms': 'ref',
             'visual_chain': False, 'docking_mode': 'vina_score', 'output_prefix': 't',
             'feat_mode': 'generalfeat', 'use_cavity_volume': False,
             'volume_prob_table_path': None, 'use_probabilistic': False,
             'prob_top_n': 5, 'prob_beam_width': 100}
cfg = Config(**args_dict)
model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config, protein_atom_feature_dim=11,
                           ligand_atom_feature_dim=13, time_emb_dim=1, ed_feature_dim=4)
acc = Accelerator(mixed_precision='bf16')
model.load_state_dict(load_compiled_model(
    model, torch.load('ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth',
                      map_location='cpu')), strict=False)
model = acc.prepare(model); model.eval()
unwrapped = acc.unwrap_model(model)
beta1 = float(unwrapped.beta1)
gamma = 1 - 0.03 ** 2

pocket = parse_pdb_file(path='casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb')
pocket_tensor = pocket_to_tensor(pocket, mode='general')

# 预解析每个 pose 的张量 + 探针位
jobs = []
for pi, m in enumerate(poses):
    xyz = m.GetConformer().GetPositions()
    elems = [a.GetAtomicNum() for a in m.GetAtoms()]
    arom = [a.GetIsAromatic() for a in m.GetAtoms()]
    feat = [T2I[(z, bool(a))] for z, a in zip(elems, arom)]
    pos_t = torch.tensor(np.array(xyz), dtype=torch.float32)
    feat_t = torch.tensor(feat, dtype=torch.float32)
    bi = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()]
    bt = [int(b.GetBondTypeAsDouble()) for b in m.GetBonds()]
    mt = {'pos': pos_t, 'atom_feature': feat_t,
          'element': torch.tensor(elems, dtype=torch.long),
          'bond_index': torch.tensor(bi, dtype=torch.long).T.contiguous() if bi else torch.zeros((2,0), dtype=torch.long),
          'bond_type': torch.tensor(bt, dtype=torch.long),
          'smiles': Chem.MolToSmiles(m)}
    # ring N 位: 两环各自的 2 个 N 中, 取 [nH](吡咯)位(替换对象)
    BMZ = Chem.MolFromSmarts('c1ccc2[nH]cnc2c1')
    BMZ2 = Chem.MolFromSmarts('c1ccc2nc[nH]c2c1')
    mts = m.GetSubstructMatches(BMZ) or m.GetSubstructMatches(BMZ2)
    sites = []
    for mt_match in mts:
        for i in mt_match:
            a = m.GetAtomWithIdx(i)
            if a.GetSymbol() == 'N' and a.GetNumExplicitHs() >= 1:
                sites.append(i)
    if len(sites) != 2:
        continue
    jobs.append((pi, m, mt, sites))

print(f'[F2] probe jobs: {len(jobs)} (2 sites each)', flush=True)
results = []
BATCH = 32
with torch.no_grad():
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        graphs, metas = [], []
        for pi, m, mt, sites in chunk:
            for s in sites:
                graphs.append(build_graphs(pocket_tensor, mt))
                metas.append((pi, s))
        for gi, g in enumerate(graphs):
            loader = ProteinLigandDataLoader([g], batch_size=1, shuffle=False)
            batch = next(iter(loader)).to(acc.device)
            bl = batch.mol_element_batch
            n = bl.shape[0]
            ktrue = batch.mol_atom_feature.long()
            logits = torch.zeros((n, K), device=acc.device)
            logits[torch.arange(n, device=acc.device), ktrue] = beta1
            theta = torch.softmax(logits, -1)
            site = metas[gi][1]
            theta[site, :] = 1.0 / K
            t = torch.ones((n, 1), device=acc.device)
            mu = batch.mol_pos * gamma / 2.0
            _, p0, _, _, _ = unwrapped.interdependency_modeling(
                time=t, pocket_pos=batch.pocket_pos / 2,
                pocket_v=batch.pocket_atom_feature.float(),
                batch_pocket=batch.pocket_element_batch,
                ed_pos=torch.empty((0, 3), device=acc.device),
                ed_v=torch.empty((0, 4), device=acc.device),
                batch_ed=batch.ed_ed_code_batch, batch_ligand=bl,
                theta_h_t=theta, mu_pos_t=mu,
                gamma_coord=torch.full((n, 1), gamma, device=acc.device),
                pocket_res=None, pocket_ele=None)
            probs = torch.softmax(torch.log(p0.clamp(min=1e-12)), -1)
            pi_, _site = metas[gi]
            row = probs[site]
            results.append({'pose': pi_, 'name': poses[pi_].GetProp('_Name'),
                            'site': int(_site),
                            'PS': round(float(row[IDX_S]), 4),
                            'top1': int(row.argmax().item())})
        if (start // BATCH) % 4 == 0:
            print(f'[F2] {start+len(chunk)}/{len(jobs)}', flush=True)

json.dump(results, open('casestudy/kras_g4/t2_denovo/phaseF2_batch_probe.json', 'w'))
best = sorted(results, key=lambda r: -r['PS'])
print(f"\n[F2] 完成 {len(results)} 探测; P[S] top10:")
for r in best[:10]:
    print(f"  pose{r['pose']:3d} site{r['site']} P[S]={r['PS']*100:5.2f}%  {r['name']}")
n_gate = sum(1 for r in results if r['PS'] > 0.10)
print(f"P[S]>10% 的探测: {n_gate}/{len(results)}")
