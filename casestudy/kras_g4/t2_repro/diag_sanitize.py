#!/usr/bin/env python3
"""Diagnose sanitize_fail in seed-mode reconstruction."""
import os, sys, random, collections
PROJECT_ROOT = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJECT_ROOT); sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import torch
from accelerate import Accelerator
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from dataloader.featurizer import GeomSeedMolOnlyPocket as _G
_orig_init = _G.__init__
def _pi(self, pocket, seed, mode='general', pocket_radius=None, extra_range=(0, 6), remain_seed=False, **kw):
    _orig_init(self, pocket=pocket, seed=seed, mode=mode, pocket_radius=pocket_radius,
               extra_range=extra_range, remain_seed=remain_seed)
_G.__init__ = _pi
from model.bfn4edmol import BFN4SBDDScoreModel as _M
_o = _M.sample_by_seed
def _p(self, **kw):
    return _o(self, **{k: v for k, v in kw.items() if k not in ('ligand_pos', 'return_traj')})
_M.sample_by_seed = _p

from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from dataloader.featurizer import (GeomSeedMolOnlyPocket, ProteinLigandDataLoader,
                                   MAP_INDEX_TO_ATOM_TYPE_AROMATIC as FEAT_TAB)
from torch.utils.data import ConcatDataset
from util.mol_utils import reconstruct_from_generated_with_seed, _neutralize_or_reject
import dataloader.featurizer as _f

random.seed(7); np.random.seed(7); torch.manual_seed(7)
_f.sample_atom_num = lambda s: 19

POCKET = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
CKPT = 'ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth'
sf = 'casestudy/kras_g4/t2_repro/inpaint_seeds/seed_mebt_etkdg.sdf'
seed_mol = next(Chem.SDMolSupplier(sf))

cfg = Config(**{'config_file': 'config.yaml', 'exp_name': 'diag', 'revision': 'default', 'debug': False,
                'resume': False, 'wandb_resume_id': None, 'empty_folder': False, 'test_only': False,
                'seed': 7, 'no_wandb': True, 'logging_level': 'warning', 'random_rot': False,
                'pos_noise_std': 0, 'pos_normalizer': 2.0, 'batch_size': 8, 'epochs': 15,
                'v_loss_weight': 1, 'lr': 5e-4, 'scheduler': 'plateau', 'weight_decay': 0,
                'max_grad_norm': 'Q', 'sigma1_coord': 0.03, 'beta1': 1.5, 't_min': 0.0001,
                'use_discrete_t': True, 'discrete_steps': 1000, 'destination_prediction': True,
                'sampling_strategy': 'end_back_pmf', 'time_emb_mode': 'simple', 'time_emb_dim': 1,
                'pos_init_mode': 'zero', 'ckpt_path': 'best', 'num_samples': 5, 'sample_steps': 100,
                'sample_num_atoms': 'ref', 'visual_chain': False, 'docking_mode': 'vina score',
                'output_prefix': 'diag', 'feat_mode': 'generalfeat', 'use_cavity_volume': False,
                'volume_prob_table_path': None, 'use_probabilistic': False, 'prob_top_n': 5,
                'prob_beam_width': 100})
model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config, protein_atom_feature_dim=11,
                           ligand_atom_feature_dim=13, time_emb_dim=1, ed_feature_dim=4)
acc = Accelerator(mixed_precision='bf16')
model.load_state_dict(load_compiled_model(model, torch.load(CKPT, map_location='cpu')), strict=False)
model = acc.prepare(model); model.eval()
unwrapped = acc.unwrap_model(model)

ds = GeomSeedMolOnlyPocket(pocket=POCKET, seed=sf, mode='general', extra_range=(-8, 8), remain_seed=True)
bd = ConcatDataset([ds] * 40)
dl = acc.prepare(ProteinLigandDataLoader(bd, batch_size=40, shuffle=False))
batch = next(iter(dl)).to(acc.device)
with torch.no_grad():
    theta_traj, sample_traj, y_traj, fe, fei = unwrapped.sample_by_seed(
        pocket_pos=batch.pocket_pos / 2, pocket_v=batch.pocket_atom_feature.float(),
        batch_pocket=batch.pocket_element_batch,
        ed_pos=torch.empty((0, 3), device=acc.device), ed_v=torch.empty((0, 4), device=acc.device),
        batch_ed=batch.ed_ed_code_batch, batch_ligand=batch.mol_element_batch, n_nodes=40,
        fixed_atom_mask=batch.mol_fixed_atom_mask, fixed_atom_pos=batch.mol_fixed_atom_pos / 2,
        fixed_atom_types=batch.mol_fixed_atom_types, sample_steps=100,
        pocket_res=None, pocket_ele=None, return_traj=False)
new_traj = (sample_traj[-1][0] * 2 + batch.mol_geom_center[:3], sample_traj[-1][1])
xyz, atomtype = new_traj
gbi = batch.mol_element_batch.cpu()
fm = batch.mol_fixed_atom_mask.cpu()

fails = collections.Counter()
examples = []
for i in range(40):
    mask = gbi == i
    sub_xyz = xyz[mask].cpu().numpy()
    sub_at = atomtype[mask].argmax(-1)
    pairs = [FEAT_TAB.get(int(j.item()), (None, None)) for j in sub_at]
    an = [p[0] for p in pairs]; ar = [p[-1] for p in pairs]
    try:
        rd = reconstruct_from_generated_with_seed(sub_xyz, an, fm[mask].numpy().astype(bool),
                                                  seed_mol, ar, dist_tol=0.4)
        smi = Chem.MolToSmiles(rd)
        if '.' in smi:
            fails['disconnected'] += 1
            examples.append(('disconnected', smi, [FEAT_TAB.get(int(j.item())) for j in sub_at]))
            continue
        neu = _neutralize_or_reject(rd)
        if neu is None:
            fails['charged'] += 1
            continue
        try:
            Chem.SanitizeMol(neu)
            fails['OK'] += 1
        except Exception as e:
            fails['sanitize: ' + str(e)[:60]] += 1
            examples.append(('sanitize', Chem.MolToSmiles(neu), str(e)[:100]))
    except Exception as e:
        fails['recon: ' + str(e)[:50]] += 1
        examples.append(('recon', str(e)[:80], [FEAT_TAB.get(int(j.item())) for j in sub_at]))

print(dict(fails))
print()
for kind, a, b in examples[:15]:
    print(f'[{kind}] {a}  |  {b}')
