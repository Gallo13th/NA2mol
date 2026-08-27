#!/usr/bin/env python3
"""Offline replay of run D round r01 (seed 443) to find how >19-atom mols arise."""
import os, sys, random, inspect, textwrap, collections
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

import dataloader.featurizer as _fmod
def _fsan(space_size):
    return random.randint(19, 19)
_fmod.sample_atom_num = _fsan
_src = textwrap.dedent(inspect.getsource(_G.__getitem__))
_new = _src.replace('total_n = max(origin_n + 6, sample_atom_num(self.pocket_size))',
                    'total_n = max(origin_n + 0, sample_atom_num(self.pocket_size))')
assert 'origin_n + 0' in _new
_ns = dict(vars(_fmod))
exec(_new, _ns)
_G.__getitem__ = _ns['__getitem__']
print('[ok] patches applied', flush=True)

from model.bfn4edmol import Config, BFN4SBDDScoreModel
from genbypkt import load_compiled_model
from dataloader.featurizer import ProteinLigandDataLoader, MAP_INDEX_TO_ATOM_TYPE_AROMATIC as FEAT_TAB
from torch.utils.data import ConcatDataset
from util.mol_utils import reconstruct_from_generated_with_seed, _neutralize_or_reject

POCKET = 'casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb'
CKPT = 'ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth'
import glob
seed_files = sorted(glob.glob('casestudy/kras_g4/t2_repro/inpaint_seeds_r2/seed_*.sdf'))
print('seeds:', [os.path.basename(s) for s in seed_files])

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 443
tag = sys.argv[2] if len(sys.argv) > 2 else f'r{seed-442:02d}'
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

cfg = Config(**{'config_file': 'config.yaml', 'exp_name': 'diag2', 'revision': 'default', 'debug': False,
                'resume': False, 'wandb_resume_id': None, 'empty_folder': False, 'test_only': False,
                'seed': seed, 'no_wandb': True, 'logging_level': 'warning', 'random_rot': False,
                'pos_noise_std': 0, 'pos_normalizer': 2.0, 'batch_size': 8, 'epochs': 15,
                'v_loss_weight': 1, 'lr': 5e-4, 'scheduler': 'plateau', 'weight_decay': 0,
                'max_grad_norm': 'Q', 'sigma1_coord': 0.03, 'beta1': 1.5, 't_min': 0.0001,
                'use_discrete_t': True, 'discrete_steps': 1000, 'destination_prediction': True,
                'sampling_strategy': 'end_back_pmf', 'time_emb_mode': 'simple', 'time_emb_dim': 1,
                'pos_init_mode': 'zero', 'ckpt_path': 'best', 'num_samples': 5, 'sample_steps': 100,
                'sample_num_atoms': 'ref', 'visual_chain': False, 'docking_mode': 'vina score',
                'output_prefix': 'diag2', 'feat_mode': 'generalfeat', 'use_cavity_volume': False,
                'volume_prob_table_path': None, 'use_probabilistic': False, 'prob_top_n': 5,
                'prob_beam_width': 100})
model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config, protein_atom_feature_dim=11,
                           ligand_atom_feature_dim=13, time_emb_dim=1, ed_feature_dim=4)
acc = Accelerator(mixed_precision='bf16')
model.load_state_dict(load_compiled_model(model, torch.load(CKPT, map_location='cpu')), strict=False)
model = acc.prepare(model); model.eval()
unwrapped = acc.unwrap_model(model)

ds_by_seed = {sf: _G(pocket=POCKET, seed=sf, mode='general',
                                        extra_range=(-8, 8), remain_seed=True) for sf in seed_files}
seed_mols = {sf: next(Chem.SDMolSupplier(sf)) for sf in seed_files}

datasets = []
i = 0
while len(datasets) < 100:
    datasets.append(ds_by_seed[seed_files[i % len(seed_files)]])
    i += 1
bd = ConcatDataset(datasets)
dl = acc.prepare(ProteinLigandDataLoader(bd, batch_size=100, shuffle=False))
batch = next(iter(dl)).to(acc.device)

# check per-graph N
gbi = batch.mol_element_batch.cpu().numpy()
sizes = collections.Counter(np.bincount(gbi))
print('per-graph atom-slot counts:', dict(sizes))
print('fixed per graph:', dict(collections.Counter(np.bincount(gbi, weights=batch.mol_fixed_atom_mask.cpu().numpy()))))

with torch.no_grad():
    theta_traj, sample_traj, y_traj, fe, fei = unwrapped.sample_by_seed(
        pocket_pos=batch.pocket_pos / 2, pocket_v=batch.pocket_atom_feature.float(),
        batch_pocket=batch.pocket_element_batch,
        ed_pos=torch.empty((0, 3), device=acc.device), ed_v=torch.empty((0, 4), device=acc.device),
        batch_ed=batch.ed_ed_code_batch, batch_ligand=batch.mol_element_batch, n_nodes=100,
        fixed_atom_mask=batch.mol_fixed_atom_mask, fixed_atom_pos=batch.mol_fixed_atom_pos / 2,
        fixed_atom_types=batch.mol_fixed_atom_types, sample_steps=100,
        pocket_res=None, pocket_ele=None, return_traj=False)
new_traj = (sample_traj[-1][0] * 2 + batch.mol_geom_center[:3], sample_traj[-1][1])
xyz, atomtype = new_traj
fm = batch.mol_fixed_atom_mask.cpu()
seed_paths = list(batch.mol_seed_path)

for i in range(100):
    mask = gbi == i
    sub_xyz = xyz[mask].cpu().numpy()
    sub_at = atomtype[mask].argmax(-1)
    pairs = [FEAT_TAB.get(int(j.item()), (None, None)) for j in sub_at]
    an = [p[0] for p in pairs]; ar = [p[-1] for p in pairs]
    sf = seed_paths[i]
    try:
        rd = reconstruct_from_generated_with_seed(sub_xyz, an, fm[mask].numpy().astype(bool),
                                                  seed_mols[sf], ar, dist_tol=0.4)
        smi = Chem.MolToSmiles(rd)
        if '.' in smi:
            continue
        neu = _neutralize_or_reject(rd)
        if neu is None:
            continue
        try:
            Chem.SanitizeMol(neu)
        except Exception:
            continue
        if neu.GetNumAtoms() != 19:
            print(f'graph {i}: slots={len(an)} out_atoms={neu.GetNumAtoms()} {Chem.MolToSmiles(neu)[:70]}')
    except Exception as e:
        pass
print('done scan')
