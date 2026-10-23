#!/usr/bin/env python3
"""Harvest a real generation trajectory that produces the 2,4-diaryl-quinoline core.

`sample()` already accumulates the full per-step belief state; genbypkt throws all but the
final frame away. This runs the sampler directly, keeps the trajectory, screens the finished
molecules for the target core, and saves for each hit:

  core_idx    atom indices forming the 2,4-diaryl-quinoline core in the batch ordering
  mu_pos      [T+1, N, 3]  belief positions per step (the real path)
  theta_h     [T+1, N, K]  belief type distributions per step
  smiles      the finished molecule

These are then fed to sample_by_seed(given_traj=...) so the core is conditioned on the path
the model actually took, instead of the analytic pseudo path gamma(t) * x_true.

Usage:
  CUDA_VISIBLE_DEVICES=5 python harvest_trajectory.py --target squash \
      --out case_study/clivia_squash/traj_squash.pt --n_batch 200
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger                    # noqa: E402
from rdkit.Chem import AllChem                      # noqa: E402
from accelerate import Accelerator                  # noqa: E402
from torch.utils.data import ConcatDataset          # noqa: E402

RDLogger.DisableLog('rdApp.*')

from dataloader.featurizer import ProteinLigandDataLoader, GeomSeedMolOnlyPocket  # noqa: E402
from model.bfn4edmol import Config, BFN4SBDDScoreModel                            # noqa: E402
from genbypkt import load_compiled_model                                          # noqa: E402
from util.mol_utils import reconstruct_from_generated, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB  # noqa: E402

CKPT = "ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth"
TARGETS = {
    'clivia': ('clivia_8HZE_pocket.pdb', 'clivia_8HZE_dummy.sdf'),
    'squash': ('squash_7KVU_pocket.pdb', 'squash_7KVU_dummy.sdf'),
}
QUIN = Chem.MolFromSmarts('c1ccc2ncccc2c1')
# the model's classes are NOT atomic numbers: map class index -> atomic number


def diaryl_positions(m):
    out = set()
    for mt in m.GetSubstructMatches(QUIN):
        for lab, i in (('C2', 5), ('C4', 7)):
            a = m.GetAtomWithIdx(mt[i])
            for nb in a.GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in mt and nb.GetSymbol() in 'CN':
                    out.add(lab)
    return out


def build_model():
    args_dict = {
        'config_file': 'config.yaml', 'exp_name': 'harvest', 'revision': 'default',
        'debug': False, 'resume': False, 'wandb_resume_id': None, 'empty_folder': False,
        'test_only': False, 'seed': 42, 'no_wandb': True, 'logging_level': 'warning',
        'random_rot': False, 'pos_noise_std': 0, 'pos_normalizer': 2.0, 'batch_size': 8,
        'epochs': 15, 'v_loss_weight': 1, 'lr': 5e-4, 'scheduler': 'plateau',
        'weight_decay': 0, 'max_grad_norm': 'Q', 'sigma1_coord': 0.03, 'beta1': 1.5,
        't_min': 0.0001, 'use_discrete_t': True, 'discrete_steps': 1000,
        'destination_prediction': True, 'sampling_strategy': 'end_back_pmf',
        'time_emb_mode': 'simple', 'time_emb_dim': 1, 'pos_init_mode': 'zero',
        'ckpt_path': 'best', 'num_samples': 5, 'sample_steps': 100,
        'sample_num_atoms': 'ref', 'visual_chain': False, 'docking_mode': 'vina_score',
        'output_prefix': 'harvest',
    }
    cfg = Config(**args_dict)
    model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config,
                               protein_atom_feature_dim=11, ligand_atom_feature_dim=13,
                               time_emb_dim=1, ed_feature_dim=4)
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    sd = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(load_compiled_model(model, sd), strict=False)
    model = acc.prepare(model)
    model.eval()
    return model, acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', choices=list(TARGETS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_batch', type=int, default=200,
                    help='number of sampler batches; each yields batch_size molecules')
    ap.add_argument('--sample_steps', type=int, default=100)
    a = ap.parse_args()

    pocket = str(SCRIPT_DIR / TARGETS[a.target][0])
    dummy = str(SCRIPT_DIR / TARGETS[a.target][1])
    bs = 8
    model, acc = build_model()
    unwrapped = acc.unwrap_model(model)

    # The dataset is created once (its constructor parses the pocket); __getitem__
    # re-randomises, so a fresh DataLoader per iteration yields new molecules. Building
    # ConcatDataset([ds] * bs) once and iterating it gives exactly one batch, which is why
    # genbypkt recreates the dataloader inside its while loop.
    ds = GeomSeedMolOnlyPocket(pocket=pocket, seed=dummy, mode='general', extra_range=(-8, 8))
    batch_dataset = ConcatDataset([ds] * bs)

    found = []
    for nb in range(a.n_batch):
        dl = ProteinLigandDataLoader(batch_dataset, batch_size=bs, shuffle=False)
        batch = next(iter(dl)).to(acc.device)
        with torch.no_grad():
            theta_traj, sample_traj, _, _, _ = unwrapped.sample(
                pocket_pos=batch.pocket_pos / 2,
                pocket_v=batch.pocket_atom_feature.float(),
                batch_pocket=batch.pocket_element_batch,
                ed_pos=torch.empty((0, 3), device=acc.device),
                ed_v=torch.empty((0, 4), device=acc.device),
                batch_ed=batch.ed_ed_code_batch,
                batch_ligand=batch.mol_element_batch,
                n_nodes=bs,
                sample_steps=a.sample_steps,
                pocket_res=None, pocket_ele=None, return_traj=False,
            )
        # reconstructed molecules, per segment of batch_ligand
        seg = batch.mol_element_batch.cpu().numpy()
        pos = sample_traj[-1][0].detach().cpu().numpy() * 2 + \
            batch.mol_geom_center[:3].cpu().numpy()
        typ = sample_traj[-1][1].detach().cpu().numpy()
        for b in range(bs):
            idx = np.where(seg == b)[0]
            if len(idx) < 5:
                continue
            nums = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(int(np.argmax(typ[i])), (6, False))[0]
                    for i in idx]
            try:
                mol = reconstruct_from_generated(pos[idx].tolist(), nums)
            except Exception:
                continue
            if mol is None:
                continue
            if diaryl_positions(mol) != {'C2', 'C4'}:
                continue
            core = set()
            for mt in mol.GetSubstructMatches(QUIN):
                core.update(mt)
                for p in (5, 7):
                    for nb2 in mol.GetAtomWithIdx(mt[p]).GetNeighbors():
                        if nb2.GetIsAromatic() and nb2.GetIdx() not in mt:
                            core.add(nb2.GetIdx())
            found.append(dict(
                batch=nb, seg=int(b),
                idx_global=idx.tolist(),
                core_idx=sorted(idx[c] for c in core if c < len(idx)),
                smiles=Chem.MolToSmiles(mol),
                mu_pos=torch.stack([f[0] for f in theta_traj]).cpu(),    # [T+1, N, 3]
                theta_h=torch.stack([f[1] for f in theta_traj]).cpu(),   # [T+1, N, K]
            ))
            print(f'[hit] batch {nb} mol {b}: {Chem.MolToSmiles(mol)}', flush=True)
            if len(found) >= 8:
                break
        if len(found) >= 8:
            break
        if nb % 20 == 0:
            print(f'  batch {nb}/{a.n_batch}, hits {len(found)}', flush=True)

    if not found:
        print('no core hit found; increase --n_batch')
        return 1
    torch.save(found, a.out)
    print(f'\nsaved {len(found)} trajectories -> {a.out}')
    for f in found:
        print(f'  core atoms {len(f["core_idx"]):2d}  {f["smiles"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
