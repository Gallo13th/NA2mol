#!/usr/bin/env python3
"""Cold-start conditioning generation in a pocket with the NA2smol BFN model.

This is the benchmark-style path: the pocket is the only input, and N molecules are
drawn i.i.d. from the model's conditional distribution. It does NOT run the
docking/refinement design loop of ``genbypkt.generate`` -- that loop selects and
grafts molecules, which is a different question from "what does the model
generate in this pocket".

Usage (from the rna2smol project root):
    python casestudy/8hze/sample_molecules.py \
        --pocket casestudy/8hze/8HZE_pocket.pdb \
        --dummy  casestudy/8hze/8HZE_dummy_ligand.sdf \
        --n 1000 --batch-size 100 --steps 100 \
        --out output/8hze_samples
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from rdkit import Chem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dataloader.featurizer import GeomSeedMolOnlyPocket, ProteinLigandDataLoader
from model.bfn4edmol import Config, BFN4SBDDScoreModel
from util.mol_utils import sampletraj_to_sdf, sampletraj_to_sdf_byseed
from torch.utils.data import ConcatDataset


def load_weights(model, state_dict):
    """Load a checkpoint that may carry torch.compile prefixes; ignore shape mismatches."""
    target = model.state_dict()
    clean = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    keep = {k: v for k, v in clean.items() if k in target and target[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    print(f'[i] loaded {len(keep)}/{len(target)} parameter tensors')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pocket', required=True)
    ap.add_argument('--dummy', required=True, help='SDF whose atom marks the pocket centre')
    ap.add_argument('--ckpt', default='ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_'
                                    'train_best_val_loss_step_111429.pth')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rng-seed', type=int, default=1234)
    ap.add_argument('--seed', help='SDF whose atoms are held fixed (scaffold-anchored mode)')
    args = ap.parse_args()

    import random
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)

    os.makedirs(args.out, exist_ok=True)

    cfg = Config(config_file='config.yaml')
    model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config,
                               protein_atom_feature_dim=11,
                               ligand_atom_feature_dim=13, time_emb_dim=1,
                               ed_feature_dim=4)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    state = torch.load(args.ckpt, map_location='cpu')
    load_weights(model, state)
    model = accelerator.prepare(model)
    model.eval()
    net = accelerator.unwrap_model(model)

    seeded = bool(args.seed)
    dataset = GeomSeedMolOnlyPocket(pocket=args.pocket, seed=args.seed or args.dummy,
                                    mode='general', extra_range=(-8, 8),
                                    remain_seed=seeded)
    seed_mol = None
    if seeded:
        seed_mol = next(Chem.SDMolSupplier(args.seed, removeHs=False))
        print(f'[i] scaffold-anchored mode: {seed_mol.GetNumAtoms()} fixed atoms from {args.seed}')
    written = 0
    t0 = time.time()
    max_attempts = 10 * max(1, (args.n + args.batch_size - 1) // args.batch_size)
    attempts = 0
    while written < args.n:
        attempts += 1
        if attempts > max_attempts:
            print(f'[w] stopping after {attempts} batches with {written}/{args.n} molecules',
                  flush=True)
            break
        b = min(args.batch_size, args.n - written)
        batch_ds = ConcatDataset([dataset] * b)
        loader = ProteinLigandDataLoader(batch_ds, batch_size=b, shuffle=False)
        loader = accelerator.prepare(loader)
        for batch in loader:
            batch = batch.to(accelerator.device)
            n_nodes = int(batch.mol_element_batch.shape[0])
            with torch.no_grad():
                if seeded:
                    _, sample_traj, _, _, _ = net.sample_by_seed(
                        pocket_pos=batch.pocket_pos / 2,
                        pocket_v=batch.pocket_atom_feature.float(),
                        batch_pocket=batch.pocket_element_batch,
                        ed_pos=torch.empty((0, 3), device=accelerator.device),
                        ed_v=torch.empty((0, 4), device=accelerator.device),
                        batch_ed=batch.ed_ed_code_batch,
                        batch_ligand=batch.mol_element_batch,
                        n_nodes=n_nodes,
                        fixed_atom_mask=batch.mol_fixed_atom_mask,
                        fixed_atom_pos=batch.mol_fixed_atom_pos / 2,
                        fixed_atom_types=batch.mol_fixed_atom_types,
                        sample_steps=args.steps,
                        pocket_res=None, pocket_ele=None, return_traj=False)
                else:
                    _, sample_traj, _, _, _ = net.sample(
                        pocket_pos=batch.pocket_pos / 2,
                        pocket_v=batch.pocket_atom_feature.float(),
                        batch_pocket=batch.pocket_element_batch,
                        ed_pos=torch.empty((0, 3), device=accelerator.device),
                        ed_v=torch.empty((0, 4), device=accelerator.device),
                        batch_ed=batch.ed_ed_code_batch,
                        batch_ligand=batch.mol_element_batch,
                        n_nodes=n_nodes,
                        sample_steps=args.steps,
                        pocket_res=None, pocket_ele=None, return_traj=False)
            traj = (sample_traj[-1][0] * 2 + batch.mol_geom_center[:3], sample_traj[-1][1])
            outdir = os.path.join(args.out, f'batch_{written:05d}')
            if seeded:
                sampletraj_to_sdf_byseed(
                    {'trajs': traj, 'batch_idx': batch.mol_element_batch.cpu()}, outdir,
                    [seed_mol] * b, batch.mol_fixed_atom_mask, dist_tol=0.4)
            else:
                sampletraj_to_sdf({'trajs': traj, 'batch_idx': batch.mol_element_batch.cpu()}, outdir)
            n_new = len([f for f in os.listdir(outdir) if f.endswith('.sdf')])
            written += n_new
            dt = time.time() - t0
            print(f'[i] batch done: +{n_new} molecules, total {written}/{args.n}, '
                  f'{dt:.1f}s elapsed ({dt / max(written, 1):.2f}s/mol)', flush=True)
            break                      # one batch per while-iteration
    print(f'[OK] {(time.time() - t0) / 60:.1f} min total')


if __name__ == '__main__':
    main()
