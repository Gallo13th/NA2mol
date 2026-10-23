#!/usr/bin/env python3
"""Cold-start (free) generation for the Clivia and Squash aptamer pockets.

Mirrors case_study/kras_g4/run_generation_ber102.py exactly: inject a pre-built
model + accelerator into genbypkt's namespace and call genbypkt.generate() with
seed_used=False (no seed/fluorophore guidance), which is the free-generation path
used for the benchmark.

Test purpose: the five quinoline series compounds (mol1-mol5) were found to be
fluorescently active on Clivia and Squash. This asks whether unconstrained
generation from the aptamer pockets reproduces them de novo.

Usage (one process per GPU):
    CUDA_VISIBLE_DEVICES=5 python case_study/clivia_squash/run_generation.py \
        --target clivia --seed 42 --output_prefix clivia_s0 --total_target 1000
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # rna2smol/
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from accelerate import Accelerator  # noqa: E402

TARGETS = {
    'clivia': {'pdb': '8HZE', 'pocket': 'clivia_8HZE_pocket.pdb', 'dummy': 'clivia_8HZE_dummy.sdf'},
    'squash': {'pdb': '7KVU', 'pocket': 'squash_7KVU_pocket.pdb', 'dummy': 'squash_7KVU_dummy.sdf'},
}
SEED = "case_study/clivia_squash/seed_2_4_diaryl_quinoline.sdf"
CKPT = "ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', choices=list(TARGETS), required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_prefix', type=str, required=True)
    ap.add_argument('--total_target', type=int, default=1000,
                    help="Target number of good-affinity molecules; monkey-patched "
                         "into genbypkt.generate at runtime.")
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--pin_scaffold', action='store_true',
                    help='pin the seed core for every iteration, not just the first')
    ap.add_argument('--seed_scaffold', action='store_true',
                    help='constrain generation to the shared 2,4-diaryl-quinoline core')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the model. Off by default: it costs many '
                         'minutes on the shared filesystem and only buys per-step speed.')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = TARGETS[args.target]
    pocket_pdb = os.path.join(SCRIPT_DIR, cfg['pocket'])
    dummy_sdf = os.path.join(SCRIPT_DIR, cfg['dummy'])
    for p, desc in [(pocket_pdb, 'Pocket PDB'), (dummy_sdf, 'Dummy SDF'), (CKPT, 'Checkpoint')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'{desc} not found: {p}')

    from model.bfn4edmol import Config, BFN4SBDDScoreModel
    import genbypkt
    from genbypkt import load_compiled_model

    args_dict = {
        'config_file': 'config.yaml', 'exp_name': 'clivia_squash_generation',
        'revision': 'default', 'debug': False, 'resume': False,
        'wandb_resume_id': None, 'empty_folder': False, 'test_only': False,
        'seed': args.seed, 'no_wandb': True, 'logging_level': 'warning',
        'random_rot': False, 'pos_noise_std': 0, 'pos_normalizer': 2.0,
        'batch_size': 8, 'epochs': 15, 'v_loss_weight': 1, 'lr': 5e-4,
        'scheduler': 'plateau', 'weight_decay': 0, 'max_grad_norm': 'Q',
        'sigma1_coord': 0.03, 'beta1': 1.5, 't_min': 0.0001,
        'use_discrete_t': True, 'discrete_steps': 1000,
        'destination_prediction': True, 'sampling_strategy': 'end_back_pmf',
        'time_emb_mode': 'simple', 'time_emb_dim': 1, 'pos_init_mode': 'zero',
        'ckpt_path': 'best', 'num_samples': 5, 'sample_steps': 100,
        'sample_num_atoms': 'ref', 'visual_chain': False,
        'docking_mode': 'vina_score', 'output_prefix': args.output_prefix,
    }
    conf = Config(**args_dict)

    print('[1/4] constructing model', flush=True)
    model = BFN4SBDDScoreModel(
        net_config=conf.dynamics.net_config,
        protein_atom_feature_dim=11, ligand_atom_feature_dim=13,
        time_emb_dim=1, ed_feature_dim=4,
    )
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    print('[2/4] loading checkpoint', flush=True)
    state_dict = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(load_compiled_model(model, state_dict), strict=False)
    if args.compile:
        print('[compile] torch.compile(mode=reduce-overhead) ...', flush=True)
        model = torch.compile(model, mode='reduce-overhead')
    print('[3/4] moving model to GPU', flush=True)
    model = accelerator.prepare(model)
    model.eval()

    if args.pin_scaffold:
        genbypkt.PIN_SCAFFOLD = True
        print('[pin] scaffold pinned for every iteration (genbypkt.PIN_SCAFFOLD=True)', flush=True)
    genbypkt.accelerator = accelerator
    genbypkt.model = model

    print('=' * 70)
    print(f'  Free generation: {args.target} ({cfg["pdb"]})')
    print(f'  pocket: {cfg["pocket"]}  dummy: {cfg["dummy"]}')
    print(f'  output_prefix={args.output_prefix}  total_target={args.total_target}')
    print(f'  seed={args.seed}  CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    print('  Mode: cold start, seed_used=False (no fluorophore guidance)')
    print('=' * 70)

    print('[4/4] generating', flush=True)
    t0 = time.time()
    task_ligand = os.path.join(SCRIPT_DIR, f"seed_{args.target}_2_4_diaryl.sdf") if args.seed_scaffold else dummy_sdf
    if args.seed_scaffold:
        print(f'[seed] constrained to {task_ligand}', flush=True)
    genbypkt.generate((task_ligand, pocket_pdb),
                      output_prefix=args.output_prefix,
                      seed_used=args.seed_scaffold,
                      total_target=args.total_target)
    print(f'\n[DONE] {args.target} in {time.time()-t0:.1f}s -> output/{args.output_prefix}/')


if __name__ == '__main__':
    main()
