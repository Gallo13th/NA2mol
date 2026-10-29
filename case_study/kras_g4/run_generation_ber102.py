#!/usr/bin/env python3
"""
Wrapper script to generate ligand molecules for the KRAS G4 pocket (7X8M).

Pure cold-start binding generation (no fluorophore guidance, seed_used=False).
Mirrors case_study/fmn/run_generation.py: injects a pre-built model + accelerator
into genbypkt's namespace and calls ``genbypkt.generate()``.

Inputs (all pre-built by extract_pocket.py):
  - pocket PDB : case_study/kras_g4/7X8M_ber101_pocket.pdb  (MODEL-1 nucleotides
                 within 14 A of the berberine BER101 ligand center; single state)
  - dummy SDF  : case_study/kras_g4/dummy_ligand.sdf (single C at BER101 center)
  - checkpoint : onlypkt_ftwithrna_oncrossdocked (...generalfeat)

Cold-start atom count is clamped to randint(20,28) by dataloader/featurizer
(matching the 25 heavy atoms of the target cyanine dye). Aromatic ammonium
ions (quinolinium N+) are PRESERVED by util/mol_utils._neutralize_or_reject
so charged cyanine ligands survive; only non-aromatic ions are rejected.

genbypkt.generate's default scale (total_target=10000, max iteration=150) is
patched at runtime via inspect+exec to a per-run --total_target; the genbypkt.py
source file itself is never edited, so the 10000/150 defaults remain intact for
all other callers. This enables 4-GPU parallel runs (distinct seeds -> distinct
molecule streams; distinct output_prefix -> no collisions).

Usage (from project root), one process per GPU:
    CUDA_VISIBLE_DEVICES=0 python case_study/kras_g4/run_generation.py \\
        --seed 42 --output_prefix kras_g4_s0 --total_target 2500
"""

import sys
import os
import argparse
import random
import time
import inspect
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # rna2smol/
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from accelerate import Accelerator


def main():
    parser = argparse.ArgumentParser(
        description="Cold-start ligand generation for KRAS G4 (7X8M BER101) pocket.")
    parser.add_argument('--seed', type=int, default=42,
                        help=" RNG seed (random / numpy / torch). Distinct per GPU"
                             " -> distinct molecule streams. Controls featurizer"
                             " randint(20,28) atom-count draw.")
    parser.add_argument('--output_prefix', type=str, default='kras_g4_output',
                        help="genbypkt.generate output_prefix (distinct per GPU to"
                             " avoid overwriting each other's output/ dirs).")
    parser.add_argument('--total_target', type=int, default=10000,
                        help="Target # of good-affinity (Affinity<0) molecules."
                             " Monkey-patched into genbypkt.generate at runtime;"
                             " genbypkt.py source is left untouched.")
    parser.add_argument('--gpu', type=int, default=None,
                        help="Logical GPU id, logged for bookkeeping. Actual device"
                             " is selected via the CUDA_VISIBLE_DEVICES env var set"
                             " at launch (cannot be changed after CUDA init).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[✓] RNG seeds set: random/numpy/torch = {args.seed}")

    pocket_pdb = "case_study/kras_g4/7X8M_ber102c_pocket.pdb"
    dummy_sdf  = "case_study/kras_g4/dummy_ber102.sdf"
    ckpt_path  = "ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth"

    for p, desc in [(pocket_pdb, "Pocket PDB"), (dummy_sdf, "Dummy ligand SDF"),
                    (ckpt_path, "Checkpoint")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{desc} not found: {p}")

    from model.bfn4edmol import Config, BFN4SBDDScoreModel
    import genbypkt
    from genbypkt import load_compiled_model

    args_dict = {
        'config_file': 'config.yaml', 'exp_name': 'kras_g4_generation',
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
    cfg = Config(**args_dict)

    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11, ligand_atom_feature_dim=13,
        time_emb_dim=1, ed_feature_dim=4,
    )

    print(f"[•] Loading checkpoint: {ckpt_path}")
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(load_compiled_model(model, state_dict), strict=False)
    print("[•] Compiling model (mode=reduce-overhead)...")
    model = torch.compile(model, mode="reduce-overhead")
    model = accelerator.prepare(model)
    model.eval()
    print("[✓] Model loaded and ready.")

    genbypkt.accelerator = accelerator
    genbypkt.model = model
    print("[✓] Injected model & accelerator into genbypkt namespace.")

    task = (dummy_sdf, pocket_pdb)
    print(f"[•] Task: ligand={dummy_sdf}, pocket={pocket_pdb}")

    print("=" * 70)
    print("  Starting ligand generation for KRAS G4 pocket (7X8M BER101)...")
    print("  Model: onlypkt_ftwithrna_oncrossdocked (generalfeat)")
    print(f"  Output prefix: {args.output_prefix}")
    print(f"  total_target: {args.total_target}  (seed={args.seed}, gpu={args.gpu},"
          f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print("  Mode: cold-start (seed_used=False, NO fluor guidance)")
    print("  Atom count: randint(20,28); aromatic ammonium PRESERVED")
    print("=" * 70)

    t0 = time.time()
    genbypkt.generate(
        task,
        output_prefix=args.output_prefix,
        seed_used=False,
        total_target=args.total_target,
    )
    print(f"\n[DONE] Generation complete in {time.time()-t0:.1f}s. "
          f"Check output/{args.output_prefix}/")


if __name__ == '__main__':
    main()
