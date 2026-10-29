#!/usr/bin/env python3
"""
Wrapper script to generate ligand molecules for the sp1 RNA pocket.

Uses genbypkt.py's generate() function WITHOUT modifying any core project files.
All customizations are done via runtime injection (monkey-patching module globals).

Usage (run from project root):
    python casestudy/sp1/run_generation.py

Requirements:
    - Model checkpoint at: ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth
    - Pocket PDB at: casestudy/sp1/sp1_recp_pocket.pdb
    - obabel (OpenBabel) available in PATH
    - smina static binary at: util/smina.static
"""

import sys
import os
import numpy as np
import torch

# ── Resolve project root and switch to it ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # rna2smol/
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from rdkit import Chem
from rdkit.Geometry import Point3D
from Bio.PDB import PDBParser
from accelerate import Accelerator


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Compute pocket geometric center from sp1_recp_pocket.pdb
# ═══════════════════════════════════════════════════════════════════════════

def compute_pocket_center(pdb_path: str) -> np.ndarray:
    """Compute the geometric center (mean) of all atoms in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)
    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.coord)
    if not coords:
        raise ValueError(f"No atoms found in {pdb_path}")
    return np.mean(coords, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Create a dummy ligand SDF (single carbon at pocket center)
# ═══════════════════════════════════════════════════════════════════════════

def create_dummy_sdf(center: np.ndarray, output_path: str) -> str:
    """
    Create a minimal SDF file with a single carbon atom at `center`.
    This satisfies genbypkt's requirement for a ligand SDF to:
      (a) compute the pocket center for residue extraction,
      (b) serve as the initial seed for cold-start generation.
    """
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))  # Carbon
    mol = mol.GetMol()

    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, Point3D(float(center[0]), float(center[1]), float(center[2])))
    mol.AddConformer(conf)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()
    print(f"[✓] Created dummy ligand SDF: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Main generation entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cold-start generation for the sp1 pocket.')
    parser.add_argument('--total_target', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--iter_cap', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    # --- Paths ---
    pocket_pdb = "casestudy/sp1/sp1_recp_pocket.pdb"
    dummy_sdf  = "casestudy/sp1/dummy_ligand.sdf"
    ckpt_path  = "ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth"

    # Validate inputs
    for p, desc in [(pocket_pdb, "Pocket PDB"), (ckpt_path, "Checkpoint")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{desc} not found: {p}")

    # --- Step 4a: Compute center & create dummy SDF ---
    center = compute_pocket_center(pocket_pdb)
    print(f"[•] Pocket geometric center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    create_dummy_sdf(center, dummy_sdf)

    # --- Step 4b: Build Config (exact same defaults as genbypkt main block) ---
    from model.bfn4edmol import Config, BFN4SBDDScoreModel
    import genbypkt
    from genbypkt import load_compiled_model

    args_dict = {
        'config_file': 'config.yaml',
        'exp_name': 'sp1_generation',
        'revision': 'default',
        'debug': False,
        'resume': False,
        'wandb_resume_id': None,
        'empty_folder': False,
        'test_only': False,
        'seed': 1234,
        'no_wandb': True,
        'logging_level': 'warning',
        'random_rot': False,
        'pos_noise_std': 0,
        'pos_normalizer': 2.0,
        'batch_size': 8,
        'epochs': 15,
        'v_loss_weight': 1,
        'lr': 5e-4,
        'scheduler': 'plateau',
        'weight_decay': 0,
        'max_grad_norm': 'Q',
        'sigma1_coord': 0.03,
        'beta1': 1.5,
        't_min': 0.0001,
        'use_discrete_t': True,
        'discrete_steps': 1000,
        'destination_prediction': True,
        'sampling_strategy': 'end_back_pmf',
        'time_emb_mode': 'simple',
        'time_emb_dim': 1,
        'pos_init_mode': 'zero',
        'ckpt_path': 'best',
        'num_samples': 5,
        'sample_steps': 100,
        'sample_num_atoms': 'ref',
        'visual_chain': False,
        'docking_mode': 'vina_score',
        'output_prefix': 'sp1_output',
    }

    cfg = Config(**args_dict)

    # --- Step 4c: Build model ---
    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11,
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4,
    )

    # --- Step 4d: Load checkpoint with Accelerator ---
    print(f"[•] Loading checkpoint: {ckpt_path}")
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(
        load_compiled_model(model, state_dict),
        strict=False,
    )

    print("[•] Compiling model (mode=reduce-overhead)...")
    model = torch.compile(model, mode="reduce-overhead")
    model = accelerator.prepare(model)
    model.eval()
    print("[✓] Model loaded and ready.")

    # --- Step 4e: Inject model & accelerator into genbypkt's namespace ---
    # generate() references `model` and `accelerator` as module-level globals.
    genbypkt.accelerator = accelerator
    genbypkt.model = model
    print("[✓] Injected model & accelerator into genbypkt namespace.")

    # --- Step 4f: Build task tuple ---
    # task format: (mol_file, protein_file)
    #   mol_file    → dummy ligand SDF (for pocket center + seed)
    #   protein_file → the pre-extracted pocket PDB
    task = (dummy_sdf, pocket_pdb)
    print(f"[•] Task: ligand={dummy_sdf}, pocket={pocket_pdb}")

    # --- Step 4g: Run generation ---
    print("=" * 70)
    print("  Starting ligand generation for sp1 pocket...")
    print("  Model: onlypkt_ftwithrna_oncrossdocked (generalfeat)")
    print("  Output prefix: sp1_output")
    print("  Mode: cold-start (no seed fragments)")
    print("=" * 70)

    genbypkt.generate(
        task,
        output_prefix='sp1_output',
        seed_used=False,
        total_target=args.total_target,
        batch_size=args.batch_size,
        iter_cap=args.iter_cap,
    )

    print("\n[DONE] Generation complete. Check output/sp1_output/")


if __name__ == '__main__':
    main()
