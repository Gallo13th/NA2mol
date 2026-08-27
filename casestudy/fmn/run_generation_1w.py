#!/usr/bin/env python3
"""
Wrapper script to generate ligand molecules for the 1FMN RNA aptamer pocket.

Pure cold-start binding generation (no fluorophore guidance). Mirrors
casestudy/8hzk/run_generation.py: injects a pre-built model + accelerator into
genbypkt's namespace and calls ``genbypkt.generate()``.

The dummy ligand (casestudy/fmn/dummy_ligand.sdf) is a single carbon placed at
the FMN natural-ligand geometric center of MODEL 1. The pocket input is the
pre-extracted single-state pocket (1FMN_pocket_extracted_clean.pdb, MODEL/ENDMDL
stripped); genbypkt.generate() then re-extracts the <14 A RNA pocket and writes
the *_pocket.pdb / *_pocket_addh.pdb used for smina minimization.

Usage (run from project root):
    python casestudy/fmn/run_generation.py

Requirements:
    - Checkpoint at ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth
    - Pocket PDB at casestudy/fmn/1FMN_pocket_extracted_clean.pdb
    - Full PDB at casestudy/fmn/1FMN.pdb (FMN center only)
    - dummy ligand SDF at casestudy/fmn/dummy_ligand.sdf
    - obabel (OpenBabel) and util/smina.static available
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
# Step 1: Compute FMN natural-ligand geometric center from the full PDB
# ═══════════════════════════════════════════════════════════════════════════

def compute_fmn_center(pdb_path: str) -> np.ndarray:
    """Geometric center of the FMN HETATM ligand in MODEL 1 of 1FMN.pdb.

    Uses only MODEL 1 (first model) so the center reflects a single NMR state
    rather than the average across the whole ensemble.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", pdb_path)
    model1 = list(structure.get_models())[0]
    coords = []
    for residue in model1.get_residues():
        if residue.resname.strip().upper() == "FMN":
            for atom in residue:
                coords.append(atom.coord)
    if not coords:
        raise ValueError(f"No FMN HETATM residues found in MODEL 1 of {pdb_path}")
    return np.mean(coords, axis=0)


def create_dummy_sdf(center: np.ndarray, output_path: str) -> str:
    """
    Create a minimal SDF with a single carbon atom at `center`.
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()
    print(f"[✓] Created dummy ligand SDF: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Runtime patch — BFN4SBDDScoreModel.sample / sample_by_seed
# ═══════════════════════════════════════════════════════════════════════════
# genbypkt.generate() passes `return_traj=False` to model.sample() and
# model.sample_by_seed(), but BFN4SBDDScoreModel does NOT accept it.
# Patch both methods to strip the extra kwarg.

from model.bfn4edmol import BFN4SBDDScoreModel as _BFN4SBDDScoreModel

_original_sample = _BFN4SBDDScoreModel.sample
_original_sample_by_seed = _BFN4SBDDScoreModel.sample_by_seed


def _patched_sample(self, pocket_pos, pocket_v, batch_pocket, ed_pos, ed_v,
                    batch_ed, batch_ligand, n_nodes, sample_steps=1000,
                    desc='', ligand_pos=None, pocket_res=None, pocket_ele=None,
                    **kwargs):
    return _original_sample(
        self, pocket_pos=pocket_pos, pocket_v=pocket_v,
        batch_pocket=batch_pocket, ed_pos=ed_pos, ed_v=ed_v,
        batch_ed=batch_ed, batch_ligand=batch_ligand, n_nodes=n_nodes,
        sample_steps=sample_steps, desc=desc, ligand_pos=ligand_pos,
        pocket_res=pocket_res, pocket_ele=pocket_ele)


def _patched_sample_by_seed(self, pocket_pos, pocket_v, batch_pocket,
                            ed_pos, ed_v, batch_ed, batch_ligand, n_nodes,
                            fixed_atom_mask, fixed_atom_pos, fixed_atom_types,
                            sample_steps=1000, desc='', ligand_pos=None,
                            pocket_res=None, pocket_ele=None, **kwargs):
    return _original_sample_by_seed(
        self, pocket_pos=pocket_pos, pocket_v=pocket_v,
        batch_pocket=batch_pocket, ed_pos=ed_pos, ed_v=ed_v,
        batch_ed=batch_ed, batch_ligand=batch_ligand, n_nodes=n_nodes,
        fixed_atom_mask=fixed_atom_mask, fixed_atom_pos=fixed_atom_pos,
        fixed_atom_types=fixed_atom_types, sample_steps=sample_steps,
        desc=desc, ligand_pos=ligand_pos, pocket_res=pocket_res,
        pocket_ele=pocket_ele)


_BFN4SBDDScoreModel.sample = _patched_sample
_BFN4SBDDScoreModel.sample_by_seed = _patched_sample_by_seed
print("[✓] Patched BFN4SBDDScoreModel.sample / sample_by_seed to accept extra kwargs")


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Main generation entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # --- Paths ---
    # Use the pre-extracted, single-state pocket (MODEL/ENDMDL stripped) rather
    # than the full NMR ensemble: cleaner residue set, no duplicate states.
    pocket_pdb = "casestudy/fmn/1FMN_pocket_extracted_clean.pdb"
    full_pdb   = "casestudy/fmn/1FMN.pdb"          # full RNA + FMN (for center calc)
    dummy_sdf  = "casestudy/fmn/dummy_ligand.sdf"  # single C at FMN MODEL-1 center
    ckpt_path  = "ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth"

    # Validate inputs
    for p, desc in [(pocket_pdb, "Pocket PDB"), (full_pdb, "Full PDB (center)"),
                    (ckpt_path, "Checkpoint")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{desc} not found: {p}")

    # --- Step 4a: (Re)compute FMN MODEL-1 center & write dummy SDF ---
    center = compute_fmn_center(full_pdb)
    print(f"[•] FMN natural-ligand geometric center (MODEL 1): ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    create_dummy_sdf(center, dummy_sdf)

    # --- Step 4b: Build Config (same defaults as genbypkt main block / sp1) ---
    from model.bfn4edmol import Config, BFN4SBDDScoreModel
    import genbypkt
    from genbypkt import load_compiled_model

    args_dict = {
        'config_file': 'config.yaml',
        'exp_name': 'fmn_generation',
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
        'output_prefix': 'fmn_pocket',
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
    #   mol_file     → dummy ligand SDF (at FMN MODEL-1 center → drives pocket extraction)
    #   protein_file → pre-extracted single-state pocket PDB. generate() re-extracts
    #                  the <14 A pocket (keeps all 15 RNA residues) and writes
    #                  *_pocket.pdb / *_pocket_addh.pdb for smina.
    task = (dummy_sdf, pocket_pdb)
    print(f"[•] Task: ligand={dummy_sdf}, pocket={pocket_pdb}")

    # --- Step 4g: Run generation (pure cold-start, no fluor guidance) ---
    print("=" * 70)
    print("  Starting ligand generation for 1FMN RNA aptamer pocket...")
    print("  Model: onlypkt_ftwithrna_oncrossdocked (generalfeat)")
    print("  Output prefix: fmn_1w (NEUTRAL-ONLY reconstruction, no ions)")
    print("  Mode: cold-start (seed_used=False, NO fluor guidance)")
    print("=" * 70)

    genbypkt.generate(
        task,
        output_prefix='fmn_1w',
        seed_used=False,
    )

    print("\n[DONE] Generation complete. Check output/fmn_1w/")


if __name__ == '__main__':
    main()
