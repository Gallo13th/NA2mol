#!/usr/bin/env python3
"""
Wrapper to run ligand generation for the 8HZK RNA pocket with optional
SPEC-02 fluorophore guidance.

Mirrors casestudy/sp1/run_generation.py: it injects a pre-built model +
accelerator into genbypkt's namespace and calls ``genbypkt.generate()``. The
two new flags (``--use_fluor_guidance`` / ``--use_fluor_replace``) are forwarded
to ``generate()``; when both are off the behaviour is identical to the original
pipeline (SPEC 02 §5 "default fallback").

Usage (run from project root):
    # fluorophore-guided generation (match + completeness + sumlogp seeds)
    python casestudy/8hzk/run_generation.py --use_fluor_guidance --use_fluor_replace
    # plain generation (original behaviour)
    python casestudy/8hzk/run_generation.py

Requirements:
    - Checkpoint at ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth
    - Pocket PDB at casestudy/8hzk/8HZK_pocket.pdb
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


def compute_pocket_center(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)
    coords = [a.coord for a in structure.get_atoms()]
    if not coords:
        raise ValueError(f"No atoms in {pdb_path}")
    return np.mean(coords, axis=0)


def create_dummy_sdf(center, output_path):
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol = mol.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, Point3D(float(center[0]), float(center[1]), float(center[2])))
    mol.AddConformer(conf)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w = Chem.SDWriter(output_path)
    w.write(mol)
    w.close()
    print(f"[✓] Created dummy ligand SDF: {output_path}")
    return output_path


# ── Runtime patches (same idea as sp1 wrapper) ─────────────────────────────
from dataloader.featurizer import GeomSeedMolOnlyPocket as _GeomSeedMolOnlyPocket
_original_init = _GeomSeedMolOnlyPocket.__init__

def _patched_init(self, pocket, seed, mode='general', pocket_radius=None,
                  extra_range=(0, 6), remain_seed=False, **kwargs):
    _original_init(self, pocket, seed, mode=mode, pocket_radius=pocket_radius,
                   extra_range=extra_range, remain_seed=remain_seed)
_GeomSeedMolOnlyPocket.__init__ = _patched_init
print("[✓] Patched GeomSeedMolOnlyPocket.__init__ to accept extra kwargs")

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fluor_guidance', action='store_true')
    parser.add_argument('--use_fluor_replace', action='store_true')
    args = parser.parse_args()

    pocket_pdb = 'casestudy/8hzk/8HZK_pocket.pdb'
    dummy_sdf = 'casestudy/8hzk/dummy_ligand.sdf'
    ckpt_path = 'ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth'

    for p, desc in [(pocket_pdb, 'Pocket PDB'), (ckpt_path, 'Checkpoint')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{desc} not found: {p}")

    center = compute_pocket_center(pocket_pdb)
    print(f"[•] Pocket center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    create_dummy_sdf(center, dummy_sdf)

    from model.bfn4edmol import Config, BFN4SBDDScoreModel
    import genbypkt
    from genbypkt import load_compiled_model

    args_dict = {
        'config_file': 'config.yaml', 'exp_name': '8hzk_generation', 'revision': 'default',
        'debug': False, 'resume': False, 'wandb_resume_id': None, 'empty_folder': False,
        'test_only': False, 'seed': 1234, 'no_wandb': True, 'logging_level': 'warning',
        'random_rot': False, 'pos_noise_std': 0, 'pos_normalizer': 2.0, 'batch_size': 8,
        'epochs': 15, 'v_loss_weight': 1, 'lr': 5e-4, 'scheduler': 'plateau',
        'weight_decay': 0, 'max_grad_norm': 'Q', 'sigma1_coord': 0.03, 'beta1': 1.5,
        't_min': 0.0001, 'use_discrete_t': True, 'discrete_steps': 1000,
        'destination_prediction': True, 'sampling_strategy': 'end_back_pmf',
        'time_emb_mode': 'simple', 'time_emb_dim': 1, 'pos_init_mode': 'zero',
        'ckpt_path': 'best', 'num_samples': 5, 'sample_steps': 100,
        'sample_num_atoms': 'ref', 'visual_chain': False, 'docking_mode': 'vina_score',
        'output_prefix': '8hzk_output', 'feat_mode': 'generalfeat',
        'use_cavity_volume': False, 'volume_prob_table_path': None,
        'use_probabilistic': False, 'prob_top_n': 5, 'prob_beam_width': 100,
    }
    cfg = Config(**args_dict)
    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config, protein_atom_feature_dim=11,
        ligand_atom_feature_dim=13, time_emb_dim=1, ed_feature_dim=4)

    print(f"[•] Loading checkpoint: {ckpt_path}")
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(load_compiled_model(model, state_dict), strict=False)
    model = torch.compile(model, mode="reduce-overhead")
    model = accelerator.prepare(model)
    model.eval()
    print("[✓] Model loaded and ready.")

    genbypkt.accelerator = accelerator
    genbypkt.model = model
    print("[✓] Injected model & accelerator into genbypkt namespace.")

    task = (dummy_sdf, pocket_pdb)
    print("=" * 70)
    print("  Starting ligand generation for 8HZK pocket...")
    print(f"  fluor_guidance={args.use_fluor_guidance}  fluor_replace={args.use_fluor_replace}")
    print("=" * 70)

    genbypkt.generate(
        task,
        output_prefix='8hzk_output',
        fragment_replace=False,
        seed_used=False,
        feat_mode='generalfeat',
        use_probabilistic=False,
        prob_top_n=5,
        prob_beam_width=100,
        use_cavity_volume=False,
        volume_prob_table_path=None,
        use_fluor_guidance=args.use_fluor_guidance,
        use_fluor_replace=args.use_fluor_replace,
    )
    print("\n[DONE] Generation complete. Check output/8hzk_output/")


if __name__ == '__main__':
    main()
