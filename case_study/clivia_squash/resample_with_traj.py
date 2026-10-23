#!/usr/bin/env python3
"""Regenerate substituents with the core conditioned on a REAL trajectory.

Takes one harvested trajectory (see harvest_trajectory.py), pins the atoms that form the
2,4-diaryl-quinoline core, and calls sample_by_seed with given_traj = the actual per-step
belief states the model visited during free generation.

This is the difference from the earlier failed seeded runs: those let sample_by_seed build
an analytic pseudo path mu(t) = gamma(t) * x_true for the fixed atoms, which is a state the
network never visits; here the fixed atoms follow the path it really took.

Usage:
  NA2SMOL_ATOM_RANGE=27,27 CUDA_VISIBLE_DEVICES=5 python resample_with_traj.py \
      --traj case_study/clivia_squash/traj_squash.pt --hit 0 \
      --out_prefix traj_squash_h0 --n_batch 150
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

from rdkit import Chem, RDLogger                     # noqa: E402
from accelerate import Accelerator                   # noqa: E402
from torch.utils.data import ConcatDataset           # noqa: E402

RDLogger.DisableLog('rdApp.*')

from dataloader.featurizer import ProteinLigandDataLoader, GeomSeedMolOnlyPocket  # noqa: E402
from model.bfn4edmol import Config, BFN4SBDDScoreModel                            # noqa: E402
from genbypkt import load_compiled_model                                          # noqa: E402
from util.mol_utils import (reconstruct_from_generated,                            # noqa: E402
                            MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB)

CKPT = "ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth"
TARGETS = {
    'clivia': ('clivia_8HZE_pocket.pdb', 'clivia_8HZE_dummy.sdf'),
    'squash': ('squash_7KVU_pocket.pdb', 'squash_7KVU_dummy.sdf'),
}


def full_ring_system_core(h, n_atoms):
    """Indices of ALL atoms in the three connected ring systems.

    The harvested core_idx holds only the quinoline plus the two ipso carbons of the aryl
    rings. The target skeleton is the quinoline fused system plus both complete aryl rings
    (22 atoms for the diaryl-quinoline series), so expand each ipso atom to its whole ring.
    """
    from rdkit import Chem
    from util.mol_utils import reconstruct_from_generated, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB
    pos = h['mu_pos'][-1].numpy()
    typ = h['theta_h'][-1].numpy()
    sel = h['idx_global']
    P = pos[sel] * 2
    nums = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(int(np.argmax(typ[g])), (6, False))[0]
            for g in sel]
    mol = reconstruct_from_generated(P.tolist(), nums)
    if mol is None:
        return sorted({h['idx_global'].index(g) for g in h['core_idx']})
    ri = mol.GetRingInfo()
    QUIN = Chem.MolFromSmarts('c1ccc2ncccc2c1')
    atoms = set()
    for mt in mol.GetSubstructMatches(QUIN):
        atoms.update(mt)
        for p_ in (5, 7):
            for nb in mol.GetAtomWithIdx(mt[p_]).GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in mt:
                    atoms.add(nb.GetIdx())
                    for ring in ri.AtomRings():
                        if nb.GetIdx() in ring:
                            atoms.update(ring)
    return sorted(atoms)


def _load_overwrite_builder():
    import importlib.util
    path = SCRIPT_DIR / 'overwrite_core_mol.py'
    spec = importlib.util.spec_from_file_location('ocm', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['ocm'] = m
    spec.loader.exec_module(m)
    return m


def build_model():
    args_dict = {
        'config_file': 'config.yaml', 'exp_name': 'resample', 'revision': 'default',
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
        'output_prefix': 'resample',
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
    ap.add_argument('--target', choices=list(TARGETS), default='squash')
    ap.add_argument('--traj', required=True)
    ap.add_argument('--hit', type=int, default=0)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--n_batch', type=int, default=150)
    ap.add_argument('--sample_steps', type=int, default=100)
    ap.add_argument('--write_sdf', action='store_true', default=True)
    a = ap.parse_args()

    hits = torch.load(a.traj, map_location='cpu', weights_only=False)
    h = hits[a.hit]
    n_atoms = len(h['idx_global'])
    print(f'hit {a.hit}: {h["smiles"]}')
    print(f'  {n_atoms} atoms, core = {len(h["core_idx"])} atoms')

    # one molecule's real trajectory: [T+1, n_atoms, 3] and [T+1, n_atoms, K]
    mu1 = h['mu_pos']                                   # [T+1, N_total, 3]
    th1 = h['theta_h']
    sel = torch.tensor(h['idx_global'])
    mu_mol = mu1[:, sel, :].float()                     # [T+1, n_atoms, 3]
    th_mol = th1[:, sel, :].float()                     # [T+1, n_atoms, K]
    K = th_mol.shape[-1]

    # core indices relative to this molecule
    core_rel = full_ring_system_core(h, n_atoms)
    print(f'  core at relative indices {core_rel}')

    # reference molecule: the harvested core with its true connectivity, same atom order
    ref_pos = mu_mol[-1].numpy() * 2.0
    zs = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(int(th_mol[-1, k].argmax()), (6, False))[0]
          for k in range(n_atoms)]
    ref_mol = reconstruct_from_generated(ref_pos.tolist(), zs)
    if ref_mol is None:
        print('reference reconstruction failed'); return 1
    OCM = _load_overwrite_builder()

    bs = 8
    pocket = str(SCRIPT_DIR / TARGETS[a.target][0])
    dummy = str(SCRIPT_DIR / TARGETS[a.target][1])
    model, acc = build_model()
    unwrapped = acc.unwrap_model(model)
    device = acc.device

    # tile the single-molecule trajectory across the batch
    given_mu = mu_mol.unsqueeze(1).repeat(1, bs, 1, 1).reshape(mu_mol.shape[0], bs * n_atoms, 3)
    given_th = th_mol.unsqueeze(1).repeat(1, bs, 1, 1).reshape(th_mol.shape[0], bs * n_atoms, K)

    ds = GeomSeedMolOnlyPocket(pocket=pocket, seed=dummy, mode='general', extra_range=(-8, 8))
    batch_dataset = ConcatDataset([ds] * bs)

    outdir = SCRIPT_DIR / f'output_traj/{a.out_prefix}'
    outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    # one writer for the whole run (re-opening truncates the file), and no kekulisation:
    # molecules whose core could not be kekulised would otherwise raise on write
    sdf_path = outdir / 'molecules.sdf'
    writer = Chem.SDWriter(str(sdf_path))
    writer.SetKekulize(False)
    for nb in range(a.n_batch):
        dl = ProteinLigandDataLoader(batch_dataset, batch_size=bs, shuffle=False)
        batch = next(iter(dl)).to(device)
        if batch.mol_element_batch.numel() != bs * n_atoms:
            print(f'  batch {nb}: atom count changed, skipping'); continue
        # fixed mask / positions / types for the core in every segment
        mask = torch.zeros(bs * n_atoms, dtype=torch.bool, device=device)
        # sample_by_seed wants one row per fixed atom across the WHOLE batch, in mask order
        fpos = torch.zeros(bs * len(core_rel), 3, device=device)
        ftyp = torch.zeros(bs * len(core_rel), dtype=torch.long, device=device)
        slot = 0
        for b in range(bs):
            for j, c in enumerate(core_rel):
                mask[b * n_atoms + c] = True
                fpos[slot] = mu_mol[-1, c]
                ftyp[slot] = int(th_mol[-1, c].argmax())
                slot += 1
        with torch.no_grad():
            theta_traj, sample_traj, _, _, _ = unwrapped.sample_by_seed(
                pocket_pos=batch.pocket_pos / 2,
                pocket_v=batch.pocket_atom_feature.float(),
                batch_pocket=batch.pocket_element_batch,
                ed_pos=torch.empty((0, 3), device=device),
                ed_v=torch.empty((0, 4), device=device),
                batch_ed=batch.ed_ed_code_batch,
                batch_ligand=batch.mol_element_batch,
                n_nodes=bs,
                fixed_atom_mask=mask,
                fixed_atom_pos=fpos,          # already model space: mu_mol is normalised
                fixed_atom_types=ftyp,
                sample_steps=a.sample_steps,
                pocket_res=None, pocket_ele=None,
                fix_atom_types=True,
                given_traj=(given_mu, given_th),
            )
        # write with the known core overwritten, not re-perceived by OpenBabel
        P = (sample_traj[-1][0].detach().cpu().numpy() * 2.0
             + batch.mol_geom_center[:3].detach().cpu().numpy())
        T = sample_traj[-1][1].detach().cpu().numpy()
        for b in range(bs):
            lo, hi = b * n_atoms, (b + 1) * n_atoms
            zzb = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(int(np.argmax(T[k])), (6, False))[0]
                   for k in range(lo, hi)]
            mol = OCM.build_with_overwritten_core(P[lo:hi], zzb, core_rel, ref_mol)
            if mol is None:
                continue
            mol.SetProp('_Name', f'{a.out_prefix}_{written}')
            try:
                writer.write(mol)
                written += 1
            except Exception as e:
                if failed < 5:
                    print(f'  write failed ({type(e).__name__}): skipped')
                failed += 1
        if nb % 10 == 0:
            print(f'  batch {nb}/{a.n_batch} ({written} molecules)', flush=True)

    writer.close()
    print(f'\nwrote {written} molecules to {sdf_path}  ({failed} failed)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
