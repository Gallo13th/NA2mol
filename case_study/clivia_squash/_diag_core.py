#!/usr/bin/env python3
"""Diagnostic: does the core survive resampling with a real trajectory?

Checks the RAW sampler output (before any molecule reconstruction) for the fixed core atoms,
so a failure can be attributed to the sampler or to reconstruction rather than guessed at.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = '/home/data/users/gaoletian/rna2smol'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from torch.utils.data import ConcatDataset

spec = importlib.util.spec_from_file_location(
    'R', f'{ROOT}/case_study/clivia_squash/resample_with_traj.py')
R = importlib.util.module_from_spec(spec)
sys.modules['R'] = R
spec.loader.exec_module(R)

from dataloader.featurizer import ProteinLigandDataLoader, GeomSeedMolOnlyPocket  # noqa: E402

HIT = int(os.environ.get('HIT', '3'))
h = torch.load(f'{ROOT}/case_study/clivia_squash/traj_squash.pt',
               map_location='cpu', weights_only=False)[HIT]
n = len(h['idx_global'])
sel = torch.tensor(h['idx_global'])
mu = h['mu_pos'][:, sel, :].float()
th = h['theta_h'][:, sel, :].float()
K = th.shape[-1]
core = sorted(R.full_ring_system_core(h, n))
print(f'hit {HIT}: {n} atoms, core {len(core)} -> {h["smiles"]}')

model, acc = R.build_model()
unw = acc.unwrap_model(model)
dev = acc.device
bs = 8
ds = GeomSeedMolOnlyPocket(pocket=f'{ROOT}/case_study/clivia_squash/squash_7KVU_pocket.pdb',
                           seed=f'{ROOT}/case_study/clivia_squash/squash_7KVU_dummy.sdf',
                           mode='general', extra_range=(-8, 8))
bd = ConcatDataset([ds] * bs)
batch = next(iter(ProteinLigandDataLoader(bd, batch_size=bs, shuffle=False))).to(dev)
print('batch atoms', batch.mol_element_batch.numel(), 'expect', bs * n)

gmu = mu.unsqueeze(1).repeat(1, bs, 1, 1).reshape(mu.shape[0], bs * n, 3)
gth = th.unsqueeze(1).repeat(1, bs, 1, 1).reshape(th.shape[0], bs * n, K)
mask = torch.zeros(bs * n, dtype=torch.bool)
fp = torch.zeros(bs * len(core), 3)
ft = torch.zeros(bs * len(core), dtype=torch.long)
s = 0
for b in range(bs):
    for c in core:
        mask[b * n + c] = True
        fp[s] = mu[-1, c]
        ft[s] = int(th[-1, c].argmax())
        s += 1

with torch.no_grad():
    _, st, _, _, _ = unw.sample_by_seed(
        pocket_pos=batch.pocket_pos / 2,
        pocket_v=batch.pocket_atom_feature.float(),
        batch_pocket=batch.pocket_element_batch,
        ed_pos=torch.empty((0, 3), device=dev),
        ed_v=torch.empty((0, 4), device=dev),
        batch_ed=batch.ed_ed_code_batch,
        batch_ligand=batch.mol_element_batch,
        n_nodes=bs,
        fixed_atom_mask=mask.to(dev),
        fixed_atom_pos=fp.to(dev),
        fixed_atom_types=ft.to(dev),
        sample_steps=100,
        pocket_res=None, pocket_ele=None,
        fix_atom_types=True,
        given_traj=(gmu, gth),
    )
out = st[-1][0].cpu()
outt = st[-1][1].cpu()
idx = [c for c in core]
print('\n--- core fidelity in the RAW sample output (segment 0) ---')
print('  pos RMSD vs harvested :',
      round(float(torch.sqrt(((out[idx] - mu[-1, core]) ** 2).sum(1).mean())), 4))
print('  type agreement        :',
      int((outt[idx].argmax(-1) == th[-1, core].argmax(-1)).sum()), '/', len(core))
print('  harvested core types  :', th[-1, core].argmax(-1).tolist())
print('  output    core types  :', outt[idx].argmax(-1).tolist())
