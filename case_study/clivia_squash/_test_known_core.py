#!/usr/bin/env python3
"""Validate known_core_mol: rebuilding a harvested hit must preserve the core exactly."""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = '/home/data/users/gaoletian/rna2smol'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

spec = importlib.util.spec_from_file_location('K', f'{ROOT}/case_study/clivia_squash/known_core_mol.py')
K = importlib.util.module_from_spec(spec); sys.modules['K'] = K; spec.loader.exec_module(K)

from util.mol_utils import reconstruct_from_generated, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB

QUIN = Chem.MolFromSmarts('c1ccc2ncccc2c1')


def full_core(ref, h):
    ri = ref.GetRingInfo()
    atoms = set()
    for mt in ref.GetSubstructMatches(QUIN):
        atoms.update(mt)
        for p in (5, 7):
            for nb in ref.GetAtomWithIdx(mt[p]).GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in mt:
                    atoms.add(nb.GetIdx())
                    for ring in ri.AtomRings():
                        if nb.GetIdx() in ring:
                            atoms.update(ring)
    if not atoms:
        atoms = {h['idx_global'].index(g) for g in h['core_idx']}
    return sorted(atoms)


hits = torch.load(f'{ROOT}/case_study/clivia_squash/traj_squash.pt',
                  map_location='cpu', weights_only=False)
print(f'{"hit":>3} {"atoms":>5} {"core":>4}  ref-core  built-core  whole-match  smiles')
for i, h in enumerate(hits):
    pos = h['mu_pos'][-1].numpy()
    typ = h['theta_h'][-1].numpy()
    sel = h['idx_global']
    P = pos[sel] * 2.0
    zs = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(int(np.argmax(typ[g])), (6, False))[0] for g in sel]
    syms = [Chem.GetPeriodicTable().GetElementSymbol(z) for z in zs]

    ref = reconstruct_from_generated(P.tolist(), zs)
    if ref is None:
        print(f'{i:3d}  reference reconstruction failed'); continue
    core = full_core(ref, h)
    core_smi = Chem.MolToSmiles(K.core_submol(ref, core)) if K.core_submol(ref, core) else 'n/a'

    built = K.build_mol_with_known_core(P, syms, core, ref)
    if built is None:
        print(f'{i:3d} {len(zs):5d} {len(core):4d}  {core_smi[:20]:20s}  BUILD FAILED'); continue
    sub = K.core_submol(built, core)
    refsub = K.core_submol(ref, core)
    same_core = (sub is not None and refsub is not None and
                 Chem.MolToSmiles(sub) == Chem.MolToSmiles(refsub))
    same_whole = Chem.MolToSmiles(built) == Chem.MolToSmiles(ref)
    print(f'{i:3d} {len(zs):5d} {len(core):4d}  {core_smi[:20]:20s}  '
          f'{"exact" if same_core else "DIFF ":5s}       {"yes" if same_whole else "no":3s}'
          f'    {Chem.MolToSmiles(built)[:44]}')
