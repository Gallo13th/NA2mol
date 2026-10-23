#!/usr/bin/env python3
"""Compare the two core-preserving builders on the harvested hits."""
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

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

K = load('K', f'{ROOT}/case_study/clivia_squash/known_core_mol.py')
O = load('O', f'{ROOT}/case_study/clivia_squash/overwrite_core_mol.py')
from util.mol_utils import reconstruct_from_generated, MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB

QUIN = Chem.MolFromSmarts('c1ccc2ncccc2c1')


def full_core(ref):
    ri = ref.GetRingInfo(); atoms = set()
    for mt in ref.GetSubstructMatches(QUIN):
        atoms.update(mt)
        for p in (5, 7):
            for nb in ref.GetAtomWithIdx(mt[p]).GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in mt:
                    atoms.add(nb.GetIdx())
                    for ring in ri.AtomRings():
                        if nb.GetIdx() in ring:
                            atoms.update(ring)
    return sorted(atoms)


hits = torch.load(f'{ROOT}/case_study/clivia_squash/traj_squash.pt',
                  map_location='cpu', weights_only=False)
print(f'{"hit":>3} {"core":>4}  {"ref-core kept (A)":>18} {"(B)":>6}  {"frags A":>7} {"frags B":>7}  B smiles')
for i, h in enumerate(hits):
    pos = h['mu_pos'][-1].numpy(); typ = h['theta_h'][-1].numpy(); sel = h['idx_global']
    P = pos[sel] * 2.0
    zs = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(int(np.argmax(typ[g])), (6, False))[0] for g in sel]
    syms = [Chem.GetPeriodicTable().GetElementSymbol(z) for z in zs]
    ref = reconstruct_from_generated(P.tolist(), zs)
    if ref is None:
        print(f'{i:3d} reference failed'); continue
    core = full_core(ref)
    if not core:
        print(f'{i:3d} no core detected'); continue
    refsub = K.core_submol(ref, core)
    ref_smi = Chem.MolToSmiles(refsub) if refsub else None

    a = K.build_mol_with_known_core(P, syms, core, ref)
    b = O.build_with_overwritten_core(P, zs, core, ref)

    def kept(m):
        if m is None: return 'FAIL'
        s = K.core_submol(m, core)
        if s is None: return 'unsanitised'
        return 'exact' if Chem.MolToSmiles(s) == ref_smi else 'DIFF'

    def nfrag(m):
        if m is None: return '-'
        return len(Chem.GetMolFrags(m))

    print(f'{i:3d} {len(core):4d}  {kept(a):>18} {kept(b):>6}  {nfrag(a):7} {nfrag(b):7}  '
          f'{Chem.MolToSmiles(b)[:46] if b else "n/a"}')
