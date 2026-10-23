#!/usr/bin/env python3
"""Build a molecule from generated coordinates with the core's connectivity overwritten.

Read in the literal sense: let OpenBabel perceive the whole graph from the generated
coordinates, then *replace* the core's bonds with the known ones from the reference. The
core therefore cannot be altered by bond perception; only the substituents are perceived.

  pos       [n, 3] generated coordinates (real Angstrom)
  zs        [n]    atomic numbers
  core_idx  atoms belonging to the fixed core
  ref_mol   RDKit molecule carrying the known core connectivity, same atom order
"""
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem

from util.mol_utils import reconstruct_from_generated


def build_with_overwritten_core(pos, zs, core_idx: Iterable[int],
                                ref_mol: Chem.Mol) -> Optional[Chem.Mol]:
    pos = np.asarray(pos, dtype=float)
    nuts = [int(z) for z in zs]
    core = set(int(i) for i in core_idx)

    # 1. openbabel perceives the full graph, substituents included
    mol = reconstruct_from_generated(pos.tolist(), nuts)
    if mol is None:
        return None
    if mol.GetNumAtoms() != len(nuts):
        return None

    # 2. overwrite: drop every bond touching a core-core pair, then re-add the known ones
    rw = Chem.RWMol(mol)
    for b in list(rw.GetBonds()):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in core and j in core:
            rw.RemoveBond(i, j)
    for b in ref_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in core and j in core and i < rw.GetNumAtoms() and j < rw.GetNumAtoms():
            if rw.GetBondBetweenAtoms(i, j) is None:
                try:
                    rw.AddBond(i, j, b.GetBondType())
                except Exception:
                    pass

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.SanitizeMol(out, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception:
            return None
    return out
