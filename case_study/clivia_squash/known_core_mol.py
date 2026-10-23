#!/usr/bin/env python3
"""Build a molecule from generated coordinates, keeping a known core's connectivity.

OpenBabel's bond perception (`reconstruct_from_generated`) re-derives the whole graph and
does not reliably return the core that was fed in. Here the core is *given*: its bonds are
copied verbatim from a reference molecule whose atom ordering matches the sampled tensors.
Bonds are only perceived where they are genuinely unknown, i.e. for the substituent atoms
and their attachment to the core.

  pos       [n, 3] generated coordinates (already in real Angstrom)
  symbols   [n]    element symbols
  core_idx  atoms belonging to the fixed core
  ref_mol   RDKit molecule with the known connectivity, same atom order as pos

Bond perception for the free atoms is a distance criterion with covalent radii, added
greedily shortest-first under a valence cap, so partially connected substituents degrade
gracefully instead of producing a disconnected graph.
"""
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

PT = Chem.GetPeriodicTable()
COVALENT = {z: PT.GetRcovalent(z) for z in range(1, 119)}
FACTOR = 1.3


def _valence_cap(z: int) -> int:
    v = PT.GetDefaultValence(z)
    return v if v > 0 else 4


def build_mol_with_known_core(pos, symbols, core_idx: Iterable[int],
                              ref_mol: Chem.Mol,
                              factor: float = FACTOR) -> Optional[Chem.Mol]:
    pos = np.asarray(pos, dtype=float)
    n = len(symbols)
    core = set(int(i) for i in core_idx)
    if n == 0 or len(core) == 0:
        return None

    rw = Chem.RWMol()
    for sym in symbols:
        rw.AddAtom(Chem.Atom(sym))
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(*pos[i]))
    rw.AddConformer(conf, assignId=True)

    # 1. core bonds copied verbatim from the reference
    zs = [PT.GetAtomicNumber(s) for s in symbols]
    used = np.zeros(n, dtype=int)
    for b in ref_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in core and j in core and i < n and j < n:
            try:
                rw.AddBond(i, j, b.GetBondType())
                used[i] += 1; used[j] += 1
            except Exception:
                pass

    # 2. perceive only the unknown bonds, shortest first
    free = [i for i in range(n) if i not in core]
    cands = []
    for i in free:
        for j in range(n):
            if j == i:
                continue
            if j in core and i in core:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            lim = factor * (COVALENT.get(zs[i], 1.5) + COVALENT.get(zs[j], 1.5))
            if 0.4 < d < lim:
                cands.append((d, i, j))
    cands.sort(key=lambda t: t[0])

    def degree(k):
        return used[k]

    for d, i, j in cands:
        if degree(i) >= _valence_cap(zs[i]) or degree(j) >= _valence_cap(zs[j]):
            continue
        if rw.GetBondBetweenAtoms(i, j) is not None:
            continue
        try:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
            used[i] += 1; used[j] += 1
        except Exception:
            pass

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception:
            return None
    return mol


def core_submol(ref_mol: Chem.Mol, core_idx: Iterable[int]) -> Chem.Mol:
    """The core as its own molecule, for reporting / comparison."""
    core = set(int(i) for i in core_idx)
    rw = Chem.RWMol()
    order = {}
    for i in range(ref_mol.GetNumAtoms()):
        if i in core:
            order[i] = rw.AddAtom(Chem.Atom(ref_mol.GetAtomWithIdx(i).GetSymbol()))
    for b in ref_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in order and j in order:
            rw.AddBond(order[i], order[j], b.GetBondType())
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m
