#!/usr/bin/env python3
"""Structural check that the specified core is intact and has NOT been extended.

The series is built from **three ring systems**: the fused quinoline (two rings) and two
separate aryl rings, each attached by a single bond. A generated molecule that closes an
extra ring between the core and the substituents (a fused cyclopentaquinoline, say) still
contains the core atoms but is no longer the specified skeleton - it breaks the three-ring-
system rule.

Counting rings does not catch this: the fused hit also has four rings. What differs is how
they are fused, so the check counts **ring systems** - connected components of the ring
adjacency graph, where two rings are joined if they share a bond (>= 2 atoms).

  targets (mol1-mol4) : 3 systems -> [2 rings, 1 ring, 1 ring]
  4-ring fused hits   : 1 system  -> [4 rings]
"""
from typing import Optional, Tuple

from rdkit import Chem


def ring_systems(mol: Chem.Mol):
    """Connected components of rings, joined when they share a bond."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = len(rings)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            # share a bond <=> at least two atoms in common
            if len(rings[i] & rings[j]) >= 2:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[rings[i] for i in g] for g in groups.values()]


def check_three_ring_systems(mol: Chem.Mol,
                             core_idx=None) -> Tuple[bool, str]:
    """True only if the molecule has exactly the specified 3-ring-system skeleton."""
    systems = ring_systems(mol)
    sizes = sorted((len(s) for s in systems), reverse=True)

    if core_idx is not None:
        core = set(int(i) for i in core_idx)
        # no ring may mix core and non-core atoms: that is a closure onto the substituents
        for ring in mol.GetRingInfo().AtomRings():
            rs = set(ring)
            if rs & core and rs - core:
                return False, 'ring closes between core and substituents'

    if len(systems) != 3:
        return False, f'{len(systems)} ring systems (want 3; sizes {sizes})'
    if sizes != [2, 1, 1]:
        return False, f'ring-system sizes {sizes} (want [2, 1, 1])'

    # the single-ring systems must be six-membered aromatic rings (the two aryls)
    for s in systems:
        if len(s) == 1:
            r = s[0]
            if len(r) != 6:
                return False, f'aryl ring is {len(r)}-membered (want 6)'
            if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r):
                return False, 'aryl ring is not aromatic'
    # the fused system must be the quinoline: 10 atoms, one N
    fused = next(s for s in systems if len(s) == 2)
    atoms = set().union(*fused)
    if len(atoms) != 10:
        return False, f'fused system has {len(atoms)} atoms (quinoline needs 10)'
    if sum(1 for a in atoms if mol.GetAtomWithIdx(a).GetSymbol() == 'N') != 1:
        return False, 'fused system is not a quinoline (N count)'
    return True, 'ok'
