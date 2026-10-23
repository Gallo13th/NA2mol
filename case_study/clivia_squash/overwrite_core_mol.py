#!/usr/bin/env python3
"""Build a molecule from generated coordinates with the core's connectivity overwritten.

Order of work, per the requirement "do not let OpenBabel rebuild the core, overwrite it":

  1. assemble the molecule from the generated coordinates and element types
  2. derive bond orders for the *substituents* with RDKit's rdDetermineBonds, which is
     stricter about valence than OpenBabel's PerceiveBondOrders and therefore yields far
     fewer un-sanitisable molecules. The total charge is unknown, so a small ladder of
     charges is tried and the first consistent assignment wins.
  3. delete every core-core bond and re-add the reference bonds verbatim, so the core
     connectivity cannot be altered by perception
  4. sanitise; if that fails, fall back to OpenBabel perception for the whole graph and
     overwrite the core the same way
"""
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

CHARGE_LADDER = (0, -1, 1, -2, 2)


def _raw_mol(pos, zs):
    rw = Chem.RWMol()
    for z in zs:
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(zs))
    from rdkit.Geometry import Point3D
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    rw.AddConformer(conf, assignId=True)
    return rw


def _overwrite_core(mol, core, ref_mol):
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
    return rw.GetMol()


def _clear_noimplicit(mol):
    """rdDetermineBonds sets NoImplicit on every atom (it works with explicit H). Since the
    molecule has no H atoms, leaving the flag set yields radical-laden SMILES like [C][c];
    clearing it lets RDKit fill implicit hydrogens from the determined valences."""
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumRadicalElectrons(0)
    return mol


def _sanitise(mol):
    for flags in (Chem.SANITIZE_ALL, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE):
        try:
            Chem.SanitizeMol(mol, flags)
            return mol
        except Exception:
            continue
    return None


def build_with_overwritten_core(pos, zs, core_idx: Iterable[int],
                                ref_mol: Chem.Mol) -> Optional[Chem.Mol]:
    from rdkit.Chem import rdDetermineBonds

    pos = np.asarray(pos, dtype=float)
    zs = [int(z) for z in zs]
    core = set(int(i) for i in core_idx)
    if len(pos) == 0 or not core:
        return None

    # --- route 1: RDKit bond-order determination, charge ladder ---
    for q in CHARGE_LADDER:
        try:
            rw = _raw_mol(pos, zs)
            m = rw.GetMol()
            rdDetermineBonds.DetermineBonds(m, charge=q)
        except Exception:
            continue
        m = _clear_noimplicit(m)
        m = _clear_noimplicit(m)
        m = _overwrite_core(m, core, ref_mol)
        out = _sanitise(m)
        if out is not None:
            return out

    # --- route 2: connectivity only, then OpenBabel-style single bonds ---
    try:
        rw = _raw_mol(pos, zs)
        m = rw.GetMol()
        rdDetermineBonds.DetermineConnectivity(m)
        m = _clear_noimplicit(m)
        m = _clear_noimplicit(m)
        m = _overwrite_core(m, core, ref_mol)
        out = _sanitise(m)
        if out is not None:
            return out
    except Exception:
        pass

    # --- route 3: OpenBabel perception for the whole graph, core overwritten ---
    try:
        from util.mol_utils import reconstruct_from_generated
        m = reconstruct_from_generated(pos.tolist(), zs)
        if m is None or m.GetNumAtoms() != len(zs):
            return None
        m = _overwrite_core(m, core, ref_mol)
        return _sanitise(m)
    except Exception:
        return None
