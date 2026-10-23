#!/usr/bin/env python3
"""Repair bond orders that rdDetermineBonds gets wrong on ambiguous centres.

Most common: a carboxylic acid is written as a gem-diol because the C-OH and C=O oxygens
are geometrically indistinguishable. That inflates the apparent difference to the target
series, which carries methoxy/hydroxyl rather than acid groups.

  C(-OH)(-OH)  ->  C(=O)(-OH)      gem-diol -> carboxylic acid

Also repairs the protonated carbonyl `[O+]=C-OH` back to `C(=O)OH`, and the `[O+]=C` that
appears on quinoline rings.
"""
from typing import Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

GEM_DIOL = Chem.MolFromSmarts('[CX4;!$(C=*);!$(C#*)](-[OH1])(-[OH1])')
HYDROXY_CARBONYL = Chem.MolFromSmarts('[CX3](=[O+])(-[OH1])')
HYDROXY_IMINIUM = Chem.MolFromSmarts('[CX3](=[O+])')


def repair_bond_orders(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], int]:
    """Return (repaired mol, number of repairs)."""
    if mol is None:
        return None, 0
    rw = Chem.RWMol(mol)
    fixed = 0

    # 1. gem-diol -> carboxylic acid
    for match in list(rw.GetSubstructMatches(GEM_DIOL)):
        c, o1, o2 = match[0], match[1], match[2]
        b = rw.GetBondBetweenAtoms(c, o1)
        if b is None:
            continue
        oa = rw.GetAtomWithIdx(o1)
        ca = rw.GetAtomWithIdx(c)
        if oa.GetTotalNumHs() == 0:
            continue
        oa.SetNoImplicit(True)
        oa.SetNumExplicitHs(0)
        ca.SetNoImplicit(False)
        b.SetBondType(Chem.BondType.DOUBLE)
        fixed += 1
        break                      # re-sanitise before the next one

    # 2. protonated carbonyl -> neutral carbonyl
    for match in list(rw.GetSubstructMatches(HYDROXY_CARBONYL)):
        c, o = match[0], match[1]
        a = rw.GetAtomWithIdx(o)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        fixed += 1
        break

    if fixed == 0:
        return mol, 0

    out = rw.GetMol()
    for flags in (Chem.SANITIZE_ALL, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE):
        try:
            Chem.SanitizeMol(out, flags)
            return out, fixed
        except Exception:
            continue
    return None, 0


def count_gem_diols(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(GEM_DIOL))
