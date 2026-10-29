#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPEC 02 — Probabilistic fluorophore guidance.

``FluorGuidance`` loads the targeted fluorophore building-block library
(``fragment_library/targeted_fluorophore.pkl`` produced by SPEC 01) and
provides three capabilities used to steer post-generation post-processing:

1. ``match``       — full-library SMARTS scan + per-hit ``sumlogp`` scoring
                     driven by the BFN per-atom probability matrix ``P``.
2. ``is_complete_dpa`` — binary D/A/π completeness (molecule-level filter).
3. ``replace``     — geometric superposition: for each role, take the
                     highest-``sumlogp`` hit and overwrite the generated
                     scaffold coordinates with the (SVD-aligned) library 3D
                     conformer. ``sumlogp`` decides *which* block wins; the
                     geometry decides *how* to align.

``sumlogp`` is used for ranking/selection only; molecule-level filtering is
purely the binary D/A/π completeness (SPEC 02 §2.5, §4.4).
"""
import os
import math
import pickle

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog('rdApp.*')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIB = os.path.join(REPO_ROOT, 'fragment_library', 'targeted_fluorophore.pkl')

# Reuse the canonical 13-class (atomic_num, is_aromatic) -> index table.
from util.mol_utils import MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX as MAP_ATOM_TYPE_AROMATIC_TO_INDEX

ROLES = ('donor', 'acceptor', 'bridge')
# The pkl uses plural keys (SPEC 01 build script) while roles are singular.
_PLURAL = {'donor': 'donors', 'acceptor': 'acceptors', 'bridge': 'bridges'}


class FluorGuidance:
    """Probabilistic fluorophore matching / scoring / replacement."""

    def __init__(self, library_path=DEFAULT_LIB, epsilon=1e-6):
        self.epsilon = epsilon
        with open(library_path, 'rb') as f:
            self.library = pickle.load(f)

        # Flatten all components, keeping a stable global index, and pre-compile
        # their SMARTS patterns (main, plus optional ``smarts_alt``).
        self.components = []        # list of dicts
        self.patterns = []          # parallel list of (pat0, pat1_or_None)
        for role in ROLES:
            for comp in self.library.get(_PLURAL[role], []):
                pats = []
                smarts = comp.get('smarts')
                if smarts:
                    p = Chem.MolFromSmarts(smarts)
                    if p is not None:
                        pats.append(p)
                alt = comp.get('smarts_alt')
                if alt:
                    p = Chem.MolFromSmarts(alt)
                    if p is not None:
                        pats.append(p)
                if not pats:
                    continue
                # Cache a parsed 3D reference conformer (first valid molblock).
                lib_mol = self._first_3d_mol(comp)
                self.components.append({
                    'role': role,
                    'smiles': comp.get('smiles'),
                    'skeleton': comp.get('skeleton'),
                    'freq': comp.get('freq', 0),
                    'molblocks': comp.get('molblocks', []),
                    'lib_mol': lib_mol,
                    'n_pats': len(pats),
                })
                self.patterns.append(tuple(pats))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _first_3d_mol(comp):
        for mb in comp.get('molblocks', []):
            try:
                m = Chem.MolFromMolBlock(mb)
                if m is not None and m.GetNumConformers() > 0:
                    return m
            except Exception:
                continue
        return None

    @staticmethod
    def _type_index(atom):
        """(atomic_num, is_aromatic) -> 13-class index, or None if unknown."""
        return MAP_ATOM_TYPE_AROMATIC_TO_INDEX.get(
            (atom.GetAtomicNum(), atom.GetIsAromatic()))

    # -------------------------------------------------------------- 1) match
    def match(self, mol, P, orig_idx_map):
        """
        Full-library scan of ``mol`` with per-atom probability matrix ``P``.

        Args:
            mol: RDKit Mol (heavy atoms) reconstructed from the generation.
            P: np.ndarray [N_gen, 13] probability matrix for this molecule.
            orig_idx_map: array-like mapping mol-atom-index -> index into P
                          (length == mol.GetNumAtoms()).

        Returns:
            dict {role: [hit, ...]} where each hit is
            ``{'comp_i', 'role', 'smiles', 'smarts_idx', 'atom_indices',
               'sumlogp', 'lib_mol', 'lib_match'}``.
        """
        if mol is None or P is None or orig_idx_map is None:
            return {r: [] for r in ROLES}

        P = np.asarray(P, dtype=np.float64)
        orig_idx_map = np.asarray(orig_idx_map, dtype=np.int64)

        matches = {r: [] for r in ROLES}
        n_atoms = mol.GetNumAtoms()

        for ci, pats in enumerate(self.patterns):
            role = self.components[ci]['role']
            # Try each compiled pattern of this component.
            for pidx, pat in enumerate(pats):
                try:
                    hits = mol.GetSubstructMatches(pat)
                except Exception:
                    continue
                if not hits:
                    continue
                for atom_indices in hits:
                    sumlogp = self._sumlogp(mol, atom_indices, P, orig_idx_map)
                    if sumlogp == -math.inf:
                        continue
                    lib_mol = self.components[ci]['lib_mol']
                    lib_match = None
                    if lib_mol is not None:
                        try:
                            lm = lib_mol.GetSubstructMatches(pat)
                            if lm:
                                lib_match = lm[0]
                        except Exception:
                            lib_match = None
                    matches[role].append({
                        'comp_i': ci,
                        'role': role,
                        'smiles': self.components[ci]['smiles'],
                        'smarts_idx': pidx,
                        'atom_indices': tuple(int(a) for a in atom_indices),
                        'sumlogp': float(sumlogp),
                        'lib_mol': lib_mol,
                        'lib_match': tuple(int(a) for a in lib_match)
                                     if lib_match is not None else None,
                    })
        return matches

    def _sumlogp(self, mol, atom_indices, P, orig_idx_map):
        """Σ log(P[orig_idx[i]][type_i] + ε) over matched atoms."""
        total = 0.0
        for a in atom_indices:
            if a >= len(orig_idx_map):
                return -math.inf
            tidx = self._type_index(mol.GetAtomWithIdx(int(a)))
            if tidx is None:
                return -math.inf
            pidx = int(orig_idx_map[a])
            if pidx < 0 or pidx >= P.shape[0]:
                return -math.inf
            p = P[pidx, tidx] + self.epsilon
            total += math.log(p)
        return total

    # ------------------------------------------------------- 2) completeness
    def is_complete_dpa(self, matches):
        """True iff donor, acceptor and bridge all have >=1 hit."""
        return (len(matches.get('donor', [])) > 0 and
                len(matches.get('acceptor', [])) > 0 and
                len(matches.get('bridge', [])) > 0)

    def best_sumlogp(self, matches):
        """Sum of the per-role highest sumlogp (a single molecule-level score).

        Roles with no hit contribute 0. Useful for ranking / diagnostics.
        """
        s = 0.0
        for r in ROLES:
            hits = matches.get(r, [])
            if hits:
                s += max(h['sumlogp'] for h in hits)
        return s

    def best_hit_per_role(self, matches):
        """Highest-sumlogp hit for each role (None if role has no hit)."""
        out = {}
        for r in ROLES:
            hits = matches.get(r, [])
            out[r] = max(hits, key=lambda h: h['sumlogp']) if hits else None
        return out

    # ---------------------------------------------------------- 3) replace
    def replace(self, mol, matches, max_iters=None):
        """
        Geometric superposition replacement (SPEC 02 §4.3).

        For each role, take the highest-sumlogp hit, SVD-align its 3D library
        conformer onto the matched generated atoms, and overwrite those atoms'
        coordinates with the aligned library coordinates. Atoms already
        replaced by a higher-priority role are skipped to avoid conflicts.

        Returns a (possibly modified) copy of ``mol``. Never raises: failures
        degrade gracefully (skip that role).
        """
        if mol is None:
            return mol
        out = Chem.Mol(mol)
        conf = out.GetConformer()
        if conf is None:
            return out

        replaced = set()
        best = self.best_hit_per_role(matches)
        for role in ROLES:
            hit = best.get(role)
            if hit is None:
                continue
            lib_mol = hit.get('lib_mol')
            lib_match = hit.get('lib_match')
            atom_indices = hit.get('atom_indices')
            if (lib_mol is None or lib_match is None or
                    not atom_indices or len(lib_match) != len(atom_indices)):
                continue

            # Skip if any target atom was already replaced.
            if any(int(a) in replaced for a in atom_indices):
                continue

            try:
                R, t, rmsd = self._svd_align(out, lib_mol, atom_indices, lib_match)
            except Exception:
                continue
            if R is None:
                continue

            lib_conf = lib_mol.GetConformer()
            for li, gi in zip(lib_match, atom_indices):
                lp = lib_conf.GetAtomPosition(int(li))
                arr = np.array([lp.x, lp.y, lp.z]) @ R.T + t
                conf.SetAtomPosition(int(gi), Point3D(float(arr[0]),
                                                      float(arr[1]),
                                                      float(arr[2])))
                replaced.add(int(gi))
        return out

    @staticmethod
    def _svd_align(target_mol, lib_mol, target_atom_indices, lib_atom_indices):
        """Rigid Kabsch align of lib points onto target points.

        Returns (R, t, rmsd) such that ``lib_pos @ R.T + t ~= target_pos``.
        """
        tconf = target_mol.GetConformer()
        lconf = lib_mol.GetConformer()
        pts_t, pts_l = [], []
        for gi, li in zip(target_atom_indices, lib_atom_indices):
            tp = tconf.GetAtomPosition(int(gi))
            lp = lconf.GetAtomPosition(int(li))
            pts_t.append([tp.x, tp.y, tp.z])
            pts_l.append([lp.x, lp.y, lp.z])
        pts_t = np.asarray(pts_t, dtype=np.float64)
        pts_l = np.asarray(pts_l, dtype=np.float64)
        if len(pts_t) < 1:
            return None, None, float('inf')

        ct = pts_t.mean(axis=0)
        cl = pts_l.mean(axis=0)
        Pm = pts_l - cl
        Qm = pts_t - ct
        H = Pm.T @ Qm
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = ct - cl @ R.T
        transformed = pts_l @ R.T + t
        rmsd = float(np.sqrt(np.mean(np.sum((transformed - pts_t) ** 2, axis=1))))
        return R, t, rmsd

    # ----------------------------------------------- fragment-level sumlogp
    def fragment_sumlogp(self, frag_mol, frag_pidx, P):
        """
        Sumlogp of a BRICS fragment against the fluorophore library.

        Args:
            frag_mol: RDKit fragment (heavy atoms).
            frag_pidx: array-like of P-indices parallel to frag_mol atoms
                       (entry may be -1 if a fragment atom has no P mapping).
            P: parent molecule probability matrix [N_gen, 13].

        Returns the best (max) sumlogp over all library hits on the fragment,
        or -inf if no pharmacophore SMARTS matches. Per SPEC 02 §4.5 this is
        the fragment ranking signal (replacing the legacy ``_SeedScore``).
        """
        if frag_mol is None or P is None or frag_pidx is None:
            return -math.inf
        P = np.asarray(P, dtype=np.float64)
        frag_pidx = np.asarray(frag_pidx, dtype=np.int64)
        best = -math.inf
        for ci, pats in enumerate(self.patterns):
            for pat in pats:
                try:
                    hits = frag_mol.GetSubstructMatches(pat)
                except Exception:
                    continue
                if not hits:
                    continue
                for atom_indices in hits:
                    total = 0.0
                    ok = True
                    for a in atom_indices:
                        if a >= len(frag_pidx):
                            ok = False
                            break
                        pidx = int(frag_pidx[a])
                        if pidx < 0:
                            ok = False
                            break
                        tidx = self._type_index(frag_mol.GetAtomWithIdx(int(a)))
                        if tidx is None:
                            ok = False
                            break
                        total += math.log(P[pidx, tidx] + self.epsilon)
                    if ok and total > best:
                        best = total
        return best
