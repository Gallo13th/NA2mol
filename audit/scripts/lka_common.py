#!/usr/bin/env python
"""Shared utilities for the RNA2smol ligand-leakage audit.

Standardisation / identity
  primary identity : canonical isomeric SMILES (RDKit MolToSmiles)
  parent identity  : largest fragment, charges neutralised (rdMolStandardize)
  scaffold         : Bemis-Murcko scaffold (no chirality)
  fingerprint      : Morgan radius 2, 2048 bit (ECFP4), no chirality
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "reproduction" / "cache" / "leakage"
AUDIT = REPO / "audit"

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

UNCHARGER = rdMolStandardize.Uncharger()
LFC = rdMolStandardize.LargestFragmentChooser()
TAUT = rdMolStandardize.TautomerEnumerator()


def mol_from_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def canon(smi):
    m = mol_from_smiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def parent_smiles(smi):
    m = mol_from_smiles(smi)
    if m is None:
        return None
    try:
        frag = rdMolStandardize.FragmentParent(m)
        frag = UNCHARGER.uncharge(frag)
        return Chem.MolToSmiles(frag)
    except Exception:
        return None


def murcko(smi):
    m = mol_from_smiles(smi)
    if m is None:
        return None
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        return s if s else None
    except Exception:
        return None


def fp_packed(smi):
    m = mol_from_smiles(smi)
    if m is None:
        return None
    try:
        fp = GEN.GetFingerprint(m)
    except Exception:
        return None
    bits = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - 48
    return np.packbits(bits)


def on_bits(smi):
    m = mol_from_smiles(smi)
    if m is None:
        return np.zeros(0, dtype=np.int64)
    fp = GEN.GetFingerprint(m)
    return np.fromiter(fp.GetOnBits(), dtype=np.int64)


def tanimoto_packed(query_packed, lib_packed, lib_pc, q_pc):
    inter = POP8[np.bitwise_and(lib_packed, query_packed)].sum(1).astype(np.int32)
    union = lib_pc + q_pc - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        t = inter / union
    return np.nan_to_num(t)


def build_fps(smiles_list):
    """Build a list of RDKit fingerprints (fast C++ bulk similarity later)."""
    fps = []
    for s in smiles_list:
        m = mol_from_smiles(s)
        if m is not None:
            try:
                fps.append(GEN.GetFingerprint(m))
            except Exception:
                pass
    return fps


def bulk_stats(qfp, fps, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Max Tanimoto + counts >= thresholds of one fingerprint vs a list (C++)."""
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(qfp, fps))
    if sims.size == 0:
        out = {"max": 0.0, "nn_id": -1}
        for th in thresholds:
            out[f"n_ge_{th}"] = 0
        return out
    j = int(sims.argmax())
    out = {"max": float(sims[j]), "nn_id": j}
    for th in thresholds:
        out[f"n_ge_{th}"] = int((sims >= th).sum())
    return out


class BitIndex:
    """Inverted-index ECFP4 nearest-neighbour search over a large library."""

    def __init__(self, prefix, packed=None, pc=None):
        z = np.load(str(prefix) + ".bitindex.npz")
        self.mol_sorted = z["mol_sorted"]
        self.ptr = z["ptr"]
        self.packed = packed
        self.pc = pc

    @classmethod
    def load(cls, prefix, with_packed=False):
        prefix = Path(prefix)
        z = np.load(str(prefix) + ".bitindex.npz")
        packed = np.load(str(prefix) + ".packed.npy") if with_packed else None
        pc = np.load(str(prefix) + ".popcnt.npy").astype(np.int32)
        self = cls.__new__(cls)
        self.mol_sorted = z["mol_sorted"]
        self.ptr = z["ptr"]
        self.packed = packed
        self.pc = pc
        return self

    @property
    def n(self):
        return len(self.pc)

    def sims(self, bits):
        """Tanimoto similarity of one fingerprint (given its on-bits) to all library rows."""
        if not hasattr(self, "_acc"):
            self._acc = np.zeros(self.n, dtype=np.int64)
        acc = self._acc
        if len(bits) == 0:
            acc.fill(0)
            return acc.astype(np.float64)
        parts = [self.mol_sorted[self.ptr[b]:self.ptr[b + 1]] for b in bits]
        idx = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
        if idx.size == 0:
            acc.fill(0)
            return acc.astype(np.float64)
        acc.fill(0)
        np.add.at(acc, idx, 1)
        qc = float(len(bits))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = acc / (self.pc + qc - acc)
        return np.nan_to_num(t)

    def query_stats(self, bits, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
        t = self.sims(bits)
        j = int(np.argmax(t))
        out = {"max": float(t[j]), "nn_id": j}
        for th in thresholds:
            out[f"n_ge_{th}"] = int((t >= th).sum())
        return out

    def query(self, smi, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
        m = mol_from_smiles(smi)
        if m is None:
            return None
        fp = GEN.GetFingerprint(m)
        bits = np.fromiter(fp.GetOnBits(), dtype=np.int64)
        st = self.query_stats(bits, thresholds)
        st["nn_smiles"] = self.nn_smiles
        return st

    def query(self, smi, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
        """Max similarity + counts >= thresholds for one SMILES."""
        m = mol_from_smiles(smi)
        if m is None:
            return None
        fp = GEN.GetFingerprint(m)
        bits = np.fromiter(fp.GetOnBits(), dtype=np.int64)
        _, t = self.sims_for_bits(bits)
        mx = float(t.max()) if t.size else 0.0
        nn = int(np.argmax(t)) if t.size else -1
        out = {"max": mx}
        for th in thresholds:
            out[f"n_ge_{th}"] = int((t >= th).sum())
        return out, mx, nn, t


def read_smiles_file(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(str(path), "rt") as f:
        for line in f:
            s = line.strip()
            if s and s not in ("smiles", "SMILES"):
                out.append(s)
    return out
