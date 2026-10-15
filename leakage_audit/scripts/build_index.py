#!/usr/bin/env python
"""Build an inverted bit index over ECFP4 fingerprints for fast Tanimoto.

Input : <prefix>.smiles.txt (canonical SMILES, one per row)
Output: <prefix>.bitindex.npz with
          mol_sorted : int32, molecule ids grouped by set bit
          ptr        : int64 [2049], posting-list boundaries per bit
          pc         : int16 [N], popcounts (already available separately)
        <prefix>.bitset.npy  uint8 [N,256] packed bits (already built)
"""
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from multiprocessing import Pool

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def onbits(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(0, dtype=np.int16)
    fp = GEN.GetFingerprint(m)
    return np.fromiter(fp.GetOnBits(), dtype=np.int16)


def main():
    smiles_file = sys.argv[1]
    out = sys.argv[2]
    with open(smiles_file) as f:
        smis = [l.strip() for l in f if l.strip()]
    n = len(smis)
    print("building on-bits for", n, flush=True)
    with Pool(12) as p:
        res = p.map(onbits, smis, chunksize=2000)
    lengths = np.fromiter((len(r) for r in res), dtype=np.int64, count=n)
    flat_bit = np.concatenate(res).astype(np.int32)
    flat_mol = np.repeat(np.arange(n, dtype=np.int32), lengths)
    print("nnz", flat_bit.shape[0], flush=True)
    order = np.argsort(flat_bit, kind="stable")
    bit_sorted = flat_bit[order]
    mol_sorted = flat_mol[order]
    del order, flat_bit, flat_mol
    counts = np.bincount(bit_sorted, minlength=2048)
    ptr = np.zeros(2049, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    np.savez(out + ".bitindex.npz", mol_sorted=mol_sorted, ptr=ptr)
    print("saved", out + ".bitindex.npz", flush=True)


if __name__ == "__main__":
    main()
