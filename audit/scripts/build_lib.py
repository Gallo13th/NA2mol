#!/usr/bin/env python
"""Build a packed ECFP4 fingerprint library from a SMILES file.

Usage: build_lib.py <smiles_file[.gz]> <out_prefix> [--unique]

Writes:
  <out_prefix>.smiles.txt   one canonical isomeric SMILES per row (order preserved)
  <out_prefix>.packed.npy   uint8 [N,256] packed Morgan r2 2048-bit fingerprints
  <out_prefix>.popcnt.npy   int16 [N] popcounts
  <out_prefix>.fail.txt     rows that failed parse
"""
import sys
import gzip
import time
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canon_fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        c = Chem.MolToSmiles(m)
        fp = GEN.GetFingerprint(m)
    except Exception:
        return None
    bits = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - 48
    packed = np.packbits(bits)  # 256 bytes
    return c, packed


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    unique = "--unique" in sys.argv
    t0 = time.time()
    opener = gzip.open if src.endswith(".gz") else open
    smis = []
    with opener(src, "rt") as f:
        for line in f:
            s = line.strip()
            if s and s not in ("smiles", "SMILES"):
                smis.append(s)
    print(f"[{time.time()-t0:.0f}s] read {len(smis)} raw smiles", flush=True)
    if unique:
        smis = list(dict.fromkeys(smis))
        print(f"  unique -> {len(smis)}", flush=True)

    with Pool(12) as p:
        res = p.map(canon_fp, smis, chunksize=2000)
    print(f"[{time.time()-t0:.0f}s] parsed", flush=True)

    cans, packs, fails = [], [], []
    seen = set()
    for s, r in zip(smis, res):
        if r is None:
            fails.append(s)
        else:
            if unique and r[0] in seen:
                continue
            seen.add(r[0])
            cans.append(r[0])
            packs.append(r[1])
    print(f"  canonical unique -> {len(cans)}", flush=True)
    packed = np.stack(packs)
    popcnt = np.unpackbits(packed, axis=1).sum(axis=1).astype(np.int16)
    np.save(out + ".packed.npy", packed)
    np.save(out + ".popcnt.npy", popcnt)
    with open(out + ".smiles.txt", "w") as f:
        f.write("\n".join(cans) + "\n")
    with open(out + ".fail.txt", "w") as f:
        f.write("\n".join(fails) + ("\n" if fails else ""))
    print(f"[{time.time()-t0:.0f}s] done N={len(cans)} fails={len(fails)} "
          f"packed={packed.nbytes/1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
