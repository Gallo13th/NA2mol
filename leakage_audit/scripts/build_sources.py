#!/usr/bin/env python
"""Build per-source metadata for the leakage audit.

Sources
  ft_train            fine-tuning ligands, split == train
  ft_train_protein    ft_train, role ft_protein(*)
  ft_train_rna        ft_train, role ft_rna
  ft_val              fine-tuning ligands, split == val
  ft_unused           fine-tuning ligands, split == unused
  ft_dude_dropped     fine-tuning ligands, split == train_dude_dropped
  pretrain            ligand-only pretraining library (canonical unique)

For each source writes a pickle with canonical/parent/scaffold sets and an
npy pair with packed ECFP4 fingerprints + popcounts (small sources only;
pretrain fingerprints are already cached by build_lib.py).
"""
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import CACHE, REPO, canon, parent_smiles, murcko

FT_TABLE = REPO / "training_ligands_leakage_table.csv"


def meta_worker(smi):
    return parent_smiles(smi), murcko(smi)


def build_meta(name, smiles_list, out_dir, pool):
    smiles_list = list(dict.fromkeys(smiles_list))
    print(f"[{name}] {len(smiles_list)} unique smiles", flush=True)
    t = time.time()
    res = pool.map(meta_worker, smiles_list, chunksize=2000)
    parents = {p for p, s in res if p}
    scaffolds = {s for p, s in res if s}
    obj = dict(name=name, n=len(smiles_list), canon=set(smiles_list),
               parent=parents, scaffold=scaffolds)
    with open(out_dir / f"src_{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    print(f"[{name}] canonical={len(obj['canon'])} parent={len(parents)} "
          f"scaffold={len(scaffolds)} ({time.time()-t:.0f}s)", flush=True)


def build_packed(name, smiles_list, out_dir):
    from lka_common import fp_packed
    arr = [fp_packed(s) for s in smiles_list]
    arr = [a for a in arr if a is not None]
    packed = np.stack(arr)
    pc = np.unpackbits(packed, axis=1).sum(1).astype(np.int16)
    np.save(out_dir / f"src_{name}.packed.npy", packed)
    np.save(out_dir / f"src_{name}.popcnt.npy", pc)
    print(f"[{name}] packed {packed.shape}", flush=True)


def main():
    out_dir = CACHE
    ft = pd.read_csv(FT_TABLE, usecols=["canonical", "split", "role"])
    sources = {}
    sources["ft_train"] = ft[ft.split == "train"]["canonical"].tolist()
    sources["ft_train_protein"] = ft[(ft.split == "train") &
                                     ft.role.str.startswith("ft_protein")]["canonical"].tolist()
    sources["ft_train_rna"] = ft[(ft.split == "train") &
                                 (ft.role == "ft_rna")]["canonical"].tolist()
    sources["ft_val"] = ft[ft.split == "val"]["canonical"].tolist()
    sources["ft_unused"] = ft[ft.split == "unused"]["canonical"].tolist()
    sources["ft_dude_dropped"] = ft[ft.split == "train_dude_dropped"]["canonical"].tolist()
    pre = [l.strip() for l in open(CACHE / "pretrain.smiles.txt") if l.strip()]
    sources["pretrain"] = pre

    with Pool(12) as pool:
        for name, smis in sources.items():
            build_meta(name, smis, out_dir, pool)

    for name, smis in sources.items():
        if name == "pretrain":
            continue  # already packed by build_lib.py
        build_packed(name, list(dict.fromkeys(smis)), out_dir)


if __name__ == "__main__":
    main()
