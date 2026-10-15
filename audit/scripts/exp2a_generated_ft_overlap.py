#!/usr/bin/env python
"""Experiment IIa (all generated molecules, cheap): exact / parent / scaffold
matches to the fine-tuning training set and the pretraining library, plus
maximum ECFP4-Tanimoto to the fine-tuning training set (RDKit bulk).

Output: tables/generated_overlap_molecules.csv (canonical + ft-train similarity
        + exact/parent/scaffold flags; pretrain similarity added by IIb).
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import (REPO, AUDIT, CACHE, GEN, build_fps, bulk_stats, mol_from_smiles,
                        parent_smiles, murcko)

RDLogger.DisableLog("rdApp.*")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
_G = {}


def ft_order():
    ft = pd.read_csv(REPO / "training_ligands_leakage_table.csv",
                     usecols=["canonical", "split"])
    return list(dict.fromkeys(ft[ft.split == "train"]["canonical"].tolist()))


def load_globals():
    _G["ft_meta"] = pickle.load(open(CACHE / "src_ft_train.pkl", "rb"))
    _G["pre_meta"] = pickle.load(open(CACHE / "src_pretrain.pkl", "rb"))
    _G["ft_fps"] = build_fps(ft_order())


def worker(smis):
    out = []
    fm, pm = _G["ft_meta"], _G["pre_meta"]
    for smi in smis:
        m = mol_from_smiles(smi)
        if m is None:
            continue
        qfp = GEN.GetFingerprint(m)
        par = parent_smiles(smi)
        sca = murcko(smi)
        rec = {"canonical": smi, "parent": par,
               "exact_ft_train": bool(smi in fm["canon"]),
               "parent_ft_train": bool(par and par in fm["parent"]),
               "scaffold_ft_train": bool(sca and sca in fm["scaffold"]),
               "exact_pretrain": bool(smi in pm["canon"]),
               "parent_pretrain": bool(par and par in pm["parent"]),
               "scaffold_pretrain": bool(sca and sca in pm["scaffold"])}
        st = bulk_stats(qfp, _G["ft_fps"], THRESHOLDS)
        rec["maxts_ft_train"] = st["max"]
        for th in THRESHOLDS:
            rec[f"n_ge_{th}_ft_train"] = st[f"n_ge_{th}"]
        out.append(rec)
    return out


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--report-every", type=int, default=5000)
    args = ap.parse_args()
    t0 = time.time()
    load_globals()
    print(f"[{time.time()-t0:6.1f}s] globals loaded", flush=True)
    smis = [l.strip() for l in open(CACHE / "gen_unique.smiles.txt") if l.strip()]
    if args.limit:
        smis = smis[:args.limit]
    print(f"[{time.time()-t0:6.1f}s] unique generated {len(smis)}", flush=True)
    chunks = [smis[i:i + args.chunk] for i in range(0, len(smis), args.chunk)]
    total, done, res = len(smis), 0, []
    with Pool(args.workers) as p:
        for sub in p.imap_unordered(worker, chunks, chunksize=1):
            res.append(sub)
            done += len(sub)
            if done % args.report_every < args.chunk or done >= total:
                el = time.time() - t0
                rate = done / el
                print(f"[{el:6.1f}s] {done}/{total} ({100*done/total:4.1f}%) "
                      f"{rate:6.1f} rec/s  ETA {(total-done)/rate/60:5.1f} min", flush=True)
    mol = pd.DataFrame([r for sub in res for r in sub])
    mol.to_csv(AUDIT / "tables" / "generated_overlap_molecules.csv", index=False)
    print(f"[{time.time()-t0:6.1f}s] wrote generated_overlap_molecules.csv {mol.shape}", flush=True)


if __name__ == "__main__":
    main()
