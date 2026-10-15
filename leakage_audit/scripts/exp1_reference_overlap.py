#!/usr/bin/env python
"""Experiment I: quantify benchmark-active overlap with training sources.

For every reference active (SMRT-Net + DUD-E) computes, against each source:
  exact identity match (canonical), parent-normalised match, scaffold match,
  maximum ECFP4-Tanimoto similarity and nearest-neighbour provenance.

Small sources use RDKit bulk similarity (C++); the pretraining library uses the
inverted bit index. Outputs
  tables/reference_active_overlap.csv    one row per reference active
  tables/reference_overlap_summary.csv   ligand-level and target-level summary
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import (REPO, AUDIT, CACHE, BitIndex, GEN, build_fps, bulk_stats,
                        mol_from_smiles)

RDLogger.DisableLog("rdApp.*")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SOURCES_SMALL = ["ft_train", "ft_train_protein", "ft_train_rna",
                 "ft_val", "ft_unused", "ft_dude_dropped"]
SOURCES = SOURCES_SMALL + ["pretrain"]
_G = {}


def small_lib_order():
    ft = pd.read_csv(REPO / "training_ligands_leakage_table.csv",
                     usecols=["canonical", "split", "role"])

    def uniq(lst):
        return list(dict.fromkeys(lst))
    return {
        "ft_train": uniq(ft[ft.split == "train"]["canonical"].tolist()),
        "ft_train_protein": uniq(ft[(ft.split == "train") & ft.role.str.startswith("ft_protein")]["canonical"].tolist()),
        "ft_train_rna": uniq(ft[(ft.split == "train") & (ft.role == "ft_rna")]["canonical"].tolist()),
        "ft_val": uniq(ft[ft.split == "val"]["canonical"].tolist()),
        "ft_unused": uniq(ft[ft.split == "unused"]["canonical"].tolist()),
        "ft_dude_dropped": uniq(ft[ft.split == "train_dude_dropped"]["canonical"].tolist()),
    }


def load_globals():
    order = small_lib_order()
    meta = {s: pickle.load(open(CACHE / f"src_{s}.pkl", "rb")) for s in SOURCES}
    libs = {}
    for s in SOURCES_SMALL:
        fps = build_fps(order[s])
        assert len(fps) == len(order[s]), (s, len(fps), len(order[s]))
        libs[s] = (fps, order[s])
    _G.update(meta=meta, libs=libs,
              pre_index=BitIndex.load(CACHE / "pretrain"),
              pre_smiles=[l.strip() for l in open(CACHE / "pretrain.smiles.txt")])


def worker(rows):
    out = []
    for r in rows:
        rec = dict(r)
        canon, par, sca = r["canonical"], r["parent"], r["murcko_scaffold"]
        m = mol_from_smiles(canon)
        qfp = GEN.GetFingerprint(m)
        qbits = np.fromiter(qfp.GetOnBits(), dtype=np.int64)
        for s in SOURCES:
            mm = _G["meta"][s]
            rec[f"exact_{s}"] = bool(canon in mm["canon"])
            rec[f"parent_{s}"] = bool(par and par in mm["parent"])
            rec[f"scaffold_{s}"] = bool(sca and sca in mm["scaffold"])
        for s in SOURCES_SMALL:
            fps, order = _G["libs"][s]
            st = bulk_stats(qfp, fps, THRESHOLDS)
            rec[f"maxts_{s}"] = st["max"]
            rec[f"nn_{s}"] = order[st["nn_id"]] if st["nn_id"] >= 0 else ""
            for th in THRESHOLDS:
                rec[f"n_ge_{th}_{s}"] = st[f"n_ge_{th}"]
        st = _G["pre_index"].query_stats(qbits, THRESHOLDS)
        rec["maxts_pretrain"] = st["max"]
        rec["nn_pretrain"] = _G["pre_smiles"][st["nn_id"]] if st["max"] > 0 else ""
        for th in THRESHOLDS:
            rec[f"n_ge_{th}_pretrain"] = st[f"n_ge_{th}"]
        out.append(rec)
    return out


def summarize(df):
    rows = []
    for bench, g in df.groupby("benchmark"):
        n_lig = g["canonical"].nunique()
        n_tgt = g["target_id"].nunique()
        for s in SOURCES:
            rows.append(dict(benchmark=bench, source=s, level="ligand", n=n_lig,
                             exact=int(g[f"exact_{s}"].sum()), parent=int(g[f"parent_{s}"].sum()),
                             scaffold=int(g[f"scaffold_{s}"].sum())))
            rows.append(dict(benchmark=bench, source=s, level="target", n=n_tgt,
                             exact=int(g.groupby("target_id")[f"exact_{s}"].any().sum()),
                             parent=int(g.groupby("target_id")[f"parent_{s}"].any().sum()),
                             scaffold=int(g.groupby("target_id")[f"scaffold_{s}"].any().sum())))
            col = f"maxts_{s}"
            for th in THRESHOLDS:
                nlig = int((g.groupby("canonical")[col].first() >= th).sum())
                ntgt = int(g.groupby("target_id")[col].max().ge(th).sum())
                rows.append(dict(benchmark=bench, source=s, level=f"ligand_ts>={th}",
                                 n=n_lig, exact=nlig, parent=0, scaffold=0))
                rows.append(dict(benchmark=bench, source=s, level=f"target_ts>={th}",
                                 n=n_tgt, exact=ntgt, parent=0, scaffold=0))
    return pd.DataFrame(rows)


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N actives (timing)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--report-every", type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    load_globals()
    print(f"[{time.time()-t0:6.1f}s] globals loaded", flush=True)
    ref = pd.read_csv(AUDIT / "tables" / "reference_actives.csv")
    ref = ref[ref.parse_ok].reset_index(drop=True)
    if args.limit:
        ref = ref.head(args.limit)
    print(f"[{time.time()-t0:6.1f}s] reference actives {ref.shape}", flush=True)
    records = ref.to_dict("records")
    chunks = [records[i:i + args.chunk] for i in range(0, len(records), args.chunk)]
    total = len(records)
    done = 0
    res = []
    last = time.time()
    with Pool(args.workers) as p:
        for sub in p.imap_unordered(worker, chunks, chunksize=1):
            res.append(sub)
            done += len(sub)
            if done % args.report_every < args.chunk or done >= total:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else float("inf")
                print(f"[{el:6.1f}s] {done}/{total} ({100*done/total:4.1f}%) "
                      f"{rate:6.1f} rec/s  ETA {eta/60:5.1f} min", flush=True)
    out = pd.DataFrame([r for sub in res for r in sub])
    out.to_csv(AUDIT / "tables" / "reference_active_overlap.csv", index=False)
    print(f"[{time.time()-t0:6.1f}s] wrote reference_active_overlap.csv {out.shape}", flush=True)
    summ = summarize(out)
    summ.to_csv(AUDIT / "tables" / "reference_overlap_summary.csv", index=False)
    print(summ[summ.level.isin(["ligand", "target"])].to_string(index=False))


if __name__ == "__main__":
    main()
