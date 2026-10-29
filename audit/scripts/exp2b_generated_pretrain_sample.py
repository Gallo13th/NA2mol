#!/usr/bin/env python
"""Experiment IIb: maximum ECFP4-Tanimoto to the pretraining library for a
stratified sample of generated molecules (6 physical cores make the full
613k-query sweep impractical; the sample is used to estimate fractions with
bootstrap confidence intervals).

Outputs
  cache/gen_sample.smiles.txt
  tables/generated_overlap_pretrain_sample.csv
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import AUDIT, CACHE, BitIndex, on_bits

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
PER_GROUP = 4000
SEED = 0
_G = {}


def load_globals():
    _G["pre_index"] = BitIndex.load(CACHE / "pretrain")
    _G["pre_smiles"] = [l.strip() for l in open(CACHE / "pretrain.smiles.txt")]


def worker(smis):
    out = []
    for smi in smis:
        st = _G["pre_index"].query_stats(on_bits(smi), THRESHOLDS)
        rec = {"canonical": smi, "maxts_pretrain": st["max"],
               "nn_pretrain": _G["pre_smiles"][st["nn_id"]] if st["max"] > 0 else ""}
        for th in THRESHOLDS:
            rec[f"n_ge_{th}_pretrain"] = st[f"n_ge_{th}"]
        out.append(rec)
    return out


def build_sample():
    gen = pd.read_csv(AUDIT / "tables" / "generated_long.csv")
    rng = np.random.default_rng(SEED)
    sample = set()
    for (b, m), sub in gen.groupby(["benchmark", "method"]):
        u = np.array(sorted(sub["canonical"].unique()), object)
        take = min(PER_GROUP, len(u))
        sample.update(u[rng.choice(len(u), take, replace=False)].tolist())
    sample = sorted(sample)
    with open(CACHE / "gen_sample.smiles.txt", "w") as f:
        f.write("\n".join(sample) + "\n")
    return sample


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    load_globals()
    print(f"[{time.time()-t0:6.1f}s] globals loaded", flush=True)
    smis = build_sample()
    print(f"[{time.time()-t0:6.1f}s] stratified sample {len(smis)}", flush=True)
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
    out = pd.DataFrame([r for sub in res for r in sub])
    out.to_csv(AUDIT / "tables" / "generated_overlap_pretrain_sample.csv", index=False)
    print(f"[{time.time()-t0:6.1f}s] wrote generated_overlap_pretrain_sample.csv {out.shape}", flush=True)


if __name__ == "__main__":
    main()
