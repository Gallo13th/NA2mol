#!/usr/bin/env python
"""Combine Experiment IIa (full) and IIb (stratified pretrain sample) into
tables/generated_overlap_summary.csv.

Fine-tuning overlap is exact for all unique generated molecules. Pretraining
similarity fractions are estimates from the stratified sample (with Wilson
95% intervals) because the full 613k x 2.29M sweep exceeds the available
compute; exact-identity pretraining overlap is still computed for all molecules.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import AUDIT, CACHE

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    gen = pd.read_csv(AUDIT / "tables" / "generated_long.csv")
    mol = pd.read_csv(AUDIT / "tables" / "generated_overlap_molecules.csv")
    sample = pd.read_csv(AUDIT / "tables" / "generated_overlap_pretrain_sample.csv")
    g = gen.merge(mol, on="canonical", how="left")
    sample_smi = set(sample["canonical"])
    sm = sample.set_index("canonical")

    rows = []
    for (bench, method), sub in g.groupby(["benchmark", "method"]):
        u = sub.drop_duplicates("canonical")
        rec = dict(benchmark=bench, method=method,
                   n_unique=len(u), n_records=len(sub))
        # fine-tuning: full
        for src in ["ft_train"]:
            rec[f"exact_{src}_uniq"] = int(u[f"exact_{src}"].sum())
            rec[f"exact_{src}_frac"] = round(u[f"exact_{src}"].mean(), 4)
            rec[f"parent_{src}_uniq"] = int(u[f"parent_{src}"].sum())
            rec[f"scaffold_{src}_uniq"] = int(u[f"scaffold_{src}"].sum())
            for th in THRESHOLDS:
                rec[f"ts{th}_{src}_frac"] = round(float((u[f"maxts_{src}"] >= th).mean()), 4)
        # pretraining: exact identity full, similarity from sample
        rec["exact_pretrain_uniq"] = int(u["exact_pretrain"].sum())
        rec["exact_pretrain_frac"] = round(u["exact_pretrain"].mean(), 4)
        rec["parent_pretrain_uniq"] = int(u["parent_pretrain"].sum())
        rec["scaffold_pretrain_uniq"] = int(u["scaffold_pretrain"].sum())
        n_s = int(u["canonical"].isin(sample_smi).sum())
        rec["pretrain_sample_n"] = n_s
        vals = sm.reindex(u.loc[u["canonical"].isin(sample_smi), "canonical"])["maxts_pretrain"].dropna()
        for th in THRESHOLDS:
            if n_s:
                k = int((vals >= th).sum())
                lo, hi = wilson(k, n_s)
                rec[f"ts{th}_pretrain_frac"] = round(k / n_s, 4)
                rec[f"ts{th}_pretrain_lo"] = round(lo, 4)
                rec[f"ts{th}_pretrain_hi"] = round(hi, 4)
            else:
                rec[f"ts{th}_pretrain_frac"] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / "tables" / "generated_overlap_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
