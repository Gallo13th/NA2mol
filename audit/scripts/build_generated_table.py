#!/usr/bin/env python
"""Build the generated-molecule table from the evaluation CSVs.

Outputs
  tables/generated_long.csv   unique (benchmark, method, target, canonical) rows
  cache/gen_unique.smiles.txt global unique canonical SMILES (query list)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import AUDIT, CACHE, REPO, canon

NOTE = REPO / "notebooks"
SMRT_CSV = NOTE / "output" / "smrtbench_combined_results.csv"
DUDE_CSV = NOTE / "dude_stat_all.csv"

SMRT_MAIN = {"RNA2smol (Only General Feat)", "MolCRAFT", "Pocket2Mol", "ED2Mol"}
DUDE_MAIN = {"RNA2smol", "MolCRAFT", "Pocket2Mol", "ed2mol"}


def tag(benchmark, method):
    main = SMRT_MAIN if benchmark == "SMRT-Net" else DUDE_MAIN
    if method == "RNA2smol (Only General Feat)" and benchmark == "SMRT-Net":
        return "RNA2smol"
    if method == "ed2mol":
        return "ED2Mol"
    if method == "MolCRAFT (Original)":
        return "MolCRAFT-Orig"
    if method in main:
        return method
    return "ablation:" + method


def main():
    parts = []
    for benchmark, path, mcol, tcol in [
        ("SMRT-Net", SMRT_CSV, "mode", "pdb_id"),
        ("DUD-E", DUDE_CSV, "mode", "pdb_id"),
    ]:
        df = pd.read_csv(path, usecols=[mcol, tcol, "SMILES"])
        df = df.dropna(subset=["SMILES"])
        df["benchmark"] = benchmark
        df["target_id"] = df[tcol].astype(str).str.lower()
        df["raw_method"] = df[mcol].str.strip()
        df = df.drop(columns=[mcol, tcol])
        parts.append(df)
    g = pd.concat(parts, ignore_index=True)
    print("gen records", g.shape)
    print("canonicalising...", flush=True)
    g["canonical"] = g["SMILES"].map(canon)
    print("fail", int(g["canonical"].isna().sum()))
    g = g.dropna(subset=["canonical"])
    g["method"] = [tag(b, m) for b, m in zip(g["benchmark"], g["raw_method"])]
    g = g.drop(columns=["SMILES", "raw_method"]).drop_duplicates(
        ["benchmark", "method", "target_id", "canonical"])
    g.to_csv(AUDIT / "tables" / "generated_long.csv", index=False)
    uniq = sorted(g["canonical"].unique())
    with open(CACHE / "gen_unique.smiles.txt", "w") as f:
        f.write("\n".join(uniq) + "\n")
    print("generated_long", g.shape, "unique canonical", len(uniq))
    print(g.groupby(["benchmark", "method"]).canonical.nunique())


if __name__ == "__main__":
    main()
