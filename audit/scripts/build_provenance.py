#!/usr/bin/env python
"""Deliverable 4: link recovery-contributing generated molecules to their closest
target active and their closest fine-tuning / pretraining neighbours.

Recovery-contributing = molecules whose ECFP4-Tanimoto to a target active is
>= 0.25 (the lower end of the manuscript's recovery window). Output:
  tables/recovery_molecule_provenance.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lka_common import AUDIT, REPO, canon

NOTE = REPO / "notebooks"
SMRT_CSV = NOTE / "output" / "smrtbench_combined_results.csv"
DUDE_CSV = NOTE / "dude_stat_all.csv"
MAIN = {"SMRT-Net": {"RNA2smol (Only General Feat)": "RNA2smol", "MolCRAFT": "MolCRAFT",
                     "Pocket2Mol": "Pocket2Mol", "ED2Mol": "ED2Mol"},
        "DUD-E": {"RNA2smol": "RNA2smol", "MolCRAFT": "MolCRAFT",
                  "Pocket2Mol": "Pocket2Mol", "ed2mol": "ED2Mol"}}
LO = 0.25


def load_eval(bench, path):
    df = pd.read_csv(path, usecols=["mode", "pdb_id", "SMILES", "max_ecfp4_tanimoto", "active_smi"])
    df = df.dropna(subset=["SMILES", "max_ecfp4_tanimoto"])
    df = df[df["max_ecfp4_tanimoto"] >= LO]
    df["benchmark"] = bench
    df["method"] = df["mode"].str.strip().map(MAIN[bench])
    df = df.dropna(subset=["method"])
    df["target_id"] = df["pdb_id"].astype(str).str.lower()
    return df


def main():
    ev = pd.concat([load_eval("SMRT-Net", SMRT_CSV), load_eval("DUD-E", DUDE_CSV)], ignore_index=True)
    print("recovery-contributing records", len(ev))
    ev["canonical"] = ev["SMILES"].map(canon)
    ev = ev.dropna(subset=["canonical"])
    mol = pd.read_csv(AUDIT / "tables" / "generated_overlap_molecules.csv")
    keep_mol = [c for c in ["canonical", "exact_ft_train", "parent_ft_train", "scaffold_ft_train",
                            "exact_pretrain", "maxts_ft_train"] if c in mol.columns]
    ev = ev.merge(mol[keep_mol], on="canonical", how="left")
    samp = AUDIT / "tables" / "generated_overlap_pretrain_sample.csv"
    if samp.exists():
        s = pd.read_csv(samp)[["canonical", "maxts_pretrain"]]
        ev = ev.merge(s, on="canonical", how="left")
    out = ev.rename(columns={"max_ecfp4_tanimoto": "sim_to_target_active",
                             "active_smi": "closest_target_active"})[
        ["benchmark", "method", "target_id", "canonical", "sim_to_target_active",
         "closest_target_active", "maxts_ft_train", "exact_ft_train", "maxts_pretrain",
         "exact_pretrain"]].drop_duplicates(["benchmark", "method", "target_id", "canonical"])
    out.to_csv(AUDIT / "tables" / "recovery_molecule_provenance.csv", index=False)
    print("wrote recovery_molecule_provenance.csv", out.shape)
    print(out.groupby(["benchmark", "method"]).size())


if __name__ == "__main__":
    main()
