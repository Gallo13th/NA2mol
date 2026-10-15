#!/usr/bin/env python
"""Build the reference-active table for SMRT-Net and DUD-E.

SMRT-Net actives: notebooks/RNALigBench/activesmi/{pdb}_pos.smi (fallback {pdb}.smi)
DUD-E actives    : notebooks/DUDE_active_compounds.csv, label == 'active'
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_ids import structure_id
from lka_common import AUDIT, REPO, canon, parent_smiles, murcko

NOTE = REPO / "notebooks"
SMRT_ACT_DIR = NOTE / "RNALigBench" / "activesmi"
DUDE_CSV = REPO / "release_package_v1" / "notebooks" / "DUDE_active_compounds.csv"
SMRT_EVAL = NOTE / "output" / "smrtbench_combined_results.csv"
DUDE_EVAL = NOTE / "dude_stat_all.csv"


def smrt_targets():
    df = pd.read_csv(SMRT_EVAL, usecols=["pdb_id", "pocket"])
    m = df.drop_duplicates("pdb_id").set_index("pdb_id")["pocket"].to_dict()
    return m


def build_smrt():
    rows = []
    for pdb, pocket in smrt_targets().items():
        path = None
        for pat in (f"{pdb}_pos.smi", f"{pdb}.smi"):
            p = SMRT_ACT_DIR / pat
            if p.exists():
                path = p
                break
        if path is None:
            continue
        for smi in path.read_text().splitlines():
            smi = smi.strip()
            if not smi:
                continue
            rows.append(dict(benchmark="SMRT-Net", target_id=str(pdb).lower(),
                             pocket=pocket, original_smiles=smi,
                             source=str(path.relative_to(REPO))))
    return pd.DataFrame(rows)


def build_dude():
    df = pd.read_csv(DUDE_CSV, usecols=["pdb_id", "canonical_smiles", "label"], dtype={"pdb_id": str})
    df["pdb_id"] = df["pdb_id"].map(lambda x: structure_id(x, repair_known=True))
    df = df[df["label"] == "active"]
    df = df.drop_duplicates(["pdb_id", "canonical_smiles"])
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(benchmark="DUD-E", target_id=r["pdb_id"], pocket=None,
                         original_smiles=r["canonical_smiles"], source="DUDE_active_compounds.csv"))
    return pd.DataFrame(rows)


def main():
    smrt = build_smrt()
    dude = build_dude()
    df = pd.concat([smrt, dude], ignore_index=True)
    df["canonical"] = df["original_smiles"].map(canon)
    df["parent"] = df["original_smiles"].map(parent_smiles)
    df["murcko_scaffold"] = df["original_smiles"].map(murcko)
    df["parse_ok"] = df["canonical"].notna()
    out = AUDIT / "tables" / "reference_actives.csv"
    df.to_csv(out, index=False)
    print("wrote", out, df.shape)
    print(df.groupby("benchmark").agg(targets=("target_id", "nunique"),
                                      actives=("canonical", "nunique"),
                                      records=("canonical", "size")))
    print("parse failures:", int((~df.parse_ok).sum()))


if __name__ == "__main__":
    main()
