#!/usr/bin/env python
"""Batch RNAsmol GPU scoring for 1FMN case study.

Scores every ligand in a multi-record SDF on GPU, writes a TSV with
P(binder)/logit/SMILES/atom_count, also scores the FMN reference, and
prints summary statistics.

Reuses the verbatim graph builder + model loader from rnasmol_score.py
(no preprocessing changes -> matches released weights).

Usage
-----
    conda activate rna2smol
    cd casestudy/fmn
    python batch_score_rnasmol.py \\
        --ligand ../../output/fmn_pocket/dummy_ligand/dummy_ligand.sdf \\
        --fmn    FMN_ligand.sdf \\
        --out    rnasmol_scores.tsv
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
RNASMOL_ROOT = os.path.join(HERE, "RNAsmol")
RNASMOL_PKG = os.path.join(RNASMOL_ROOT, "rnasmol")
for _p in (RNASMOL_PKG, RNASMOL_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from rdkit import Chem

from rnasmol_score import (
    build_data_from_mol, load_model, ligands_from_file,
    FMN_APTAMER_SEQ, DEFAULT_MODEL,
)
from rnasmol_score_gpu import score_data


def score_supplier(model, suppl, rna_seq, device, label=""):
    """Iterate an SDF supplier and score each mol on GPU.
    Returns list of dicts."""
    rows = []
    n_tot = 0
    n_bad = 0
    t0 = time.perf_counter()
    for i, m in enumerate(suppl):
        if m is None:
            n_bad += 1
            continue
        n_tot += 1
        try:
            data = build_data_from_mol(m, rna_seq)
            p_binder, pred, logits = score_data(model, data, device)
        except Exception as e:
            n_bad += 1
            print(f"[batch]{label} rec {i} failed: {e}", file=sys.stderr)
            continue
        smi = Chem.MolToSmiles(m)
        natoms = m.GetNumAtoms()
        props = m.GetPropsAsDict() if m.HasProp("_Name") else {}
        aff = props.get("minimizedAffinity", "")
        it = props.get("Iteration", "")
        rows.append({
            "idx": i,
            "name": m.GetProp("_Name") if m.HasProp("_Name") else f"mol{i}",
            "smiles": smi,
            "natoms": natoms,
            "aff": aff,
            "iter": it,
            "p_binder": p_binder,
            "logit_neg": logits[1] - logits[0],
            "pred": pred,
        })
        if (n_tot % 500) == 0:
            dt = time.perf_counter() - t0
            print(f"[batch]{label} {n_tot} mols in {dt:.1f}s "
                  f"({dt/n_tot*1000:.2f} ms/mol)", flush=True)
    dt = time.perf_counter() - t0
    print(f"[batch]{label} DONE {n_tot} mols, {n_bad} bad, "
          f"{dt:.1f}s ({dt/max(n_tot,1)*1000:.2f} ms/mol)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand", required=True,
                    help="multi-record SDF of generated ligands")
    ap.add_argument("--fmn", default=os.path.join(HERE, "FMN_ligand.sdf"))
    ap.add_argument("--out", required=True, help="output TSV")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rna-seq", default=FMN_APTAMER_SEQ)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available(), \
        "GPU required for this script"
    print(f"[batch] torch={torch.__version__} dev={device} "
          f"({torch.cuda.get_device_name(0)})")
    print(f"[batch] model={os.path.relpath(args.model, HERE)}")
    print(f"[batch] RNA ({len(args.rna_seq)} nt)={args.rna_seq}")

    model = load_model(args.model, device)
    print(f"[batch] model loaded; param dev="
          f"{next(model.parameters()).device}")

    # 1) FMN reference
    print("\n[batch] === scoring FMN reference ===")
    fmn_suppl = Chem.SDMolSupplier(args.fmn, removeHs=True, sanitize=True)
    fmn_rows = score_supplier(model, fmn_suppl, args.rna_seq, device,
                              label="FMN")
    for r in fmn_rows:
        print(f"[batch] FMN  P(binder)={r['p_binder']:.4f} "
              f"logit={r['logit_neg']:+.3f} pred={r['pred']}")

    # 2) Generated ligands
    print("\n[batch] === scoring generated ligands ===")
    gen_suppl = Chem.SDMolSupplier(args.ligand, removeHs=True, sanitize=True)
    gen_rows = score_supplier(model, gen_suppl, args.rna_seq, device,
                              label="GEN")

    # 3) Write TSV (FMN first, then generated sorted by p_binder desc)
    header = ["source", "rank", "idx", "name", "smiles", "natoms",
              "p_binder", "logit_neg", "pred", "aff", "iter"]
    fmn_sorted = sorted(fmn_rows, key=lambda r: -r["p_binder"])
    gen_sorted = sorted(gen_rows, key=lambda r: -r["p_binder"])
    with open(args.out, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for rank, r in enumerate(fmn_sorted, 1):
            fh.write("\t".join([
                "FMN", str(rank), str(r["idx"]), r["name"], r["smiles"],
                str(r["natoms"]), f"{r['p_binder']:.6f}",
                f"{r['logit_neg']:+.6f}", str(r["pred"]),
                str(r["aff"]), str(r["iter"]),
            ]) + "\n")
        for rank, r in enumerate(gen_sorted, 1):
            fh.write("\t".join([
                "GEN", str(rank), str(r["idx"]), r["name"], r["smiles"],
                str(r["natoms"]), f"{r['p_binder']:.6f}",
                f"{r['logit_neg']:+.6f}", str(r["pred"]),
                str(r["aff"]), str(r["iter"]),
            ]) + "\n")
    print(f"\n[batch] TSV -> {args.out}")

    # 4) Stats on generated P(binder)
    ps = np.array([r["p_binder"] for r in gen_rows], dtype=float)
    print("\n[batch] === P(binder) stats (generated, n={}) ===".format(len(ps)))
    print(f"  min    = {ps.min():.4f}")
    print(f"  10%    = {np.percentile(ps, 10):.4f}")
    print(f"  25%    = {np.percentile(ps, 25):.4f}")
    print(f"  median = {np.median(ps):.4f}")
    print(f"  75%    = {np.percentile(ps, 75):.4f}")
    print(f"  90%    = {np.percentile(ps, 90):.4f}")
    print(f"  max    = {ps.max():.4f}")
    print(f"  mean   = {ps.mean():.4f}")
    for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
        n = int((ps >= thr).sum())
        print(f"  P >= {thr:.2f} : {n:5d} ({n/len(ps)*100:.2f}%)")

    if fmn_rows:
        fmn_p = max(r["p_binder"] for r in fmn_rows)
        # FMN rank in the merged distribution
        merged = np.append(ps, fmn_p)
        rank = int((merged > fmn_p).sum()) + 1
        print(f"\n[batch] FMN P(binder) = {fmn_p:.4f} -> "
              f"rank {rank} of {len(merged)} (incl. FMN)")

    print("\n[batch] DONE")


if __name__ == "__main__":
    main()
