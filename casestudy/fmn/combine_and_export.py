#!/usr/bin/env python
"""Fuse rnasmol P(binder) + rdock SCORE into a combined ranking for the
1FMN case study, and emit top_binders.sdf (top-100 by combined score).

Inputs
------
* rnasmol_scores.tsv   -- every generated mol + FMN, with p_binder + logit
* rdock_top300_scores.tsv -- per-ligand best rdock SCORE/INTER for top-300
* top300.sdf           -- 3D coords (H included) of the top-300 by rnasmol
* FMN_ligand.sdf       -- reference ligand
* FMN rdock SCORE      -- taken from rdock/FMN_redock.sd (best SCORE)

Method
------
Among the 300 docked ligands:
  rnasmol_norm  = minmax(logit_neg)        in [0,1], higher=better
  rdock_norm    = minmax(-SCORE)           in [0,1], higher=better
  combined      = 0.5*rnasmol_norm + 0.5*rdock_norm
FMN is added as a reference row with its actual values.

Outputs
-------
* top_binders.sdf            -- top-100 (3D, H included) sorted by combined
* top_binders_combined.tsv   -- every docked ligand + FMN with all scores
* top20_report.md            -- markdown table for the report
"""
import os
import sys
import argparse
import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rnasmol_full(tsv):
    """gen_idx -> {p_binder, logit_neg, smiles, natoms, rank}"""
    out = {}
    with open(tsv) as fh:
        header = fh.readline().rstrip().split("\t")
        gi = header.index("idx")
        smi = header.index("smiles")
        na = header.index("natoms")
        pb = header.index("p_binder")
        lg = header.index("logit_neg")
        rk = header.index("rank")
        af = header.index("aff")
        for ln in fh:
            f = ln.rstrip().split("\t")
            if f[0] != "GEN":
                continue
            out[int(f[gi])] = {
                "smiles": f[smi],
                "natoms": int(f[na]),
                "p_binder": float(f[pb]),
                "logit_neg": float(f[lg]),
                "rank": int(f[rk]),
                "aff": f[af],
            }
    return out


def load_rdock(tsv):
    """gen_idx -> {rdock_SCORE, rdock_INTER, rnasmol_p_binder, orig_idx}"""
    out = {}
    with open(tsv) as fh:
        next(fh)
        for ln in fh:
            f = ln.rstrip().split("\t")
            out[int(f[2])] = {            # gen_idx
                "orig_idx":       int(f[0]),
                "rnasmol_rank":   int(f[1]),
                "rnasmol_p":      float(f[3]),
                "rdock_SCORE":    float(f[4]),
                "rdock_INTER":    float(f[5]),
                "n_poses":        int(f[6]),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnasmol-tsv", default="rnasmol_scores.tsv")
    ap.add_argument("--rdock-tsv",   default="rdock_top300_scores.tsv")
    ap.add_argument("--top-sdf",     default="top300.sdf")
    ap.add_argument("--fmn-sdf",     default="FMN_ligand.sdf")
    ap.add_argument("--fmn-score",   type=float, default=-17.157,
                    help="FMN best rdock SCORE (from FMN_redock.sd)")
    ap.add_argument("--fmn-inter",   type=float, default=-16.787)
    ap.add_argument("--fmn-p",       type=float, default=1.0)
    ap.add_argument("--fmn-logit",   type=float, default=10.893)
    ap.add_argument("--out-sdf",     default="top_binders.sdf")
    ap.add_argument("--out-tsv",     default="top_binders_combined.tsv")
    ap.add_argument("--out-md",      default="top20_report.md")
    ap.add_argument("--top-n",       type=int, default=100)
    args = ap.parse_args()

    rna_full = load_rnasmol_full(args.rnasmol_tsv)
    rdock    = load_rdock(args.rdock_tsv)
    print(f"[combine] rnasmol rows (GEN): {len(rna_full)}")
    print(f"[combine] rdock rows        : {len(rdock)}")

    # build per-ligand rows for all 300 docked
    rows = []
    for gid, r in rdock.items():
        meta = rna_full.get(gid, {})
        rows.append({
            "gen_idx":   gid,
            "orig_idx":  r["orig_idx"],
            "smiles":    meta.get("smiles", ""),
            "natoms":    meta.get("natoms", -1),
            "aff":       meta.get("aff", ""),
            "p_binder":  r["rnasmol_p"],
            "logit_neg": meta.get("logit_neg", float("nan")),
            "rdock_SCORE": r["rdock_SCORE"],
            "rdock_INTER": r["rdock_INTER"],
        })

    # minmax normalize logit and -SCORE over the 300
    logits = np.array([r["logit_neg"] for r in rows], dtype=float)
    scores = np.array([r["rdock_SCORE"] for r in rows], dtype=float)
    def minmax(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
    rnasmol_norm = minmax(logits)        # higher=better
    rdock_norm   = minmax(-scores)      # higher=better (less positive SCORE)
    combined     = 0.5 * rnasmol_norm + 0.5 * rdock_norm
    for i, r in enumerate(rows):
        r["rnasmol_norm"] = float(rnasmol_norm[i])
        r["rdock_norm"]   = float(rdock_norm[i])
        r["combined"]     = float(combined[i])

    # add FMN reference (not normalized against the 300, use raw)
    fmn_row = {
        "gen_idx":   -1,
        "orig_idx":  -1,
        "smiles":    Chem.MolToSmiles(
            next(Chem.SDMolSupplier(args.fmn_sdf, removeHs=True,
                                    sanitize=True))),
        "natoms":    31,
        "aff":       "",
        "p_binder":  args.fmn_p,
        "logit_neg": args.fmn_logit,
        "rdock_SCORE": args.fmn_score,
        "rdock_INTER": args.fmn_inter,
        "rnasmol_norm": float("nan"),
        "rdock_norm":   float("nan"),
        "combined":     float("nan"),
        "is_fmn":       True,
    }
    for r in rows:
        r["is_fmn"] = False

    # rank within the 300 (rank 1 = best)
    rows_by_p     = sorted(rows, key=lambda r: -r["p_binder"])
    rows_by_logit = sorted(rows, key=lambda r: -r["logit_neg"])
    rows_by_score = sorted(rows, key=lambda r:  r["rdock_SCORE"])
    rows_by_inter = sorted(rows, key=lambda r:  r["rdock_INTER"])
    rows_by_comb  = sorted(rows, key=lambda r: -r["combined"])
    rank_p     = {r["gen_idx"]: i+1 for i, r in enumerate(rows_by_p)}
    rank_logit = {r["gen_idx"]: i+1 for i, r in enumerate(rows_by_logit)}
    rank_score = {r["gen_idx"]: i+1 for i, r in enumerate(rows_by_score)}
    rank_inter = {r["gen_idx"]: i+1 for i, r in enumerate(rows_by_inter)}
    rank_comb  = {r["gen_idx"]: i+1 for i, r in enumerate(rows_by_comb)}
    for r in rows:
        r["rank_p"]     = rank_p[r["gen_idx"]]
        r["rank_logit"] = rank_logit[r["gen_idx"]]
        r["rank_score"] = rank_score[r["gen_idx"]]
        r["rank_inter"] = rank_inter[r["gen_idx"]]
        r["rank_comb"]  = rank_comb[r["gen_idx"]]

    # write the full combined TSV (sorted by combined desc), FMN appended last
    cols = ["gen_idx", "rank_comb", "rank_p", "rank_score", "combined",
            "p_binder", "logit_neg", "rdock_SCORE", "rdock_INTER",
            "rnasmol_norm", "rdock_norm", "natoms", "aff", "smiles"]
    with open(args.out_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows_by_comb:
            fh.write("\t".join([
                str(r["gen_idx"]), str(r["rank_comb"]), str(r["rank_p"]),
                str(r["rank_score"]),
                f"{r['combined']:.4f}",
                f"{r['p_binder']:.6f}",
                f"{r['logit_neg']:+.4f}",
                f"{r['rdock_SCORE']:.4f}",
                f"{r['rdock_INTER']:.4f}",
                f"{r['rnasmol_norm']:.4f}",
                f"{r['rdock_norm']:.4f}",
                str(r["natoms"]),
                str(r["aff"]), r["smiles"],
            ]) + "\n")
        # FMN reference (ranks not meaningful across the 300; print separately)
        fh.write("\t".join([
            "FMN", "FMN_ref", "FMN_ref", "FMN_ref",
            "FMN_ref",
            f"{fmn_row['p_binder']:.6f}",
            f"{fmn_row['logit_neg']:+.4f}",
            f"{fmn_row['rdock_SCORE']:.4f}",
            f"{fmn_row['rdock_INTER']:.4f}",
            "nan", "nan", "31", "", fmn_row["smiles"],
        ]) + "\n")
    print(f"[combine] combined TSV -> {args.out_tsv}")

    # FMN rank in rdock SCORE distribution (lower=better):
    # count docked ligands with SCORE <= FMN's -17.157
    n_better_score = sum(1 for r in rows if r["rdock_SCORE"] < args.fmn_score)
    n_better_inter = sum(1 for r in rows if r["rdock_INTER"] < args.fmn_inter)
    print(f"[combine] FMN rdock SCORE={args.fmn_score:.3f}: "
          f"{n_better_score}/300 docked are better (FMN rdock-rank ~ "
          f"{n_better_score + 1})")
    print(f"[combine] FMN rdock INTER={args.fmn_inter:.3f}: "
          f"{n_better_inter}/300 docked are better (FMN inter-rank ~ "
          f"{n_better_inter + 1})")

    # pick top-N by combined -> write SDF (3D + H from top300.sdf)
    top_n = rows_by_comb[:args.top_n]
    wanted_orig = {r["orig_idx"] for r in top_n}
    order = {r["orig_idx"]: k for k, r in enumerate(top_n)}
    suppl = Chem.SDMolSupplier(args.top_sdf, removeHs=False, sanitize=False)
    kept = {}
    for i, m in enumerate(suppl):
        if i in wanted_orig:
            kept[i] = m
    print(f"[combine] recovered {len(kept)}/{len(wanted_orig)} mols from "
          f"{args.top_sdf}")

    w = Chem.SDWriter(args.out_sdf)
    for r in top_n:
        m = kept.get(r["orig_idx"])
        if m is None:
            print(f"[combine] WARN missing orig_idx {r['orig_idx']}",
                  file=sys.stderr)
            continue
        try:
            m.SetProp("combined_rank",  str(r["rank_comb"]))
            m.SetProp("combined_score", f"{r['combined']:.4f}")
            m.SetProp("rnasmol_p_binder", f"{r['p_binder']:.6f}")
            m.SetProp("rnasmol_logit",    f"{r['logit_neg']:+.6f}")
            m.SetProp("rnasmol_norm",     f"{r['rnasmol_norm']:.4f}")
            m.SetProp("rdock_SCORE",      f"{r['rdock_SCORE']:.4f}")
            m.SetProp("rdock_INTER",      f"{r['rdock_INTER']:.4f}")
            m.SetProp("rdock_norm",       f"{r['rdock_norm']:.4f}")
            m.SetProp("gen_idx",          str(r["gen_idx"]))
            m.SetProp("smiles",           r["smiles"])
        except Exception:
            pass
        w.write(m)
    w.close()
    print(f"[combine] top-{len(top_n)} SDF -> {args.out_sdf}")

    # also append FMN to the SDF for visual reference
    fmn_supp = Chem.SDMolSupplier(args.fmn_sdf, removeHs=False, sanitize=False)
    fmn_m = next(iter(fmn_supp))
    if fmn_m is not None:
        try:
            fmn_m.SetProp("combined_rank",  "FMN_ref")
            fmn_m.SetProp("combined_score", "FMN_ref")
            fmn_m.SetProp("rnasmol_p_binder", f"{fmn_row['p_binder']:.6f}")
            fmn_m.SetProp("rdock_SCORE",      f"{fmn_row['rdock_SCORE']:.4f}")
            fmn_m.SetProp("rdock_INTER",      f"{fmn_row['rdock_INTER']:.4f}")
            fmn_m.SetProp("smiles",           fmn_row["smiles"])
        except Exception:
            pass
        with Chem.SDWriter(args.out_sdf.replace(".sdf", "_withFMN.sdf")) as ww:
            # re-open the top-N SDF and copy through, then add FMN
            for m in Chem.SDMolSupplier(args.out_sdf, removeHs=False,
                                        sanitize=False):
                ww.write(m)
            ww.write(fmn_m)
        print(f"[combine] FMN-appended SDF -> "
              f"{args.out_sdf.replace('.sdf', '_withFMN.sdf')}")

    # top-20 markdown report
    top20 = rows_by_comb[:20]
    with open(args.out_md, "w") as fh:
        fh.write("# 1FMN case study -- top-20 combined binders\n\n")
        fh.write("Combined = 0.5 * minmax(logit) + 0.5 * minmax(-rdock_SCORE), "
                 "computed over the 300 docked ligands.\n")
        fh.write(f"FMN reference: P={fmn_row['p_binder']:.4f}, "
                 f"rdock SCORE={fmn_row['rdock_SCORE']:.3f}, "
                 f"INTER={fmn_row['rdock_INTER']:.3f}. "
                 f"FMN rdock-SCORE-rank among 300 docked ~ "
                 f"{n_better_score+1} (lower=better).\n\n")
        fh.write("| rank | gen_idx | SMILES | natoms | rnasmol P | "
                 "rnasmol logit | rdock SCORE | rdock INTER | combined | "
                 "rdock rank |\n")
        fh.write("|------|---------|--------|--------|-----------|"
                 "---------------|-------------|-------------|----------|"
                 "------------|\n")
        for k, r in enumerate(top20, 1):
            fh.write(f"| {k} | {r['gen_idx']} | `{r['smiles']}` | "
                     f"{r['natoms']} | {r['p_binder']:.4f} | "
                     f"{r['logit_neg']:+.2f} | {r['rdock_SCORE']:.2f} | "
                     f"{r['rdock_INTER']:.2f} | {r['combined']:.3f} | "
                     f"{r['rank_score']} |\n")
        fh.write(f"\n## Top-100 SDF\n`{os.path.abspath(args.out_sdf)}`\n")
    print(f"[combine] top-20 markdown -> {args.out_md}")


if __name__ == "__main__":
    main()
