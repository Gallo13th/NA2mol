#!/usr/bin/env python
"""Select top-N molecules (by rnasmol P(binder), ties broken by logit)
from the original generated SDF and write them out as a new SDF for rdock.

Reads the TSV produced by batch_score_rnasmol.py and the original SDF.
Preserves the original 3D coordinates + SDF properties.
"""
import os
import sys
import argparse

from rdkit import Chem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="rnasmol_scores.tsv")
    ap.add_argument("--sdf", required=True, help="original generated SDF")
    ap.add_argument("--out", required=True, help="output top-N SDF")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--source", default="GEN",
                    help="filter by source column (GEN/FMN)")
    args = ap.parse_args()

    # 1) read TSV, pick top-N by (p_binder desc, logit_neg desc)
    rows = []
    with open(args.tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_idx = header.index("idx")
        i_src = header.index("source")
        i_p = header.index("p_binder")
        i_l = header.index("logit_neg")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if f[i_src] != args.source:
                continue
            rows.append((int(f[i_idx]), float(f[i_p]), float(f[i_l])))
    rows.sort(key=lambda t: (-t[1], -t[2]))
    top = rows[:args.n]
    wanted = {idx for idx, _, _ in top}
    # preserve top ordering for the writer
    order = {idx: k for k, (idx, _, _) in enumerate(top)}
    print(f"[select] {len(rows)} {args.source} rows -> top-{len(top)} "
          f"P range [{top[-1][1]:.6f}, {top[0][1]:.6f}]")

    # 2) iterate the original SDF once, keep only wanted records
    suppl = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=False)
    kept = {}
    n_seen = 0
    for i, m in enumerate(suppl):
        if m is None:
            continue
        if i in wanted:
            kept[i] = m
        n_seen += 1
    print(f"[select] scanned {n_seen} SDF records, "
          f"recovered {len(kept)}/{len(wanted)}")

    # 3) write top-N in score order, attach rnasmol props for downstream
    w = Chem.SDWriter(args.out)
    for idx, p, l in top:
        m = kept.get(idx)
        if m is None:
            print(f"[select] WARN missing idx {idx}", file=sys.stderr)
            continue
        # props are on the mol already (minimizedAffinity etc.); add rnasmol
        try:
            m.SetProp("rnasmol_p_binder", f"{p:.6f}")
            m.SetProp("rnasmol_logit",    f"{l:+.6f}")
            m.SetProp("rnasmol_rank",     f"{order[idx]}")
            m.SetProp("gen_idx",          f"{idx}")
        except Exception:
            pass
        w.write(m)
    w.close()
    print(f"[select] -> {args.out}")


if __name__ == "__main__":
    main()
