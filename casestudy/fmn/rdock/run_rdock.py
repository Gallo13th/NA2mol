#!/usr/bin/env python
"""
rDock docking wrapper for the 1FMN/FMN case study (batch screening).

IMPORTANT (this rDock build): rbdock/rbcavity SEGFAULT on any explicit
hydrogen atom. Therefore every ligand SDF must be stripped of ALL hydrogens
(3D coordinates of heavy atoms preserved) before docking. This wrapper does
that automatically and then runs rbdock.

Receptor + cavity are prepared ONCE and live next to this script:
  receptor.mol2   1FMN pocket, NO hydrogen, kekulized (ar->1/2)
  receptor.as     precomputed cavity (sphere @ FMN centroid, r=12, 194 A^3)
  receptor.prm    receptor parameter file (SCORE section)
  dock.prm        docking protocol (GA + MC + simplex)

Usage (run inside the rna2smol conda env, which has rbdock + rdkit):
    conda activate rna2smol
    python run_rdock.py --ligand generated_mols.sdf --out docked.sdf -n 3
    python run_rdock.py --ligand generated_mols.sdf --out docked.sdf -n 5 \\
                        --receptor receptor.prm --protocol dock.prm
"""
import os
import sys
import shutil
import subprocess
import tempfile
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

def strip_h_to_sdf(in_sdf, out_sdf):
    """Read an SDF, strip ALL hydrogens (keep heavy-atom 3D), write out_sdf.
    Requires the rna2smol env (rdkit). Returns number of molecules written."""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(in_sdf, removeHs=False, sanitize=False)
    n = 0
    w = Chem.SDWriter(out_sdf)
    for i, m in enumerate(suppl):
        if m is None:
            print(f"[run_rdock] WARN: SDF record {i} unreadable, skipped",
                  file=sys.stderr)
            continue
        try:
            mh = Chem.AddHs(m)            # ensure full H
            mh = Chem.RemoveHs(mh)        # then strip all H (keeps heavy 3D)
            Chem.SanitizeMol(mh)
        except Exception:
            # fall back: just drop atoms whose atomic number == 1
            em = Chem.RWMol(m)
            to_remove = [a.GetIdx() for a in em.GetAtoms() if a.GetAtomicNum() == 1]
            for idx in sorted(to_remove, reverse=True):
                em.RemoveAtom(idx)
            mh = em.GetMol()
        mh.SetProp("_Name", m.GetProp("_Name")
                   if m.HasProp("_Name") else f"mol{i}")
        w.write(mh)
        n += 1
    w.close()
    return n


def run_rbdock(ligand_noh_sdf, out_root, receptor_prm, protocol_prm, n_runs, seed):
    cmd = ["rbdock", "-r", receptor_prm, "-p", protocol_prm,
           "-i", ligand_noh_sdf, "-o", out_root, "-n", str(n_runs)]
    if seed is not None:
        cmd += ["-s", str(seed)]
    print("[run_rdock] " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0:
        # 139 = segfault; most likely cause is leftover H atoms in the ligand
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        if p.returncode == 139:
            print("[run_rdock] ERROR: rbdock segfaulted (exit 139). This rDock "
                  "build crashes on explicit H atoms -- check that the ligand "
                  "was fully dehydrogenated.", file=sys.stderr)
        raise SystemExit(p.returncode)
    return p.stdout


def parse_scores(out_sd):
    """Return list of (record_idx, score_total, score_inter) from rbdock SD."""
    scores = []
    if not os.path.exists(out_sd):
        return scores
    idx = 0
    cur_total = cur_inter = None
    with open(out_sd) as fh:
        for ln in fh:
            if ln.startswith(">  <SCORE>"):
                cur_total = float(next(fh).strip())
            elif ln.startswith(">  <SCORE.INTER>"):
                cur_inter = float(next(fh).strip())
                idx += 1
                scores.append((idx, cur_total, cur_inter))
                cur_total = cur_inter = None
    return scores


def main():
    ap = argparse.ArgumentParser(description="rDock screening wrapper (1FMN)")
    ap.add_argument("--ligand", required=True, help="input ligand SDF (may have H)")
    ap.add_argument("--out", default="docked.sd", help="output SD root (.sd)")
    ap.add_argument("--receptor", default="receptor.prm")
    ap.add_argument("--protocol", default="dock.prm")
    ap.add_argument("-n", "--nruns", type=int, default=3,
                    help="dock poses per ligand (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    receptor_prm = os.path.join(HERE, args.receptor)
    protocol_prm = os.path.join(HERE, args.protocol)
    for f in (receptor_prm, protocol_prm):
        if not os.path.exists(f):
            print(f"[run_rdock] ERROR missing {f}", file=sys.stderr)
            sys.exit(1)
    as_file = receptor_prm.replace(".prm", ".as")
    if not os.path.exists(as_file):
        print(f"[run_rdock] ERROR missing cavity {as_file}; run rbcavity first",
              file=sys.stderr)
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="rdock_")
    try:
        noh = os.path.join(tmpdir, "ligand_noh.sdf")
        n = strip_h_to_sdf(args.ligand, noh)
        print(f"[run_rdock] {n} ligands (H stripped) -> {noh}")

        out_root = os.path.join(tmpdir, "out")
        run_rbdock(noh, out_root, receptor_prm, protocol_prm,
                   args.nruns, args.seed)

        out_sd = out_root + ".sd"
        final = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
        shutil.copy(out_sd, final)
        print(f"[run_rdock] docked poses -> {final}")

        scores = parse_scores(out_sd)
        if scores:
            print(f"\n{'record':>7} {'SCORE':>10} {'INTER':>10}")
            best = min(scores, key=lambda t: t[1])
            for idx, tot, inter in scores:
                mark = "  <-- best" if (idx, tot) == (best[0], best[1]) else ""
                print(f"{idx:>7} {tot:>10.3f} {inter:>10.3f}{mark}")
            print(f"\n[run_rdock] best total SCORE = {best[1]:.3f} "
                  f"(more negative = better)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
