#!/usr/bin/env python
"""Parallel rDock wrapper: split the input SDF into K shards, run rbdock
on each shard in parallel (one process per shard), then merge the per-shard
.sd outputs into a single output .sd and write a per-ligand best-score TSV.

Each shard process inherits OMP_NUM_THREADS from the env. With K shards
and OMP_NUM_THREADS=T, peak load is K*T cores.

Usage
-----
    python run_rdock_parallel.py \\
        --ligand top300.sdf --out docked_top300.sd --shards 16 -n 3
"""
import os
import sys
import shutil
import argparse
import subprocess
import tempfile
import time
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))


def split_sdf(in_sdf, n_shards, tmpdir):
    """Split an SDF into n_shards via round-robin. Returns
    (paths, member_lists) where member_lists[k] is the ordered list of
    original-SDF record indices stored in shard k."""
    suppl = Chem.SDMolSupplier(in_sdf, removeHs=False, sanitize=False)
    paths = [os.path.join(tmpdir, f"shard_{k:02d}.sdf") for k in range(n_shards)]
    writers = [Chem.SDWriter(p) for p in paths]
    members = [[] for _ in range(n_shards)]
    n = 0
    for i, m in enumerate(suppl):
        if m is None:
            continue
        k = i % n_shards
        writers[k].write(m)
        members[k].append(i)
        n += 1
    for w in writers:
        w.close()
    return paths, members, n


def shard_scores_by_position(sd_path, n_runs):
    """rbdock emits exactly n_runs poses per input ligand, in input order.
    Return list of (input_position_in_shard, best_SCORE, best_INTER,
                    list_of_(SCORE,INTER))."""
    # collect all pose score blocks
    poses = []
    cur = {}
    if os.path.exists(sd_path):
        with open(sd_path) as fh:
            for ln in fh:
                if ln.startswith(">  <SCORE>"):
                    cur["SCORE"] = float(next(fh).strip())
                elif ln.startswith(">  <SCORE.INTER>"):
                    cur["SCORE.INTER"] = float(next(fh).strip())
                elif ln.startswith("$$$$"):
                    poses.append(cur)
                    cur = {}
    # group every n_runs consecutive poses -> one ligand
    out = []
    for i in range(0, len(poses), n_runs):
        chunk = poses[i:i + n_runs]
        if not chunk:
            break
        best = min(chunk, key=lambda d: d.get("SCORE", float("inf")))
        out.append({
            "pos": i // n_runs,
            "best_SCORE": best.get("SCORE"),
            "best_INTER": best.get("SCORE.INTER"),
            "all": chunk,
        })
    return out


def run_shard(shard_sdf, out_root, receptor_prm, protocol_prm, n_runs, seed,
              cwd, env):
    cmd = ["rbdock", "-r", receptor_prm, "-p", protocol_prm,
           "-i", shard_sdf, "-o", out_root, "-n", str(n_runs)]
    if seed is not None:
        cmd += ["-s", str(seed)]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return p


def parse_sd_records(sd_path):
    """Yield dict_of_score_props for each pose (legacy helper, unused)."""
    scores = []
    if not os.path.exists(sd_path):
        return scores
    cur = {}
    with open(sd_path) as fh:
        for ln in fh:
            if ln.startswith(">  <SCORE>"):
                cur["SCORE"] = float(next(fh).strip())
            elif ln.startswith(">  <SCORE.INTER>"):
                cur["SCORE.INTER"] = float(next(fh).strip())
            elif ln.startswith("$$$$"):
                scores.append(dict(cur))
                cur = {}
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--out", required=True, help="merged output .sd path")
    ap.add_argument("--receptor", default="receptor.prm")
    ap.add_argument("--protocol", default="dock.prm")
    ap.add_argument("-n", "--nruns", type=int, default=3)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tsv", default=None,
                    help="write per-ligand best score TSV here")
    ap.add_argument("--keep-temps", action="store_true")
    args = ap.parse_args()

    receptor_prm = (args.receptor if os.path.isabs(args.receptor)
                    else os.path.join(HERE, args.receptor))
    protocol_prm = (args.protocol if os.path.isabs(args.protocol)
                    else os.path.join(HERE, args.protocol))
    for f in (receptor_prm, protocol_prm):
        if not os.path.exists(f):
            print(f"ERROR missing {f}", file=sys.stderr); sys.exit(1)
    as_file = receptor_prm.replace(".prm", ".as")
    if not os.path.exists(as_file):
        print(f"ERROR missing cavity {as_file}", file=sys.stderr); sys.exit(1)

    # strip H into a clean SDF first (mirror run_rdock.strip_h_to_sdf)
    from run_rdock import strip_h_to_sdf
    tmpdir = tempfile.mkdtemp(prefix="rdock_par_")
    noh = os.path.join(tmpdir, "all_noh.sdf")
    n_total = strip_h_to_sdf(args.ligand, noh)
    print(f"[par] {n_total} ligands (H stripped) -> {noh}")

    # also build a positional table: original_idx -> (rnasmol_rank, gen_idx,
    # p_binder, smiles) by reading the *input* SDF (top300.sdf) in order
    suppl_in = Chem.SDMolSupplier(args.ligand, removeHs=False, sanitize=False)
    pos_meta = []
    for i, m in enumerate(suppl_in):
        d = m.GetPropsAsDict() if m is not None else {}
        pos_meta.append({
            "rnasmol_rank": d.get("rnasmol_rank", i),
            "gen_idx":      d.get("gen_idx", i),
            "rnasmol_p_binder": d.get("rnasmol_p_binder", float("nan")),
        })

    shard_paths, members, _ = split_sdf(noh, args.shards, tmpdir)
    print(f"[par] split into {args.shards} shards (round-robin)")

    # launch one rbdock per shard
    env = os.environ.copy()
    t0 = time.perf_counter()
    procs = []
    for k, sp in enumerate(shard_paths):
        if os.path.getsize(sp) == 0:
            continue
        out_root = os.path.join(tmpdir, f"out_{k:02d}")
        p = subprocess.Popen(
            ["rbdock", "-r", receptor_prm, "-p", protocol_prm,
             "-i", sp, "-o", out_root, "-n", str(args.nruns),
             "-s", str(args.seed)],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        procs.append((k, sp, out_root, p))
        print(f"[par] launched shard {k:02d} (pid {p.pid})")

    # wait
    for k, sp, out_root, p in procs:
        out, err = p.communicate()
        rc = p.returncode
        sd = out_root + ".sd"
        n_poses = 0
        if os.path.exists(sd):
            with open(sd) as fh:
                n_poses = sum(1 for ln in fh if ln.startswith("$$$$"))
        tag = "OK" if rc == 0 else f"RC={rc}"
        print(f"[par] shard {k:02d} done {tag} poses={n_poses}")
    dt = time.perf_counter() - t0
    print(f"[par] all shards done in {dt:.1f}s")

    # merge all shard .sd into a single .sd (plain cat works for SD files)
    final = (args.out if os.path.isabs(args.out)
             else os.path.join(os.getcwd(), args.out))
    with open(final, "w") as fh:
        for k, sp, out_root, _ in procs:
            sd = out_root + ".sd"
            if os.path.exists(sd):
                with open(sd) as sfh:
                    fh.write(sfh.read())
    print(f"[par] merged poses -> {final}")

    # parse per-pose scores positionally; group poses by ligand
    by_lig = {}  # original_idx -> dict
    n_pose_total = 0
    for k, sp, out_root, _ in procs:
        sd = out_root + ".sd"
        shard_inputs = members[k]  # original_idx list for this shard
        lig_scores = shard_scores_by_position(sd, args.nruns)
        for entry in lig_scores:
            pos = entry["pos"]
            if pos >= len(shard_inputs):
                continue
            orig_idx = shard_inputs[pos]
            meta = pos_meta[orig_idx] if orig_idx < len(pos_meta) else {}
            n_pose_total += len(entry["all"])
            cur = by_lig.get(orig_idx)
            new_score = entry["best_SCORE"]
            if cur is None or (new_score is not None and
                               new_score < cur["rdock_SCORE"]):
                by_lig[orig_idx] = {
                    "orig_idx": orig_idx,
                    "rnasmol_rank": meta.get("rnasmol_rank", orig_idx),
                    "gen_idx":      meta.get("gen_idx", orig_idx),
                    "rnasmol_p_binder": meta.get("rnasmol_p_binder",
                                                 float("nan")),
                    "rdock_SCORE":  new_score,
                    "rdock_INTER":  entry["best_INTER"],
                    "n_poses":      len(entry["all"]),
                }
    print(f"[par] {n_pose_total} poses, {len(by_lig)} unique ligands")

    tsv = args.tsv or (os.path.splitext(final)[0] + "_scores.tsv")
    with open(tsv, "w") as fh:
        fh.write("orig_idx\trnasmol_rank\tgen_idx\trnasmol_p_binder\t"
                 "rdock_SCORE\trdock_INTER\tn_poses\n")
        for orig_idx, rec in sorted(by_lig.items(),
                                    key=lambda kv: kv[1]["rdock_SCORE"]):
            fh.write(f"{orig_idx}\t{rec['rnasmol_rank']}\t{rec['gen_idx']}\t"
                     f"{rec['rnasmol_p_binder']:.6f}\t"
                     f"{rec['rdock_SCORE']:.4f}\t"
                     f"{rec['rdock_INTER']:.4f}\t{rec['n_poses']}\n")
    print(f"[par] per-ligand best scores -> {tsv}")

    if not args.keep_temps:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
