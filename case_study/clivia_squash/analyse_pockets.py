#!/usr/bin/env python3
"""Cluster the extracted Clivia/Squash pockets into distinct binding sites.

Two pockets are the same site if they superpose well. Superposition uses the set of
nucleotide residues shared by both pockets, matched through the RNA backbone trace, which
is comparable even when residue numbering differs between entries.

Outputs:
  - a within-entry residue-overlap table (are the multiple ligand copies one site or several?)
  - a cross-pocket RMSD matrix and single-linkage clusters at a fixed cut-off
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POCKETS = HERE / 'pockets'
BACKBONE = {'P', "P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "O4'"}


def load(path):
    """Return {(chain, resnum, resname): {atom_name: xyz}} for nucleotide residues."""
    res = {}
    for line in path.read_text().splitlines():
        if not line.startswith('ATOM'):
            continue
        rn = line[17:20].strip()
        if rn not in {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}:
            continue
        # chain is dropped: each pocket comes from one chain of one structure, and
        # chain IDs differ between entries for the same physical site
        key = (line[22:26].strip(), rn)
        res.setdefault(key, {})[line[12:16].strip()] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return res


def kabsch(P, Q):
    """RMSD after optimal superposition of P onto Q."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1, 1, d])
    R = V @ D @ Wt
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


def pocket_rmsd(a, b):
    """Superpose on shared backbone atoms of shared residues; return (rmsd, n_res_shared)."""
    shared = set(a) & set(b)
    if len(shared) < 4:
        return None, len(shared)
    P, Q = [], []
    for k in sorted(shared):
        for an in BACKBONE:
            if an in a[k] and an in b[k]:
                P.append(a[k][an]); Q.append(b[k][an])
    if len(P) < 12:
        return None, len(shared)
    return kabsch(np.array(P), np.array(Q)), len(shared)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cutoff', type=float, default=2.0)
    a = ap.parse_args()

    summary = json.loads((HERE / 'pockets_summary.json').read_text())
    info = {s['file'][:-4]: s for s in summary}
    structs = {}
    for f in sorted(POCKETS.glob('*.pdb')):
        structs[f.stem] = load(f)
    names = [s['file'][:-4] for s in summary if s['file'][:-4] in structs]

    # ---- within-entry: are the multiple ligand copies the same site? ----
    print('=== within-entry residue overlap (Jaccard of residue sets) ===')
    by_entry = {}
    for n in names:
        by_entry.setdefault(info[n]['pdb'], []).append(n)
    for pid, group in by_entry.items():
        if len(group) < 2:
            print(f'  {pid}: 1 ligand instance')
            continue
        for x, y in itertools.combinations(group, 2):
            sx, sy = set(structs[x]), set(structs[y])
            j = len(sx & sy) / len(sx | sy) if sx | sy else 0
            print(f'  {pid}: {x.split("_",1)[1]:16s} vs {y.split("_",1)[1]:16s} Jaccard={j:.2f}')

    # ---- cross-pocket RMSD ----
    print('\n=== cross-pocket superposition (shared residues, backbone atoms) ===')
    n = len(names)
    rmsd = np.full((n, n), np.nan)
    shared = np.zeros((n, n), dtype=int)
    for i, j in itertools.combinations(range(n), 2):
        r, ns = pocket_rmsd(structs[names[i]], structs[names[j]])
        shared[i, j] = shared[j, i] = ns
        if r is not None:
            rmsd[i, j] = rmsd[j, i] = r

    # single-linkage clustering at the cutoff
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in itertools.combinations(range(n), 2):
        if not np.isnan(rmsd[i, j]) and rmsd[i, j] <= a.cutoff:
            parent[find(i)] = find(j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(names[i])

    print(f'\n=== distinct sites at RMSD <= {a.cutoff} A ===')
    for cid, members in enumerate(clusters.values(), 1):
        apt = Counter(info[m]['aptamer'] for m in members)
        print(f'cluster {cid}: {len(members)} pockets  {dict(apt)}')
        for m in members:
            s = info[m]
            print(f'    {m:22s} {s["n_residues"]:2d} res  {s["composition"]}')

    np.save(HERE / 'pocket_rmsd.npy', rmsd)
    (HERE / 'pocket_clusters.json').write_text(json.dumps(
        {'cutoff': a.cutoff,
         'clusters': {f'cluster_{i+1}': v for i, v in enumerate(clusters.values())},
         'names': names}, indent=2))
    print(f'\nwrote pocket_rmsd.npy and pocket_clusters.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
