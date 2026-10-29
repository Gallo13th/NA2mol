#!/usr/bin/env python
"""Score arbitrary ligand SMILES with the RNAsmol ensemble against the Squash aptamer.

RNAsmol (sequence-based RNA-ligand binder classifier): RNA aptamer sequence + ligand 2D graph
-> P(binder).  We use the same 6-checkpoint ensemble and the same verbatim graph builder as
case_study/clivia_squash/score_with_rnasmol.py, but accept an arbitrary candidate file.

usage (rnasmol env):
  CUDA_VISIBLE_DEVICES=5 python score_candidates_rnasmol.py \
      --ligands ../HierMolGen/logs/fluo/final/squash_fluo_candidates.csv \
      --out ../HierMolGen/logs/fluo/final/squash_rnasmol_scores.csv --limit 0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fmn"))

import rnasmol_score as RS  # noqa: E402
from score_with_rnasmol import CHAIN, CHECKPOINTS, PDB_FOR_SEQ, rna_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ligands', required=True,
                    help='.csv/.smi (SMILES in first column) or .sdf')
    ap.add_argument('--pocket', default='squash', choices=list(PDB_FOR_SEQ))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--models-dir', default=os.path.join(HERE, '..', 'fmn', 'RNAsmol', 'saved_model'))
    args = ap.parse_args()

    seq = rna_sequence(PDB_FOR_SEQ[args.pocket], CHAIN[args.pocket])
    if not seq:
        raise SystemExit('empty RNA sequence')
    print(f'[rna] {args.pocket} chain {CHAIN[args.pocket]}: {len(seq)} nt')
    print(f'[rna] {seq}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = {}
    for ck in CHECKPOINTS:
        path = os.path.join(args.models_dir, ck)
        models[ck] = RS.load_model(path, device)
        print(f'[model] {ck} loaded', flush=True)

    rows = []
    t0 = time.time()
    for i, (name, mol) in enumerate(RS.ligands_from_file(args.ligands)):
        if args.limit and i >= args.limit:
            break
        try:
            data = RS.build_data_from_mol(mol, seq)
        except Exception as e:
            rows.append({'name': name, 'smiles': '', 'error': f'graph:{type(e).__name__}'})
            continue
        rec = {'name': name, 'smiles': RS.Chem.MolToSmiles(mol)}
        probs = []
        for ck, model in models.items():
            try:
                p, pred, logits = RS.score_data(model, data, device)
            except Exception as e:
                p = float('nan')
            rec[ck.replace('.pt', '')] = round(float(p), 4)
            probs.append(float(p))
        arr = np.array([p for p in probs if np.isfinite(p)])
        rec['rnasmol_mean'] = round(float(arr.mean()), 4) if len(arr) else float('nan')
        rec['rnasmol_max'] = round(float(arr.max()), 4) if len(arr) else float('nan')
        rec['rnasmol_n'] = int(len(arr))
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f'  scored {i+1} ({time.time()-t0:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = ['name', 'smiles', 'rnasmol_mean', 'rnasmol_max', 'rnasmol_n'] + \
           [ck.replace('.pt', '') for ck in CHECKPOINTS] + ['error']
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    n_ok = sum(1 for r in rows if 'error' not in r)
    print(f'[DONE] {len(rows)} molecules ({n_ok} scored) in {time.time()-t0:.0f}s -> {args.out}')
    if n_ok:
        means = np.array([r['rnasmol_mean'] for r in rows if 'error' not in r])
        print(f'  mean P(binder): median {np.median(means):.3f}  >=0.5: {(means>=0.5).sum()}/{n_ok}')


if __name__ == '__main__':
    main()
