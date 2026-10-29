#!/usr/bin/env python3
"""Ground-truth redock validation (protocol-credibility gate).

Runs rbdock of the co-crystal ligand back into its own cavity, ranks poses by
SCORE.INTER (the true binding term; INTRA/RESTR are ignored for ranking), and
measures heavy-atom RMSD of the best pose against the crystal pose:
  - RMSD(in-place) : direct coordinate RMSD, same frame as receptor  <- PASS gate
  - RMSD(aligned)  : Kabsch rigid-body aligned (same atom order)

rDock's SD writer can garble bond orders (e.g. quaternary aromatic N+
demoted), so pose coordinates are read positionally and attached to the
chemically-correct reference mol copy. Atom order preservation is verified
by the per-atom element sequence.

PASS criterion: best SCORE.INTER pose RMSD(in-place) < 2.5 A AND
SCORE.INTER < -10 kcal/mol (loose sanity).

Usage:
  python redock_validation.py --system ber102_system.prm --ref ber102_ref.sdf \
      --out-dir . --tag ber102 [-n 10] [-s 42] [--protocol $RBT_ROOT/data/scripts/dock_solv.prm]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog('rdApp.*')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, help='system prm (absolute path)')
    ap.add_argument('--ref', required=True, help='co-crystal ligand SD (absolute path)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('-n', '--n-runs', type=int, default=10)
    ap.add_argument('-s', '--seed', type=int, default=42)
    ap.add_argument('--protocol', default=None)
    ap.add_argument('--rmsd-cut', type=float, default=2.5)
    ap.add_argument('--score-cut', type=float, default=-10.0)
    return ap.parse_args()


def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid-body alignment of P onto Q (same atom order)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Prot = Pc @ R
    return float(np.sqrt(((Prot - Qc) ** 2).sum(axis=1).mean()))


def main():
    args = parse_args()
    rbt_root = os.environ.get('RBT_ROOT')
    if not rbt_root:
        sys.exit('[FATAL] RBT_ROOT not exported')
    protocol = args.protocol or os.path.join(rbt_root, 'data/scripts/dock_solv.prm')

    out = os.path.join(os.path.realpath(args.out_dir), f'redock_{args.tag}')
    cmd = ['rbdock', '-i', os.path.realpath(args.ref), '-o', out,
           '-r', os.path.realpath(args.system), '-p', protocol,
           '-n', str(args.n_runs), '-s', str(args.seed)]
    print('[cmd] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)

    sd = out + '.sd'
    ref = next(Chem.SDMolSupplier(os.path.realpath(args.ref), removeHs=True))
    if ref is None:
        sys.exit('[FATAL] ref SD not readable')
    ref_symbols = [a.GetSymbol() for a in ref.GetAtoms()]
    ref_xyz = ref.GetConformer().GetPositions()

    rows, poses = [], {}
    for i, m in enumerate(Chem.SDMolSupplier(sd, removeHs=False, sanitize=False)):
        if m is None:
            continue
        # rDock output may retain polar H while the ref SD heavy count is the
        # comparison basis: align positions to the ref heavy-atom sequence
        heavy_idx = [j for j, a in enumerate(m.GetAtoms()) if a.GetSymbol() != 'H']
        syms = [m.GetAtomWithIdx(j).GetSymbol() for j in heavy_idx]
        if syms != ref_symbols:
            sys.exit(f'[FATAL] pose {i + 1} element order differs from ref - '
                     f'rDock did not preserve atom order')
        props = m.GetPropsAsDict()

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float('nan')

        all_xyz = m.GetConformer().GetPositions()
        xyz = all_xyz[heavy_idx]
        poses[i + 1] = xyz
        rows.append({'run': i + 1, 'SCORE.INTER': num(props.get('SCORE.INTER')),
                     'SCORE.INTRA': num(props.get('SCORE.INTRA')),
                     'SCORE.RESTR': num(props.get('SCORE.RESTR')),
                     'SCORE': num(props.get('SCORE'))})
    if not rows:
        sys.exit('[FATAL] rbdock produced no readable records')
    rows.sort(key=lambda r: r['SCORE.INTER'])
    best_row = rows[0]
    xyz = poses[best_row['run']]

    rmsd_inplace = float(np.sqrt(((xyz - ref_xyz) ** 2).sum(axis=1).mean()))
    rmsd_aligned = kabsch_rmsd(xyz, ref_xyz)
    passed = (rmsd_inplace < args.rmsd_cut) and (best_row['SCORE.INTER'] < args.score_cut)
    verdict = 'PASS' if passed else 'FAIL'

    # also write the best pose as a chemically-correct SDF for downstream use
    best_mol = Chem.Mol(ref)
    conf = Chem.Conformer(ref.GetNumAtoms())
    for j, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(j, Point3D(float(x), float(y), float(z)))
    best_mol.RemoveAllConformers()
    best_mol.AddConformer(conf, assignId=True)
    best_mol.SetProp('_Name', f'{args.tag}_best_INTER_{best_row["SCORE.INTER"]:.2f}')
    best_sd = os.path.join(os.path.realpath(args.out_dir), f'redock_{args.tag}_best.sdf')
    w = Chem.SDWriter(best_sd)
    w.write(best_mol)
    w.close()

    print('\n===== redock validation =====')
    print(f"runs scored: {len(rows)}  (ranked by SCORE.INTER)")
    for r in rows:
        print(f"  run {r['run']:>2}: INTER {r['SCORE.INTER']:9.3f}  "
              f"INTRA {r['SCORE.INTRA']:8.3f}  RESTR {r['SCORE.RESTR']:8.3f}")
    print(f"best SCORE.INTER = {best_row['SCORE.INTER']:.3f} (run {best_row['run']})")
    print(f"RMSD in-place    = {rmsd_inplace:.3f} A   <- gate (< {args.rmsd_cut})")
    print(f"RMSD aligned     = {rmsd_aligned:.3f} A")
    print(f"VERDICT: {verdict}")

    report = {'cmd': cmd, 'tag': args.tag, 'protocol': protocol,
              'n_runs': len(rows), 'scores': rows,
              'best': {'run': best_row['run'], 'SCORE.INTER': best_row['SCORE.INTER'],
                       'rmsd_inplace': rmsd_inplace, 'rmsd_aligned': rmsd_aligned},
              'cutoffs': {'rmsd': args.rmsd_cut, 'score_inter': args.score_cut},
              'verdict': verdict, 'best_pose_sdf': best_sd}
    rp = os.path.join(os.path.realpath(args.out_dir), f'redock_report_{args.tag}.json')
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[saved] {rp} / {best_sd}')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
