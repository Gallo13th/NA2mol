#!/usr/bin/env python3
"""sp1 final: rDock scoring of 159 exact-m1 hits (and reference m1).

Uses the protocol's SCORE.INTER as the binding term; ranks hits by their best
pose; also redocks the m1 reference pose itself as anchor (sanity). Batch on
CUDA-free CPU; ~10 runs each.
"""
import os, sys, json, glob, subprocess, tempfile
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)

SYSTEM = 'casestudy/sp1/rdock/sp1_system.prm'
OUTDIR_ROOT = 'casestudy/sp1/rdock'

def rbdock(lig_sdf, tag, n=10, seed=42):
    prm = os.path.realpath(SYSTEM)
    prm_data = os.path.join(os.environ['RBT_ROOT'], 'data/scripts/dock_solv.prm')
    cmd = ['rbdock', '-r', prm, '-i', os.path.realpath(lig_sdf), '-o', tag,
           '-p', prm_data, '-n', str(n), '-s', str(seed)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return res.stdout, res.stderr

def parse_poses(tag):
    """given tag as base path, read <tag>.sd (may not exist) - index rows."""
    rows = []
    sd = f'{tag}.sd'
    if not os.path.exists(sd):
        return rows
    suppl = Chem.SDMolSupplier(sd, removeHs=True, sanitize=True)
    for m in suppl:
        if m is None or not m.HasProp('SCORE.INTER'):
            continue
        try:
            score = float(m.GetProp('SCORE.INTER'))
            rows.append((score, m))
        except Exception:
            continue
    return rows

def main():
    recs = json.load(open('casestudy/sp1/gen_runA/results.json'))['results']
    m1hits = [r for r in recs if r['exact']]
    print(f'[sp1] {len(m1hits)} m1 hits')

    # make single-mol SDFs for each hit (up to 159)
    mols_by_name = {}
    for m in Chem.SDMolSupplier('casestudy/sp1/gen_runA/valid_mols.sdf', removeHs=True, sanitize=True):
        if m is not None and m.HasProp('_Name'):
            mols_by_name[m.GetProp('_Name')] = m
    print(f'[sp1] loaded {len(mols_by_name)} mols from valid_mols.sdf')

    # Batch: review the 159 hits, but dock subset (first 30) for time,
    # plus reference = m1_ref_pose
    subset = m1hits[:30]
    scored = []
    for r in subset:
        m = mols_by_name.get(r['name'])
        if m is None:
            continue
        uid = r['name'].replace('.', '_')
        tmp_f = os.path.realpath(f'{OUTDIR_ROOT}/tmp_{uid}.sdf')
        w = Chem.SDWriter(tmp_f)
        w.write(m)
        w.close()
        tag = os.path.realpath(f'{OUTDIR_ROOT}/dock_{uid}')
        stdout, stderr = rbdock(tmp_f, tag)
        rows = parse_poses(tag)
        if rows:
            best = max(rows, key=lambda x: x[0]) if False else rows[np.argmax([x[0] for x in rows])]
            scored.append({'name': r['name'], 'smiles': r['smiles'],
                           'best_INTER': round(best[0], 2),
                           'n_poses': len(rows)})
            print(f"  [dock] {r['name']} INTER={best[0]:.2f} poses={len(rows)}", flush=True)
        else:
            stdout_t = stdout[-300:] if stdout else ''
            print(f"  [fail] {r['name']}: no poses. {stdout_t[:120]}", flush=True)
        os.remove(tmp_f)
    json.dump({'scored': scored,
               'system': SYSTEM},
              open(f'{OUTDIR_ROOT}/rdock_m1_30.json', 'w'), indent=2)
    print(f'\n[sum] docked {len(scored)}/{len(subset)}')
    if scored:
        for s in sorted(scored, key=lambda x: -x['best_INTER'])[:15]:
            print(f"  {s['best_INTER']:8.2f}  {s['name']}")

if __name__ == '__main__':
    main()
