"""Compute the method-independent ECFP4 recovery baseline r_ref.

The radar's ECFP4 axis needs a calibration that carries no information about any of the
compared generators. The per-method permutation null does not qualify: it mixes in that
method's molecule count, size distribution and, through the molecule-to-target assignment,
target-specific structure, so two methods with identical observed recovery can be scored
differently. Here the null is estimated from a screening-molecule library instead --- the
PDBBind-derived CrossDocked ligands --- assigned at random to the fixed target universes. The
result is one scalar per benchmark, shared by every method.

    python compute_rref.py --library crossdock --sample 5000 --permutations 1000

Writes data/benchmark/output/tables/radar/rref.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')

ROOT = Path('/home/data/users/gaoletian/rna2smol')
sys.path.insert(0, str(ROOT / 'analysis'))
import radar  # noqa: E402  (recovery_from_maxima, scatter_maxima, ECFP4_RANGE)

DATA = ROOT / 'data' / 'benchmark'
OUT = DATA / 'output' / 'tables' / 'radar' / 'rref.json'

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def rdkit_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return m if m is not None else None


def fingerprints(smiles_list):
    from rdkit.Chem import DataStructs
    fps = []
    for s in smiles_list:
        m = rdkit_mol(s)
        if m is not None:
            fps.append(_GEN.GetFingerprint(m))
    arr = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


# ── target universes and their actives ──────────────────────────────────────
def universes():
    dude = DATA / 'DUDE_active_compounds.csv'
    d = pd.read_csv(dude, usecols=['pdb_id', 'canonical_smiles'], low_memory=False)
    dude_actives = {str(k): g['canonical_smiles'].dropna().unique().tolist()
                    for k, g in d.groupby(d['pdb_id'].astype(str))}

    sm = pd.read_csv(DATA / 'output' / 'smrtbench_combined_results.csv',
                     usecols=['pdb_id', 'active_smi'], low_memory=False).dropna(subset=['active_smi'])
    rna_actives = {str(k): g['active_smi'].dropna().unique().tolist()
                   for k, g in sm.groupby(sm['pdb_id'].astype(str))}
    return {'dude': dude_actives, 'smrtbench': rna_actives}


# ── reference molecule library ──────────────────────────────────────────────
def crossdock_smiles(limit: int | None = None) -> list[str]:
    """Ligand SMILES from the PDBBind-derived CrossDocked LMDB."""
    import lmdb
    import pickle
    env = lmdb.open(str(ROOT / 'data/crossdock/crossdocked_diffedmol_total'),
                    readonly=True, lock=False)
    out = []
    _seen = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            try:
                rec = pickle.loads(value)
                if not out and not _seen:
                    _seen.append(1)
                    print('record type:', type(rec), flush=True)
                    if isinstance(rec, dict):
                        print('record keys:', list(rec)[:15], flush=True)
                # the ligand record is nested: rec['mol'] holds smiles/element/pos/...
                smi = None
                if isinstance(rec, dict):
                    mol = rec.get('mol')
                    if isinstance(mol, dict) and mol.get('smiles'):
                        smi = mol['smiles']
                    else:
                        for field in ('smiles', 'ligand_smiles', 'SMI', 'smi'):
                            if rec.get(field):
                                smi = rec[field]
                                break
                if smi:
                    out.append(smi)
            except Exception:
                continue
            if limit and len(out) >= limit * 2:
                break
    env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--library', default='crossdock', choices=['crossdock', 'bkg'])
    ap.add_argument('--sample', type=int, default=5000)
    ap.add_argument('--permutations', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.library == 'crossdock':
        # the LMDB holds far more records than the null needs; stop early rather than
        # walking the whole store for a 5,000-molecule sample
        pool = crossdock_smiles(limit=a.sample * 3)
    else:
        pool = pd.read_csv(DATA / 'bkg_smiles.csv', sep='\t')['SMILES'].dropna().tolist()
    rng = np.random.default_rng(a.seed)
    if len(pool) > a.sample:
        pool = [pool[i] for i in rng.choice(len(pool), a.sample, replace=False)]
    print(f'reference library: {a.library}, {len(pool)} molecules', flush=True)

    ref_fp = fingerprints(pool)
    print(f'fingerprints: {ref_fp.shape}', flush=True)

    universes_ = universes()
    result = {}
    for bench, actives in universes_.items():
        targets = sorted(actives)
        n_targets = len(targets)
        # S[i, j] = best ECFP4-Tanimoto of reference molecule i against target j's actives
        ref_f = ref_fp
        S = np.zeros((ref_fp.shape[0], n_targets), dtype=np.float32)
        for j, t in enumerate(targets):
            act = fingerprints(actives[t])
            if act.size == 0:
                continue
            # float32: a uint8 matmul wraps at 256 and silently truncates the overlap.
            # Tanimoto, not Dice: inter / (|A| + |B| - inter).
            inter = ref_f.astype(np.float32) @ act.astype(np.float32).T
            denom = (ref_f.astype(np.float32).sum(1)[:, None]
                     + act.astype(np.float32).sum(1)[None, :] - inter)
            S[:, j] = (inter / np.maximum(denom, 1.0)).max(axis=1)
        print(f'{bench}: S {S.shape}, median max_sim {np.median(S.max(1)):.3f}', flush=True)

        assign = np.arange(ref_fp.shape[0]) % n_targets
        vals = np.empty(a.permutations)
        for k in range(a.permutations):
            vals[k] = radar.recovery_from_maxima(
                radar.scatter_maxima(S, rng.permutation(assign), n_targets))
        result[bench] = dict(r_ref=float(vals.mean()), sd=float(vals.std(ddof=1)),
                             n_reference=len(pool), n_targets=n_targets,
                             permutations=a.permutations)
        print(f'{bench}: r_ref = {vals.mean():.4f} (sd {vals.std(ddof=1):.4f})', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
