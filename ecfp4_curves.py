"""Compare ECFP4 recovery curves of the methods against the reference library.

Recovery is the area under a cumulative retention curve: for each target R(t) = 1[max >= t],
and s = (1/(0.50-0.25)) * integral_0.25^0.50 R(t) dt, whose closed form is
clip((max - 0.25)/0.25, 0, 1) averaged over targets. Two curves can share an area while
crossing, so the shape has to be looked at before choosing how to weight it.

Prints, for tau = 0.25..0.50, the fraction of the fixed universe recovered by each method and
by the reference library under random assignment, plus the area of each curve.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')
ROOT = Path('/home/data/users/gaoletian/rna2smol')
SB = Path('/home/data/users/gaoletian/sbdd_baselines')
DATA = ROOT / 'data' / 'benchmark'

TAUS = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ── per-target maxima for the published methods ────────────────────────────
def method_maxima(which: str) -> dict[str, np.ndarray]:
    """mode -> array of per-target best ECFP4-TS, over the fixed universe."""
    if which == 'dude':
        path, uni = DATA / 'dude_stat_all.csv', None
        d = pd.read_csv(path, usecols=['mode', 'pdb_id', 'max_ecfp4_tanimoto'], low_memory=False)
        uni = sorted(pd.read_csv(DATA / 'DUDE_active_compounds.csv',
                                 usecols=['pdb_id'], low_memory=False)['pdb_id'].astype(str).unique())
    else:
        path = DATA / 'output' / 'smrtbench_combined_results.csv'
        d = pd.read_csv(path, usecols=['mode', 'pdb_id', 'max_ecfp4_tanimoto'], low_memory=False)
        sm = pd.read_csv(path, usecols=['pdb_id', 'active_smi'], low_memory=False).dropna(subset=['active_smi'])
        uni = sorted(sm['pdb_id'].astype(str).unique())

    idx = {t: i for i, t in enumerate(uni)}
    out = {}
    for mode, g in d.groupby('mode'):
        mx = np.full(len(uni), -np.inf)
        for tid, gg in g.groupby(g['pdb_id'].astype(str)):
            if tid in idx:
                v = pd.to_numeric(gg['max_ecfp4_tanimoto'], errors='coerce').max()
                if np.isfinite(v):
                    mx[idx[tid]] = v
        out[mode] = mx
    return out


# ── per-target maxima for the three new baselines ──────────────────────────
def baseline_maxima(which: str) -> dict[str, np.ndarray]:
    if which == 'dude':
        uni = sorted(pd.read_csv(DATA / 'DUDE_active_compounds.csv',
                                 usecols=['pdb_id'], low_memory=False)['pdb_id'].astype(str).unique())
    else:
        sm = pd.read_csv(DATA / 'output' / 'smrtbench_combined_results.csv',
                         usecols=['pdb_id', 'active_smi'], low_memory=False).dropna(subset=['active_smi'])
        uni = sorted(sm['pdb_id'].astype(str).unique())
    idx = {t: i for i, t in enumerate(uni)}
    stems = {'diffsbdd': 'diffsbdd', 'graphbp': 'graphbp_a1500', 'flag': 'flag'}
    out = {}
    for label, stem in stems.items():
        p = SB / 'smoke_out' / f'ecfp4_recovery_{stem}.csv'
        mx = np.full(len(uni), -np.inf)
        if p.is_file():
            d = pd.read_csv(p)
            d = d[d['bench'] == which]
            for r in d.itertuples(index=False):
                tid = str(r.target).split('_')[0]
                if tid in idx:
                    mx[idx[tid]] = max(mx[idx[tid]], float(r.max_sim))
        out[label] = mx
    return out


# ── reference library null curve ────────────────────────────────────────────
def ref_curve(which: str, library: str = 'crossdock', sample: int = 5000,
              permutations: int = 400, seed: int = 0) -> dict:
    import pickle
    import lmdb
    rng = np.random.default_rng(seed)

    if which == 'dude':
        d = pd.read_csv(DATA / 'DUDE_active_compounds.csv', usecols=['pdb_id', 'canonical_smiles'], low_memory=False)
        act = {str(k): g['canonical_smiles'].dropna().unique().tolist() for k, g in d.groupby(d['pdb_id'].astype(str))}
    else:
        sm = pd.read_csv(DATA / 'output' / 'smrtbench_combined_results.csv',
                         usecols=['pdb_id', 'active_smi'], low_memory=False).dropna(subset=['active_smi'])
        act = {str(k): g['active_smi'].dropna().unique().tolist() for k, g in sm.groupby(sm['pdb_id'].astype(str))}

    env = lmdb.open(str(ROOT / 'data/crossdock/crossdocked_diffedmol_total'), readonly=True, lock=False)
    pool = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            try:
                m = pickle.loads(value)['mol']
                if m.get('smiles'):
                    pool.append(m['smiles'])
            except Exception:
                continue
            if len(pool) >= sample * 3:
                break
    env.close()
    pool = [pool[i] for i in rng.choice(len(pool), sample, replace=False)]

    def fps(smis):
        out = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                out.append(_GEN.GetFingerprint(m))
        a = np.zeros((len(out), 2048), dtype=np.float32)
        from rdkit.Chem import DataStructs
        for i, fp in enumerate(out):
            DataStructs.ConvertToNumpyArray(fp, a[i])
        return a

    ref = fps(pool)
    targets = sorted(act)
    S = np.zeros((ref.shape[0], len(targets)), dtype=np.float32)
    for j, t in enumerate(targets):
        a = fps(act[t])
        if a.size == 0:
            continue
        inter = ref @ a.T
        denom = ref.sum(1)[:, None] + a.sum(1)[None, :] - inter
        S[:, j] = (inter / np.maximum(denom, 1.0)).max(axis=1)

    assign0 = np.arange(ref.shape[0]) % len(targets)
    curves = np.zeros((permutations, len(TAUS)))
    areas = np.zeros(permutations)
    for k in range(permutations):
        asg = rng.permutation(assign0)
        mx = np.full(len(targets), -np.inf)
        np.maximum.at(mx, asg, S[np.arange(S.shape[0]), asg])
        curves[k] = [(mx >= t).mean() for t in TAUS]
        areas[k] = area_of(mx)          # same closed form the methods use
    return dict(curve=curves.mean(0), area=float(areas.mean()), area_sd=float(areas.std(ddof=1)))


def area_of(maxima: np.ndarray) -> float:
    return float(np.clip((maxima - 0.25) / 0.25, 0.0, 1.0).mean())


def main() -> int:
    for which in ('dude', 'smrtbench'):
        print(f'=== {which} : fraction of universe recovered at each tau ===')
        rows = []
        for label, mx in method_maxima(which).items():
            rows.append((label, [(mx >= t).mean() for t in TAUS], area_of(mx)))
        for label, mx in baseline_maxima(which).items():
            rows.append((label, [(mx >= t).mean() for t in TAUS], area_of(mx)))
        print(f'{"method":26s} ' + ' '.join(f'{t:5.2f}' for t in TAUS) + '    area')
        for label, c, a in sorted(rows, key=lambda r: -r[2]):
            print(f'{label:26s} ' + ' '.join(f'{v:5.3f}' for v in c) + f'  {a:6.3f}')
        print()
    print('computing reference-library null curve (this takes a few minutes) ...', flush=True)
    for which in ('dude', 'smrtbench'):
        r = ref_curve(which)
        print(f'reference library {which}: ' + ' '.join(f'{v:5.3f}' for v in r['curve']) + f'  area {r["area"]:6.3f} (sd {r["area_sd"]:.3f})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
