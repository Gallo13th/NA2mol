#!/usr/bin/env python3
"""Compute the manuscript's seven radar axes per method and draw the radar panels.

Implements Methods, "Radar-chart normalisation" (see analysis/radar.py for the spec):
six property-retention axes pooling molecules, plus an ECFP4 recovery-gain axis that
weights targets equally and is standardised against a shuffled molecule-to-target null.

Writes
  data/benchmark/output/tables/radar/radar_scores_<benchmark>.csv   the seven scores
  data/benchmark/output/tables/radar/radar_null_<benchmark>.csv     permutation detail
  reproduction/figures/radar_<benchmark>.{pdf,png}                  the panel

    python3 analysis/radar_profiles.py [--permutations 1000] [--benchmark both]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import radar as R  # noqa: E402

RDLogger.DisableLog('rdApp.*')

DATA = ROOT / 'data' / 'benchmark'
OUT_TABLES = DATA / 'output' / 'tables' / 'radar'
OUT_FIGS = ROOT / 'reproduction' / 'figures'

BENCHMARKS = {
    'smrtbench': {
        'csv': DATA / 'output' / 'smrtbench_combined_results.csv',
        'title': 'RNA test set',
        'methods': ['RNA2smol (Only General Feat)', 'RNA2smol (Baseline)',
                    'Pretrain Only', 'Pretrain+Constrain', 'MolCRAFT',
                    'Pocket2Mol', 'ED2Mol'],
        'ablation': ['RNA2smol (Baseline)', 'Pretrain Only',
                     'Pretrain+Constrain', 'RNA2smol (Only General Feat)'],
    },
    'dude': {
        'csv': DATA / 'dude_stat_all.csv',
        # The manuscript's fixed universe is "101 targets with reference actives in
        # the fixed reference set": DUDE_active_compounds.csv. The per-row
        # active_smi in dude_stat_all.csv is a much smaller subset (42-332 vs
        # 57-1,647 unique actives per target) and gives a different recovery.
        'actives_csv': DATA / 'DUDE_active_compounds.csv',
        'actives_smiles_col': 'canonical_smiles',
        'title': 'DUD-E protein',
        'methods': ['RNA2smol', 'MolCRAFT', 'Pocket2Mol', 'ed2mol'],
        'ablation': [],
    },
}
DISPLAY = {
    'RNA2smol (Only General Feat)': 'NA2smol',
    'RNA2smol (Baseline)': 'Baseline',
    'Pretrain Only': 'Pretrain',
    'Pretrain+Constrain': 'Pretrain\n+Constrain',
    'RNA2smol': 'NA2smol',
    'ed2mol': 'ED2Mol',
}

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FP_CACHE: dict[str, object] = {}


def fp(smiles: str):
    """Morgan radius 2, 2048 bits, no chirality (manuscript's ECFP4 definition)."""
    if smiles not in _FP_CACHE:
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        try:
            _FP_CACHE[smiles] = _GEN.GetFingerprint(m) if m is not None else None
        except Exception:
            _FP_CACHE[smiles] = None
    return _FP_CACHE[smiles]


def actives_by_target(frame: pd.DataFrame, smiles_col: str = 'active_smi',
                      label_col: str | None = None) -> dict[str, list]:
    if label_col and label_col in frame.columns:
        frame = frame[frame[label_col] == 'active']
    out = {}
    for tid, grp in frame.groupby('pdb_id'):
        smis = [s for s in grp[smiles_col].dropna().unique() if isinstance(s, str)]
        fps = [f for f in (fp(s) for s in smis) if f is not None]
        if fps:
            out[str(tid)] = fps
    return out


def _bit_matrix(fps) -> np.ndarray:
    """(n, 2048) uint8 bit matrix from RDKit fingerprints; None rows stay zero."""
    M = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, f in enumerate(fps):
        if f is not None:
            M[i] = np.frombuffer(f.ToBitString().encode(), dtype=np.uint8) - 48
    return M


def similarity_matrix(smiles: pd.Series, actives: dict[str, list],
                      targets: list[str]) -> np.ndarray:
    """S[i, j] = molecule i's best ECFP4-TS against target j's reference actives.

    Tanimoto = |M & A| / (|M| + |A| - |M & A|), so intersections come from an
    integer matmul over the bit matrices. The loop is over targets, not over
    actives, so peak memory is bounded by the largest single target's active set
    (1,647 for DUD-E) rather than the 38,759-actives total.
    """
    n_mol = len(smiles)
    S = np.zeros((n_mol, len(targets)), dtype=np.float32)
    fps = [fp(s) for s in smiles]
    for j, tid in enumerate(targets):
        acts = actives.get(tid)
        if not acts:
            continue
        A = _bit_matrix(acts)
        Ai = A.astype(np.float32)
        pop_a = A.sum(axis=1, dtype=np.float32)[None, :]
        for lo in range(0, n_mol, 20000):
            hi = min(lo + 20000, n_mol)
            Mc = _bit_matrix(fps[lo:hi])
            inter = Mc.astype(np.float32) @ Ai.T
            pop_m = Mc.sum(axis=1, dtype=np.float32)[:, None]
            union = pop_m + pop_a - inter
            sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            S[lo:hi, j] = sim.max(axis=1)
            del Mc, inter, sim, union
    return S


def score_benchmark(key: str, permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = BENCHMARKS[key]
    t0 = time.time()
    cols = ['mode', 'pdb_id', 'SMILES', 'affinity', 'rmsd', 'qed', 'sa_score',
            'tpsa', 'strain', 'active_smi']
    # pdb_id must stay a string: identifiers such as 1e66 and 3e37 are parsed
    # as scientific notation otherwise, and stop matching the actives file.
    df = pd.read_csv(cfg['csv'], dtype={'pdb_id': str},
                     usecols=lambda c: c in cols)
    df = df[df['mode'].isin(cfg['methods'])].copy()
    df['SEI'] = R.sei(df)
    print(f'{key}: {len(df):,} rows, methods={df["mode"].nunique()}', flush=True)

    # the fixed universe: every target seen for any compared method
    targets = sorted(df['pdb_id'].astype(str).unique())
    tid_index = {t: i for i, t in enumerate(targets)}
    print(f'  target universe: {len(targets)}', flush=True)
    if cfg.get('actives_csv'):
        af = pd.read_csv(cfg['actives_csv'], dtype={'pdb_id': str},
                         usecols=lambda c: c in ('pdb_id', cfg['actives_smiles_col'], 'label'))
        acts = actives_by_target(af, cfg['actives_smiles_col'], 'label' if 'label' in af else None)
    else:
        acts = actives_by_target(df)
    print(f'  targets with reference actives: {len(acts)}  ({time.time()-t0:.0f}s)', flush=True)

    rows, null_rows = [], []
    for m in cfg['methods']:
        sub = df[df['mode'] == m].reset_index(drop=True)
        if sub.empty:
            continue
        t1 = time.time()
        scores = R.property_scores(sub)

        sub = sub[sub['SMILES'].notna()].reset_index(drop=True)
        S = similarity_matrix(sub['SMILES'], acts, targets)
        assign = np.array([tid_index[str(t)] for t in sub['pdb_id']], dtype=np.int64)
        ec = R.ecfp4_gain(S, assign, len(targets),
                          permutations=permutations, seed=0)
        print(f'  {m:34s} obs={ec["recovery_observed"]:.4f} '
              f'rand={ec["recovery_random"]:.4f}  ({time.time()-t1:.0f}s)', flush=True)

        row = {'benchmark': key, 'method': m, 'n_molecules': int(len(sub)),
               'n_targets': len(targets)}
        row.update(scores)
        row.update(R.property_summaries(sub))   # raw pooled values for the source table
        row['recovery_observed'] = ec['recovery_observed']
        row['recovery_random'] = ec['recovery_random']
        row['recovery_random_sd'] = ec['recovery_random_sd']
        rows.append(row)
        null_rows.append({'benchmark': key, 'method': m,
                          'n_permutations': ec['n_permutations'],
                          'recovery_random_mean': ec['recovery_random'],
                          'recovery_random_sd': ec['recovery_random_sd']})

    tbl = pd.DataFrame(rows).set_index('method')
    # s_m = (obs - rand) / (best - rand), clipped to [0,1]
    tbl['ecfp4'] = pd.Series(
        R.normalise_ecfp4(tbl['recovery_observed'].to_dict(),
                          tbl['recovery_random'].to_dict()))
    tbl['n_targets_universe'] = len(targets)
    tbl = tbl.reset_index()
    print(f'  total {time.time()-t0:.0f}s', flush=True)
    return tbl, pd.DataFrame(null_rows)


def draw(tbl: pd.DataFrame, title: str, out_stem: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / 'analysis'))
    from style import (METHOD_COLORS, METHOD_LABELS, INK, GRID,  # noqa: E402
                       setup_publication_style)
    setup_publication_style()

    axes = R.AXIS_ORDER
    labels = [R.SEVEN_LABELS[a] for a in axes]
    n = len(axes)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig = plt.figure(figsize=(3.1, 3.1))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_offset(np.pi / 2)      # first axis at the top
    ax.set_theta_direction(-1)          # clockwise, matching the manuscript
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 3.0)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1', '2', '3'], fontsize=5.5, color='#8a9299')
    ax.set_rlabel_position(180 / n)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines['polar'].set_color('#c8ced3')

    closed = np.append(theta, theta[0])
    for _, row in tbl.iterrows():
        m = row['method']
        s = np.array([row[a] for a in axes], dtype=float)
        r = R.radial(np.nan_to_num(s, nan=0.0))
        r = np.append(r, r[0])
        colour = METHOD_COLORS.get(m, '#606060')
        label = METHOD_LABELS.get(m, DISPLAY.get(m, m)).replace('\n', ' ')
        ax.plot(closed, r, color=colour, linewidth=1.15, marker='o',
                markersize=2.6, label=label)
        ax.fill(closed, r, color=colour, alpha=0.06)
    ax.set_title(title, fontsize=8, pad=12, color=INK)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
              ncol=min(4, len(tbl)), frameon=False, fontsize=6, handlelength=1.2)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_stem.with_suffix('.png'), dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out_stem.with_suffix('.pdf').relative_to(ROOT), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--permutations', type=int, default=1000)
    ap.add_argument('--benchmark', choices=['smrtbench', 'dude', 'both'], default='both')
    a = ap.parse_args()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    keys = ['smrtbench', 'dude'] if a.benchmark == 'both' else [a.benchmark]
    for key in keys:
        tbl, null = score_benchmark(key, a.permutations)
        tbl.to_csv(OUT_TABLES / f'radar_scores_{key}.csv', index=False, float_format='%.6f')
        null.to_csv(OUT_TABLES / f'radar_null_{key}.csv', index=False, float_format='%.6f')
        print(tbl[['method', 'qed', 'log10_strain', 'sa_score', 'rmsd',
                   'affinity', 'ecfp4', 'sei']].to_string(index=False), flush=True)
        draw(tbl, BENCHMARKS[key]['title'], OUT_FIGS / f'radar_{key}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
