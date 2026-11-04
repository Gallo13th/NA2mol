"""Extend the Fig. 2 source tables to the three new SBDD baselines.

Fig. 2 compares methods on four things beyond the radar: ECFP4 recovery as a function of
threshold, retention through the four-tier screening cascade, scaffold breadth, and
per-target retention. All four are appended here for DiffSBDD, GraphBP and FLAG, computed
from the same molecules and with the same rules as the published methods.

Sources
  molecules   ~/sbdd_baselines/radar_export/new_baselines_molecules.csv
  ECFP4       ~/sbdd_baselines/smoke_out/ecfp4_recovery_<stem>.csv  (per-target max_sim)

Rules, taken from analysis/style.py FUNNEL_STEPS and the manuscript's cascade:
  All         every molecule
  Pharma.     affinity < -8, QED > 0.5, SA < 5
  +SEI        Pharma. and SEI > 0.1
  Strict      affinity < -9, QED > 0.5, SA < 5, SEI > 0.15
  +Strain     affinity < -9, QED > 0.6, SA < 4, SEI > 0.2, strain < 200

`targets_ge5` and `hit_mols_total` need molecule-level similarities, which the export does
not carry for these methods; they are written as NaN and must not be used for them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')
ROOT = Path('/home/data/users/gaoletian/rna2smol')
SB = Path('/home/data/users/gaoletian/sbdd_baselines')
DATA = ROOT / 'data' / 'benchmark'
TABLES = DATA / 'output' / 'tables'
MOL = SB / 'radar_export' / 'new_baselines_molecules.csv'

METHODS = ['DiffSBDD', 'GraphBP', 'FLAG']
ECFP4_STEM = {'DiffSBDD': 'diffsbdd', 'GraphBP': 'graphbp_a1500', 'FLAG': 'flag'}
STEPS = [
    ('All', lambda d: np.ones(len(d), bool)),
    ('aff<-8,QED>0.5,SA<5', lambda d: (d.affinity < -8) & (d.qed > 0.5) & (d.sa_score < 5)),
    ('+SEI>0.1', lambda d: (d.affinity < -8) & (d.qed > 0.5) & (d.sa_score < 5) & (d.SEI > 0.1)),
    ('aff<-9,QED>0.5,SA<5,SEI>0.15',
     lambda d: (d.affinity < -9) & (d.qed > 0.5) & (d.sa_score < 5) & (d.SEI > 0.15)),
    ('aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200',
     lambda d: ((d.affinity < -9) & (d.qed > 0.6) & (d.sa_score < 4)
                & (d.SEI > 0.2) & (d.strain < 200))),
]


def molecules() -> pd.DataFrame:
    d = pd.read_csv(MOL, low_memory=False)
    for c in ('affinity', 'rmsd', 'qed', 'sa_score', 'tpsa', 'strain', 'SEI'):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['SEI'] = d.get('SEI', -d['affinity'] / d['tpsa'].replace(0, np.nan))
    return d[d['mode'].isin(METHODS)]


def universes() -> dict[str, list[str]]:
    dude = sorted(pd.read_csv(DATA / 'DUDE_active_compounds.csv',
                              usecols=['pdb_id'], low_memory=False)['pdb_id'].astype(str).unique())
    sm = pd.read_csv(DATA / 'output' / 'smrtbench_combined_results.csv',
                     usecols=['pdb_id', 'active_smi'], low_memory=False).dropna(subset=['active_smi'])
    return {'dude': dude, 'smrtbench': sorted(sm['pdb_id'].astype(str).unique())}


def ecfp4_rows(bench: str, uni: list[str], taus) -> list[dict]:
    rows = []
    for method in METHODS:
        p = SB / 'smoke_out' / f'ecfp4_recovery_{ECFP4_STEM[method]}.csv'
        maxima = {t: -np.inf for t in uni}
        if p.is_file():
            d = pd.read_csv(p)
            d = d[d['bench'] == bench]
            for r in d.itertuples(index=False):
                tid = str(r.target).split('_')[0]
                if tid in maxima:
                    maxima[tid] = max(maxima[tid], float(r.max_sim))
        m = np.array([maxima[t] for t in uni])
        for tau in taus:
            rows.append(dict(method=method, tau=tau,
                             successful_targets_ge1_molecule=int((m >= tau).sum()),
                             successful_targets_ge5_molecules=np.nan,
                             total_evaluable_targets=len(uni),
                             hit_mols_total=np.nan))
    return rows


def funnel_rows(bench: str, mol: pd.DataFrame) -> list[dict]:
    rows = []
    for method in METHODS:
        sub = mol[(mol['mode'] == method) & (mol['bench'] == bench)]
        if sub.empty:
            continue
        for step, rule in STEPS:
            keep = sub[rule(sub)]
            rows.append(dict(flow='strain_aware', method=method, filter_step=step,
                             mols_retained=int(len(keep)),
                             targets_ge1=int(keep['pdb_id'].nunique()),
                             targets_ge5=np.nan))
    return rows


def scaffold_rows(mol: pd.DataFrame) -> list[dict]:
    rows = []
    for method in METHODS:
        rec = {'method': method}
        for bench, prefix in (('smrtbench', 'smrt'), ('dude', 'dude')):
            smis = mol[(mol['mode'] == method) & (mol['bench'] == bench)]['SMILES'].dropna().unique()
            scaffolds = set()
            for s in smis:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    continue
                try:
                    scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
                except Exception:
                    continue
            rec[f'{prefix}_unique_smiles'] = int(len(smis))
            rec[f'{prefix}_bm_scaffolds'] = int(len(scaffolds))
            rec[f'{prefix}_internal_diversity'] = (len(scaffolds) / len(smis)) if len(smis) else np.nan
        rows.append(rec)
    return rows


def append(path: Path, rows: list[dict], drop: list[str]) -> None:
    df = pd.read_csv(path)
    df = df[~df['method'].isin(drop)]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path, index=False)
    print(f'  {path.relative_to(ROOT)}: {len(df)} rows')


def main() -> int:
    mol = molecules()
    uni = universes()
    print(f'molecules: {len(mol)} rows from {MOL.name}')

    smrt_ecfp4 = TABLES / 'smrtbench' / 't1b_ecfp4_target_success_by_threshold.csv'
    taus = sorted(pd.read_csv(smrt_ecfp4)['tau'].unique())
    print(f'ECFP4 tau grid: {len(taus)} values')

    append(TABLES / 'smrtbench' / 't1b_ecfp4_target_success_by_threshold.csv',
           ecfp4_rows('smrtbench', uni['smrtbench'], taus), METHODS)
    append(TABLES / 'dude' / 'd2_ecfp4_target_success_by_threshold.csv',
           ecfp4_rows('dude', uni['dude'], taus), METHODS)
    append(TABLES / 'smrtbench' / 't6_virtual_screening_funnel.csv',
           funnel_rows('smrtbench', mol), METHODS)
    append(TABLES / 'dude' / 'd7_virtual_screening_funnel.csv',
           funnel_rows('dude', mol), METHODS)
    append(DATA / 'output' / 'chemspace_targeted' / 'scaffold_breadth_table.csv',
           scaffold_rows(mol), METHODS)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
