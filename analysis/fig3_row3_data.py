"""Data for Fig. 3 panels b and g: recovery curves and metric sensitivity to the reference set.

Writes reproduction/figures/fig3_recovery_curves.csv and fig3_row3_metrics.csv.

Panel b needs, per configuration, the fraction of RNA targets with at least one molecule above an
ECFP4-TS threshold. Panel g needs, per configuration and per reference-set definition, the value of
recovery, QED, SA score and strain computed on the targets that survive that definition.

Reference-set definitions follow Extended Data Table 6: a target is dropped when every one of its
reference actives has ECFP4-TS >= 0.5 to the fine-tuning partition (FT) or to the ligand-only
pretraining library (PT).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reproduction' / 'figures'
COMBINED = ROOT / 'data/benchmark/output/smrtbench_combined_results.csv'
ACTIVES = ROOT / 'audit/tables/reference_active_overlap.csv'

CONFIGS = ['RNA2smol (Baseline)', 'Pretrain Only', 'Pretrain+Constrain',
           'RNA2smol (Only General Feat)']
WINDOW = (0.25, 0.50)
TAUS = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.50])


def target_recovery(maxima: np.ndarray) -> float:
    return float(np.clip((maxima - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0, 1).mean())


def main() -> int:
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df['mode'].isin(CONFIGS)].copy()
    print(f'{len(df):,} molecules x {df["mode"].nunique()} configurations, '
          f'{df["pdb_id"].nunique()} targets')

    # per-target best ECFP4-TS for each configuration
    best = (df.groupby(['mode', 'pdb_id'])['max_ecfp4_tanimoto'].max().unstack(0))
    targets = list(best.index)

    # panel b: fraction of targets recovered at each threshold
    rows = []
    for m in CONFIGS:
        col = best[m].to_numpy(dtype=float)
        for tau in TAUS:
            rows.append({'mode': m, 'tau': tau,
                         'targets_recovered': float((col >= tau).sum())})
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / 'fig3_recovery_curves.csv', index=False)
    print('wrote fig3_recovery_curves.csv')

    # reference-set definitions from the active-level audit
    act = pd.read_csv(ACTIVES, low_memory=False)
    act = act[(act['benchmark'] == 'SMRT-Net') & (act['parse_ok'] == True)]  # noqa: E712
    ft_ok = act['maxts_ft_train'] < 0.5
    pt_ok = act['maxts_pretrain'] < 0.5
    sets = {
        'All actives': set(targets),
        'FT-similar removed': set(act.loc[ft_ok, 'target_id']),
        'PT-similar removed': set(act.loc[pt_ok, 'target_id']),
    }
    for name, s in sets.items():
        print(f'{name:22s} {len(s)} targets')

    def pooled_median(sub: pd.DataFrame, col: str) -> float:
        v = pd.to_numeric(sub[col], errors='coerce').dropna()
        return float(v.median()) if len(v) else float('nan')

    rows = []
    for name, keep in sets.items():
        for m in CONFIGS:
            sub = df[df['mode'] == m]
            # recovery on that definition: mean over the surviving targets (all targets counted)
            maxima = np.array([best.loc[t, m] if t in best.index else np.nan
                               for t in sorted(keep)], dtype=float)
            rec = float(np.nanmean(np.clip((maxima - WINDOW[0]) /
                                          (WINDOW[1] - WINDOW[0]), 0, 1)))
            mols = sub[sub['pdb_id'].isin(keep)]
            rows.append({
                'mode': m, 'definition': name, 'n_targets': len(keep),
                'recovery': rec,
                'qed': pooled_median(mols, 'qed'),
                'sa_score': pooled_median(mols, 'sa_score'),
                'strain': pooled_median(mols, 'strain'),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'fig3_row3_metrics.csv', index=False)
    print(out.round(3).to_string(index=False))
    print('wrote fig3_row3_metrics.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# ── per-method version of the reference-set panel (main figure panel g) ──────────
def method_reference_panel() -> None:
    """Values of seven benchmarked methods on the three reference-set target subsets.

    Recovery comes from the overlap audit (identical universes to Extended Data Table 6);
    QED, SA and strain are pooled medians over the molecules of the surviving targets,
    combining the four original benchmark runs with the three later generators.
    """
    out = ROOT / 'reproduction' / 'figures' / 'fig3_method_reference_panel.csv'
    methods = ['NA2smol', 'ED2Mol', 'MolCRAFT', 'Pocket2Mol', 'DiffSBDD', 'FLAG', 'GraphBP']
    mode_map = {'RNA2smol (Only General Feat)': 'NA2smol', 'ED2Mol': 'ED2Mol',
                'ed2mol': 'ED2Mol',
                'MolCRAFT': 'MolCRAFT', 'Pocket2Mol': 'Pocket2Mol',
                'DiffSBDD': 'DiffSBDD', 'FLAG': 'FLAG', 'GraphBP': 'GraphBP'}
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df['mode'].isin(mode_map)].copy()
    df['method'] = df['mode'].map(mode_map)
    extra = pd.read_csv('/home/data/users/gaoletian/sbdd_baselines/radar_export/'
                        'new_baselines_molecules.csv', low_memory=False)
    extra = extra[(extra['bench'] == 'smrtbench') & extra['mode'].isin(mode_map)]
    extra = extra.assign(method=extra['mode'].map(mode_map))
    keep = ['method', 'pdb_id', 'qed', 'sa_score', 'strain']
    df = pd.concat([df[keep], extra[keep]], ignore_index=True)
    for c in ('qed', 'sa_score', 'strain'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    act = pd.read_csv(ACTIVES, low_memory=False)
    act = act[(act['benchmark'] == 'SMRT-Net') & (act['parse_ok'] == True)]  # noqa: E712
    sets = {
        'All actives': set(act['target_id']),
        'FT-similar removed': set(act.loc[act['maxts_ft_train'] < 0.5, 'target_id']),
        'PT-similar removed': set(act.loc[act['maxts_pretrain'] < 0.5, 'target_id']),
    }
    rec = pd.read_csv(ROOT / 'audit/tables/recovery_overlap_unified.csv')
    rec = rec[rec['benchmark'] == 'SMRT-Net']
    rec = rec.pivot_table(index='method', columns='filter', values='recovery')
    rec = rec.rename(columns={'original': 'All actives', 'ft_T0.5': 'FT-similar removed',
                              'pretrain_T0.5': 'PT-similar removed'})

    rows = []
    for name, keep_targets in sets.items():
        for m in methods:
            sub = df[(df['method'] == m) & (df['pdb_id'].isin(keep_targets))]
            rows.append({'method': m, 'definition': name, 'n_targets': len(keep_targets),
                         'recovery': float(rec.loc[m, name]) if m in rec.index else float('nan'),
                         'qed': float(sub['qed'].dropna().median()),
                         'sa_score': float(sub['sa_score'].dropna().median()),
                         'strain': float(sub['strain'].dropna().median())})
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(res.round(3).to_string(index=False))
    print('wrote fig3_method_reference_panel.csv')


def recovery_curves_by_definition() -> None:
    """Recovery curves on each reference-set target subset, for every method.

    For a target j, the curve value at threshold tau is 1 if any molecule assigned to j reaches
    ECFP4-TS >= tau; the panel averages those indicators over the targets of the subset, so the
    three curves are directly comparable even though the subsets differ in size.
    """
    out = ROOT / 'reproduction' / 'figures' / 'fig3_recovery_curves_by_def.csv'
    # the per-molecule ECFP4-TS is carried by the four runs in the combined table; the three
    # later generators are not needed for this panel (NA2smol and ED2Mol are the ones that swap)
    methods = ['NA2smol', 'ED2Mol', 'MolCRAFT', 'Pocket2Mol']
    mode_map = {'RNA2smol (Only General Feat)': 'NA2smol', 'ED2Mol': 'ED2Mol',
                'MolCRAFT': 'MolCRAFT', 'Pocket2Mol': 'Pocket2Mol'}
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df['mode'].isin(mode_map)].copy()
    df['method'] = df['mode'].map(mode_map)
    best = df.groupby(['method', 'pdb_id'])['max_ecfp4_tanimoto'].max()

    act = pd.read_csv(ACTIVES, low_memory=False)
    act = act[(act['benchmark'] == 'SMRT-Net') & (act['parse_ok'] == True)]  # noqa: E712
    sets = {
        'All actives': sorted(set(act['target_id'])),
        'FT-similar removed': sorted(set(act.loc[act['maxts_ft_train'] < 0.5, 'target_id'])),
        'PT-similar removed': sorted(set(act.loc[act['maxts_pretrain'] < 0.5, 'target_id'])),
    }
    rows = []
    for name, keep in sets.items():
        for m in methods:
            if m not in best.index.get_level_values(0):
                continue
            vals = np.array([best.get((m, t), np.nan) for t in keep], dtype=float)
            for tau in TAUS:
                rows.append({'method': m, 'definition': name, 'n_targets': len(keep),
                             'tau': tau,
                             'frac_targets': float(np.nanmean(vals >= tau))})
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(res.groupby(['method', 'definition'])['frac_targets'].max().unstack().round(3)
          .to_string())
    print('wrote fig3_recovery_curves_by_def.csv')
