#!/usr/bin/env python
"""
Export all numerical data underlying visualizations as CSV tables.
Exports every numerical table behind the manuscript figures.
Output: data/benchmark/output/tables/{smrtbench,ringfix,dude}/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sascorer = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / 'data' / 'benchmark'
# Tables are data, not code: they belong under data/, not next to this script.
OUT_DIR = DATA_DIR / 'output' / 'tables'
EVAL_DIR = DATA_DIR / 'eval_fixed_rings'

# ── Helpers ─────────────────────────────────────────────────────────────────
def agg_across_targets(data, col, agg='median'):
    per_target = data.groupby(['mode', 'pdb_id'])[col].median().reset_index()
    across = per_target.groupby('mode')[col].agg(['mean', 'std', 'count']).reset_index()
    across.columns = ['mode', f'{col}_mean', f'{col}_std', f'{col}_n_targets']
    return across

def save_csv(df, subdir, filename):
    d = OUT_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    df.to_csv(path, index=False, float_format='%.6f')
    print(f"  Saved: {path}")

def bootstrap_mean_ci(values, n_boot=10000, seed=0, alpha=0.05):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))

def safe_wilcoxon_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2 or np.allclose(diffs, 0):
        return np.nan
    try:
        return float(wilcoxon(diffs, alternative='two-sided', zero_method='wilcox').pvalue)
    except ValueError:
        return np.nan

def per_target_series(data, method, metric, tau=None):
    sub = data[data['mode'] == method]
    if metric == 'ecfp4_target_success_ge_tau':
        return sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
            lambda x: int((x >= tau).sum() >= 1))
    if tau is not None:
        return sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(lambda x: int((x >= tau).sum()))
    return sub.groupby('pdb_id')[metric].median()

def paired_stats_table(data, reference, comparators, metric_specs, seed_offset=0):
    rows = []
    seed = seed_offset
    for spec in metric_specs:
        metric = spec['metric']
        tau = spec.get('tau')
        direction = spec['direction']
        ref = per_target_series(data, reference, metric, tau=tau)
        for comp in comparators:
            other = per_target_series(data, comp, metric, tau=tau)
            common = sorted(set(ref.index) & set(other.index))
            if len(common) == 0:
                continue
            r = ref.loc[common].astype(float)
            c = other.loc[common].astype(float)
            diffs = r - c
            ci_low, ci_high = bootstrap_mean_ci(diffs.values, seed=seed)
            seed += 1
            if direction == 'higher':
                improved = int((diffs > 0).sum())
            else:
                improved = int((diffs < 0).sum())
            rows.append({
                'reference': reference,
                'comparator': comp,
                'metric': metric,
                'tau': tau if tau is not None else np.nan,
                'direction': direction,
                'n_targets': len(common),
                'reference_mean': r.mean(),
                'comparator_mean': c.mean(),
                'mean_delta_reference_minus_comparator': diffs.mean(),
                'delta_ci95_low': ci_low,
                'delta_ci95_high': ci_high,
                'wilcoxon_p_two_sided': safe_wilcoxon_p(diffs.values),
                'targets_improved': improved,
                'targets_tied': int((diffs == 0).sum()),
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# 1. RNA test-set tables
# ══════════════════════════════════════════════════════════════════════════════
def export_smrtbench():
    print("=" * 60)
    print("Exporting SMRTBench tables...")
    print("=" * 60)

    csv_path = DATA_DIR / 'output' / 'smrtbench_combined_results.csv'
    if not csv_path.exists():
        print("  SKIP: smrtbench_combined_results.csv not found")
        return
    df = pd.read_csv(csv_path)

    MAIN_METHODS = [
        'RNA2smol (Only General Feat)',
        'RNA2smol (Baseline)',
        'MolCRAFT',
        'Pocket2Mol',
        'ED2Mol',
    ]
    ABLATION_METHODS = [
        ' Allfeat Cst', ' Allfeat Nocst',
        ' Generalfeat Cst', ' Generalfeat Nocst',
    ]
    ALL_METHODS = MAIN_METHODS + ABLATION_METHODS
    df = df[df['mode'].isin(ALL_METHODS)]
    df['SEI'] = -df['affinity'] / df['tpsa'].replace(0, np.nan)
    df['SEI'] = df['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
    df['LLE'] = -df['affinity'] - df['logp']
    df['LBE'] = df['affinity'] / df['heavy_atoms'].replace(0, np.nan)

    # ── Table 1a: Main comparison — per-metric mean±std across targets ──────
    main_df = df[df['mode'].isin(MAIN_METHODS)]
    plot_metrics = ['affinity', 'SEI', 'LLE', 'rmsd', 'LBE', 'max_ecfp4_tanimoto', 'strain']
    rows = []
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for col in plot_metrics:
            pt = sub.groupby('pdb_id')[col].median()
            rows.append({
                'method': method, 'metric': col,
                'mean': pt.mean(), 'std': pt.std(),
                'n_targets': len(pt), 'n_mols': len(sub),
            })
    t1a = pd.DataFrame(rows)
    save_csv(t1a, 'smrtbench', 't1_main_comparison.csv')

    # ── Table 1b: ECFP4 thresholded reference recovery ─────────────────────
    TAU_VALS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    rows = []
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for tau in TAU_VALS:
            per_target_hits = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: int((x >= t).sum()))
            per_target_rate = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: (x >= t).mean())
            rows.append({
                'method': method, 'tau': tau,
                'hit_mols_total': int((sub['max_ecfp4_tanimoto'] >= tau).sum()),
                'targets_ge1_hit': int((per_target_hits >= 1).sum()),
                'targets_ge5_hits': int((per_target_hits >= 5).sum()),
                'total_evaluable_targets': int(per_target_hits.shape[0]),
                'per_target_hits_mean': per_target_hits.mean(),
                'per_target_hits_std': per_target_hits.std(),
                'per_target_hits_median': per_target_hits.median(),
                'hit_rate_mean': per_target_rate.mean(),
                'hit_rate_std': per_target_rate.std(),
            })
    ecfp4_threshold_counts = pd.DataFrame(rows)
    save_csv(ecfp4_threshold_counts, 'smrtbench', 't1b_ecfp4_threshold_counts.csv')
    save_csv(
        ecfp4_threshold_counts[[
            'method', 'tau', 'targets_ge1_hit', 'targets_ge5_hits',
            'total_evaluable_targets', 'hit_mols_total'
        ]].rename(columns={
            'targets_ge1_hit': 'successful_targets_ge1_molecule',
            'targets_ge5_hits': 'successful_targets_ge5_molecules',
        }),
        'smrtbench',
        't1b_ecfp4_target_success_by_threshold.csv',
    )
    save_csv(ecfp4_threshold_counts, 'smrtbench', 't1b_ecfp4_hitrate.csv')

    # ── Table 1c: ECFP4 top-N median ────────────────────────────────────────
    TOP_N = [1, 3, 5, 10, 20, 50, 100]
    rows = []
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for n in TOP_N:
            per_target = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, n=n: x.nlargest(min(n, len(x))).median())
            rows.append({
                'method': method, 'top_n': n,
                'median_mean': per_target.mean(),
                'median_std': per_target.std(),
            })
    save_csv(pd.DataFrame(rows), 'smrtbench', 't1c_ecfp4_topn.csv')

    # ── Table 1d: Paired per-target tests and bootstrap CIs ─────────────────
    metric_specs = [
        {'metric': 'affinity', 'direction': 'lower'},
        {'metric': 'qed', 'direction': 'higher'},
        {'metric': 'sa_score', 'direction': 'lower'},
        {'metric': 'rmsd', 'direction': 'lower'},
        {'metric': 'SEI', 'direction': 'higher'},
        {'metric': 'LLE', 'direction': 'higher'},
        {'metric': 'max_ecfp4_tanimoto', 'direction': 'higher'},
        {'metric': 'strain', 'direction': 'lower'},
        {'metric': 'ecfp4_hits_ge_tau', 'tau': 0.30, 'direction': 'higher'},
        {'metric': 'ecfp4_target_success_ge_tau', 'tau': 0.30, 'direction': 'higher'},
    ]
    main_stats = paired_stats_table(
        main_df,
        'RNA2smol (Only General Feat)',
        [m for m in MAIN_METHODS if m != 'RNA2smol (Only General Feat)'],
        metric_specs,
        seed_offset=1000,
    )
    save_csv(main_stats, 'smrtbench', 't1d_per_target_paired_stats.csv')

    # ── Table 2: Ablation comparison ────────────────────────────────────────
    abl_methods = ['RNA2smol (Only General Feat)'] + ABLATION_METHODS + ['RNA2smol (Baseline)']
    abl_df = df[df['mode'].isin(abl_methods)]
    rows = []
    for method in abl_methods:
        sub = abl_df[abl_df['mode'] == method]
        for col in plot_metrics:
            pt = sub.groupby('pdb_id')[col].median()
            rows.append({
                'method': method, 'metric': col,
                'mean': pt.mean(), 'std': pt.std(),
                'n_targets': len(pt), 'n_mols': len(sub),
            })
    save_csv(pd.DataFrame(rows), 'smrtbench', 't2_ablation_comparison.csv')

    ablation_stats = paired_stats_table(
        abl_df,
        'RNA2smol (Only General Feat)',
        [m for m in abl_methods if m != 'RNA2smol (Only General Feat)'],
        metric_specs,
        seed_offset=2000,
    )
    save_csv(ablation_stats, 'smrtbench', 't2b_ablation_paired_stats.csv')

    # ── Table 3: Per-method × per-target heatmap data (affinity/SEI/rmsd) ───
    for col in [c for c in ['affinity', 'rmsd', 'SEI', 'strain'] if c in main_df.columns]:
        pivot = main_df.groupby(['mode', 'pdb_id'])[col].median().reset_index()
        ht = pivot.pivot(index='mode', columns='pdb_id', values=col)
        ht = ht.reindex(MAIN_METHODS)
        ht.index.name = 'method'
        save_csv(ht.reset_index(), 'smrtbench', f't3_heatmap_{col}.csv')

    # ── Table 4: Property distribution summary stats ─────────────────────────
    prop_cols = ['qed', 'logp', 'tpsa', 'sa_score']
    rows = []
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for col in prop_cols:
            vals = sub[col].dropna()
            rows.append({
                'method': method, 'property': col,
                'mean': vals.mean(), 'std': vals.std(),
                'median': vals.median(), 'q1': vals.quantile(0.25),
                'q3': vals.quantile(0.75), 'n': len(vals),
            })
    # Add active molecule reference stats
    active_props = main_df.groupby(['pdb_id', 'active_smi']).first().reset_index()
    for smi in active_props['active_smi'].dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for col, val in [('qed', QED.qed(mol)), ('logp', Crippen.MolLogP(mol)),
                          ('tpsa', Descriptors.TPSA(mol))]:
            pass
    save_csv(pd.DataFrame(rows), 'smrtbench', 't4_property_stats.csv')

    # ── Table 5: MW-binned performance ──────────────────────────────────────
    mw_bins = ['0–120', '120-180', '180–300', '300-420', '≥420']
    rows = []
    for method in ALL_METHODS:
        sub = df[df['mode'] == method]
        for mb in mw_bins:
            mb_data = sub[sub['mw_bin'] == mb]
            if len(mb_data) == 0:
                continue
            for col in [c for c in ['affinity', 'SEI', 'strain'] if c in mb_data.columns]:
                per_target = mb_data.groupby('pdb_id')[col].median()
                rows.append({
                    'method': method, 'mw_bin': mb, 'metric': col,
                    'mean': per_target.mean(), 'std': per_target.std(),
                    'n_targets': len(per_target), 'n_mols': len(mb_data),
                })
    save_csv(pd.DataFrame(rows), 'smrtbench', 't5_mw_binned.csv')

    # ── Table 6: Virtual screening funnel ────────────────────────────────────
    # The default manuscript funnel avoids using pose-relaxation RMSD as the
    # final geometry gate. RMSD is a minimise-in-place stability readout, whereas
    # PoseCheck strain energy is closer to the conformational-plausibility risk
    # that practical RNA virtual screening needs to control.
    strain_aware_filter_steps = [
        ('All', None),
        ('aff<-6', lambda d: d['affinity'] < -6),
        ('aff<-7', lambda d: d['affinity'] < -7),
        ('+QED>0.3', lambda d: (d['affinity'] < -7) & (d['qed'] > 0.3)),
        ('QED>0.4,+SA<5', lambda d: (d['affinity'] < -7) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
        ('aff<-8,QED>0.5,SA<5', lambda d: (d['affinity'] < -8) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
        ('+SEI>0.1', lambda d: (d['affinity'] < -8) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.1)),
        ('aff<-9,QED>0.5,SA<5,SEI>0.15', lambda d: (d['affinity'] < -9) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.15)),
        ('aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200',
         lambda d: (d['affinity'] < -9) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2) & (d['strain'] < 200)),
    ]

    rmsd_reference_filter_steps = [
        ('All', None),
        ('aff<-6', lambda d: d['affinity'] < -6),
        ('aff<-7', lambda d: d['affinity'] < -7),
        ('+RMSD<2.5', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.5)),
        ('RMSD<2.0', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0)),
        ('+QED>0.3', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.3)),
        ('QED>0.4,+SA<5', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
        ('aff<-8,RMSD<1.5,QED>0.5,SA<5', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
        ('+SEI>0.1', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.1)),
        ('aff<-9,RMSD<1.0,QED>0.5,SA<5,SEI>0.15', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.15)),
        ('aff<-9,RMSD<1.0,QED>0.6,SA<4,SEI>0.2', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2)),
    ]

    def virtual_screening_rows(filter_steps, flow_name):
        rows = []
        for method in MAIN_METHODS:
            sub = main_df[main_df['mode'] == method]
            for step_name, flt in filter_steps:
                filtered = sub if flt is None else sub[flt(sub)]
                per_target_counts = filtered.groupby('pdb_id').size()
                rows.append({
                    'flow': flow_name,
                    'method': method,
                    'filter_step': step_name,
                    'mols_retained': len(filtered),
                    'targets_ge1': int((per_target_counts >= 1).sum()),
                    'targets_ge5': int((per_target_counts >= 5).sum()),
                })
        return pd.DataFrame(rows)

    save_csv(
        virtual_screening_rows(strain_aware_filter_steps, 'strain_aware'),
        'smrtbench',
        't6_virtual_screening_funnel.csv',
    )
    save_csv(
        virtual_screening_rows(rmsd_reference_filter_steps, 'rmsd_reference'),
        'smrtbench',
        't6b_virtual_screening_funnel_rmsd_reference.csv',
    )

    # ── Table 7a: Radar profile — raw values per method ─────────────────────
    radar_metrics = ['affinity', 'SEI', 'LLE', 'LBE', 'rmsd',
                     'qed', 'sa_score', 'rotatable_bonds',
                     'max_ecfp4_tanimoto', 'strain']
    rows = []
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for col in radar_metrics:
            if col in sub.columns:
                pt = sub.groupby('pdb_id')[col].median()
                rows.append({
                    'method': method, 'metric': col,
                    'mean': pt.mean(), 'std': pt.std(), 'n_targets': len(pt),
                })
    # Bond JS divergence (try to load from cache)
    import pickle
    bond_cache = DATA_DIR / 'output' / 'cache_bond_js.pkl'
    if bond_cache.exists():
        with open(bond_cache, 'rb') as f:
            bond_js = pickle.load(f)
        for method in MAIN_METHODS:
            if method in bond_js:
                for k, v in bond_js[method].items():
                    rows.append({'method': method, 'metric': k,
                                 'mean': v, 'std': 0, 'n_targets': 0})
    save_csv(pd.DataFrame(rows), 'smrtbench', 't7a_radar_profile_raw.csv')

    # ── Table 7b: Radar normalized values (rank) ────────────────────────────
    # Compute rank normalization
    from scipy.spatial.distance import jensenshannon

    RADAR_DIRS = {
        'affinity': 'lower', 'SEI': 'higher', 'LLE': 'higher', 'LBE': 'lower',
        'rmsd': 'lower', 'qed': 'higher', 'sa_score': 'lower',
        'rotatable_bonds': 'lower', 'max_ecfp4_tanimoto': 'higher',
        'bond_len_js': 'lower', 'bond_angle_js': 'lower', 'strain': 'lower',
    }
    RADAR_FLOOR = 0.25
    radar_metrics_all = radar_metrics + ['bond_len_js', 'bond_angle_js']

    # Build profile dict
    profile = {}
    for method in MAIN_METHODS:
        profile[method] = {}
        sub = main_df[main_df['mode'] == method]
        for col in radar_metrics:
            if col in sub.columns:
                profile[method][col] = sub.groupby('pdb_id')[col].median().mean()
        if bond_cache.exists() and method in bond_js:
            profile[method].update(bond_js[method])

    rank_rows, mm_rows = [], []
    for method in MAIN_METHODS:
        rank_row = {'method': method}
        mm_row = {'method': method}
        for col in radar_metrics_all:
            vals = {m: profile[m].get(col, np.nan) for m in MAIN_METHODS}
            valid = {k: v for k, v in vals.items() if not np.isnan(v)}
            if len(valid) < 2:
                rank_row[f'{col}_rank_norm'] = (1.0 + RADAR_FLOOR) / 2
                mm_row[f'{col}_minmax_norm'] = (1.0 + RADAR_FLOOR) / 2
                continue
            # Rank
            ascending = (RADAR_DIRS[col] == 'lower')
            sorted_m = sorted(valid, key=valid.get, reverse=not ascending)
            r = sorted_m.index(method) + 1
            rank_row[f'{col}_rank_norm'] = 1.0 - (r - 1) * (1.0 - RADAR_FLOOR) / (len(sorted_m) - 1)
            # MinMax
            min_v, max_v = min(valid.values()), max(valid.values())
            v = profile[method].get(col, np.nan)
            if np.isnan(v) or max_v - min_v < 1e-8:
                mm_row[f'{col}_minmax_norm'] = (1.0 + RADAR_FLOOR) / 2
            elif RADAR_DIRS[col] == 'higher':
                mm_row[f'{col}_minmax_norm'] = RADAR_FLOOR + (1.0 - RADAR_FLOOR) * (v - min_v) / (max_v - min_v)
            else:
                mm_row[f'{col}_minmax_norm'] = RADAR_FLOOR + (1.0 - RADAR_FLOOR) * (1.0 - (v - min_v) / (max_v - min_v))
            rank_row[f'{col}_raw'] = profile[method].get(col, np.nan)
            mm_row[f'{col}_raw'] = profile[method].get(col, np.nan)
        rank_rows.append(rank_row)
        mm_rows.append(mm_row)
    save_csv(pd.DataFrame(rank_rows), 'smrtbench', 't7b_radar_rank_normalized.csv')
    save_csv(pd.DataFrame(mm_rows), 'smrtbench', 't7c_radar_minmax_normalized.csv')

    # ── Table 8: Ablation radar profile raw ──────────────────────────────────
    rows = []
    for method in ABLATION_METHODS:
        sub = df[df['mode'] == method]
        for col in radar_metrics:
            if col in sub.columns:
                pt = sub.groupby('pdb_id')[col].median()
                rows.append({
                    'method': method, 'metric': col,
                    'mean': pt.mean(), 'std': pt.std(), 'n_targets': len(pt),
                })
    if bond_cache.exists():
        for method in ABLATION_METHODS:
            if method in bond_js:
                for k, v in bond_js[method].items():
                    rows.append({'method': method, 'metric': k,
                                 'mean': v, 'std': 0, 'n_targets': 0})
    save_csv(pd.DataFrame(rows), 'smrtbench', 't8_ablation_radar_profile.csv')

    print("  SMRTBench tables done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ring Fix tables (visualize_fix.py)
# ══════════════════════════════════════════════════════════════════════════════
def export_ringfix():
    print("=" * 60)
    print("Exporting Ring Fix tables...")
    print("=" * 60)

    # Load our fix evaluation data
    all_rows = []
    for ds_name, ds_dir in [('smrtnet', EVAL_DIR / 'smrtnet_onlygeneral_0504'),
                              ('dude', EVAL_DIR / 'dude_onlygeneral_0506')]:
        for csv_path in sorted(ds_dir.glob('*/*_comparison.csv')):
            tdf = pd.read_csv(csv_path)
            tdf['dataset'] = ds_name
            tdf['target'] = csv_path.parent.name
            all_rows.append(tdf)
    full = pd.concat(all_rows, ignore_index=True)
    full = full[(full['orig_affinity'] < 0) & (full['fixed_affinity'] < 0)].copy()
    changed = full[full['smi_changed'] == True].copy()

    changed['dqed'] = changed['fixed_qed'] - changed['orig_qed']
    changed['dsa'] = changed['fixed_sa'] - changed['orig_sa']
    changed['dlogp'] = changed['fixed_logp'] - changed['orig_logp']
    changed['dmw'] = changed['fixed_mw'] - changed['orig_mw']
    changed['drmsd'] = changed['fixed_rmsd'] - changed['orig_rmsd']
    changed['daffinity'] = changed['fixed_affinity'] - changed['orig_affinity']

    rna = changed[changed['dataset'] == 'smrtnet']
    dude = changed[changed['dataset'] == 'dude']
    rna_all = full[full['dataset'] == 'smrtnet']
    dude_all = full[full['dataset'] == 'dude']

    # ── Table R1: Fix rate summary ───────────────────────────────────────────
    rows = []
    for ds_name, ds_changed, ds_all in [('RNA (smrtnet)', rna, rna_all),
                                          ('DUDE (protein)', dude, dude_all)]:
        rows.append({
            'dataset': ds_name,
            'total_mols': len(ds_all),
            'fixed_mols': len(ds_changed),
            'fix_rate_pct': len(ds_changed) / len(ds_all) * 100,
        })
    save_csv(pd.DataFrame(rows), 'ringfix', 'r1_fix_rate_summary.csv')

    # ── Table R2: Property delta summary stats ──────────────────────────────
    delta_cols = ['daffinity', 'dqed', 'dsa', 'dlogp', 'dmw', 'drmsd']
    delta_labels = ['delta_affinity', 'delta_qed', 'delta_sa', 'delta_logp', 'delta_mw', 'delta_rmsd']
    rows = []
    for ds_name, ds_data in [('RNA', rna), ('DUDE', dude)]:
        for dcol, dlab in zip(delta_cols, delta_labels):
            vals = ds_data[dcol].dropna()
            rows.append({
                'dataset': ds_name, 'property': dlab,
                'mean': vals.mean(), 'std': vals.std(),
                'median': vals.median(), 'q1': vals.quantile(0.25),
                'q3': vals.quantile(0.75),
                'pct_improved': (vals < 0).mean() * 100 if 'affinity' in dlab or 'sa' in dlab or 'rmsd' in dlab else None,
                'n': len(vals),
            })
    save_csv(pd.DataFrame(rows), 'ringfix', 'r2_property_deltas.csv')

    # ── Table R3: Before/after scatter stats (R², MAE) ──────────────────────
    scatter_pairs = [
        ('orig_affinity', 'fixed_affinity', 'affinity'),
        ('orig_sa', 'fixed_sa', 'sa_score'),
        ('orig_qed', 'fixed_qed', 'qed'),
        ('orig_logp', 'fixed_logp', 'logp'),
    ]
    rows = []
    for ds_name, ds_data in [('RNA', rna), ('DUDE', dude)]:
        for orig_c, fixed_c, label in scatter_pairs:
            valid = ds_data[[orig_c, fixed_c]].dropna()
            r2 = valid[orig_c].corr(valid[fixed_c]) ** 2 if len(valid) > 1 else np.nan
            mae = (valid[fixed_c] - valid[orig_c]).abs().mean()
            rows.append({
                'dataset': ds_name, 'property': label,
                'R_squared': r2, 'MAE': mae, 'n': len(valid),
            })
    save_csv(pd.DataFrame(rows), 'ringfix', 'r3_scatter_stats.csv')

    # ── Cross-method data (if smrtbench data available) ──────────────────────
    smrt_path = DATA_DIR / 'output' / 'smrtbench_combined_results.csv'
    has_cross = smrt_path.exists()
    if not has_cross:
        print("  SKIP cross-method tables: smrtbench_combined_results.csv not found")
    else:
        smrt = pd.read_csv(smrt_path)
        smrt_pdb_set = set(smrt['pdb_id'].unique())
        SMRT_METHODS = ['RNA2smol (Only General Feat)', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']

        def target_to_pdb(t):
            if '_ligand' in t: return t.split('_')[0]
            elif 'dude_' in t:
                parts = t.split('_'); return '_'.join(parts[2:]) if len(parts) > 2 else parts[-1]
            return None

        full['pdb_id'] = full['target'].apply(target_to_pdb)
        common_pdbs = set(full[full['dataset'] == 'smrtnet']['pdb_id'].unique()) & smrt_pdb_set

        # Build cross-method aggregates
        def agg_our_per_pdb(data, prefix):
            sub = data[data['smi_changed'] == True]
            return sub.groupby('pdb_id').agg(
                affinity=(f'{prefix}_affinity', 'median'),
                rmsd=(f'{prefix}_rmsd', 'median'),
                qed=(f'{prefix}_qed', 'median'),
                sa_score=(f'{prefix}_sa', 'median'),
                logp=(f'{prefix}_logp', 'median'),
            ).reset_index()

        before_agg = agg_our_per_pdb(full[full['dataset'] == 'smrtnet'], 'orig')
        after_agg = agg_our_per_pdb(full[full['dataset'] == 'smrtnet'], 'fixed')
        before_agg = before_agg[before_agg['pdb_id'].isin(common_pdbs)]
        after_agg = after_agg[after_agg['pdb_id'].isin(common_pdbs)]

        smrt_common = smrt[(smrt['pdb_id'].isin(common_pdbs)) & (smrt['mode'].isin(SMRT_METHODS))]
        smrt_per_target = smrt_common.groupby(['mode', 'pdb_id']).agg(
            affinity=('affinity', 'median'),
            rmsd=('rmsd', 'median'),
            qed=('qed', 'median'),
            sa_score=('sa_score', 'median'),
            logp=('logp', 'median'),
        ).reset_index()

        # Build cross-method table: one row per (method, pdb_id)
        cross_rows = []
        for _, row in before_agg.iterrows():
            cross_rows.append({'method': 'Before Fix', 'pdb_id': row['pdb_id'],
                               'affinity': row['affinity'], 'rmsd': row['rmsd'],
                               'qed': row['qed'], 'sa_score': row['sa_score'],
                               'logp': row['logp']})
        for _, row in after_agg.iterrows():
            cross_rows.append({'method': 'After Fix', 'pdb_id': row['pdb_id'],
                               'affinity': row['affinity'], 'rmsd': row['rmsd'],
                               'qed': row['qed'], 'sa_score': row['sa_score'],
                               'logp': row['logp']})
        for _, row in smrt_per_target.iterrows():
            cross_rows.append({'method': row['mode'], 'pdb_id': row['pdb_id'],
                               'affinity': row['affinity'], 'rmsd': row['rmsd'],
                               'qed': row['qed'], 'sa_score': row['sa_score'],
                               'logp': row['logp']})
        cross_df = pd.DataFrame(cross_rows)
        CROSS_METHOD_ORDER = ['Before Fix', 'After Fix', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']

        # ── Table R4: Cross-method per-target mean properties ───────────────
        rows = []
        for method in CROSS_METHOD_ORDER:
            sub = cross_df[cross_df['method'] == method]
            for col in ['affinity', 'rmsd', 'qed', 'sa_score', 'logp']:
                vals = sub[col].dropna()
                rows.append({
                    'method': method, 'property': col,
                    'mean': vals.mean(), 'std': vals.std(),
                    'n_targets': len(vals),
                })
        save_csv(pd.DataFrame(rows), 'ringfix', 'r4_cross_method_summary.csv')

        # ── Table R5: Cross-method per-target detail (wide format) ──────────
        wide_rows = []
        for pdb in sorted(common_pdbs):
            row = {'pdb_id': pdb}
            for method in CROSS_METHOD_ORDER:
                sub = cross_df[(cross_df['method'] == method) & (cross_df['pdb_id'] == pdb)]
                if len(sub) > 0:
                    prefix = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    row[f'{prefix}_affinity'] = sub['affinity'].values[0]
                    row[f'{prefix}_rmsd'] = sub['rmsd'].values[0]
                    row[f'{prefix}_sa_score'] = sub['sa_score'].values[0]
                    row[f'{prefix}_qed'] = sub['qed'].values[0]
                    row[f'{prefix}_logp'] = sub['logp'].values[0]
            wide_rows.append(row)
        save_csv(pd.DataFrame(wide_rows), 'ringfix', 'r5_cross_per_target_detail.csv')

        # ── Table R6: Per-method × per-target heatmap (affinity, rmsd) ──────
        for col in ['affinity', 'rmsd']:
            ht = cross_df[cross_df['method'].isin(CROSS_METHOD_ORDER)]
            pivot = ht.pivot(index='method', columns='pdb_id', values=col)
            pivot = pivot.reindex(CROSS_METHOD_ORDER)
            pivot.index.name = 'method'
            save_csv(pivot.reset_index(), 'ringfix', f'r6_heatmap_{col}.csv')

        # ── Table R7: Summary statistics text data ──────────────────────────
        rows = [
            {'stat': 'total_binders_evaluated', 'value': len(full)},
            {'stat': 'actually_fixed', 'value': len(changed)},
            {'stat': 'fix_pct', 'value': len(changed) / len(full) * 100},
            {'stat': 'rna_total', 'value': len(rna_all)},
            {'stat': 'rna_fixed', 'value': len(rna)},
            {'stat': 'rna_fix_pct', 'value': len(rna) / len(rna_all) * 100},
            {'stat': 'rna_mean_delta_affinity', 'value': rna['daffinity'].mean()},
            {'stat': 'rna_mean_delta_sa', 'value': rna['dsa'].mean()},
            {'stat': 'rna_pct_better_affinity', 'value': (rna['daffinity'].dropna() < 0).mean() * 100},
            {'stat': 'rna_pct_better_sa', 'value': (rna['dsa'].dropna() < 0).mean() * 100},
            {'stat': 'dude_total', 'value': len(dude_all)},
            {'stat': 'dude_fixed', 'value': len(dude)},
            {'stat': 'dude_fix_pct', 'value': len(dude) / len(dude_all) * 100},
            {'stat': 'dude_mean_delta_affinity', 'value': dude['daffinity'].mean()},
            {'stat': 'dude_mean_delta_sa', 'value': dude['dsa'].mean()},
            {'stat': 'dude_pct_better_affinity', 'value': (dude['daffinity'].dropna() < 0).mean() * 100},
            {'stat': 'dude_pct_better_sa', 'value': (dude['dsa'].dropna() < 0).mean() * 100},
        ]
        save_csv(pd.DataFrame(rows), 'ringfix', 'r7_summary_stats.csv')

        # ── Table R8: SA score per target (top 30 improved) ──────────────────
        per_target_sa = changed[changed['dataset'] == 'smrtnet'].groupby('target').agg(
            orig_sa=('orig_sa', 'median'),
            fixed_sa=('fixed_sa', 'median'),
            dsa=('dsa', 'mean'),
            n=('smi_changed', 'count'),
        ).reset_index()
        per_target_sa['pdb_id'] = per_target_sa['target'].apply(
            lambda t: t.split('_')[0] if '_ligand' in t else t)
        per_target_sa = per_target_sa.sort_values('dsa')
        if has_cross:
            mc_sa = smrt_common[smrt_common['mode'] == 'MolCRAFT'].groupby('pdb_id')['sa_score'].median()
            per_target_sa['molcraft_sa'] = per_target_sa['pdb_id'].map(mc_sa)
        save_csv(per_target_sa, 'ringfix', 'r8_sa_per_target.csv')

        # ── Table R9: Virtual screening funnel (before/after fix) ────────────
        rna_full_data = full[full['dataset'] == 'smrtnet'].copy()
        rna_full_data = rna_full_data[rna_full_data['pdb_id'].isin(common_pdbs)]

        funnel_rows = []
        for _, row in rna_full_data.iterrows():
            funnel_rows.append({'method': 'Before Fix', 'pdb_id': row['pdb_id'],
                               'affinity': row['orig_affinity'], 'rmsd': row['orig_rmsd'],
                               'qed': row['orig_qed'], 'sa_score': row['orig_sa']})
        for _, row in rna_full_data.iterrows():
            funnel_rows.append({'method': 'After Fix', 'pdb_id': row['pdb_id'],
                               'affinity': row['fixed_affinity'], 'rmsd': row['fixed_rmsd'],
                               'qed': row['fixed_qed'], 'sa_score': row['fixed_sa']})
        smrt_funnel = smrt[smrt['pdb_id'].isin(common_pdbs)]
        SMRT_FUNNEL_METHODS = ['MolCRAFT', 'Pocket2Mol', 'ED2Mol']
        for _, row in smrt_funnel[smrt_funnel['mode'].isin(SMRT_FUNNEL_METHODS)].iterrows():
            funnel_rows.append({'method': row['mode'], 'pdb_id': row['pdb_id'],
                               'affinity': row['affinity'], 'rmsd': row['rmsd'],
                               'qed': row['qed'], 'sa_score': row['sa_score']})
        funnel_df = pd.DataFrame(funnel_rows)
        funnel_methods = ['Before Fix', 'After Fix', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']

        filter_steps_fix = [
            ('All', None),
            ('aff<-6', lambda d: d['affinity'] < -6),
            ('aff<-7', lambda d: d['affinity'] < -7),
            ('+RMSD<2.5', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.5)),
            ('RMSD<2.0', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0)),
            ('+QED>0.3', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.3)),
            ('QED>0.4,+SA<5', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
            ('aff<-8,RMSD<1.5,QED>0.5', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
            ('+SA<4', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 4)),
            ('aff<-9,RMSD<1.0,QED>0.6', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4)),
            ('QED>0.7,SA<3', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.7) & (d['sa_score'] < 3)),
        ]
        rows = []
        for method in funnel_methods:
            sub = funnel_df[funnel_df['method'] == method]
            for step_name, flt in filter_steps_fix:
                filtered = sub if flt is None else sub[flt(sub)]
                n_targets_5 = (filtered.groupby('pdb_id').size() >= 5).sum()
                rows.append({
                    'method': method, 'filter_step': step_name,
                    'mols_retained': len(filtered),
                    'targets_ge5': n_targets_5,
                })
        save_csv(pd.DataFrame(rows), 'ringfix', 'r9_virtual_screening_funnel.csv')

        # ── Table R10: Top-N ECFP4 enrichment ────────────────────────────────
        # Try to load from cache
        ecfp4_cache = DATA_DIR / 'output' / 'cache_ecfp4_enrich.csv'
        if ecfp4_cache.exists():
            ecfp4_df = pd.read_csv(ecfp4_cache)
            enrich_rows = []
            for _, row in ecfp4_df.iterrows():
                if not pd.isna(row['orig_ecfp4']):
                    enrich_rows.append({'method': 'Before Fix', 'pdb_id': row['pdb_id'],
                                       'ecfp4': row['orig_ecfp4']})
                if not pd.isna(row['fixed_ecfp4']):
                    enrich_rows.append({'method': 'After Fix', 'pdb_id': row['pdb_id'],
                                       'ecfp4': row['fixed_ecfp4']})
            for smrt_mode in ['MolCRAFT', 'Pocket2Mol', 'ED2Mol']:
                sub = smrt[(smrt['mode'] == smrt_mode) & (smrt['pdb_id'].isin(common_pdbs))]
                for _, row in sub.iterrows():
                    if not pd.isna(row['max_ecfp4_tanimoto']):
                        enrich_rows.append({'method': smrt_mode, 'pdb_id': row['pdb_id'],
                                           'ecfp4': row['max_ecfp4_tanimoto']})
            enrich_df = pd.DataFrame(enrich_rows)

            TOP_N = [1, 3, 5, 10, 20, 50, 100]
            enrich_methods = ['Before Fix', 'After Fix', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']
            per_pdb = {}
            for method in enrich_methods:
                sub = enrich_df[enrich_df['method'] == method]
                for pdb_id, grp in sub.groupby('pdb_id'):
                    vals_sorted = grp['ecfp4'].sort_values(ascending=False).values
                    key = (method, pdb_id)
                    per_pdb[key] = {}
                    for n in TOP_N:
                        top_n = vals_sorted[:min(n, len(vals_sorted))]
                        per_pdb[key][f'top{n}'] = np.median(top_n)

            rows = []
            for method in enrich_methods:
                for n in TOP_N:
                    vals = [per_pdb[(method, p)][f'top{n}'] for p in common_pdbs
                            if (method, p) in per_pdb]
                    if vals:
                        rows.append({
                            'method': method, 'top_n': n,
                            'mean': np.mean(vals), 'std': np.std(vals),
                            'n_targets': len(vals),
                        })
            save_csv(pd.DataFrame(rows), 'ringfix', 'r10_ecfp4_topn_enrichment.csv')
        else:
            print("  SKIP ECFP4 enrichment: cache not found (run visualize_fix.py first)")

        # ── Table R11: Radar profile raw values ──────────────────────────────
        radar_methods_fix = ['Before Fix', 'After Fix', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']

        # Build per-molecule radar data (simplified: from cross_df aggregates)
        radar_profile = {}
        for method in radar_methods_fix:
            sub = cross_df[cross_df['method'] == method]
            radar_profile[method] = {
                'affinity': sub['affinity'].mean(),
                'rmsd': sub['rmsd'].mean(),
                'sa_score': sub['sa_score'].mean(),
                'qed': sub['qed'].mean(),
                'logp': sub['logp'].mean(),
            }
        # Also compute SEI, LLE, LBE from cross_df
        for method in radar_methods_fix:
            sub = cross_df[cross_df['method'] == method]
            # need tpsa and heavy_atoms — approximate from smrt data
            method_smrt = method if method not in ('Before Fix', 'After Fix') else 'RNA2smol (Only General Feat)'
            smrt_method_data = smrt[(smrt['mode'] == method_smrt) & (smrt['pdb_id'].isin(common_pdbs))]
            if len(smrt_method_data) > 0:
                avg_tpsa = smrt_method_data['tpsa'].median()
                avg_heavy = smrt_method_data['heavy_atoms'].median()
            else:
                avg_tpsa = avg_heavy = np.nan
            radar_profile[method]['SEI'] = -sub['affinity'].mean() / avg_tpsa if avg_tpsa and avg_tpsa > 0 else np.nan
            radar_profile[method]['LLE'] = -sub['affinity'].mean() - sub['logp'].mean()
            radar_profile[method]['LBE'] = sub['affinity'].mean() / avg_heavy if avg_heavy and avg_heavy > 0 else np.nan

        # Bond JS (try cache)
        bond_cache_fix = DATA_DIR / 'output' / 'cache_bond_js_fix.pkl'
        if bond_cache_fix.exists():
            import pickle
            with open(bond_cache_fix, 'rb') as f:
                bond_js_fix = pickle.load(f)
            for method in radar_methods_fix:
                if method in bond_js_fix:
                    radar_profile[method].update(bond_js_fix[method])

        rows = []
        for method in radar_methods_fix:
            for metric, val in radar_profile[method].items():
                rows.append({
                    'method': method, 'metric': metric, 'value': val if not (isinstance(val, float) and np.isnan(val)) else None,
                })
        save_csv(pd.DataFrame(rows), 'ringfix', 'r11_radar_profile_raw.csv')

    print("  Ring Fix tables done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DUD-E tables
# ══════════════════════════════════════════════════════════════════════════════
def export_dude():
    print("=" * 60)
    print("Exporting DUDE tables...")
    print("=" * 60)

    csv_path = DATA_DIR / 'dude_stat_all.csv'
    if not csv_path.exists():
        print("  SKIP: dude_stat_all.csv not found")
        return
    df = pd.read_csv(csv_path)
    df['SEI'] = -df['affinity'] / df['tpsa'].replace(0, np.nan)
    df['SEI'] = df['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
    df['LLE'] = -df['affinity'] - df['logp']
    df['LBE'] = df['affinity'] / df['heavy_atoms'].replace(0, np.nan)
    df['mw_bin'] = pd.cut(df['mw'], bins=[0, 120, 180, 300, 420, 2000],
                           labels=['0–120', '120-180', '180–300', '300-420', '≥420'])

    METHODS = ['RNA2smol', 'MolCRAFT', 'MolCRAFT (Original)', 'Pocket2Mol', 'ed2mol']
    d = df[df['mode'].isin(METHODS)]

    # ── Table D1: Main comparison — per-metric mean±std across targets ──────
    plot_metrics = [c for c in ['affinity', 'SEI', 'LLE', 'rmsd', 'LBE', 'max_ecfp4_tanimoto', 'strain'] if c in d.columns]
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        for col in plot_metrics:
            pt = sub.groupby('pdb_id')[col].median()
            rows.append({
                'method': method, 'metric': col,
                'mean': pt.mean(), 'std': pt.std(),
                'n_targets': len(pt), 'n_mols': len(sub),
            })
    save_csv(pd.DataFrame(rows), 'dude', 'd1_main_comparison.csv')

    # ── Table D2: ECFP4 target-level recovery at multiple τ ────────────────
    TAU_VALS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        for tau in TAU_VALS:
            per_target_hits = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: int((x >= t).sum()))
            per_target = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: (x >= t).mean())
            rows.append({
                'method': method, 'tau': tau,
                'successful_targets_ge1_molecule': int((per_target_hits >= 1).sum()),
                'successful_targets_ge5_molecules': int((per_target_hits >= 5).sum()),
                'total_evaluable_targets': int(per_target_hits.shape[0]),
                'hit_mols_total': int((sub['max_ecfp4_tanimoto'] >= tau).sum()),
                'hit_rate_mean': per_target.mean(),
                'hit_rate_std': per_target.std(),
            })
    d2 = pd.DataFrame(rows)
    save_csv(d2, 'dude', 'd2_ecfp4_target_success_by_threshold.csv')
    save_csv(d2, 'dude', 'd2_ecfp4_hitrate.csv')

    # ── Table D3: ECFP4 Top-N median ────────────────────────────────────────
    TOP_N = [1, 3, 5, 10, 20, 50, 100]
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        for n in TOP_N:
            per_target = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, n=n: x.nlargest(min(n, len(x))).median())
            rows.append({
                'method': method, 'top_n': n,
                'median_mean': per_target.mean(),
                'median_std': per_target.std(),
            })
    save_csv(pd.DataFrame(rows), 'dude', 'd3_ecfp4_topn.csv')

    # ── Table D4: Per-method × per-target heatmap data ──────────────────────
    for col in [c for c in ['affinity', 'rmsd', 'SEI', 'strain'] if c in d.columns]:
        pivot = d.groupby(['mode', 'pdb_id'])[col].median().reset_index()
        ht = pivot.pivot(index='mode', columns='pdb_id', values=col)
        ht = ht.reindex(METHODS)
        ht.index.name = 'method'
        save_csv(ht.reset_index(), 'dude', f'd4_heatmap_{col}.csv')

    # ── Table D5: Property distribution summary stats ────────────────────────
    prop_cols = ['qed', 'logp', 'tpsa', 'sa_score']
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        for col in prop_cols:
            vals = sub[col].dropna()
            rows.append({
                'method': method, 'property': col,
                'mean': vals.mean(), 'std': vals.std(),
                'median': vals.median(), 'q1': vals.quantile(0.25),
                'q3': vals.quantile(0.75), 'n': len(vals),
            })
    save_csv(pd.DataFrame(rows), 'dude', 'd5_property_stats.csv')

    # ── Table D6: MW-binned performance ─────────────────────────────────────
    mw_bins = ['0–120', '120-180', '180–300', '300-420', '≥420']
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        for mb in mw_bins:
            mb_data = sub[sub['mw_bin'] == mb]
            if len(mb_data) == 0:
                continue
            for col in [c for c in ['affinity', 'SEI', 'strain'] if c in mb_data.columns]:
                per_target = mb_data.groupby('pdb_id')[col].median()
                rows.append({
                    'method': method, 'mw_bin': mb, 'metric': col,
                    'mean': per_target.mean(), 'std': per_target.std(),
                    'n_targets': len(per_target), 'n_mols': len(mb_data),
                })
    save_csv(pd.DataFrame(rows), 'dude', 'd6_mw_binned.csv')

    # ── Table D7: Virtual screening funnel ───────────────────────────────────
    # Same strain-aware funnel as SMRT-Net (no RMSD dependence in the default
    # manuscript flow); the RMSD-based flow is retained below as an audit table.
    strain_aware_filter_steps = [
        ('All', None),
        ('aff<-6', lambda d_: d_['affinity'] < -6),
        ('aff<-7', lambda d_: d_['affinity'] < -7),
        ('+QED>0.3', lambda d_: (d_['affinity'] < -7) & (d_['qed'] > 0.3)),
        ('QED>0.4,+SA<5', lambda d_: (d_['affinity'] < -7) & (d_['qed'] > 0.4) & (d_['sa_score'] < 5)),
        ('aff<-8,QED>0.5,SA<5', lambda d_: (d_['affinity'] < -8) & (d_['qed'] > 0.5) & (d_['sa_score'] < 5)),
        ('+SEI>0.1', lambda d_: (d_['affinity'] < -8) & (d_['qed'] > 0.5) & (d_['sa_score'] < 5) & (d_['SEI'] > 0.1)),
        ('aff<-9,QED>0.5,SA<5,SEI>0.15', lambda d_: (d_['affinity'] < -9) & (d_['qed'] > 0.5) & (d_['sa_score'] < 5) & (d_['SEI'] > 0.15)),
        ('aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200',
         lambda d_: (d_['affinity'] < -9) & (d_['qed'] > 0.6) & (d_['sa_score'] < 4) & (d_['SEI'] > 0.2) & (d_['strain'] < 200)),
    ]
    rmsd_reference_filter_steps = [
        ('All', None),
        ('aff<-6', lambda d_: d_['affinity'] < -6),
        ('aff<-7', lambda d_: d_['affinity'] < -7),
        ('+RMSD<2.5', lambda d_: (d_['affinity'] < -7) & (d_['rmsd'] < 2.5)),
        ('RMSD<2.0', lambda d_: (d_['affinity'] < -7) & (d_['rmsd'] < 2.0)),
        ('+QED>0.3', lambda d_: (d_['affinity'] < -7) & (d_['rmsd'] < 2.0) & (d_['qed'] > 0.3)),
        ('QED>0.4,+SA<5', lambda d_: (d_['affinity'] < -7) & (d_['rmsd'] < 2.0) & (d_['qed'] > 0.4) & (d_['sa_score'] < 5)),
        ('aff<-8,RMSD<1.5,QED>0.5,SA<5', lambda d_: (d_['affinity'] < -8) & (d_['rmsd'] < 1.5) & (d_['qed'] > 0.5) & (d_['sa_score'] < 5)),
        ('+SEI>0.1', lambda d_: (d_['affinity'] < -8) & (d_['rmsd'] < 1.5) & (d_['qed'] > 0.5) & (d_['sa_score'] < 5) & (d_['SEI'] > 0.1)),
        ('aff<-9,RMSD<1.0,QED>0.5,SA<5,SEI>0.15', lambda d_: (d_['affinity'] < -9) & (d_['rmsd'] < 1.0) & (d_['qed'] > 0.5) & (d_['sa_score'] < 5) & (d_['SEI'] > 0.15)),
        ('aff<-9,RMSD<1.0,QED>0.6,SA<4,SEI>0.2', lambda d_: (d_['affinity'] < -9) & (d_['rmsd'] < 1.0) & (d_['qed'] > 0.6) & (d_['sa_score'] < 4) & (d_['SEI'] > 0.2)),
    ]

    def dude_virtual_screening_rows(filter_steps, flow_name):
        rows = []
        for method in METHODS:
            sub = d[d['mode'] == method]
            for step_name, flt in filter_steps:
                filtered = sub if flt is None else sub[flt(sub)]
                per_target_counts = filtered.groupby('pdb_id').size()
                rows.append({
                    'flow': flow_name,
                    'method': method,
                    'filter_step': step_name,
                    'mols_retained': len(filtered),
                    'targets_ge1': int((per_target_counts >= 1).sum()),
                    'targets_ge5': int((per_target_counts >= 5).sum()),
                })
        return pd.DataFrame(rows)

    save_csv(
        dude_virtual_screening_rows(strain_aware_filter_steps, 'strain_aware'),
        'dude',
        'd7_virtual_screening_funnel.csv',
    )
    save_csv(
        dude_virtual_screening_rows(rmsd_reference_filter_steps, 'rmsd_reference'),
        'dude',
        'd7b_virtual_screening_funnel_rmsd_reference.csv',
    )

    # ── Table D8: Radar profile raw values ───────────────────────────────────
    radar_metrics = ['affinity', 'SEI', 'LLE', 'LBE', 'rmsd',
                     'qed', 'sa_score', 'rotatable_bonds',
                     'max_ecfp4_tanimoto', 'strain']
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        for col in radar_metrics:
            if col in sub.columns:
                pt = sub.groupby('pdb_id')[col].median()
                rows.append({
                    'method': method, 'metric': col,
                    'mean': pt.mean(), 'std': pt.std(), 'n_targets': len(pt),
                })
    # Bond JS divergence (try cache)
    import pickle
    bond_cache = DATA_DIR / 'output' / 'cache_dude_bond_js.pkl'
    RADAR_DIRS = {
        'affinity': 'lower', 'SEI': 'higher', 'LLE': 'higher', 'LBE': 'lower',
        'rmsd': 'lower', 'qed': 'higher', 'sa_score': 'lower',
        'rotatable_bonds': 'lower', 'max_ecfp4_tanimoto': 'higher',
        'bond_len_js': 'lower', 'bond_angle_js': 'lower', 'strain': 'lower',
    }
    RADAR_FLOOR = 0.25
    if bond_cache.exists():
        with open(bond_cache, 'rb') as f:
            bond_js = pickle.load(f)
        profile = {}
        for method in METHODS:
            profile[method] = {}
            sub = d[d['mode'] == method]
            for col in radar_metrics:
                if col in sub.columns:
                    profile[method][col] = sub.groupby('pdb_id')[col].median().mean()
            if method in bond_js:
                profile[method].update(bond_js[method])
                for k, v in bond_js[method].items():
                    rows.append({'method': method, 'metric': k,
                                 'mean': v, 'std': 0, 'n_targets': 0})

        # ── Table D8b: Radar rank normalized ───────────────────────────────
        radar_metrics_all = radar_metrics + ['bond_len_js', 'bond_angle_js']
        rank_rows, mm_rows = [], []
        for method in METHODS:
            rank_row = {'method': method}
            mm_row = {'method': method}
            for col in radar_metrics_all:
                vals = {m: profile[m].get(col, np.nan) for m in METHODS}
                valid = {k: v for k, v in vals.items() if not np.isnan(v)}
                rank_row[f'{col}_raw'] = profile[method].get(col, np.nan)
                mm_row[f'{col}_raw'] = profile[method].get(col, np.nan)
                if len(valid) < 2:
                    rank_row[f'{col}_rank_norm'] = (1.0 + RADAR_FLOOR) / 2
                    mm_row[f'{col}_minmax_norm'] = (1.0 + RADAR_FLOOR) / 2
                    continue
                ascending = (RADAR_DIRS[col] == 'lower')
                sorted_m = sorted(valid, key=valid.get, reverse=not ascending)
                r = sorted_m.index(method) + 1
                rank_row[f'{col}_rank_norm'] = 1.0 - (r - 1) * (1.0 - RADAR_FLOOR) / (len(sorted_m) - 1)
                min_v, max_v = min(valid.values()), max(valid.values())
                v = profile[method].get(col, np.nan)
                if np.isnan(v) or max_v - min_v < 1e-8:
                    mm_row[f'{col}_minmax_norm'] = (1.0 + RADAR_FLOOR) / 2
                elif RADAR_DIRS[col] == 'higher':
                    mm_row[f'{col}_minmax_norm'] = RADAR_FLOOR + (1.0 - RADAR_FLOOR) * (v - min_v) / (max_v - min_v)
                else:
                    mm_row[f'{col}_minmax_norm'] = RADAR_FLOOR + (1.0 - RADAR_FLOOR) * (1.0 - (v - min_v) / (max_v - min_v))
            rank_rows.append(rank_row)
            mm_rows.append(mm_row)
        save_csv(pd.DataFrame(rank_rows), 'dude', 'd8b_radar_rank_normalized.csv')
        save_csv(pd.DataFrame(mm_rows), 'dude', 'd8c_radar_minmax_normalized.csv')

    save_csv(pd.DataFrame(rows), 'dude', 'd8a_radar_profile_raw.csv')

    # ── Table D9: Rotatable bonds pass rate ──────────────────────────────────
    rows = []
    for method in METHODS:
        sub = d[d['mode'] == method]
        per_target = sub.groupby('pdb_id').apply(
            lambda x: (x['rotatable_bonds'] <= 7).mean(), include_groups=False)
        rows.append({
            'method': method,
            'rotb_pass_rate_mean': per_target.mean(),
            'rotb_pass_rate_std': per_target.std(),
        })
    save_csv(pd.DataFrame(rows), 'dude', 'd9_rotatable_bonds_pass_rate.csv')

    print("  DUDE tables done.\n")


# ══════════════════════════════════════════════════════════════════════════════

# ── Completeness gate ───────────────────────────────────────────────────────
# Several blocks above are guarded by `if <cache>.exists():`. When a cache is
# missing the block is skipped silently, which yields a PARTIAL table set that
# only fails later inside a figure script (d8b/d8c missing breaks
# fig2_cross_benchmark.py). Fail loudly here instead.
REQUIRED_TABLES = [
    'smrtbench/t1_main_comparison.csv',
    'smrtbench/t1b_ecfp4_target_success_by_threshold.csv',
    'smrtbench/t1d_per_target_paired_stats.csv',
    'smrtbench/t2_ablation_comparison.csv',
    'smrtbench/t2b_ablation_paired_stats.csv',
    'smrtbench/t6_virtual_screening_funnel.csv',
    'smrtbench/t7a_radar_profile_raw.csv',
    'smrtbench/t7b_radar_rank_normalized.csv',
    'smrtbench/t8_ablation_radar_profile.csv',
    'dude/d1_main_comparison.csv',
    'dude/d2_ecfp4_target_success_by_threshold.csv',
    'dude/d7_virtual_screening_funnel.csv',
    'dude/d8a_radar_profile_raw.csv',
    'dude/d8b_radar_rank_normalized.csv',
    'dude/d8c_radar_minmax_normalized.csv',
]


def check_completeness() -> int:
    missing = [t for t in REQUIRED_TABLES if not (OUT_DIR / t).exists()]
    if missing:
        print("\n" + "=" * 68)
        print(f"INCOMPLETE: {len(missing)} expected table(s) were not produced.")
        print("A guarded block was skipped, most likely a missing cache:")
        for m in missing:
            print(f"  missing {m}")
        print("\nProducers (see analysis/README.md):")
        print("  data/benchmark/output/cache_bond_js.pkl       <- analysis/prepare_bond_js.py")
        print("  data/benchmark/output/cache_dude_bond_js.pkl  <- analysis/prepare_bond_js_dude.py")
        print("  or run: python3 reproduction/scripts/build_bond_js_caches.py")
        print("=" * 68)
        return 2
    print(f"\nCompleteness check: all {len(REQUIRED_TABLES)} expected tables present.")
    return 0


if __name__ == '__main__':
    # Clear previous output first. Without this, a skipped guarded block leaves the
    # PREVIOUS run's file in place, the completeness check passes on stale data, and
    # the problem only surfaces later inside a figure script. Everything under
    # OUT_DIR is derived, so removing it is safe.
    import shutil
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print(f"cleared {OUT_DIR}")

    export_smrtbench()
    export_ringfix()
    export_dude()
    print(f"\nAll tables saved to: {OUT_DIR}")
    raise SystemExit(check_completeness())
