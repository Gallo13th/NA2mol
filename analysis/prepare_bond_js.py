"""
7-figure visualization for comparing molecular generation methods on SMRTBench.
Output: PNG files in data/benchmark/output/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# ── Setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
# Data lives under data/, not next to this script.
OUT_DIR = ROOT / 'data' / 'benchmark' / 'output'
OUT_DIR.mkdir(exist_ok=True)
SMRT_FIG_DIR = OUT_DIR / 'smrtbench'
SMRT_FIG_DIR.mkdir(exist_ok=True)
FONT_PATH = Path(__file__).resolve().parent.parent / 'NotoSansCJKsc-Regular.otf'

# Nature-style global rcParams
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica', 'Noto Sans CJK SC'],
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
})

if FONT_PATH.exists():
    from matplotlib.font_manager import fontManager
    fontManager.addfont(str(FONT_PATH))
    # Prepend CJK font so Chinese characters render correctly
    _cjk = 'Noto Sans CJK SC'
    if _cjk not in plt.rcParams['font.sans-serif']:
        plt.rcParams['font.sans-serif'] = [_cjk] + plt.rcParams['font.sans-serif']

def _panel_label(ax, label, x=-0.06, y=1.02):
    """Add bold panel label (a), (b), etc. in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left')

# ── Method groups & colors ───────────────────────────────────────────────
MAIN_METHODS = [
    'RNA2smol (Only General Feat)',
    'RNA2smol (Baseline)',
    'MolCRAFT',
    'Pocket2Mol',
    'ED2Mol',
]
ABLATION_METHODS = [
    ' Allfeat Cst',
    ' Allfeat Nocst',
    ' Generalfeat Cst',
    ' Generalfeat Nocst',
]
ALL_METHODS = MAIN_METHODS + ABLATION_METHODS
METHOD_ORDER_MAIN = MAIN_METHODS

# Nature Wong colorblind-friendly palette
MAIN_COLORS = {
    'RNA2smol (Only General Feat)':     '#D55E00',  # vermilion (full model, most prominent)
    'RNA2smol (Baseline)':              '#E69F00',  # orange (ablated baseline)
    'MolCRAFT':                          '#0072B2',  # blue
    'Pocket2Mol':                        '#009E73',  # bluish green
    'ED2Mol':                            '#CC79A7',  # purple
}
# Ablation: progressive grayscale (darker = more complete features)
ABLATION_COLORS = {
    ' Allfeat Cst':       '#666666',
    ' Allfeat Nocst':     '#888888',
    ' Generalfeat Cst':   '#AAAAAA',
    ' Generalfeat Nocst': '#CCCCCC',
}
METHOD_COLORS = {**MAIN_COLORS, **ABLATION_COLORS}

# Display names: Only General Feat = complete model → just "RNA2smol"
METHOD_DISPLAY = {
    'RNA2smol (Baseline)':          'RNA2smol (Baseline)',
    'RNA2smol (Only General Feat)': 'RNA2smol',
    'MolCRAFT':                     'MolCRAFT',
    'Pocket2Mol':                   'Pocket2Mol',
    'ED2Mol':                       'ED2Mol',
}
def _short_name(m):
    return METHOD_DISPLAY.get(m, m)

ABLATION_LINESTYLES = {
    ' Allfeat Cst':       '-',
    ' Allfeat Nocst':     '--',
    ' Generalfeat Cst':   '-.',
    ' Generalfeat Nocst': ':',
}

# ── Metrics metadata ─────────────────────────────────────────────────────
# (col, label, direction: 'higher'=better when larger, 'lower'=better when smaller)
METRICS_LIST = [
    ('affinity',           'Affinity (kcal/mol)',  'lower'),
    ('SEI',                'SEI',                   'higher'),
    ('LLE',                'LLE',                   'higher'),
    ('rmsd',               'RMSD (Å)',              'lower'),
    ('LBE',                'LBE',                   'lower'),
    ('max_ecfp4_tanimoto', 'Max ECFP4 Tanimoto',    'higher'),
    ('qed',                'QED',                   'higher'),
    ('sa_score',           'SA Score',              'lower'),
]
METRIC_LABEL = {col: label for col, label, _ in METRICS_LIST}
METRIC_DIRECTION = {col: d for col, _, d in METRICS_LIST}

SCORE_METRICS = ['affinity', 'SEI', 'LLE', 'rmsd', 'LBE', 'max_ecfp4_tanimoto']
PROP_METRICS = ['qed', 'logp', 'tpsa', 'sa_score']

# ── Data loading ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(OUT_DIR / 'smrtbench_combined_results.csv')
df = df[df['mode'].isin(ALL_METHODS)]
# Recompute SEI and LBE for safety
df['SEI'] = -df['affinity'] / df['tpsa'].replace(0, np.nan)
df['SEI'] = df['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
df['LLE'] = -df['affinity'] - df['logp']
df['LBE'] = df['affinity'] / df['heavy_atoms'].replace(0, np.nan)
print(f"Loaded {len(df)} rows, {df['mode'].nunique()} methods, {df['pdb_id'].nunique()} targets.")

# Aggregate: median per (mode, pdb_id), then aggregate across targets
def agg_across_targets(data, col, agg='median'):
    """Median per target, then mean across targets."""
    per_target = data.groupby(['mode', 'pdb_id'])[col].median().reset_index()
    across = per_target.groupby('mode')[col].agg(['mean', 'std']).reset_index()
    across.columns = ['mode', f'{col}_mean', f'{col}_std']
    return across

# ── Helper: save figure ──────────────────────────────────────────────────
def savefig(name, tight=True):
    path = OUT_DIR / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(SMRT_FIG_DIR / name, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Main comparison — grouped bar chart
# ══════════════════════════════════════════════════════════════════════════
def fig1_main_barchart():
    print("\n[Fig 1] Main comparison bar chart")
    main_df = df[df['mode'].isin(MAIN_METHODS)]

    plot_metrics = ['affinity', 'SEI', 'LLE', 'rmsd', 'LBE', 'max_ecfp4_tanimoto']
    all_agg = []
    for col in plot_metrics:
        agg = agg_across_targets(main_df, col)
        agg['metric'] = col
        all_agg.append(agg)
    combined = pd.concat(all_agg, ignore_index=True)

    n_methods = len(MAIN_METHODS)
    width = 0.16
    offsets = np.linspace(-(n_methods-1)*width/2, (n_methods-1)*width/2, n_methods)

    def _draw_panel(panel_metrics, filename):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(bottom=0.22)
        vals_all, errs_all = [], []
        for i, method in enumerate(MAIN_METHODS):
            vals, errs = [], []
            for col in panel_metrics:
                row = combined[(combined['mode'] == method) & (combined['metric'] == col)]
                v = row[f'{col}_mean'].values[0] if len(row) > 0 else np.nan
                e = row[f'{col}_std'].values[0] if len(row) > 0 else np.nan
                vals.append(v)
                errs.append(e)
            vals_all.extend(vals)
            errs_all.extend(errs)
            ax.bar(np.arange(len(panel_metrics)) + offsets[i], vals, width,
                   yerr=errs, color=MAIN_COLORS[method],
                   capsize=2, label=_short_name(method), edgecolor='white', linewidth=0.3,
                   error_kw={'linewidth': 0.8})
        ax.set_xticks(np.arange(len(panel_metrics)))
        ax.set_xticklabels([METRIC_LABEL[c] for c in panel_metrics])
        ax.axhline(y=0, color='black', linewidth=0.5)
        a_min = min(v - e for v, e in zip(vals_all, errs_all) if not np.isnan(v))
        a_max = max(v + e for v, e in zip(vals_all, errs_all) if not np.isnan(v))
        pa_pad = (a_max - a_min) * 0.12
        ax.set_ylim(a_min - pa_pad, a_max + pa_pad)
        ax.legend(frameon=True, edgecolor='#cccccc', fontsize=9)
        savefig(filename)

    _draw_panel(['affinity', 'SEI', 'LLE'], 'fig1a_main_docking.png')
    _draw_panel(['rmsd', 'LBE', 'max_ecfp4_tanimoto'], 'fig1b_main_other.png')

# ══════════════════════════════════════════════════════════════════════════
# Figure 1b: ECFP4 Tanimoto similarity — thresholded recovery & top-N analysis
# ══════════════════════════════════════════════════════════════════════════
def fig1b_ecfp4_ts():
    print("\n[Fig 1b] ECFP4 Tanimoto similarity: thresholded recovery & top-N median")
    main_df = df[df['mode'].isin(MAIN_METHODS)]

    TAU_VALS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    hit_rate_data = {tau: {} for tau in TAU_VALS}
    hit_rate_errs = {tau: {} for tau in TAU_VALS}
    hit_count_data = {tau: {} for tau in TAU_VALS}
    hit_target_data = {tau: {} for tau in TAU_VALS}
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for tau in TAU_VALS:
            hits = sub['max_ecfp4_tanimoto'] >= tau
            per_target_hits = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: int((x >= t).sum()))
            per_target = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: (x >= t).mean())
            hit_count_data[tau][method] = int(hits.sum())
            hit_target_data[tau][method] = int((per_target_hits >= 1).sum())
            hit_rate_data[tau][method] = per_target.mean()
            hit_rate_errs[tau][method] = per_target.std()

    TOP_N = [10, 50, 100]
    topn_data = {n: {} for n in TOP_N}
    topn_errs = {n: {} for n in TOP_N}
    for method in MAIN_METHODS:
        sub = main_df[main_df['mode'] == method]
        for n in TOP_N:
            per_target = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x: x.nlargest(min(n, len(x))).median())
            topn_data[n][method] = per_target.mean()
            topn_errs[n][method] = per_target.std()

    width = 0.16
    offsets = np.linspace(-(len(MAIN_METHODS)-1)*width/2, (len(MAIN_METHODS)-1)*width/2, len(MAIN_METHODS))

    # Panel A: target-level thresholded recovery across ECFP4 thresholds.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(TAU_VALS))
    for method in MAIN_METHODS:
        vals = [hit_target_data[tau][method] for tau in TAU_VALS]
        ax.plot(x, vals, color=MAIN_COLORS[method],
                linewidth=2.5, marker='o', markersize=7,
                label=_short_name(method))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in TAU_VALS])
    ax.set_xlabel('ECFP4 Tanimoto threshold')
    ax.set_ylabel('Targets with ≥1 recovered molecule')
    ax.set_ylim(0, main_df['pdb_id'].nunique() + 1)
    ax.grid(alpha=0.15, linestyle='--')
    ax.legend(frameon=True, edgecolor='#cccccc', fontsize=8, ncol=2)
    savefig('fig1c_ecfp4_target_success.png')

    # Supplementary diagnostic: molecule-hit totals at the primary tau.
    primary_tau = 0.30
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = MAIN_METHODS
    x_counts = np.arange(len(methods))
    counts = [hit_count_data[primary_tau][m] for m in methods]
    targets = [hit_target_data[primary_tau][m] for m in methods]
    bars = ax.bar(x_counts, counts, width=0.65,
                  color=[MAIN_COLORS[m] for m in methods],
                  edgecolor='white', linewidth=0.4)
    ax.set_xticks(x_counts)
    ax.set_xticklabels([_short_name(m) for m in methods], rotation=15, ha='right')
    ax.set_ylabel(f'Molecules with ECFP4 ≥ {primary_tau:.2f}')
    ax.set_ylim(0, max(counts) * 1.22 if counts else 1)
    ax.grid(axis='y', alpha=0.15, linestyle='--')
    for bar, n_tgt in zip(bars, targets):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + max(counts) * 0.03,
                f'{int(y)}\n{n_tgt} targets',
                ha='center', va='bottom', fontsize=8)
    savefig('fig1c_ecfp4_mol_hitcounts.png')

    # Legacy panel: hit rate at multiple τ, retained for supplementary inspection.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(bottom=0.22)
    x = np.arange(len(TAU_VALS))
    for i, method in enumerate(MAIN_METHODS):
        vals = [hit_rate_data[tau][method] for tau in TAU_VALS]
        errs = [hit_rate_errs[tau][method] for tau in TAU_VALS]
        ax.bar(x + offsets[i], vals, width, yerr=errs, color=MAIN_COLORS[method],
               capsize=2, edgecolor='white', linewidth=0.3,
               error_kw={'linewidth': 0.8}, label=_short_name(method))
    ax.set_xticks(x)
    ax.set_xticklabels([f'τ={t}' for t in TAU_VALS])
    ax.set_ylabel('Hit rate')
    ax.set_ylim(0, None)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(frameon=True, edgecolor='#cccccc', fontsize=9)
    savefig('fig1c_ecfp4_hitrate.png')

    # Panel B: Top-N median tanimoto
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(bottom=0.22)
    x = np.arange(len(TOP_N))
    for i, method in enumerate(MAIN_METHODS):
        vals = [topn_data[n][method] for n in TOP_N]
        errs = [topn_errs[n][method] for n in TOP_N]
        ax.bar(x + offsets[i], vals, width, yerr=errs, color=MAIN_COLORS[method],
               capsize=2, edgecolor='white', linewidth=0.3,
               error_kw={'linewidth': 0.8}, label=_short_name(method))
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{n}\nmedian' for n in TOP_N])
    ax.set_ylabel('ECFP4 Tanimoto')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(frameon=True, edgecolor='#cccccc', fontsize=9)
    savefig('fig1d_ecfp4_topn.png')

# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Ablation comparison — grouped bar chart
# ══════════════════════════════════════════════════════════════════════════
def fig2_ablation_barchart():
    print("\n[Fig 2] Ablation comparison bar chart")
    methods_abl = ['RNA2smol (Only General Feat)'] + ABLATION_METHODS + ['RNA2smol (Baseline)']
    abl_df = df[df['mode'].isin(methods_abl)]

    plot_metrics = ['affinity', 'SEI', 'LLE', 'rmsd', 'LBE', 'max_ecfp4_tanimoto']
    all_agg = []
    for col in plot_metrics:
        agg = agg_across_targets(abl_df, col)
        agg['metric'] = col
        all_agg.append(agg)
    combined = pd.concat(all_agg, ignore_index=True)

    n_methods = len(methods_abl)
    width = 0.14
    offsets = np.linspace(-(n_methods-1)*width/2, (n_methods-1)*width/2, n_methods)
    abl_color_cycle = {m: ABLATION_COLORS.get(m, MAIN_COLORS.get(m, '#333333')) for m in methods_abl}

    def _draw_panel(panel_metrics, filename):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(bottom=0.22)
        vals_all, errs_all = [], []
        for i, method in enumerate(methods_abl):
            vals, errs = [], []
            for col in panel_metrics:
                row = combined[(combined['mode'] == method) & (combined['metric'] == col)]
                v = row[f'{col}_mean'].values[0] if len(row) > 0 else np.nan
                e = row[f'{col}_std'].values[0] if len(row) > 0 else np.nan
                vals.append(v)
                errs.append(e)
            vals_all.extend(vals)
            errs_all.extend(errs)
            is_highlight = method in {'RNA2smol (Only General Feat)', 'RNA2smol (Baseline)'}
            ax.bar(np.arange(len(panel_metrics)) + offsets[i], vals, width,
                   yerr=errs, color=abl_color_cycle[method],
                   capsize=2, label=_short_name(method), edgecolor='white', linewidth=0.3,
                   alpha=1.0 if is_highlight else 0.85,
                   error_kw={'linewidth': 0.8})
        ax.set_xticks(np.arange(len(panel_metrics)))
        ax.set_xticklabels([METRIC_LABEL[c] for c in panel_metrics])
        ax.axhline(y=0, color='black', linewidth=0.5)
        a_min = min(v - e for v, e in zip(vals_all, errs_all) if not np.isnan(v))
        a_max = max(v + e for v, e in zip(vals_all, errs_all) if not np.isnan(v))
        pa_pad = (a_max - a_min) * 0.12
        ax.set_ylim(a_min - pa_pad, a_max + pa_pad)
        ax.legend(frameon=True, edgecolor='#cccccc', fontsize=9)
        savefig(filename)

    _draw_panel(['affinity', 'SEI', 'LLE'], 'fig2a_ablation_docking.png')
    _draw_panel(['rmsd', 'LBE', 'max_ecfp4_tanimoto'], 'fig2b_ablation_other.png')

# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Method × Target heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig3_heatmap():
    print("\n[Fig 3] Method x Target heatmap")
    main_df = df[df['mode'].isin(MAIN_METHODS)]

    heat_metrics = ['affinity', 'rmsd', 'SEI']
    heat_units = ['kcal/mol', 'Å', '']
    filenames = ['fig3a_heatmap_affinity.png', 'fig3b_heatmap_rmsd.png', 'fig3c_heatmap_sei.png']

    for ax_idx, col in enumerate(heat_metrics):
        pivot = main_df.groupby(['mode', 'pdb_id'])[col].median().reset_index()
        heatmap_data = pivot.pivot(index='mode', columns='pdb_id', values=col)

        if 'RNA2smol (Baseline)' in heatmap_data.index:
            ref = heatmap_data.loc['RNA2smol (Baseline)']
            ascending = True if col == 'rmsd' else False
            col_order = ref.sort_values(ascending=ascending).index
            heatmap_data = heatmap_data[col_order]
        heatmap_data = heatmap_data.reindex(MAIN_METHODS)

        vmin, vmax = heatmap_data.min().min(), heatmap_data.max().max()
        vcenter = heatmap_data.stack().median()
        direction = METRIC_DIRECTION.get(col, 'higher')
        cmap = 'RdBu' if direction == 'higher' else 'RdBu_r'

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heatmap_data, ax=ax, cmap=cmap,
                    annot=False, linewidths=0.8, linecolor='white',
                    vmin=vmin, vmax=vmax, center=vcenter,
                    cbar_kws={'shrink': 0.7, 'label': heat_units[ax_idx] if heat_units[ax_idx] else ''})
        ax.set_xlabel('Target')
        ax.set_ylabel('Method')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
        ax.set_yticklabels([_short_name(m.get_text()) for m in ax.get_yticklabels()])
        savefig(filenames[ax_idx])

# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Molecular property distribution KDE/violin
# ══════════════════════════════════════════════════════════════════════════
def fig4_property_dist():
    print("\n[Fig 4] Property distribution")
    main_df = df[df['mode'].isin(MAIN_METHODS)].copy()

    active_props = main_df.groupby(['pdb_id', 'active_smi']).first().reset_index()
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED
    from rdkit.Contrib.SA_Score import sascorer

    active_qed, active_logp, active_tpsa, active_sa = [], [], [], []
    for smi in active_props['active_smi'].dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            active_qed.append(QED.qed(mol))
            active_logp.append(Crippen.MolLogP(mol))
            active_tpsa.append(Descriptors.TPSA(mol))
            try:
                active_sa.append(sascorer.calculateScore(mol))
            except:
                pass

    plot_props = ['qed', 'logp', 'tpsa', 'sa_score']
    active_data = {'qed': active_qed, 'logp': active_logp,
                   'tpsa': active_tpsa, 'sa_score': active_sa}
    short_names = [_short_name(m) for m in MAIN_METHODS]
    filenames = ['fig4a_qed.png', 'fig4b_logp.png', 'fig4c_tpsa.png', 'fig4d_sa.png']

    for ax_idx, col in enumerate(plot_props):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_data = [main_df[main_df['mode'] == m][col].dropna().values
                     for m in MAIN_METHODS]
        positions = np.arange(len(MAIN_METHODS))
        bp = ax.boxplot(plot_data, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops={'color': '#333333', 'linewidth': 1.2},
                        whiskerprops={'linewidth': 0.8},
                        capprops={'linewidth': 0.8},
                        boxprops={'linewidth': 0.8})

        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(MAIN_COLORS[MAIN_METHODS[i]])
            patch.set_alpha(0.7)

        ref_vals = active_data.get(col, [])
        if ref_vals:
            ref = np.median(ref_vals)
            ax.axhline(y=ref, color='#333333', linestyle='--', linewidth=1.2, alpha=0.8)
            ax.text(len(MAIN_METHODS) - 0.45, ref, 'active\nmedian', fontsize=9,
                    va='center', color='#333333', style='italic')

        ax.set_xticks(positions)
        ax.set_xticklabels(short_names, rotation=15, ha='right')
        ax.set_ylabel(METRIC_LABEL.get(col, col))
        ax.grid(axis='y', alpha=0.15, linestyle='--')
        savefig(filenames[ax_idx])

# ══════════════════════════════════════════════════════════════════════════
# Figure 5: MW-binned line chart
# ══════════════════════════════════════════════════════════════════════════
def fig5_mw_binned():
    print("\n[Fig 5] MW-binned performance")
    mw_bins_ordered = ['0–120', '120-180', '180–300', '300-420', '≥420']

    metrics_to_plot = ['affinity', 'SEI']
    filenames = ['fig5a_mw_affinity.png', 'fig5b_mw_sei.png']

    for ax_idx, col in enumerate(metrics_to_plot):
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for method in MAIN_METHODS:
            sub = df[df['mode'] == method]
            vals = []
            for mb in mw_bins_ordered:
                mb_data = sub[sub['mw_bin'] == mb]
                if len(mb_data) > 0:
                    per_target = mb_data.groupby('pdb_id')[col].median()
                    vals.append(per_target.mean())
                else:
                    vals.append(np.nan)
            ax.plot(range(len(mw_bins_ordered)), vals,
                    color=MAIN_COLORS[method], linewidth=2.5,
                    marker='o', markersize=8, label=_short_name(method), zorder=5)

        for method in ABLATION_METHODS:
            sub = df[df['mode'] == method]
            vals = []
            for mb in mw_bins_ordered:
                mb_data = sub[sub['mw_bin'] == mb]
                if len(mb_data) > 0:
                    per_target = mb_data.groupby('pdb_id')[col].median()
                    vals.append(per_target.mean())
                else:
                    vals.append(np.nan)
            ax.plot(range(len(mw_bins_ordered)), vals,
                    color=ABLATION_COLORS[method],
                    linestyle=ABLATION_LINESTYLES.get(method, '--'),
                    linewidth=1.2, marker='s', markersize=4,
                    label=method, alpha=0.5, zorder=3)

        ax.set_xticks(range(len(mw_bins_ordered)))
        ax.set_xticklabels(mw_bins_ordered)
        ax.set_ylabel(METRIC_LABEL[col])
        ax.set_xlabel('Molecular weight')
        ax.grid(alpha=0.15, linestyle='--')
        ax.legend(frameon=True, edgecolor='#cccccc', fontsize=8)
        savefig(filenames[ax_idx])

# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Stepwise virtual screening funnel (fine-grained)
# ══════════════════════════════════════════════════════════════════════════
def fig6_stepped_filter():
    print("\n[Fig 6] Stepwise filtering funnel")

    # Fine-grained filter cascade. Plot labels are intentionally compact; the
    # full thresholds are encoded in the predicates and reported in the text.
    filter_steps = [
        ('① All',                   None),
        ('② aff<−6',               lambda d: d['affinity'] < -6),
        ('③ aff<−7',               lambda d: d['affinity'] < -7),
        ('④ +RMSD<2.5',            lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.5)),
        ('⑤ RMSD<2.0',             lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0)),
        ('⑥ +QED>0.3',             lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.3)),
        ('⑦ QED>0.4, +SA<5',       lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
        ('⑧ aff<−8, RMSD<1.5,\nQED>0.5, SA<5', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
        ('⑨ +SEI>0.1',             lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.1)),
        ('⑩ aff<−9, RMSD<1.0,\nQED>0.5, SA<5, SEI>0.15', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.15)),
        ('⑪ aff<−9, RMSD<1.0,\nQED>0.6, SA<4, SEI>0.2', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2)),
    ]
    step_labels = [
        '1\nAll',
        '2\naff<-6',
        '3\naff<-7',
        '4\n+RMSD<2.5',
        '5\nRMSD<2.0',
        '6\n+QED>0.3',
        '7\n+SA<5',
        '8\naff<-8\nRMSD<1.5',
        '9\n+SEI>0.1',
        '10\naff<-9\nRMSD<1.0',
        '11\nQED>0.6\nSA<4',
    ]

    methods = MAIN_METHODS

    # Panel A: Total molecules surviving (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#fafafa')
    for method in methods:
        sub = df[df['mode'] == method]
        counts = []
        for _, flt in filter_steps:
            counts.append(len(sub) if flt is None else flt(sub).sum())
        ax.plot(range(len(step_labels)), counts, color=MAIN_COLORS[method],
                linewidth=2.5, marker='o', markersize=8, label=_short_name(method), zorder=5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels(step_labels, rotation=0, ha='center', fontsize=8)
    ax.set_xlabel('Filter step')
    ax.set_ylabel('Molecules retained')
    ax.grid(alpha=0.12, which='both', linestyle='--')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(frameon=True, edgecolor='#cccccc', fontsize=9, loc='lower left')
    savefig('fig6a_funnel_mols.png')

    # Panel B: Targets covered (≥5 molecules)
    fig, ax = plt.subplots(figsize=(12, 6))
    for method in methods:
        sub = df[df['mode'] == method]
        ntargets = []
        for _, flt in filter_steps:
            filtered = sub if flt is None else sub[flt(sub)]
            n = filtered.groupby('pdb_id').size()
            ntargets.append((n >= 5).sum())
        ax.plot(range(len(step_labels)), ntargets, color=MAIN_COLORS[method],
                linewidth=2.5, marker='o', markersize=8, label=_short_name(method), zorder=5)
    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels(step_labels, rotation=0, ha='center', fontsize=8)
    ax.set_xlabel('Filter step')
    ax.set_ylabel('Targets (≥5 mol.)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15, linestyle='--')
    ax.legend(frameon=True, edgecolor='#cccccc', fontsize=9, loc='upper right')
    savefig('fig6b_funnel_targets.png')

# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Radar chart — multi-dimensional profile (optimized, aligned with DUDE style)
# Two normalization variants: Rank+Floor and MinMax+Floor
# 11 metrics reordered into 3 categories: Docking, Drug-likeness, Chemical Space/Similarity
# ══════════════════════════════════════════════════════════════════════════

RADAR_METRICS_REORDERED = [
    'affinity', 'SEI', 'LLE', 'LBE', 'rmsd',
    'qed', 'sa_score', 'rotatable_bonds',
    'bond_len_js', 'bond_angle_js', 'max_ecfp4_tanimoto',
]
RADAR_DIRS = {
    'affinity': 'lower', 'SEI': 'higher', 'LLE': 'higher', 'LBE': 'lower',
    'rmsd': 'lower', 'qed': 'higher', 'sa_score': 'lower', 'rotatable_bonds': 'lower',
    'max_ecfp4_tanimoto': 'higher', 'bond_len_js': 'lower', 'bond_angle_js': 'lower',
}
RADAR_LABELS = [
    'Affinity', 'SEI', 'LLE', 'LBE', 'RMSD',
    'QED', 'SA Score', 'Rot. Bonds',
    'Bond Len.\nJS div.', 'Bond Ang.\nJS div.', 'ECFP4\nTanimoto',
]
RADAR_FLOOR = 0.25

# Category spans: (start, end, label, bg_color, text_color)
RADAR_CATEGORIES = [
    (0, 5, 'Docking', '#E8F0FE', '#1565C0'),
    (5, 8, 'Drug-likeness', '#E8F5E9', '#2E7D32'),
    (8, 11, 'Chemical Space\nSimilarity', '#F3E5F5', '#7B1FA2'),
]


def _get_label_alignment(angle_rad):
    """Return (ha, va) for a polar tick label based on its angle."""
    deg = np.degrees(angle_rad) % 360
    if 0 <= deg < 30 or 330 <= deg <= 360:
        return 'center', 'bottom'
    elif 30 <= deg < 60:
        return 'left', 'bottom'
    elif 60 <= deg < 120:
        return 'left', 'center'
    elif 120 <= deg < 150:
        return 'left', 'top'
    elif 150 <= deg < 210:
        return 'center', 'top'
    elif 210 <= deg < 240:
        return 'right', 'top'
    elif 240 <= deg < 300:
        return 'right', 'center'
    else:
        return 'right', 'bottom'


def _extract_bond_features(smiles_list, max_mols=300):
    """Extract bond lengths and angles from 3D conformers of sampled molecules."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    lengths, angles = [], []
    rng = np.random.RandomState(42)
    smis = list(smiles_list)
    if len(smis) > max_mols:
        smis = rng.choice(smis, max_mols, replace=False)
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            mol = Chem.RemoveHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                continue
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (mol.GetAtomWithIdx(ai).GetAtomicNum() > 1
                    and mol.GetAtomWithIdx(aj).GetAtomicNum() > 1):
                d = conf.GetAtomPosition(ai).Distance(conf.GetAtomPosition(aj))
                lengths.append(d)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                    if mol.GetAtomWithIdx(n.GetIdx()).GetAtomicNum() > 1]
            if len(nbrs) < 2:
                continue
            pos_c = conf.GetAtomPosition(atom.GetIdx())
            for p in range(len(nbrs)):
                for q in range(p + 1, len(nbrs)):
                    v1 = conf.GetAtomPosition(nbrs[p]) - pos_c
                    v2 = conf.GetAtomPosition(nbrs[q]) - pos_c
                    denom = v1.Length() * v2.Length()
                    if denom < 1e-12:
                        continue
                    cos_a = max(-1.0, min(1.0, v1.DotProduct(v2) / denom))
                    angles.append(np.degrees(np.arccos(cos_a)))
    return np.array(lengths), np.array(angles)


def _js_divergence_hist(sample_a, sample_b, bins=60, rng=None):
    """Jensen-Shannon distance between two 1D samples via histogram."""
    from scipy.spatial.distance import jensenshannon
    if rng is None:
        rng = (min(sample_a.min(), sample_b.min()), max(sample_a.max(), sample_b.max()))
    ha, _ = np.histogram(sample_a, bins=bins, range=rng, density=True)
    hb, _ = np.histogram(sample_b, bins=bins, range=rng, density=True)
    ha = ha + 1e-10
    hb = hb + 1e-10
    ha /= ha.sum()
    hb /= hb.sum()
    return float(jensenshannon(ha, hb))


def _compute_bond_js_for_methods(data_df, methods, cache_path):
    """Compute per-method bond length/angle JS divergence vs active molecules."""
    import pickle
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if all(m in cached for m in methods):
            return cached

    print("  Computing 3D bond features for reference (active) molecules...")
    active_smis_all = data_df['active_smi'].dropna().unique()
    ref_len, ref_ang = _extract_bond_features(active_smis_all, max_mols=200)

    results = {}
    for method in methods:
        print(f"    {_short_name(method)}...")
        smis = data_df[data_df['mode'] == method]['SMILES'].dropna().unique()
        gen_len, gen_ang = _extract_bond_features(smis, max_mols=300)
        if len(gen_len) < 10 or len(ref_len) < 10:
            results[method] = {'bond_len_js': np.nan, 'bond_angle_js': np.nan}
            continue
        len_js = _js_divergence_hist(gen_len, ref_len, bins=60, rng=(0.8, 2.5))
        ang_js = _js_divergence_hist(gen_ang, ref_ang, bins=60, rng=(0, 180))
        results[method] = {'bond_len_js': len_js, 'bond_angle_js': ang_js}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    return results


def _build_profile(data_df, methods, metrics=None):
    """Compute per-target median, then mean across targets for each method."""
    use_metrics = metrics if metrics is not None else RADAR_METRICS_REORDERED
    profile = {}
    for col in use_metrics:
        if col in ('bond_len_js', 'bond_angle_js'):
            continue
        agg = agg_across_targets(data_df, col)
        for _, row_ in agg.iterrows():
            m = row_['mode']
            if m not in profile:
                profile[m] = {}
            profile[m][col] = row_[f'{col}_mean']
    return profile


def _compute_rotb_pass_rate(data_df, methods, threshold=7):
    """Compute per-target pass rate for rotatable_bonds (≤threshold), then mean across targets."""
    results = {}
    for m in methods:
        sub = data_df[data_df['mode'] == m]
        if len(sub) == 0:
            results[m] = np.nan
            continue
        per_target = sub.groupby('pdb_id').apply(
            lambda x: (x['rotatable_bonds'] <= threshold).mean(),
            include_groups=False,
        )
        results[m] = per_target.mean()
    return results


def _normalize_rank(profile, methods, metrics=None):
    """Rank-based normalization with floor: worst → FLOOR, best → 1.0, evenly spaced."""
    use_metrics = metrics if metrics is not None else RADAR_METRICS_REORDERED
    N = len(methods)
    normed = {}
    for m in methods:
        normed[m] = []
        for col in use_metrics:
            vals = {mm: profile[mm].get(col, np.nan) for mm in methods}
            valid = {k: v for k, v in vals.items() if not np.isnan(v)}
            if not valid:
                normed[m].append((1.0 + RADAR_FLOOR) / 2)
                continue
            ascending = (RADAR_DIRS[col] == 'lower')
            sorted_methods = sorted(valid, key=valid.get, reverse=not ascending)
            rank = sorted_methods.index(m) + 1 if m in sorted_methods else len(sorted_methods)
            n_valid = len(sorted_methods)
            n = 1.0 - (rank - 1) * (1.0 - RADAR_FLOOR) / (n_valid - 1) if n_valid > 1 else (1.0 + RADAR_FLOOR) / 2
            normed[m].append(n)
    return normed


def _normalize_minmax(profile, methods, dirs=None, metrics=None, extra_vals=None):
    """Min-max normalization with floor: preserves raw-value gaps between methods."""
    use_dirs = dirs if dirs is not None else RADAR_DIRS
    use_metrics = metrics if metrics is not None else RADAR_METRICS_REORDERED
    normed = {}
    for m in methods:
        normed[m] = []
        for col in use_metrics:
            raw_vals = np.array([profile[mm].get(col, np.nan) for mm in methods])
            if extra_vals and col in extra_vals:
                raw_vals = np.array(list(raw_vals) + [extra_vals[col].get(mm, np.nan) for mm in methods])
            valid = raw_vals[~np.isnan(raw_vals)]
            if len(valid) < 2:
                normed[m].append((1.0 + RADAR_FLOOR) / 2)
                continue
            min_v, max_v = valid.min(), valid.max()
            if max_v - min_v < 1e-8:
                n = (1.0 + RADAR_FLOOR) / 2
            else:
                v = profile[m].get(col, np.nan)
                if np.isnan(v):
                    n = (1.0 + RADAR_FLOOR) / 2
                elif use_dirs[col] == 'higher':
                    n = RADAR_FLOOR + (1.0 - RADAR_FLOOR) * (v - min_v) / (max_v - min_v)
                else:
                    n = RADAR_FLOOR + (1.0 - RADAR_FLOOR) * (1.0 - (v - min_v) / (max_v - min_v))
            normed[m].append(np.clip(n, RADAR_FLOOR, 1.0))
    return normed


def _draw_radar(normed, methods, title, filename,
                metric_labels=None, category_defs=None, metrics=None):
    """Generic radar drawer with category shading and adaptive tick labels."""
    use_metrics = metrics if metrics is not None else RADAR_METRICS_REORDERED
    n_vars = len(use_metrics)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]
    theta_ref = np.linspace(0, 2 * np.pi, 100)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    labels = metric_labels if metric_labels is not None else RADAR_LABELS
    cat_defs = category_defs if category_defs is not None else RADAR_CATEGORIES

    # Build tick color map from categories
    tick_colors = {}
    for cat_item in cat_defs:
        start, end, cat_label, cat_color = cat_item[:4]
        text_color = cat_item[4] if len(cat_item) >= 5 else '#333333'
        for idx in range(start, end):
            tick_colors[idx] = text_color

    # Category background arcs
    for cat_item in cat_defs:
        start, end, cat_label, cat_color = cat_item[:4]
        text_color = cat_item[4] if len(cat_item) >= 5 else '#333333'
        arc_angles = np.linspace(angles[start], angles[end - 1], 50)
        ax.fill_between(arc_angles, 0, 1.0, color=cat_color, alpha=0.45, zorder=0)
        mid_angle = (angles[start] + angles[end - 1]) / 2
        span = angles[end - 1] - angles[start]
        mid_angle += np.radians(12 if span > np.radians(60) else 18)
        ax.text(mid_angle, 1.44, cat_label, fontsize=9, fontweight='bold',
                ha='center', va='center', color=text_color)

    for method in methods:
        values = normed[method] + [normed[method][0]]
        ax.plot(angles, values, color=MAIN_COLORS.get(method, ABLATION_COLORS.get(method, '#333333')),
                linewidth=2.2, marker='o', markersize=6, label=_short_name(method))
        ax.fill(angles, values, color=MAIN_COLORS.get(method, ABLATION_COLORS.get(method, '#333333')),
                alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.tick_params(axis='x', pad=5)
    ax.set_xticklabels(labels)
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_fontsize(8.5)
        label.set_weight('semibold')
        ha, va = _get_label_alignment(angles[idx])
        label.set_ha(ha)
        label.set_va(va)
        if idx in tick_colors:
            label.set_color(tick_colors[idx])

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['', '', '', ''], fontsize=9, color='#999999')
    ax.plot(theta_ref, np.full_like(theta_ref, RADAR_FLOOR), color='#dddddd',
            linewidth=0.6, linestyle=':', zorder=0)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.18)

    ncol = 1 if len(methods) <= 5 else 2
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08),
              frameon=True, edgecolor='#cccccc', fontsize=9, ncol=ncol)
    savefig(filename)


def fig7_radar():
    print("\n[Fig 7] Radar charts")

    # ── Pre-compute bond JS divergence (cached) ──
    bond_cache = OUT_DIR / 'cache_bond_js.pkl'
    all_methods_for_js = MAIN_METHODS + ABLATION_METHODS
    bond_js_all = _compute_bond_js_for_methods(df, all_methods_for_js, bond_cache)

    # ── Main methods ──
    main_df = df[df['mode'].isin(MAIN_METHODS)]
    main_profile = _build_profile(main_df, MAIN_METHODS)
    for m in MAIN_METHODS:
        if m in bond_js_all:
            main_profile[m].update(bond_js_all[m])

    # Rank + floor
    rank_normed = _normalize_rank(main_profile, MAIN_METHODS)
    _draw_radar(rank_normed, MAIN_METHODS, None,
                'fig7a_radar_main_rank.png')

    # MinMax + floor (use rotatable_bonds pass rate instead of raw median)
    rotb_main = _compute_rotb_pass_rate(main_df, MAIN_METHODS)
    main_profile_minmax = {m: dict(main_profile[m]) for m in MAIN_METHODS}
    for m in MAIN_METHODS:
        main_profile_minmax[m]['rotatable_bonds'] = rotb_main.get(m, np.nan)
    mm_normed = _normalize_minmax(main_profile_minmax, MAIN_METHODS,
                                  dirs={**RADAR_DIRS, 'rotatable_bonds': 'higher'})
    _draw_radar(mm_normed, MAIN_METHODS, None,
                'fig7b_radar_main_minmax.png')

    # ── Ablation methods ──
    abl_df = df[df['mode'].isin(ABLATION_METHODS)]
    abl_profile = _build_profile(abl_df, ABLATION_METHODS)
    for m in ABLATION_METHODS:
        if m in bond_js_all:
            abl_profile[m].update(bond_js_all[m])

    rank_abl = _normalize_rank(abl_profile, ABLATION_METHODS)
    _draw_radar(rank_abl, ABLATION_METHODS, None,
                'fig7c_radar_ablation_rank.png')

    rotb_abl = _compute_rotb_pass_rate(abl_df, ABLATION_METHODS)
    abl_profile_minmax = {m: dict(abl_profile[m]) for m in ABLATION_METHODS}
    for m in ABLATION_METHODS:
        abl_profile_minmax[m]['rotatable_bonds'] = rotb_abl.get(m, np.nan)
    mm_abl = _normalize_minmax(abl_profile_minmax, ABLATION_METHODS,
                               dirs={**RADAR_DIRS, 'rotatable_bonds': 'higher'})
    _draw_radar(mm_abl, ABLATION_METHODS, None,
                'fig7d_radar_ablation_minmax.png')

    # Print bond JS divergence summary
    print(f"\n  Bond Length/Angle JS divergence vs Active (lower = more reference-like):")
    print(f"  {'Method':<35} {'BondLen JS':>12} {'BondAng JS':>12}")
    print(f"  {'-'*62}")
    for m in MAIN_METHODS:
        bl = bond_js_all.get(m, {}).get('bond_len_js', np.nan)
        ba = bond_js_all.get(m, {}).get('bond_angle_js', np.nan)
        print(f"  {_short_name(m):<35} {bl:>12.4f} {ba:>12.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fig1_main_barchart()
    fig1b_ecfp4_ts()
    fig2_ablation_barchart()
    fig3_heatmap()
    fig4_property_dist()
    fig5_mw_binned()
    fig6_stepped_filter()
    fig7_radar()
    print("\nAll figures generated in:", OUT_DIR)
