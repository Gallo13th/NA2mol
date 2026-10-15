"""
Ablation-focused visualization for RNA2smol.
Nature-style, multi-panel figures covering fine-grained ablation analysis.

Chart-type rationale (by data type):
  - Categorical comparison (overview)    → grouped bar
  - Distribution comparison              → box plot + swarm overlay
  - Incremental contribution             → waterfall plot
  - Per-target paired improvement        → paired Cleveland dot plot
  - Target-level consistency             → heatmap (target × method)
  - Filter cascade                       → stepped line (log-scale funnel)
  - ECFP4 threshold sweep                → multi-curve line plot
  - Trade-off analysis                   → 2D contour / density scatter
  - MW-segmented analysis                → small-multiples faceted bar
  - Statistical significance             → pairwise P-value heatmap
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu
from pathlib import Path
import pickle
from collections import defaultdict

# ── Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'output' / 'smrtbench'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FONT_PATH = SCRIPT_DIR.parent / 'NotoSansCJKsc-Regular.otf'

# Nature-style global rcParams (exactly matching visualize_methods.py)
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
    _cjk = 'Noto Sans CJK SC'
    if _cjk not in plt.rcParams['font.sans-serif']:
        plt.rcParams['font.sans-serif'] = [_cjk] + plt.rcParams['font.sans-serif']


def _panel_label(ax, label, x=-0.06, y=1.02):
    """Add bold panel label in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left')


def savefig(name, tight=True):
    path = OUT_DIR / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {name}")


# ── Data ──────────────────────────────────────────────────────────────────
CSV_PATH = SCRIPT_DIR / 'output' / 'smrtbench_combined_results.csv'
df_full = pd.read_csv(CSV_PATH)

# New ablation names + existing
ABL_METHODS = [
    'RNA2smol (Baseline)',       # ablated: no pretrain, no contrastive, no constrain
    'Pretrain Only',             # pretrain only, no constrain, no contrastive
    'Pretrain+Constrain',        # pretrain + constrain, no contrastive
    'RNA2smol (Only General Feat)',  # full model
]
ABL_DISPLAY = {
    'RNA2smol (Baseline)':          'Baseline',
    'Pretrain Only':                'Pretrain Only',
    'Pretrain+Constrain':          'Pretrain\n+Constrain',
    'RNA2smol (Only General Feat)': 'Full Model',
}

df = df_full[df_full['mode'].isin(ABL_METHODS)].copy()

# Derived metrics
df['SEI'] = -df['affinity'] / df['tpsa'].replace(0, np.nan)
df['SEI'] = df['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
df['LLE'] = -df['affinity'] - df['logp']
df['LBE'] = df['affinity'] / df['heavy_atoms'].replace(0, np.nan)

# Nature Wong palette — ablation uses progressive color
ABL_COLORS = {
    'RNA2smol (Baseline)':          '#999999',   # grey: worst
    'Pretrain Only':                '#56B4E9',   # sky blue
    'Pretrain+Constrain':          '#E69F00',   # orange
    'RNA2smol (Only General Feat)': '#D55E00',   # vermilion: best/full model
}
ABL_COLORS_LIGHT = {
    'RNA2smol (Baseline)':          '#cccccc',
    'Pretrain Only':                '#a0d8f3',
    'Pretrain+Constrain':          '#f5c96a',
    'RNA2smol (Only General Feat)': '#f0a070',
}

METHOD_ORDER = [
    'RNA2smol (Baseline)',
    'Pretrain Only',
    'Pretrain+Constrain',
    'RNA2smol (Only General Feat)',
]

print(f"Data: {len(df)} molecules, {df['pdb_id'].nunique()} targets, modes: {df['mode'].unique().tolist()}")


# ── Helpers ───────────────────────────────────────────────────────────────
def per_target_median(data, col):
    return data.groupby('pdb_id')[col].median()


def _short(m):
    return ABL_DISPLAY.get(m, m)


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Ablation overview — grouped bar + incremental waterfall
# ══════════════════════════════════════════════════════════════════════════
def fig1_ablation_overview():
    """Multi-panel: (a) bar chart on docking+geometry, (b) bar chart on
    drug-likeness+synthesis, (c) waterfall of incremental contribution."""
    print("\n[Fig 1] Ablation overview")

    metrics_a = ['affinity', 'rmsd', 'SEI', 'LLE']
    metrics_b = ['qed', 'sa_score', 'LBE', 'max_ecfp4_tanimoto']

    def _agg_summary(metric_list):
        rows = []
        for m in METHOD_ORDER:
            sub = df[df['mode'] == m]
            for col in metric_list:
                pt = per_target_median(sub, col)
                rows.append({
                    'method': m, 'metric': col,
                    'mean': pt.mean(), 'std': pt.std(),
                    'n_targets': len(pt),
                })
        return pd.DataFrame(rows)

    combined_a = _agg_summary(metrics_a)
    combined_b = _agg_summary(metrics_b)

    n_methods = len(METHOD_ORDER)
    width = 0.18
    offsets = np.linspace(-(n_methods - 1) * width / 2, (n_methods - 1) * width / 2, n_methods)

    def _draw_panel(ax, combined, panel_metrics, title_str):
        metric_labels = {
            'affinity': 'Affinity\n(kcal/mol)', 'SEI': 'SEI', 'LLE': 'LLE',
            'rmsd': 'RMSD (Å)', 'LBE': 'LBE', 'max_ecfp4_tanimoto': 'ECFP4\nTanimoto',
            'qed': 'QED', 'sa_score': 'SA Score',
        }
        vals_all, errs_all = [], []
        for i, method in enumerate(METHOD_ORDER):
            vals, errs = [], []
            for col in panel_metrics:
                row = combined[(combined['method'] == method) & (combined['metric'] == col)]
                v = row['mean'].values[0] if len(row) > 0 else np.nan
                e = row['std'].values[0] if len(row) > 0 else np.nan
                vals.append(v); errs.append(e)
            vals_all.extend(vals); errs_all.extend(errs)
            ax.bar(np.arange(len(panel_metrics)) + offsets[i], vals, width,
                   yerr=errs, color=ABL_COLORS[method],
                   capsize=2, label=_short(method), edgecolor='white', linewidth=0.3,
                   error_kw={'linewidth': 0.8})
        ax.set_xticks(np.arange(len(panel_metrics)))
        ax.set_xticklabels([metric_labels[c] for c in panel_metrics], fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
        vmin = min(v - e for v, e in zip(vals_all, errs_all) if not np.isnan(v))
        vmax = max(v + e for v, e in zip(vals_all, errs_all) if not np.isnan(v))
        pad = (vmax - vmin) * 0.12
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_title(title_str, fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.15, linestyle='--')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    _draw_panel(ax1, combined_a, metrics_a, 'Docking & Geometry')
    _draw_panel(ax2, combined_b, metrics_b, 'Drug-likeness & Synthesis')
    ax1.legend(frameon=True, edgecolor='#cccccc', fontsize=8, loc='lower left')
    ax2.legend(frameon=True, edgecolor='#cccccc', fontsize=8, loc='lower left')
    _panel_label(ax1, 'a')
    _panel_label(ax2, 'b')
    fig.suptitle('Ablation Comparison — Overview', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_overview.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Property distribution — box plot + swarm overlay
# ══════════════════════════════════════════════════════════════════════════
def fig2_property_distribution():
    """Panel (a-d): box plots for QED, SA, logP, affinity distribution per method.
    Use box plot (robust to outliers) + median line in red."""
    print("\n[Fig 2] Property distribution comparison")

    props = ['qed', 'sa_score', 'logp', 'affinity']
    titles = ['QED', 'SA Score', 'logP', 'Docking Affinity (kcal/mol)']
    ylabels = ['QED', 'SA Score', 'logP', 'Affinity (kcal/mol)']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax_i, (ax, col, title_str, ylab) in enumerate(zip(axes, props, titles, ylabels)):
        plot_data = []
        positions = []
        for j, m in enumerate(METHOD_ORDER):
            vals = df[df['mode'] == m][col].dropna()
            if len(vals) > 2000:
                vals = vals.sample(2000, random_state=42)
            plot_data.append(vals.values)
            positions.append(j)

        bp = ax.boxplot(plot_data, positions=positions, widths=0.55,
                       patch_artist=True, showfliers=False,
                       medianprops={'color': '#cc0000', 'linewidth': 1.5},
                       whiskerprops={'linewidth': 0.8, 'color': '#555555'},
                       capprops={'linewidth': 0.8, 'color': '#555555'},
                       boxprops={'linewidth': 0.8})

        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(ABL_COLORS[METHOD_ORDER[i]])
            patch.set_alpha(0.75)

        # Add median value annotations
        for j, m in enumerate(METHOD_ORDER):
            med = df[df['mode'] == m][col].median()
            ax.text(j + 0.35, med, f'{med:.2f}', fontsize=7,
                   va='center', ha='left', color='#333333', fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels([_short(m) for m in METHOD_ORDER], rotation=15, ha='right')
        ax.set_ylabel(ylab)
        ax.set_title(title_str, fontweight='bold')
        ax.grid(axis='y', alpha=0.15, linestyle='--')

        panel_label = chr(ord('a') + ax_i)
        _panel_label(ax, panel_label)

    fig.suptitle('Ablation — Property Distribution', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_property_dist.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Incremental contribution — waterfall plot
# ══════════════════════════════════════════════════════════════════════════
def fig3_waterfall():
    """Show the incremental contribution of each component as a waterfall.
    Baseline → +Pretrain → +Constrain → +Contrastive (=Full Model)."""
    print("\n[Fig 3] Waterfall — incremental contribution decomposition")

    def _pt_median(data, col):
        return data.groupby('pdb_id')[col].median().mean()

    metrics = ['affinity', 'qed', 'sa_score', 'SEI']
    labels = ['Affinity (kcal/mol)', 'QED', 'SA Score', 'SEI']
    # direction: -1 = lower is better, +1 = higher is better
    dir_mult = {'affinity': -1, 'qed': 1, 'sa_score': -1, 'SEI': 1}

    # Compute per-component delta across shared targets
    baseline_vals = {}
    for col in metrics:
        baseline_vals[col] = per_target_median(df[df['mode'] == 'RNA2smol (Baseline)'], col)

    pretrain_vals = {}
    for col in metrics:
        pretrain_vals[col] = per_target_median(df[df['mode'] == 'Pretrain Only'], col)

    constrain_vals = {}
    for col in metrics:
        constrain_vals[col] = per_target_median(df[df['mode'] == 'Pretrain+Constrain'], col)

    full_vals = {}
    for col in metrics:
        full_vals[col] = per_target_median(df[df['mode'] == 'RNA2smol (Only General Feat)'], col)

    # Compute delta for each component on shared targets
    components = ['+Pretrain', '+Constrain', '+Contrastive']
    pairs = [
        ('Pretrain Only', 'RNA2smol (Baseline)'),
        ('Pretrain+Constrain', 'Pretrain Only'),
        ('RNA2smol (Only General Feat)', 'Pretrain+Constrain'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax_i, (ax, col, label) in enumerate(zip(axes, metrics, labels)):
        deltas = []
        ci_lows, ci_highs = [], []
        cumul = 0.0

        for comp_a, comp_b in pairs:
            a_vals = per_target_median(df[df['mode'] == comp_a], col)
            b_vals = per_target_median(df[df['mode'] == comp_b], col)
            common = sorted(set(a_vals.index) & set(b_vals.index))
            if len(common) == 0:
                continue
            diffs = (a_vals[common].values - b_vals[common].values) * dir_mult[col]
            d_mean = diffs.mean()
            # Bootstrap CI
            rng = np.random.default_rng(42)
            boots = rng.choice(diffs, size=(10000, len(diffs)), replace=True).mean(axis=1)
            ci_low = np.quantile(boots, 0.025)
            ci_high = np.quantile(boots, 0.975)
            deltas.append(d_mean)
            ci_lows.append(d_mean - ci_low)
            ci_highs.append(ci_high - d_mean)

        # Waterfall
        x = np.arange(len(components) + 1)  # 0=baseline, 1..3=components
        bottoms = np.zeros(len(components) + 1)
        heights = np.zeros(len(components) + 1)
        heights[0] = 0  # baseline as starting point
        colors_list = ['#999999']  # baseline

        for i, d in enumerate(deltas):
            bottoms[i + 1] = sum(deltas[:i]) if i > 0 else 0
            heights[i + 1] = d
            if d >= 0:
                colors_list.append('#2E7D32' if d > 0 else '#999999')
            else:
                colors_list.append('#C62828')

        # Cumulative final value
        final = sum(deltas)
        bar_labels = ['Baseline'] + components + [f'Total\n{final:+.3f}']
        x_full = np.arange(len(bar_labels) - 1)
        colors_list.append('#333333')

        # Draw the component bars
        bars = ax.bar(x, heights, bottom=bottoms, color=colors_list[:len(x)],
                     width=0.5, edgecolor='white', linewidth=0.5, zorder=3)

        # Connect with dashed lines
        y_positions = [0] + list(np.cumsum(deltas))
        ax.plot(x, y_positions, 'o-', color='#333333', linewidth=1.5,
               markersize=8, zorder=5)

        # Error bars
        for i, (low, high) in enumerate(zip(ci_lows, ci_highs)):
            if i + 1 < len(y_positions):
                ax.errorbar(i + 1, y_positions[i + 1],
                           yerr=[[low], [high]], color='#333333',
                           capsize=4, linewidth=1.2, zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels[:len(x)], rotation=15, ha='right', fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
        ax.set_ylabel(f'Δ {label} (better ↑)', fontsize=10)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.15, linestyle='--')

        # Annotate each bar
        for bar, h in zip(bars, heights):
            if abs(h) > 1e-6:
                y_pos = bar.get_y() + bar.get_height()
                va = 'bottom' if h >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                       f'{h:+.3f}', ha='center', va=va, fontsize=8.5,
                       fontweight='bold', color='#333333')

        panel_label = chr(ord('a') + ax_i)
        _panel_label(ax, panel_label)

    fig.suptitle('Ablation — Incremental Contribution (Waterfall)', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_waterfall.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Per-target paired improvement — Cleveland dot plot
# ══════════════════════════════════════════════════════════════════════════
def fig4_paired_dotplot():
    """Paired Cleveland dot plot: each target as a row, showing
    baseline→pretrain→pretrain+constrain→full model as connected dots.
    Shows which targets benefit from each component."""
    print("\n[Fig 4] Paired Cleveland dot plot")

    col = 'affinity'
    target_data = {}

    for m in METHOD_ORDER:
        sub = df[df['mode'] == m]
        target_data[m] = per_target_median(sub, col)

    # Shared targets
    common_targets = sorted(
        set(target_data[METHOD_ORDER[0]].index) &
        set(target_data[METHOD_ORDER[-1]].index)
    )

    # Sort targets by full model performance
    sort_by = target_data[METHOD_ORDER[-1]][common_targets].sort_values()
    ordered_targets = sort_by.index.tolist()

    fig, ax = plt.subplots(figsize=(8, 14))
    n_targets = len(ordered_targets)
    y_positions = np.arange(n_targets)

    for i, m in enumerate(METHOD_ORDER):
        vals = [target_data[m].get(t, np.nan) for t in ordered_targets]
        x_offset = i * 0.25
        ax.scatter(vals, y_positions + x_offset, color=ABL_COLORS[m],
                  s=50, zorder=5, label=_short(m), edgecolors='white', linewidth=0.5)

    # Connect with thin lines
    for t_idx, t in enumerate(ordered_targets):
        vals = []
        for m in METHOD_ORDER:
            v = target_data[m].get(t, np.nan)
            if not np.isnan(v):
                vals.append((METHOD_ORDER.index(m) * 0.25, v))
        if len(vals) >= 2:
            xs = [v[1] for v in vals]
            ys = [t_idx + v[0] for v in vals]
            ax.plot(xs, ys, color='#cccccc', linewidth=0.5, zorder=1, alpha=0.6)

    ax.set_yticks(y_positions + 0.375)
    ax.set_yticklabels(ordered_targets, fontsize=8, fontfamily='monospace')
    ax.set_xlabel('Docking Affinity (kcal/mol)', fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15, linestyle='--')
    ax.legend(frameon=True, edgecolor='#cccccc', fontsize=8, loc='lower left')
    _panel_label(ax, 'a')
    ax.set_title('Per-Target Docking Affinity — Paired Comparison', fontweight='bold', fontsize=13)
    savefig('fig_abl_paired_dotplot.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Target × Method heatmap — affinity improvement over baseline
# ══════════════════════════════════════════════════════════════════════════
def fig5_delta_heatmap():
    """Heatmap: per-target delta vs baseline for each method, for 4 key metrics."""
    print("\n[Fig 5] Delta heatmap vs baseline")

    baseline_pt = {}
    for col in ['affinity', 'qed', 'sa_score', 'SEI']:
        baseline_pt[col] = per_target_median(df[df['mode'] == 'RNA2smol (Baseline)'], col)

    comparators = ['Pretrain Only', 'Pretrain+Constrain', 'RNA2smol (Only General Feat)']
    metrics_hm = ['affinity', 'qed', 'sa_score', 'SEI']
    dir_mult = {'affinity': -1, 'qed': 1, 'sa_score': -1, 'SEI': 1}

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for ax_i, (ax, col) in enumerate(zip(axes, metrics_hm)):
        # Build delta matrix: rows=targets, cols=methods
        all_targets = set()
        for comp in comparators:
            comp_pt = per_target_median(df[df['mode'] == comp], col)
            all_targets |= set(baseline_pt[col].index) & set(comp_pt.index)

        all_targets = sorted(all_targets)
        delta_matrix = np.zeros((len(all_targets), len(comparators)))
        for j, comp in enumerate(comparators):
            comp_pt = per_target_median(df[df['mode'] == comp], col)
            for i, t in enumerate(all_targets):
                delta_matrix[i, j] = (comp_pt.get(t, np.nan) - baseline_pt[col].get(t, np.nan)) * dir_mult[col]

        delta_df = pd.DataFrame(delta_matrix, index=all_targets,
                                columns=[_short(c) for c in comparators])

        vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max())) if delta_matrix.size > 0 else 1
        sns.heatmap(delta_df, ax=ax, cmap='RdBu_r', center=0,
                   annot=True, fmt='.2f', linewidths=0.8, linecolor='white',
                   cbar_kws={'shrink': 0.7, 'label': f'Δ {col} (better ↑)'},
                   vmin=-vmax, vmax=vmax,
                   annot_kws={'fontsize': 7})
        ax.set_title(f'{col}\n(positive = improvement)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Target')
        ax.set_xlabel('Method')
        ax.tick_params(axis='both', labelsize=8)

        panel_label = chr(ord('a') + ax_i)
        _panel_label(ax, panel_label)

    fig.suptitle('Ablation — Per-Target Improvement over Baseline', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_delta_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Virtual screening funnel (log scale, fine-grained)
# ══════════════════════════════════════════════════════════════════════════
def fig6_stepped_funnel():
    """Stepped filtering funnel mimicking Figure 6 of paper but ablation-only."""
    print("\n[Fig 6] Virtual screening funnel")

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
    step_labels = ['1\nAll', '2\naff<-6', '3\naff<-7', '4\n+RMSD<2.5',
                   '5\nRMSD<2.0', '6\n+QED>0.3', '7\n+SA<5',
                   '8\naff<-8\nRMSD<1.5', '9\n+SEI>0.1',
                   '10\naff<-9\nRMSD<1.0', '11\nQED>0.6\nSA<4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.set_facecolor('#fafafa')
    ax2.set_facecolor('#fafafa')

    for method in METHOD_ORDER:
        sub = df[df['mode'] == method]
        counts = []
        ntargets = []
        for _, flt in filter_steps:
            filtered = sub if flt is None else sub[flt(sub)]
            counts.append(len(filtered))
            n = filtered.groupby('pdb_id').size()
            ntargets.append((n >= 1).sum())

        ax1.plot(range(len(step_labels)), counts, color=ABL_COLORS[method],
                linewidth=2.5, marker='o', markersize=8, label=_short(method), zorder=5)
        ax2.plot(range(len(step_labels)), ntargets, color=ABL_COLORS[method],
                linewidth=2.5, marker='o', markersize=8, label=_short(method), zorder=5)

    ax1.set_yscale('log')
    ax1.set_xticks(range(len(step_labels)))
    ax1.set_xticklabels(step_labels, rotation=0, ha='center', fontsize=8)
    ax1.set_xlabel('Filter step')
    ax1.set_ylabel('Molecules retained')
    ax1.grid(alpha=0.12, which='both', linestyle='--')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.legend(frameon=True, edgecolor='#cccccc', fontsize=9, loc='lower left')
    _panel_label(ax1, 'a')

    ax2.set_xticks(range(len(step_labels)))
    ax2.set_xticklabels(step_labels, rotation=0, ha='center', fontsize=8)
    ax2.set_xlabel('Filter step')
    ax2.set_ylabel('Targets (≥1 mol.)')
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.15, linestyle='--')
    ax2.legend(frameon=True, edgecolor='#cccccc', fontsize=9, loc='upper right')
    _panel_label(ax2, 'b')

    fig.suptitle('Ablation — Virtual Screening Funnel', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_funnel.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 7: ECFP4 target-level recovery curve
# ══════════════════════════════════════════════════════════════════════════
def fig7_ecfp4_recovery():
    """ECFP4 threshold sweep: (a) target-level success, (b) total molecules."""
    print("\n[Fig 7] ECFP4 recovery curve")

    TAU_VALS = np.linspace(0.10, 0.60, 51)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for method in METHOD_ORDER:
        sub = df[df['mode'] == method]
        targets_success = []
        total_hits = []
        for tau in TAU_VALS:
            per_target_hits = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: int((x >= t).sum()))
            targets_success.append(int((per_target_hits >= 1).sum()))
            total_hits.append(int((sub['max_ecfp4_tanimoto'] >= tau).sum()))

        ax1.plot(TAU_VALS, targets_success, color=ABL_COLORS[method],
                linewidth=2.5, marker='.', markersize=2, label=_short(method))
        ax2.plot(TAU_VALS, total_hits, color=ABL_COLORS[method],
                linewidth=2.5, marker='.', markersize=2, label=_short(method))

    ax1.set_xlabel('ECFP4 Tanimoto Threshold τ')
    ax1.set_ylabel('Targets with ≥1 recovered molecule')
    ax1.set_ylim(0, df['pdb_id'].nunique() + 2)
    ax1.grid(alpha=0.15, linestyle='--')
    ax1.legend(frameon=True, edgecolor='#cccccc', fontsize=9)
    _panel_label(ax1, 'a')

    ax2.set_yscale('log')
    ax2.set_xlabel('ECFP4 Tanimoto Threshold τ')
    ax2.set_ylabel('Total thresholded molecules')
    ax2.grid(alpha=0.15, linestyle='--')
    ax2.legend(frameon=True, edgecolor='#cccccc', fontsize=9)
    _panel_label(ax2, 'b')

    fig.suptitle('Ablation — ECFP4 Reference-Ligand Recovery', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_ecfp4_curve.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 8: Trade-off analysis — 2D density of Affinity vs QED
# ══════════════════════════════════════════════════════════════════════════
def fig8_tradeoff():
    """2D kernel density + marginal: where do molecules land on the
    affinity–QED and affinity–SA trade-off planes?"""
    print("\n[Fig 8] Trade-off analysis")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    pairs = [
        ('affinity', 'qed', 'Docking Affinity (kcal/mol)', 'QED',
         lambda x, y: x < -7 and y > 0.5),
        ('affinity', 'sa_score', 'Docking Affinity (kcal/mol)', 'SA Score',
         lambda x, y: x < -7 and y < 5),
        ('SEI', 'LLE', 'SEI', 'LLE',
         lambda x, y: x > 0.1 and y > 2),
        ('qed', 'sa_score', 'QED', 'SA Score',
         lambda x, y: x > 0.5 and y < 5),
    ]

    for ax_i, (ax, (col_x, col_y, xlab, ylab, good_region)) in enumerate(
        zip(axes.flat, pairs)
    ):
        for j, method in enumerate(METHOD_ORDER):
            sub = df[df['mode'] == method]
            x_vals = sub[col_x].dropna()
            y_vals = sub[col_y].dropna()
            # Subsample for speed
            if len(x_vals) > 2000:
                idx = np.random.default_rng(42).choice(len(x_vals), 2000, replace=False)
                x_vals = x_vals.iloc[idx]
                y_vals = y_vals.iloc[idx]

            # 2D KDE contour
            try:
                sns.kdeplot(x=x_vals, y=y_vals, ax=ax, color=ABL_COLORS[method],
                           levels=6, linewidths=1.2, alpha=0.7, label=_short(method))
            except Exception:
                ax.scatter(x_vals, y_vals, alpha=0.15, s=5, color=ABL_COLORS[method])

        # "Good" region shading
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        # Determine quadrant for good region
        rect_x = min(x_lim[0], -7) if col_x == 'affinity' else (0 if col_x in ('SEI', 'qed') else 2)
        ax.fill_between([rect_x, x_lim[1]], y_lim[0], y_lim[1], alpha=0.05, color='#2E7D32')

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.15, linestyle='--')
        ax.legend(frameon=True, edgecolor='#cccccc', fontsize=7)

        panel_label = chr(ord('a') + ax_i)
        _panel_label(ax, panel_label)

    fig.suptitle('Ablation — Trade-off Analysis (KDE contours)', fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_tradeoff.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 9: MW-segmented small multiples — bar charts
# ══════════════════════════════════════════════════════════════════════════
def fig9_mw_segmented():
    """Small multiples: 4 MW bins × key metrics as faceted bar charts."""
    print("\n[Fig 9] MW-segmented analysis")

    mw_bins = ['0–120', '120-180', '180–300', '300-420', '≥420']
    metrics_plot = ['affinity', 'qed', 'sa_score']
    metric_labels = ['Affinity (kcal/mol)', 'QED', 'SA Score']

    fig, axes = plt.subplots(len(mw_bins), len(metrics_plot), figsize=(18, 20))

    n_methods = len(METHOD_ORDER)
    width = 0.18
    offsets = np.linspace(-(n_methods - 1) * width / 2, (n_methods - 1) * width / 2, n_methods)

    for i, mw_bin in enumerate(mw_bins):
        for j, (col, label) in enumerate(zip(metrics_plot, metric_labels)):
            ax = axes[i, j]
            for k, method in enumerate(METHOD_ORDER):
                sub = df[(df['mode'] == method) & (df['mw_bin'] == mw_bin)]
                if len(sub) == 0:
                    continue
                pt = per_target_median(sub, col)
                mean_val = pt.mean()
                std_val = pt.std()
                bar = ax.bar(j + offsets[k], mean_val, width, yerr=std_val,
                           color=ABL_COLORS[method], capsize=2,
                           edgecolor='white', linewidth=0.3,
                           error_kw={'linewidth': 0.8})
                # Annotate n
                if k == 0:
                    ax.text(j + offsets[k], mean_val - std_val - 0.1,
                           f'n={len(sub)}', fontsize=5, ha='center', color='#666666')

            if j == 0:
                ax.set_ylabel(f'MW {mw_bin}', fontsize=11, fontweight='bold')
            if i == 0:
                ax.set_title(label, fontweight='bold', fontsize=12)
            if i == len(mw_bins) - 1:
                ax.set_xticks(np.arange(len(metrics_plot)))
            ax.grid(axis='y', alpha=0.15, linestyle='--')

    # Single legend at bottom
    legend_elements = [Line2D([0], [0], color=ABL_COLORS[m], linewidth=3,
                             label=_short(m)) for m in METHOD_ORDER]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(METHOD_ORDER),
              fontsize=10, frameon=True, edgecolor='#cccccc',
              bbox_to_anchor=(0.5, 0.005))
    fig.suptitle('Ablation — MW-Segmented Analysis', fontsize=15, fontweight='bold', y=1.005)
    savefig('fig_abl_mw_segmented.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 10: Pairwise statistical significance heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig10_significance_matrix():
    """Heatmap of pairwise Wilcoxon P-values across all metric pairs.
    Shows which ablation comparisons are statistically significant."""
    print("\n[Fig 10] Pairwise significance matrix")

    metrics_sig = ['affinity', 'qed', 'sa_score', 'SEI', 'LLE', 'rmsd', 'max_ecfp4_tanimoto']
    metric_labels_sig = ['Affinity', 'QED', 'SA Score', 'SEI', 'LLE', 'RMSD', 'ECFP4-TS']

    # Compute pairwise P-values
    pval_matrix = np.zeros((len(metrics_sig), len(METHOD_ORDER), len(METHOD_ORDER)))
    pval_matrix[:] = np.nan

    for mi, col in enumerate(metrics_sig):
        for i, m1 in enumerate(METHOD_ORDER):
            for j, m2 in enumerate(METHOD_ORDER):
                if i >= j:
                    continue
                v1 = per_target_median(df[df['mode'] == m1], col)
                v2 = per_target_median(df[df['mode'] == m2], col)
                common = sorted(set(v1.index) & set(v2.index))
                if len(common) < 3:
                    continue
                diffs = v1[common].values - v2[common].values
                try:
                    _, p = wilcoxon(diffs, alternative='two-sided', zero_method='wilcox')
                    pval_matrix[mi, i, j] = p
                    pval_matrix[mi, j, i] = p
                except ValueError:
                    pass

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for mi, (ax, col, label) in enumerate(zip(axes, metrics_sig, metric_labels_sig)):
        pmat = pval_matrix[mi]
        mask = np.triu(np.ones_like(pmat, dtype=bool), k=1)
        pmat_display = np.where(mask, pmat, np.nan)

        sns.heatmap(pmat_display, ax=ax, annot=True,
                   fmt='.1e', cmap='RdYlGn_r',
                   vmin=1e-10, vmax=1.0,
                   xticklabels=[_short(m) for m in METHOD_ORDER],
                   yticklabels=[_short(m) for m in METHOD_ORDER],
                   linewidths=0.5, linecolor='white',
                   cbar_kws={'shrink': 0.7, 'label': 'Wilcoxon P'},
                   annot_kws={'fontsize': 7})
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)

        panel_label = chr(ord('a') + mi)
        _panel_label(ax, panel_label)

    # Hide unused subplot
    if len(metrics_sig) < 8:
        axes[-1].set_visible(False)

    fig.suptitle('Ablation — Pairwise Significance Matrix\n(upper triangle: P-value, lower: nan)',
                 fontsize=15, fontweight='bold', y=1.01)
    savefig('fig_abl_significance.png')


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fig1_ablation_overview()
    fig2_property_distribution()
    fig3_waterfall()
    fig4_paired_dotplot()
    fig5_delta_heatmap()
    fig6_stepped_funnel()
    fig7_ecfp4_recovery()
    fig8_tradeoff()
    fig9_mw_segmented()
    fig10_significance_matrix()
    print(f"\nAll figures saved to: {OUT_DIR}")
