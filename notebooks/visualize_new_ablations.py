"""
Ablation comparison visualization for the two new ablation experiments:
  - Pretrain Only
  - Pretrain+Constrain

Compared against existing:
  - RNA2smol (Only General Feat)  [= the full model]
  - RNA2smol (Baseline)

Mirrors the style of visualize_methods.py for consistency.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'output' / 'smrtbench'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = SCRIPT_DIR / 'output' / 'smrtbench_combined_results.csv'
df_full = pd.read_csv(CSV_PATH)

# Define ablation method groups
ABL_METHODS = [
    'RNA2smol (Baseline)',
    'Pretrain Only',
    'Pretrain+Constrain',
    'RNA2smol (Only General Feat)',  # full model
]

# Filter to only these methods
df = df_full[df_full['mode'].isin(ABL_METHODS)].copy()

# Ensure derived metrics
df['SEI'] = -df['affinity'] / df['tpsa'].replace(0, np.nan)
df['SEI'] = df['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
df['LLE'] = -df['affinity'] - df['logp']
df['LBE'] = df['affinity'] / df['heavy_atoms'].replace(0, np.nan)

METHOD_COLORS = {
    'RNA2smol (Baseline)': '#b0b0b0',
    'Pretrain Only': '#5B9BD5',
    'Pretrain+Constrain': '#ED7D31',
    'RNA2smol (Only General Feat)': '#4472C4',
}
METHOD_ORDER = [
    'RNA2smol (Baseline)',
    'Pretrain Only',
    'Pretrain+Constrain',
    'RNA2smol (Only General Feat)',
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------------
# Helper: per-target aggregation
# ---------------------------------------------------------------------------
def per_target_median(data, col):
    return data.groupby('pdb_id')[col].median()

def agg_summary(data, col, methods=METHOD_ORDER):
    rows = []
    for m in methods:
        sub = data[data['mode'] == m]
        pt = per_target_median(sub, col)
        rows.append({'method': m, 'mean': pt.mean(), 'std': pt.std(),
                     'median': pt.median(), 'n_targets': len(pt)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Figure 1: Bar chart comparison of key metrics
# ---------------------------------------------------------------------------
def plot_ablation_barchart():
    print("Generating ablation bar chart...")
    metrics = [
        ('affinity', 'Docking Affinity (kcal/mol)', 'lower'),
        ('qed', 'QED', 'higher'),
        ('sa_score', 'SA Score', 'lower'),
        ('rmsd', 'RMSD (Å)', 'lower'),
        ('SEI', 'SEI', 'higher'),
        ('LLE', 'LLE', 'higher'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for ax, (col, label, direction) in zip(axes, metrics):
        summary = agg_summary(df, col)
        summary = summary.set_index('method').reindex(METHOD_ORDER).reset_index()
        bars = ax.bar(
            summary['method'], summary['mean'],
            yerr=summary['std'],
            color=[METHOD_COLORS[m] for m in summary['method']],
            edgecolor='black', linewidth=0.8, capsize=5,
        )
        ax.set_title(label, fontweight='bold')
        ax.set_xticklabels(summary['method'], rotation=25, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Annotate with mean values
        for bar, val in zip(bars, summary['mean']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=8, fontweight='bold')

    fig.suptitle('Ablation Comparison — Key Metrics (Per-Target Median ± Std)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_new_ablation_barchart.png', bbox_inches='tight')
    plt.close()
    print("  Saved: fig_new_ablation_barchart.png")


# ---------------------------------------------------------------------------
# Figure 2: Scatter comparison (Affinity, QED, SA)
# ---------------------------------------------------------------------------
def plot_ablation_scatter():
    print("Generating ablation scatter plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prepare per-target data for scatter
    scatter_metrics = [
        ('affinity', 'Docking Affinity (kcal/mol)'),
        ('qed', 'QED'),
        ('sa_score', 'SA Score'),
    ]

    for ax, (col, label) in zip(axes, scatter_metrics):
        # Build a dataframe of per-target values for each method
        methods_data = {}
        for m in METHOD_ORDER:
            sub = df[df['mode'] == m]
            pt = per_target_median(sub, col)
            methods_data[m] = pt

        # Use shared targets between baseline and full model
        common_targets = sorted(
            set(methods_data['RNA2smol (Baseline)'].index) &
            set(methods_data['RNA2smol (Only General Feat)'].index)
        )

        for m in METHOD_ORDER:
            vals = methods_data[m].reindex(common_targets)
            if m == 'RNA2smol (Baseline)':
                x_vals = vals.values
            else:
                y_vals = vals.values
                ax.scatter(x_vals, y_vals, alpha=0.7, s=40,
                          color=METHOD_COLORS[m], label=m, edgecolors='black', linewidth=0.3)

        # Identity line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('RNA2smol (Baseline)', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} vs Baseline', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Ablation — Per-Target Scatter vs Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_new_ablation_scatter.png', bbox_inches='tight')
    plt.close()
    print("  Saved: fig_new_ablation_scatter.png")


# ---------------------------------------------------------------------------
# Figure 3: Virtual screening funnel for ablations
# ---------------------------------------------------------------------------
def plot_ablation_funnel():
    print("Generating ablation VS funnel...")
    filter_steps = [
        ('All', None),
        ('aff<-7', lambda d: d['affinity'] < -7),
        ('+RMSD<2.0', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0)),
        ('+QED>0.4,SA<5', lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
        ('aff<-8,RMSD<1.5\nQED>0.5,SA<5', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
        ('+SEI>0.1', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.1)),
        ('aff<-9,RMSD<1.0\nQED>0.5,SA<5\nSEI>0.15', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.15)),
        ('aff<-9,RMSD<1.0\nQED>0.6,SA<4\nSEI>0.2', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (ax, metric_name) in enumerate(zip(axes, ['Molecules Retained', 'Targets Covered'])):
        for m in METHOD_ORDER:
            sub = df[df['mode'] == m]
            x_vals = []
            y_vals = []
            for step_name, flt in filter_steps:
                filtered = sub if flt is None else sub[flt(sub)]
                x_vals.append(step_name)
                if metric_name == 'Molecules Retained':
                    y_vals.append(len(filtered))
                else:
                    y_vals.append(filtered['pdb_id'].nunique())
            ax.plot(x_vals, y_vals, 'o-', color=METHOD_COLORS[m], label=m,
                    linewidth=2, markersize=7)
        ax.set_title(metric_name, fontweight='bold')
        ax.set_xticklabels(x_vals, rotation=30, ha='right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Ablation — Virtual Screening Funnel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_new_ablation_funnel.png', bbox_inches='tight')
    plt.close()
    print("  Saved: fig_new_ablation_funnel.png")


# ---------------------------------------------------------------------------
# Figure 4: ECFP4 target-level success across thresholds
# ---------------------------------------------------------------------------
def plot_ablation_ecfp4():
    print("Generating ablation ECFP4 comparison...")
    TAU_VALS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Target-level success (≥1 molecule at τ)
    ax = axes[0]
    x = np.arange(len(TAU_VALS))
    width = 0.2
    for i, m in enumerate(METHOD_ORDER):
        sub = df[df['mode'] == m]
        successes = []
        for tau in TAU_VALS:
            per_target_hits = sub.groupby('pdb_id')['max_ecfp4_tanimoto'].apply(
                lambda x, t=tau: int((x >= t).sum()))
            successes.append(int((per_target_hits >= 1).sum()))
        bars = ax.bar(x + i * width, successes, width, color=METHOD_COLORS[m],
                     label=m, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, successes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'τ={t}' for t in TAU_VALS])
    ax.set_ylabel('Successful Targets (≥1 molecule)', fontsize=11)
    ax.set_title('Target-Level ECFP4 Recovery', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Right: Total thresholded molecules
    ax = axes[1]
    for i, m in enumerate(METHOD_ORDER):
        sub = df[df['mode'] == m]
        totals = []
        for tau in TAU_VALS:
            totals.append(int((sub['max_ecfp4_tanimoto'] >= tau).sum()))
        bars = ax.bar(x + i * width, totals, width, color=METHOD_COLORS[m],
                     label=m, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha='center', fontsize=7, fontweight='bold', rotation=90)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'τ={t}' for t in TAU_VALS])
    ax.set_ylabel('Total Thresholded Molecules', fontsize=11)
    ax.set_title('Molecule-Level ECFP4 Hits', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Ablation — ECFP4 Reference-Ligand Recovery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_new_ablation_ecfp4.png', bbox_inches='tight')
    plt.close()
    print("  Saved: fig_new_ablation_ecfp4.png")


# ---------------------------------------------------------------------------
# Figure 5: Per-metric delta vs Baseline (paired targets)
# ---------------------------------------------------------------------------
def plot_ablation_delta():
    print("Generating ablation delta plots...")
    metrics = ['affinity', 'qed', 'sa_score', 'rmsd', 'SEI', 'LLE',
               'max_ecfp4_tanimoto']

    baseline_pt = {}
    sub_base = df[df['mode'] == 'RNA2smol (Baseline)']
    for col in metrics:
        baseline_pt[col] = per_target_median(sub_base, col)

    comparators = ['Pretrain Only', 'Pretrain+Constrain', 'RNA2smol (Only General Feat)']
    metric_labels = ['Affinity', 'QED', 'SA Score', 'RMSD', 'SEI', 'LLE', 'ECFP4-TS']
    directions = ['lower', 'higher', 'lower', 'lower', 'higher', 'higher', 'higher']

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6))
    for ax, col, label, direction in zip(axes, metrics, metric_labels, directions):
        x_pos = np.arange(len(comparators))
        means = []
        stds = []
        for comp in comparators:
            sub_comp = df[df['mode'] == comp]
            comp_pt = per_target_median(sub_comp, col)
            common = sorted(set(baseline_pt[col].index) & set(comp_pt.index))
            if len(common) == 0:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            diffs = comp_pt[common].values - baseline_pt[col][common].values
            means.append(diffs.mean())
            stds.append(diffs.std())

        colors = [METHOD_COLORS[c] for c in comparators]
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5,
                     edgecolor='black', linewidth=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace('RNA2smol (Only General Feat)', 'Full Model')
                           for c in comparators], rotation=25, ha='right', fontsize=8)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, means):
            if not np.isnan(val):
                y_pos = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{val:+.3f}', ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=7, fontweight='bold')

    fig.suptitle('Ablation — Δ vs Baseline (Per-Target Median, Mean ± Std across Targets)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_new_ablation_delta.png', bbox_inches='tight')
    plt.close()
    print("  Saved: fig_new_ablation_delta.png")


# ---------------------------------------------------------------------------
# Figure 6: Summary stats table exported as CSV + printed
# ---------------------------------------------------------------------------
def export_ablation_stats():
    print("\n" + "="*70)
    print("Ablation Summary Statistics")
    print("="*70)

    metrics = ['affinity', 'qed', 'sa_score', 'rmsd', 'SEI', 'LLE',
               'max_ecfp4_tanimoto']
    rows = []

    for m in METHOD_ORDER:
        sub = df[df['mode'] == m]
        for col in metrics:
            pt = per_target_median(sub, col)
            rows.append({
                'method': m, 'metric': col,
                'mean': pt.mean(), 'std': pt.std(),
                'median': pt.median(), 'n_targets': len(pt),
                'n_mols': len(sub),
            })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUTPUT_DIR / 't_new_ablation_stats.csv', index=False,
                    float_format='%.4f')
    print("  Saved: t_new_ablation_stats.csv")

    # Print key comparisons
    for col in ['affinity', 'qed', 'sa_score']:
        print(f"\n  --- {col} ---")
        for m in METHOD_ORDER:
            s = stats_df[(stats_df['method'] == m) & (stats_df['metric'] == col)]
            if len(s) > 0:
                print(f"    {m:40s}: mean={s['mean'].values[0]:.4f} ± {s['std'].values[0]:.4f}")

    # Paired stats vs baseline
    print("\n  --- Paired Δ vs Baseline (shared targets) ---")
    baseline_pt = {}
    sub_base = df[df['mode'] == 'RNA2smol (Baseline)']
    for col in metrics:
        baseline_pt[col] = per_target_median(sub_base, col)

    for comp in ['Pretrain Only', 'Pretrain+Constrain', 'RNA2smol (Only General Feat)']:
        print(f"\n  {comp}:")
        sub_comp = df[df['mode'] == comp]
        for col in ['affinity', 'qed', 'sa_score', 'SEI', 'max_ecfp4_tanimoto']:
            comp_pt = per_target_median(sub_comp, col)
            common = sorted(set(baseline_pt[col].index) & set(comp_pt.index))
            if len(common) < 2:
                continue
            diffs = comp_pt[common].values - baseline_pt[col][common].values
            try:
                p = wilcoxon(diffs, alternative='two-sided', zero_method='wilcox').pvalue
            except ValueError:
                p = np.nan
            print(f"    {col:25s}: Δ={diffs.mean():+.4f}, CI=[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}], P={p:.2e}")

    # VS funnel summary
    print("\n  --- Virtual Screening Funnel (Strictest Filter) ---")
    flt = lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2)
    for m in METHOD_ORDER:
        sub = df[df['mode'] == m]
        f = sub[flt(sub)]
        print(f"    {m:40s}: {len(f)} mols, {f['pdb_id'].nunique()} targets")

    return stats_df


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Data: {len(df)} molecules, {df['mode'].nunique()} modes, {df['pdb_id'].nunique()} targets")
    print(f"Modes: {df['mode'].unique().tolist()}")
    print()

    plot_ablation_barchart()
    plot_ablation_scatter()
    plot_ablation_funnel()
    plot_ablation_ecfp4()
    plot_ablation_delta()
    stats = export_ablation_stats()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
