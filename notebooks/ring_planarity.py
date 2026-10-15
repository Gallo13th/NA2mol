"""
Ring planarity analysis: detect saturated rings that are geometrically planar,
indicating potential aromatic→aliphatic mislabeling by the generative model.
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from rdkit import Chem
from rdkit.Chem import rdchem

# ── Paths & Config ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent / 'output'
CACHE_DIR = OUT_DIR / 'cache_ring_planarity'

# SDF directory → method label (short name for display)
SDF_METHOD_MAP = {
    'smrtnet_onlygeneral_0504': 'RNA2smol\n(Only General)',
    'ed2mol': 'ED2Mol',
    'molcraft_minimized': 'MolCRAFT',
    'abl_generalfeat_cst': 'Generalfeat\nCst',
    'abl_generalfeat_nocst': 'Generalfeat\nNocst',
    'abl_allfeat_cst': 'Allfeat\nCst',
    'abl_allfeat_nocst': 'Allfeat\nNocst',
}

# Colors matching visualize_methods.py palette
MAIN_COLORS = {
    'RNA2smol (Baseline)': '#D55E00',
    'RNA2smol (Only General Feat)': '#E69F00',
    'MolCRAFT': '#0072B2',
    'Pocket2Mol': '#009E73',
    'ED2Mol': '#CC79A7',
}
ABLATION_COLORS = ['#666666', '#888888', '#AAAAAA', '#CCCCCC']

METHOD_DISPLAY_ORDER = [
    'RNA2smol\n(Only General)', 'ED2Mol', 'MolCRAFT',
    'Generalfeat\nCst', 'Generalfeat\nNocst',
    'Allfeat\nCst', 'Allfeat\nNocst',
]

PLANAR_THRESHOLD = 0.05   # Å RMSD — below this, ring is considered planar
NEARLY_PLANAR = 0.15      # Å RMSD — below this, ring is nearly planar

# ── Global matplotlib style ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_planarity_rmsd(mol, ring_indices):
    """RMSD of ring atoms from the best-fit plane (SVD-based)."""
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in ring_indices])
    centroid = pos.mean(axis=0)
    centered = pos - centroid
    _, s, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    distances = np.abs(centered @ normal)
    return float(np.sqrt(np.mean(distances ** 2)))


def count_ring_double_bonds(mol, ring_indices):
    """Count double bonds entirely within the ring."""
    ring_set = set(ring_indices)
    n = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in ring_set and bond.GetEndAtomIdx() in ring_set:
            if bond.GetBondType() == rdchem.BondType.DOUBLE:
                n += 1
    return n


def is_fused_to_aromatic(mol, ring_indices, all_rings):
    """True if this ring shares >= 2 atoms with any aromatic ring."""
    ring_set = set(ring_indices)
    for other in all_rings:
        if set(other) == ring_set:
            continue
        shared = ring_set & set(other)
        if len(shared) >= 2:
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in other):
                return True
    return False


def analyze_molecule(mol):
    """Extract ring planarity data for one molecule. Returns list of dicts."""
    if mol is None:
        return []
    try:
        rings = mol.GetRingInfo().AtomRings()
    except Exception:
        return []
    if not rings:
        return []

    results = []
    for ring in rings:
        sz = len(ring)
        if sz < 3:
            continue
        is_arom = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        rmsd = compute_planarity_rmsd(mol, ring)
        n_db = count_ring_double_bonds(mol, ring)
        fused_arom = is_fused_to_aromatic(mol, ring, rings)

        results.append({
            'ring_size': sz,
            'is_aromatic': is_arom,
            'planarity_rmsd': rmsd,
            'n_double_bonds': n_db,
            'fused_to_aromatic': fused_arom,
        })
    return results


def _short_name(label):
    """Simplify method label for compact display."""
    return label.replace('\n', ' ')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 & 2: Load SDFs and analyze rings
# ══════════════════════════════════════════════════════════════════════════════

def load_all_rings():
    """Iterate all SDF directories, analyze every molecule, return DataFrame."""
    cache_path = CACHE_DIR / 'all_rings.pkl'
    if cache_path.exists():
        print(f"Loading cached ring data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    all_rows = []
    for sdf_dir_name, method_label in SDF_METHOD_MAP.items():
        sdf_dir = OUT_DIR / sdf_dir_name
        if not sdf_dir.exists():
            print(f"  [SKIP] {sdf_dir_name} — directory not found")
            continue

        # Find all SDF files under this directory
        sdf_files = sorted(sdf_dir.rglob('*.sdf'))
        # Filter: prefer per-target merged SDFs (e.g., basename/basename.sdf)
        merged_sdfs = [f for f in sdf_files if f.parent.name + '.sdf' == f.name]
        if not merged_sdfs:
            # Fallback: flat per-target SDFs (e.g., molcraft_minimized/*.sdf)
            merged_sdfs = sdf_files

        print(f"\n{method_label} ({sdf_dir_name}): {len(merged_sdfs)} targets")

        for sdf_path in merged_sdfs:
            target_name = sdf_path.stem  # e.g., '2ku0_A_ISI_ligand'
            # Extract clean target ID (first segment before first underscore patterns)
            parts = target_name.split('_')
            pdb_id = parts[0]  # e.g., '2ku0'

            supplier = Chem.SDMolSupplier(str(sdf_path))
            mol_idx = 0
            for mol in supplier:
                if mol is None:
                    mol_idx += 1
                    continue
                ring_data = analyze_molecule(mol)
                for rd in ring_data:
                    rd.update({
                        'method': method_label,
                        'sdf_dir': sdf_dir_name,
                        'target': target_name,
                        'pdb_id': pdb_id,
                        'mol_idx': mol_idx,
                    })
                    all_rows.append(rd)
                mol_idx += 1

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rings analyzed: {len(df)}")
    print(f"  Aromatic: {df['is_aromatic'].sum()}")
    print(f"  Non-aromatic: {(~df['is_aromatic']).sum()}")

    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"  Cached → {cache_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_stats(df):
    """Compute per-method and per-target planar-ring statistics."""
    # Focus on non-aromatic rings, sizes 5-7
    na = df[~df['is_aromatic'] & df['ring_size'].isin([5, 6, 7])].copy()
    na['is_planar'] = na['planarity_rmsd'] < PLANAR_THRESHOLD
    na['is_nearly_planar'] = na['planarity_rmsd'] < NEARLY_PLANAR
    # "Strict" = no double bonds AND not fused to aromatic
    na['strict_suspicious'] = na['is_planar'] & (na['n_double_bonds'] == 0) & (~na['fused_to_aromatic'])

    # ── Per-method stats ──
    method_stats = []
    for method in df['method'].unique():
        na_m = na[na['method'] == method]
        for sz in [5, 6, 7]:
            na_ms = na_m[na_m['ring_size'] == sz]
            if len(na_ms) == 0:
                continue
            method_stats.append({
                'method': method,
                'ring_size': sz,
                'n_nonaromatic_rings': len(na_ms),
                'n_planar': na_ms['is_planar'].sum(),
                'pct_planar': 100 * na_ms['is_planar'].mean(),
                'n_strict_suspicious': na_ms['strict_suspicious'].sum(),
                'pct_strict_suspicious': 100 * na_ms['strict_suspicious'].mean(),
                'mean_rmsd': na_ms['planarity_rmsd'].mean(),
                'median_rmsd': na_ms['planarity_rmsd'].median(),
            })
    stats_df = pd.DataFrame(method_stats)

    # ── Per-target stats (6-rings only, most informative) ──
    target_stats = []
    for method in df['method'].unique():
        na_m = na[(na['method'] == method) & (na['ring_size'] == 6)]
        for pdb in sorted(na_m['pdb_id'].unique()):
            na_t = na_m[na_m['pdb_id'] == pdb]
            if len(na_t) < 5:
                continue
            target_stats.append({
                'method': method,
                'pdb_id': pdb,
                'n_nonarom_6rings': len(na_t),
                'pct_planar': 100 * na_t['is_planar'].mean(),
                'pct_strict_suspicious': 100 * na_t['strict_suspicious'].mean(),
            })
    target_df = pd.DataFrame(target_stats)

    return stats_df, target_df, na


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def _method_color(method_label):
    """Get a consistent color for a method label."""
    # Check MAIN_COLORS for partial match
    for key, color in MAIN_COLORS.items():
        if key in method_label or method_label in key:
            return color
    # Ablation greys
    abl_labels = [lbl for lbl in METHOD_DISPLAY_ORDER if 'Generalfeat' in lbl or 'Allfeat' in lbl]
    if method_label in abl_labels:
        idx = abl_labels.index(method_label)
        return ABLATION_COLORS[min(idx, len(ABLATION_COLORS) - 1)]
    return '#333333'


def _savefig(filename):
    path = OUT_DIR / filename
    plt.savefig(path)
    print(f"  Saved: {path}")
    plt.close()


def figA_rmsd_histogram(df):
    """Histogram of planarity RMSD: aromatic vs non-aromatic, per method."""
    print("\n[Fig A] Planarity RMSD histogram")
    methods = [m for m in METHOD_DISPLAY_ORDER if m in df['method'].unique()]
    n_methods = len(methods)
    ncols = 4
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    # Shared bins
    all_rmsd = df[df['ring_size'].isin([5, 6, 7])]['planarity_rmsd']
    bins = np.linspace(0, min(0.6, all_rmsd.max() + 0.05), 60)

    for idx, method in enumerate(methods):
        ax = axes[idx]
        dm = df[(df['method'] == method) & df['ring_size'].isin([5, 6, 7])]
        arom = dm[dm['is_aromatic']]['planarity_rmsd']
        nonarom = dm[~dm['is_aromatic']]['planarity_rmsd']

        ax.hist(nonarom, bins=bins, alpha=0.7, color='#E53935', label='Non-aromatic', edgecolor='white', linewidth=0.2)
        ax.hist(arom, bins=bins, alpha=0.7, color='#1E88E5', label='Aromatic', edgecolor='white', linewidth=0.2)
        ax.axvline(x=PLANAR_THRESHOLD, color='#333333', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(x=NEARLY_PLANAR, color='#999999', linestyle=':', linewidth=0.6, alpha=0.5)

        ax.set_title(_short_name(method), fontsize=9, fontweight='bold')
        ax.set_xlabel('Planarity RMSD (Å)')
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right', frameon=True, edgecolor='#cccccc')
        ax.set_xlim(0, bins[-1])

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Ring Planarity RMSD Distribution: Aromatic vs Non-Aromatic',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _savefig('ring_planarity_figA_rmsd_histogram.png')


def figB_planar_fraction(stats_df):
    """Grouped bar chart: % planar non-aromatic rings by method and ring size."""
    print("\n[Fig B] % Planar saturated rings by method")
    methods = [m for m in METHOD_DISPLAY_ORDER if m in stats_df['method'].unique()]
    ring_sizes = [5, 6, 7]
    colors_map = {5: '#FDD835', 6: '#FB8C00', 7: '#D32F2F'}
    hatch_map = {5: '', 6: '', 7: '//'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    x = np.arange(len(methods))
    width = 0.25

    for i, sz in enumerate(ring_sizes):
        vals = []
        for m in methods:
            row = stats_df[(stats_df['method'] == m) & (stats_df['ring_size'] == sz)]
            vals.append(row['pct_planar'].values[0] if len(row) > 0 else np.nan)
        bars = ax1.bar(x + i * width, vals, width, color=colors_map[sz],
                       edgecolor='white', linewidth=0.5, label=f'{sz}-membered')
        # Annotate values above bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, color='#333333')

    ax1.set_ylabel('% Planar (RMSD < 0.05 Å)')
    ax1.set_title('All non-aromatic rings', fontsize=11)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([m.replace('\n', ' ') for m in methods], rotation=30, ha='right', fontsize=7.5)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 105)

    # Panel B: Strict (no double bonds, not fused to aromatic)
    for i, sz in enumerate(ring_sizes):
        vals = []
        for m in methods:
            row = stats_df[(stats_df['method'] == m) & (stats_df['ring_size'] == sz)]
            vals.append(row['pct_strict_suspicious'].values[0] if len(row) > 0 else np.nan)
        bars = ax2.bar(x + i * width, vals, width, color=colors_map[sz],
                       edgecolor='white', linewidth=0.5, label=f'{sz}-membered')
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, color='#333333')

    ax2.set_ylabel('% Planar & Strictly Suspicious')
    ax2.set_title('Excluding double-bond & arom-fused rings', fontsize=11)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([m.replace('\n', ' ') for m in methods], rotation=30, ha='right', fontsize=7.5)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    _savefig('ring_planarity_figB_planar_fraction.png')


def figC_target_heatmap(target_df):
    """Per-target heatmap: % planar non-aromatic 6-rings."""
    print("\n[Fig C] Per-target heatmap")

    # Pivot: rows=targets, cols=methods
    pivot = target_df.pivot_table(index='pdb_id', columns='method', values='pct_planar', aggfunc='mean')

    # Keep only targets with at least 3 methods represented
    pivot = pivot.dropna(thresh=3)

    # Sort targets by mean planar %
    pivot['_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('_mean')
    pivot = pivot.drop(columns=['_mean'])

    # Column order
    col_order = [m for m in METHOD_DISPLAY_ORDER if m in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(10, len(col_order) * 1.5),
                                    max(6, len(pivot) * 0.35)))

    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([c.replace('\n', ' ') for c in col_order], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7.5)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > 60 else '#333333'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=6.5, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('% Planar Non-Aromatic 6-Rings', fontsize=9)

    ax.set_title('% Planar Saturated 6-Membered Rings by Target and Method',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _savefig('ring_planarity_figC_target_heatmap.png')


def figD_rmsd_scatter(df, max_points=2000):
    """Scatter: planarity RMSD vs ring size, colored by method and aromaticity."""
    print("\n[Fig D] RMSD vs ring size scatter")

    dm = df[df['ring_size'].isin([4, 5, 6, 7, 8])].copy()
    # Subsample for performance if needed
    if len(dm) > max_points:
        dm = dm.sample(n=max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [m for m in METHOD_DISPLAY_ORDER if m in dm['method'].unique()]
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    for idx, method in enumerate(methods):
        dmm = dm[dm['method'] == method]
        # Aromatic: transparent
        arom = dmm[dmm['is_aromatic']]
        if len(arom) > 0:
            ax.scatter(arom['ring_size'] + np.random.uniform(-0.2, 0.2, len(arom)),
                       arom['planarity_rmsd'],
                       c=[_method_color(method)], alpha=0.15, s=8,
                       marker='o', linewidths=0, label='_nolegend_')
        # Non-aromatic: more visible
        nonarom = dmm[~dmm['is_aromatic']]
        if len(nonarom) > 0:
            ax.scatter(nonarom['ring_size'] + np.random.uniform(-0.2, 0.2, len(nonarom)),
                       nonarom['planarity_rmsd'],
                       c=[_method_color(method)], alpha=0.5, s=14,
                       marker=markers[idx % len(markers)], linewidths=0,
                       label=_short_name(method))

    ax.axhline(y=PLANAR_THRESHOLD, color='#333333', linestyle='--', linewidth=0.8, alpha=0.7,
               label=f'Planar threshold ({PLANAR_THRESHOLD} Å)')
    ax.axhline(y=NEARLY_PLANAR, color='#999999', linestyle=':', linewidth=0.6, alpha=0.5,
               label=f'Nearly planar ({NEARLY_PLANAR} Å)')

    ax.set_xlabel('Ring Size')
    ax.set_ylabel('Planarity RMSD (Å)')
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.legend(fontsize=7, loc='upper right', ncol=2, frameon=True, edgecolor='#cccccc',
              markerscale=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation: aromatic = open circles, non-aromatic = filled shapes
    ax.text(0.98, 0.95, 'Filled = non-aromatic  |  Faint open circles = aromatic',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    plt.tight_layout()
    _savefig('ring_planarity_figD_scatter.png')


# ══════════════════════════════════════════════════════════════════════════════
# Text summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(stats_df, na_df):
    """Print per-method summary table."""
    print("\n" + "=" * 90)
    print("SUMMARY: % Planar Non-Aromatic Rings by Method and Ring Size")
    print("=" * 90)

    methods = [m for m in METHOD_DISPLAY_ORDER if m in stats_df['method'].unique()]

    for sz in [5, 6, 7]:
        print(f"\n  --- {sz}-membered rings ---")
        print(f"  {'Method':<30} {'# Rings':>8} {'% Planar':>10} {'% Strict':>10} {'Mean RMSD':>10}")
        print(f"  {'-'*68}")
        for m in methods:
            row = stats_df[(stats_df['method'] == m) & (stats_df['ring_size'] == sz)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"  {_short_name(m):<30} {int(r['n_nonaromatic_rings']):>8} "
                  f"{r['pct_planar']:>9.1f}% {r['pct_strict_suspicious']:>9.1f}% "
                  f"{r['mean_rmsd']:>10.4f}")

    print(f"\n  Planarity threshold: RMSD < {PLANAR_THRESHOLD} Å")
    print(f"  'Strict' = no double bonds AND not fused to aromatic ring")

    # Global summary
    na = na_df[na_df['ring_size'].isin([5, 6, 7])]
    total = len(na)
    planar = na['is_planar'].sum()
    print(f"\n  Overall: {planar}/{total} ({100*planar/total:.1f}%) non-aromatic "
          f"5–7 membered rings are planar (RMSD < {PLANAR_THRESHOLD} Å)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading ring data...")
    df = load_all_rings()

    print("\nAggregating statistics...")
    stats_df, target_df, na_df = aggregate_stats(df)

    print_summary(stats_df, na_df)

    print("\nGenerating figures...")
    figA_rmsd_histogram(df)
    figB_planar_fraction(stats_df)
    figC_target_heatmap(target_df)
    figD_rmsd_scatter(df)

    # Save stats CSV
    stats_csv = OUT_DIR / 'ring_planarity_stats.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nStats saved: {stats_csv}")
    print("\nDone.")
