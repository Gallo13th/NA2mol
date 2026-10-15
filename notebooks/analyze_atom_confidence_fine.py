#!/usr/bin/env python
"""Fine-grained atom confidence analysis: atom type, ring, scaffold, BRICS, local geometry."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# ── Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'output'
OUT_DIR.mkdir(exist_ok=True)
SDF_PATH = SCRIPT_DIR.parent / 'downstream_project' / 'fluomolgen' / '7kvv_sp1' / 'genmols_raw.sdf'

plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica'], 'font.family': 'sans-serif',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'grid.alpha': 0.15, 'grid.linestyle': '--', 'grid.linewidth': 0.4,
})

def _panel_label(ax, label, x=-0.06, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='left')

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Extract per-atom data from all molecules
# ══════════════════════════════════════════════════════════════════════════
print("Loading SDF and computing per-atom features...")
supplier = Chem.SDMolSupplier(str(SDF_PATH))

# Ideal covalent radii (in Angstroms) for common elements
COVALENT_RADII = {6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02,
                  35: 1.20, 53: 1.39, 1: 0.31}
# Typical single bond distances (sum of radii)
ELEM_VALENCE_RADII = {6: 0.76, 7: 0.71, 8: 0.66, 9: 0.64, 15: 1.10, 16: 1.04, 17: 0.99}

all_atom_rows = []
mol_counter = 0

for mol_idx, mol in enumerate(supplier):
    if mol is None:
        continue
    try:
        conf_str = mol.GetProp('_AtomConfidenceList')
        conf_arr = np.array([float(x) for x in conf_str.split(',')])
    except Exception:
        continue

    n_atoms = mol.GetNumAtoms()
    if len(conf_arr) != n_atoms:
        continue

    mol_counter += 1
    conf = mol.GetConformer()
    ring_info = mol.GetRingInfo()

    # ── Murcko scaffold labeling ──
    is_scaffold = np.zeros(n_atoms, dtype=bool)
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is not None and scaffold_mol.GetNumAtoms() > 0:
            match = mol.GetSubstructMatch(scaffold_mol)
            if match:
                is_scaffold[list(match)] = True
    except Exception:
        pass

    # ── BRICS decomposition and junction labeling ──
    is_junction = np.zeros(n_atoms, dtype=bool)
    frag_size = np.full(n_atoms, np.nan)
    n_frags_in_mol = 0
    try:
        frags_mol = BRICS.BreakBRICSBonds(mol)
        frag_indices = Chem.GetMolFrags(frags_mol, asMols=False)
        n_frags_in_mol = len(frag_indices)
        # Junction atoms: atoms connected to atoms in other fragments
        for i, frag_i in enumerate(frag_indices):
            frag_i_set = set(frag_i)
            for a_idx in frag_i:
                frag_size[a_idx] = len(frag_i)
                for nbr in mol.GetAtomWithIdx(a_idx).GetNeighbors():
                    if nbr.GetIdx() not in frag_i_set:
                        is_junction[a_idx] = True
    except Exception:
        pass

    # ── Per-atom features ──
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos_i = conf.GetAtomPosition(idx)

        anum = atom.GetAtomicNum()

        # Neighbor distances
        nbr_dists = []
        nbr_elements = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            pos_j = conf.GetAtomPosition(nbr_idx)
            nbr_dists.append(pos_i.Distance(pos_j))
            nbr_elements.append(nbr.GetAtomicNum())

        mean_d = np.mean(nbr_dists) if nbr_dists else np.nan
        # Deviation from ideal: actual - sum of covalent radii
        ideal_dists = []
        for nbr_anum, d in zip(nbr_elements, nbr_dists):
            r1 = COVALENT_RADII.get(anum, 0.75)
            r2 = COVALENT_RADII.get(nbr_anum, 0.75)
            ideal_dists.append(d - (r1 + r2))
        mean_bond_dev = np.mean(ideal_dists) if ideal_dists else np.nan

        # Determine ring context
        ring_sizes = [len(r) for r in ring_info.AtomRings() if idx in r]
        min_ring = min(ring_sizes) if ring_sizes else 0
        is_fused = len(ring_sizes) >= 2

        # Scaffold context
        if is_scaffold[idx]:
            scaffold_ctx = 'scaffold_ring' if atom.IsInRing() else 'scaffold_linker'
        else:
            scaffold_ctx = 'substituent'

        # BRICS context
        if n_frags_in_mol <= 1:
            brics_ctx = 'no_cut'
        elif is_junction[idx]:
            brics_ctx = 'junction'
        else:
            brics_ctx = 'fragment_interior'

        all_atom_rows.append({
            'mol_idx': mol_idx,
            'atom_idx': idx,
            'confidence': conf_arr[idx],
            'element': atom.GetSymbol(),
            'atomic_num': anum,
            'hybridization': str(atom.GetHybridization()),
            'is_aromatic': atom.GetIsAromatic(),
            'is_in_ring': atom.IsInRing(),
            'min_ring_size': min_ring,
            'is_fused': is_fused,
            'n_rings': len(ring_sizes),
            'degree': atom.GetDegree(),
            'total_degree': atom.GetTotalDegree(),
            'formal_charge': atom.GetFormalCharge(),
            'num_Hs': atom.GetTotalNumHs(),
            'mean_nbr_dist': mean_d,
            'mean_bond_dev': mean_bond_dev,
            'n_nbrs': len(nbr_dists),
            'is_scaffold': is_scaffold[idx],
            'scaffold_ctx': scaffold_ctx,
            'brics_ctx': brics_ctx,
            'frag_size': frag_size[idx],
            'n_frags_in_mol': n_frags_in_mol,
        })

    # Progress
    if mol_counter % 2000 == 0:
        print(f"  Processed {mol_counter} molecules, {len(all_atom_rows):,} atoms...")

atom_df = pd.DataFrame(all_atom_rows)
print(f"  Total: {mol_counter} molecules, {len(atom_df):,} atoms")
print(f"  Scaffold atoms: {atom_df['is_scaffold'].sum():,} ({atom_df['is_scaffold'].mean()*100:.1f}%)")
print(f"  Junction atoms: {(atom_df['brics_ctx']=='junction').sum():,}")
print(f"  Atom types: {atom_df['element'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════
# Print summary tables
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D1: CONFIDENCE BY ELEMENT & HYBRIDIZATION")
print("=" * 70)

for group_name, group_df in [
    ('Element', atom_df.groupby('element')),
    ('Hybridization', atom_df.groupby('hybridization')),
]:
    print(f"\n  --- {group_name} ---")
    print(f"  {'Group':<16s} {'N':>8s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    stats = group_df['confidence'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    stats = stats.sort_values('median', ascending=False)
    for grp, row in stats.iterrows():
        print(f"  {str(grp):<16s} {int(row['count']):8,d} {row['mean']:8.4f} {row['median']:8.4f} "
              f"{row['std']:8.4f} {row['min']:8.4f} {row['max']:8.4f}")

# Cross-tab: element x hybridization
print(f"\n  --- Element x Hybridization (mean confidence) ---")
pivot = atom_df.pivot_table(values='confidence', index='element', columns='hybridization',
                             aggfunc='mean')
# Keep only common elements
common_elems = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl']
print(pivot.loc[pivot.index.isin(common_elems)].round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D2: RING MEMBERSHIP ANALYSIS")
print("=" * 70)

for group_name, col in [('Ring vs acyclic', 'is_in_ring'), ('Aromatic vs non-aromatic', 'is_aromatic')]:
    grouped = atom_df.groupby(col)['confidence']
    print(f"\n  --- {group_name} ---")
    for grp, sub in grouped:
        print(f"  {grp}: N={len(sub):,}, mean={sub.mean():.4f}, median={sub.median():.4f}, std={sub.std():.4f}")

# Ring size analysis
ring_atoms = atom_df[atom_df['is_in_ring']].copy()
ring_atoms['ring_size_bin'] = ring_atoms['min_ring_size'].apply(
    lambda x: str(x) if x in [3, 4, 5, 6] else '7+' if x >= 7 else 'none')
print(f"\n  --- Ring size ---")
ring_stats = ring_atoms.groupby('ring_size_bin')['confidence'].agg(['count', 'mean', 'median', 'std'])
ring_stats = ring_stats.reindex(['3', '4', '5', '6', '7+'])
print(ring_stats.to_string())

# Fused vs single ring
print(f"\n  --- Fused ring atoms ---")
for grp, sub in atom_df.groupby('is_fused'):
    print(f"  Fused={grp}: N={len(sub):,}, mean={sub['confidence'].mean():.4f}, "
          f"median={sub['confidence'].median():.4f}")

# Heteroatom in 6-ring
ring6 = ring_atoms[ring_atoms['min_ring_size'] == 6]
for grp, sub in ring6.groupby(ring6['element'].apply(lambda e: 'C' if e == 'C' else 'hetero (N,O,S)')):
    print(f"  6-ring {grp}: N={len(sub):,}, mean={sub['confidence'].mean():.4f}, "
          f"median={sub['confidence'].median():.4f}")


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D3: LOCAL COORDINATION ENVIRONMENT")
print("=" * 70)

for group_col in ['degree', 'total_degree']:
    print(f"\n  --- {group_col} ---")
    stats = atom_df.groupby(group_col)['confidence'].agg(['count', 'mean', 'median', 'std'])
    print(stats.to_string())

# Bond distance deviation
print(f"\n  --- Bond distance deviation vs confidence ---")
atom_df['bond_dev_bin'] = pd.cut(atom_df['mean_bond_dev'], bins=[-1, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 1.0],
                                  labels=['<−0.2', '−0.2:−0.1', '−0.1:−0.05', '−0.05:0',
                                          '0:+0.05', '+0.05:+0.1', '+0.1:+0.2', '>+0.2'])
for grp, sub in atom_df.groupby('bond_dev_bin', observed=False):
    if len(sub) > 10:
        print(f"  {grp}: N={len(sub):,}, mean_conf={sub['confidence'].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D4: SCAFFOLD vs SUBSTITUENT")
print("=" * 70)

for grp, sub in atom_df.groupby('scaffold_ctx'):
    print(f"  {grp:<20s}: N={len(sub):,}, mean={sub['confidence'].mean():.4f}, "
          f"median={sub['confidence'].median():.4f}, std={sub['confidence'].std():.4f}")

# Within scaffold ring atoms by element
scaffold_ring = atom_df[atom_df['scaffold_ctx'] == 'scaffold_ring']
print(f"\n  --- Scaffold ring atoms by element ---")
for grp, sub in scaffold_ring.groupby('element'):
    if len(sub) > 50:
        print(f"  {grp}: N={len(sub):,}, mean={sub['confidence'].mean():.4f}, "
              f"median={sub['confidence'].median():.4f}")


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D5: BRICS FRAGMENT CONTEXT")
print("=" * 70)

for grp, sub in atom_df.groupby('brics_ctx'):
    print(f"  {grp:<20s}: N={len(sub):,}, mean={sub['confidence'].mean():.4f}, "
          f"median={sub['confidence'].median():.4f}, std={sub['confidence'].std():.4f}")

# Fragment size bins
atom_df['frag_size_bin'] = pd.cut(atom_df['frag_size'], bins=[0, 5, 10, 15, 20, 50],
                                   labels=['1-5', '6-10', '11-15', '16-20', '>20'])
print(f"\n  --- Fragment size ---")
for grp, sub in atom_df.groupby('frag_size_bin', observed=False):
    if len(sub) > 10:
        print(f"  {grp}: N={len(sub):,}, mean={sub['confidence'].mean():.4f}")

# Statistical tests
print(f"\n  --- Mann-Whitney U tests ---")
u1, p1 = mannwhitneyu(atom_df[atom_df['is_scaffold']]['confidence'],
                        atom_df[~atom_df['is_scaffold']]['confidence'])
print(f"  Scaffold vs Substituent: U={u1:,.0f}, p={p1:.2e}")

ring_conf = atom_df[atom_df['is_in_ring']]['confidence']
acyc_conf = atom_df[~atom_df['is_in_ring']]['confidence']
u2, p2 = mannwhitneyu(ring_conf, acyc_conf)
print(f"  Ring vs Acyclic:         U={u2:,.0f}, p={p2:.2e}")

junction_conf = atom_df[atom_df['brics_ctx'] == 'junction']['confidence']
interior_conf = atom_df[atom_df['brics_ctx'] == 'fragment_interior']['confidence']
u3, p3 = mannwhitneyu(junction_conf, interior_conf)
print(f"  BRICS junction vs interior: U={u3:,.0f}, p={p3:.2e}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════

# Color palettes
ELEM_COLORS = {'C': '#555555', 'N': '#0072B2', 'O': '#D55E00', 'S': '#F0E442', 'F': '#56B4E9',
               'P': '#CC79A7', 'Cl': '#009E73'}
SCAFFOLD_COLORS = {'scaffold_ring': '#0072B2', 'scaffold_linker': '#56B4E9',
                   'substituent': '#E69F00'}
BRICS_COLORS = {'fragment_interior': '#0072B2', 'junction': '#D55E00', 'no_cut': '#999999'}

# ══════════════════════════════════════════════════════════════════════════
# Fig 1: Atom type confidence boxplot
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Atom type confidence")
elem_order = atom_df.groupby('element')['confidence'].median().sort_values(ascending=False).index.tolist()
main_elems = [e for e in elem_order if atom_df[atom_df['element'] == e]['confidence'].count() > 100]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Boxplot by element
ax = axes[0]; _panel_label(ax, '(a)')
data = [atom_df[atom_df['element'] == e]['confidence'].values for e in main_elems]
colors = [ELEM_COLORS.get(e, '#999999') for e in main_elems]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(main_elems, fontsize=11, fontweight='bold')
ax.set_ylabel('Atom confidence'); ax.set_xlabel('Element')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (b) By hybridization
ax = axes[1]; _panel_label(ax, '(b)')
hyb_order = ['SP3', 'SP2', 'SP', 'SP3D', 'SP3D2', 'UNSPECIFIED', 'S', 'OTHER']
hyb_data = []
hyb_labels = []
for h in hyb_order:
    vals = atom_df[atom_df['hybridization'] == h]['confidence'].values
    if len(vals) > 10:
        hyb_data.append(vals)
        hyb_labels.append(h)
bp = ax.boxplot(hyb_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
for patch in bp['boxes']:
    patch.set_facecolor('#0072B2'); patch.set_alpha(0.7)
ax.set_xticklabels(hyb_labels, fontsize=10)
ax.set_ylabel('Atom confidence'); ax.set_xlabel('Hybridization')
ax.grid(axis='y', alpha=0.15, linestyle='--')

fig.suptitle('Atom Confidence by Element and Hybridization', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_01_element_hyb.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 2: Element × Hybridization heatmap
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 2] Element x Hybridization heatmap")
import seaborn as sns

pivot = atom_df.pivot_table(values='confidence', index='element', columns='hybridization',
                             aggfunc='mean')
# Filter to elements and hybs with enough data
elem_counts = atom_df['element'].value_counts()
keep_elems = elem_counts[elem_counts > 200].index.tolist()
hyb_counts = atom_df['hybridization'].value_counts()
keep_hybs = hyb_counts[hyb_counts > 50].index.tolist()
pivot_filt = pivot.loc[pivot.index.isin(keep_elems), pivot.columns.isin(keep_hybs)]

# Also create count matrix for annotation
count_pivot = atom_df.pivot_table(values='confidence', index='element', columns='hybridization',
                                   aggfunc='count')
count_filt = count_pivot.loc[pivot_filt.index, pivot_filt.columns]

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot_filt, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Mean confidence', 'shrink': 0.8})
ax.set_title('Mean Atom Confidence: Element × Hybridization', fontsize=13, fontweight='bold')
ax.set_xlabel('Hybridization'); ax.set_ylabel('Element')
savefig('atom_conf_fine_02_element_hyb_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 3: Ring vs acyclic + aromaticity violin
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 3] Ring vs acyclic violin")
atom_df['ring_aro_label'] = atom_df.apply(
    lambda r: 'Ring+Aro' if (r['is_in_ring'] and r['is_aromatic']) else
              'Ring+NonAro' if r['is_in_ring'] else
              'Acyclic+Aro' if r['is_aromatic'] else 'Acyclic+NonAro', axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) 4-category violin
ax = axes[0]; _panel_label(ax, '(a)')
cat_order = ['Acyclic+NonAro', 'Acyclic+Aro', 'Ring+NonAro', 'Ring+Aro']
cat_colors = ['#E69F00', '#F0E442', '#0072B2', '#009E73']
data = [atom_df[atom_df['ring_aro_label'] == c]['confidence'].dropna().values for c in cat_order]
data = [d for d in data if len(d) > 0]
cat_labels = [f'{c}\n(n={len(d):,})' for c, d in zip(cat_order, data)]
cat_colors_filt = [cat_colors[cat_order.index(c)] for c, d in zip(cat_order, data) if len(d) > 0]
# Reset cat_order to match filtered data
cat_order_filt = [c for c, d in zip(cat_order, data) if len(d) > 0]
vp = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.7)
for i, (body, c) in enumerate(zip(vp['bodies'], cat_colors_filt)):
    body.set_facecolor(c); body.set_alpha(0.7)
for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
    if part in vp: vp[part].set_color('#333333')
ax.set_xticks(range(1, len(data) + 1))
ax.set_xticklabels(cat_labels, fontsize=9, rotation=10)
ax.set_ylabel('Atom confidence'); ax.grid(axis='y', alpha=0.15, linestyle='--')

# (b) Ring size breakdown
ax = axes[1]; _panel_label(ax, '(b)')
rs_bins = ['4', '5', '6', '7+']
rs_data = [ring_atoms[ring_atoms['ring_size_bin'] == s]['confidence'].values for s in rs_bins]
rs_data = [d for d in rs_data if len(d) > 0]  # filter empty
rs_labels = [f'{s}\n(n={len(d):,})' for s, d in zip(rs_bins, rs_data)]
bp = ax.boxplot(rs_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
rs_colors = ['#E69F00', '#56B4E9', '#0072B2', '#009E73']
for patch, c in zip(bp['boxes'], rs_colors[:len(rs_data)]):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(rs_labels, fontsize=9)
ax.set_xlabel('Ring size'); ax.set_ylabel('Atom confidence')
ax.grid(axis='y', alpha=0.15, linestyle='--')

fig.suptitle('Atom Confidence: Ring Membership & Aromaticity', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_03_ring_aro.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 4: Ring size × element detail
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 4] Ring size and heteroatom detail")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) 6-ring C vs N vs O confidence
ax = axes[0]; _panel_label(ax, '(a)')
ring6_elem_data = []
ring6_elem_labels = []
for e in ['C', 'N', 'O', 'S']:
    vals = ring6[ring6['element'] == e]['confidence'].values
    if len(vals) > 20:
        ring6_elem_data.append(vals)
        ring6_elem_labels.append(f'{e} (n={len(vals):,})')
vp = ax.violinplot(ring6_elem_data, showmedians=True, showextrema=True, widths=0.7)
elem_v_colors = ['#555555', '#0072B2', '#D55E00', '#F0E442']
for body, c in zip(vp['bodies'], elem_v_colors[:len(ring6_elem_data)]):
    body.set_facecolor(c); body.set_alpha(0.7)
for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
    if part in vp: vp[part].set_color('#333333')
ax.set_xticks(range(1, len(ring6_elem_data) + 1))
ax.set_xticklabels(ring6_elem_labels, fontsize=9)
ax.set_ylabel('Atom confidence'); ax.set_title('6-ring atoms by element')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (b) Fused vs single-ring, split by ring size
ax = axes[1]; _panel_label(ax, '(b)')
import numpy as np
rs_bins = ['5', '6', '7+']
x = np.arange(len(rs_bins))
width = 0.35
for i, (label, mask, color) in enumerate([
    ('Single-ring', ~ring_atoms['is_fused'], '#0072B2'),
    ('Fused (≥2 rings)', ring_atoms['is_fused'], '#D55E00'),
]):
    means = [ring_atoms[(ring_atoms['ring_size_bin'] == s) & mask]['confidence'].mean() for s in rs_bins]
    stds = [ring_atoms[(ring_atoms['ring_size_bin'] == s) & mask]['confidence'].std() for s in rs_bins]
    ax.bar(x + i * width - width / 2, means, width, yerr=stds, color=color, alpha=0.85, label=label,
           capsize=3, edgecolor='white', linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(rs_bins)
ax.set_xlabel('Ring size'); ax.set_ylabel('Mean atom confidence')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Single-ring vs fused-ring atoms')

fig.suptitle('Atom Confidence: Ring Composition & Fusion', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_04_ring_detail.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 5: Coordination environment — bond distance vs confidence
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 5] Coordination & bond distance")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) Degree vs confidence
ax = axes[0]; _panel_label(ax, '(a)')
deg_data = [atom_df[atom_df['degree'] == d]['confidence'].values for d in [1, 2, 3, 4]]
bp = ax.boxplot(deg_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
deg_colors = ['#E69F00', '#56B4E9', '#0072B2', '#009E73']
for patch, c in zip(bp['boxes'], deg_colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(['1', '2', '3', '4'])
ax.set_xlabel('Heavy-atom degree'); ax.set_ylabel('Atom confidence')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (b) Mean neighbor distance vs confidence scatter
ax = axes[1]; _panel_label(ax, '(b)')
sample = atom_df.dropna(subset=['mean_nbr_dist', 'degree']).sample(min(50000, len(atom_df)), random_state=42)
sc = ax.scatter(sample['mean_nbr_dist'], sample['confidence'], c=sample['degree'], cmap='viridis',
                alpha=0.25, s=5, edgecolors='none', vmin=0.5, vmax=4.5)
ax.set_xlabel('Mean neighbor distance (Å)'); ax.set_ylabel('Atom confidence')
cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('Degree')

# (c) Bond distance deviation vs confidence
ax = axes[2]; _panel_label(ax, '(c)')
sample2 = atom_df.dropna(subset=['mean_bond_dev']).sample(min(50000, len(atom_df)), random_state=42)
ax.hist2d(sample2['mean_bond_dev'], sample2['confidence'], bins=80, cmap='Blues',
          vmax=500, cmin=1)
ax.set_xlabel('Mean bond deviation from ideal (Å)'); ax.set_ylabel('Atom confidence')
# Add reference line at 0
ax.axvline(x=0, color='#D55E00', linestyle='--', linewidth=1, alpha=0.7)

fig.suptitle('Atom Confidence vs Local Coordination Environment', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_05_coordination.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 6: Scaffold vs Substituent comparison
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 6] Scaffold vs substituent")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Histogram overlay
ax = axes[0, 0]; _panel_label(ax, '(a)')
for ctx, color in SCAFFOLD_COLORS.items():
    vals = atom_df[atom_df['scaffold_ctx'] == ctx]['confidence'].dropna().values
    ax.hist(vals, bins=70, color=color, alpha=0.5, label=f"{ctx} (n={len(vals):,})", edgecolor='white', linewidth=0.2)
ax.set_xlabel('Atom confidence'); ax.set_ylabel('Count')
ax.legend(fontsize=8, frameon=True, edgecolor='#cccccc')
ax.set_title('Scaffold vs substituent: distribution')

# (b) Boxplot by scaffold context
ax = axes[0, 1]; _panel_label(ax, '(b)')
ctx_order = ['scaffold_ring', 'scaffold_linker', 'substituent']
data = [atom_df[atom_df['scaffold_ctx'] == c]['confidence'].values for c in ctx_order]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
for patch, c in zip(bp['boxes'], [SCAFFOLD_COLORS[x] for x in ctx_order]):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(['Scaffold\nring', 'Scaffold\nlinker', 'Substituent'], fontsize=9)
ax.set_ylabel('Atom confidence'); ax.grid(axis='y', alpha=0.15, linestyle='--')

# (c) Scaffold ring atoms by element
ax = axes[1, 0]; _panel_label(ax, '(c)')
sring_elems = ['C', 'N', 'O']
sring_data = [scaffold_ring[scaffold_ring['element'] == e]['confidence'].values for e in sring_elems]
sring_data = [d for d in sring_data if len(d) > 0]
sring_labels = [f'{e} (n={len(v):,})' for e, v in zip(sring_elems, sring_data)]
sring_colors = ['#555555', '#0072B2', '#D55E00']
if len(sring_data) > 0:
    vp = ax.violinplot(sring_data, showmedians=True, widths=0.7)
    for body, c in zip(vp['bodies'], sring_colors[:len(sring_data)]):
        body.set_facecolor(c); body.set_alpha(0.7)
    ax.set_xticks(range(1, len(sring_data) + 1))
ax.set_xticklabels(sring_labels, fontsize=9)
ax.set_ylabel('Atom confidence'); ax.set_title('Scaffold ring atoms')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (d) Substituent atoms by element
ax = axes[1, 1]; _panel_label(ax, '(d)')
subs = atom_df[atom_df['scaffold_ctx'] == 'substituent']
subs_elems = ['C', 'N', 'O', 'F', 'Br']
subs_data_raw = [(e, subs[subs['element'] == e]['confidence'].values) for e in subs_elems]
subs_data = [(e, d) for e, d in subs_data_raw if len(d) > 20]
subs_colors = ['#555555', '#0072B2', '#D55E00', '#56B4E9', '#999999']
if len(subs_data) > 0:
    vp = ax.violinplot([d for _, d in subs_data], showmedians=True, widths=0.7)
    for idx_body, _ in enumerate(subs_data):
        vp['bodies'][idx_body].set_facecolor(subs_colors[idx_body % len(subs_colors)])
        vp['bodies'][idx_body].set_alpha(0.7)
    ax.set_xticks(range(1, len(subs_data) + 1))
    ax.set_xticklabels([f'{e} (n={len(d):,})' for e, d in subs_data], fontsize=9)
ax.set_ylabel('Atom confidence'); ax.set_title('Substituent atoms')
ax.grid(axis='y', alpha=0.15, linestyle='--')

fig.suptitle('Atom Confidence: Scaffold vs Substituent Decomposition', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_06_scaffold.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 7: Substituent depth (distance from scaffold)
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 7] Substituent depth analysis")
# For substituent atoms, compute topological distance to nearest scaffold atom
# Quick approximation: use the bond distance to substitute attachment points
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) Confidence by scaffold context + element for key elements
ax = axes[0]; _panel_label(ax, '(a)')
# Boxplot: C/N/O confidence split by scaffold_ctx
ctx_elems = []
for ctx in ['scaffold_ring', 'substituent']:
    for e in ['C', 'N', 'O']:
        vals = atom_df[(atom_df['scaffold_ctx'] == ctx) & (atom_df['element'] == e)]['confidence']
        if len(vals) > 30:
            ctx_elems.append((f'{e}\n{ctx}', vals.values))
labels = [l for l, _ in ctx_elems]
bp_data = [d for _, d in ctx_elems]
bp = ax.boxplot(bp_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
ring_colors = {'scaffold_ring': '#0072B2', 'substituent': '#E69F00'}
for i, (label, _) in enumerate(ctx_elems):
    ctx = 'scaffold_ring' if 'scaffold_ring' in label else 'substituent'
    bp['boxes'][i].set_facecolor(ring_colors[ctx])
    bp['boxes'][i].set_alpha(0.8)
ax.set_xticklabels(labels, fontsize=8, rotation=15)
ax.set_ylabel('Atom confidence'); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('C/N/O: Scaffold vs Substituent')

# (b) Num hydrogens vs confidence
ax = axes[1]; _panel_label(ax, '(b)')
h_data = [atom_df[atom_df['num_Hs'] == h]['confidence'].values for h in range(5)]
bp = ax.boxplot(h_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
h_colors = ['#D55E00', '#E69F00', '#56B4E9', '#0072B2', '#009E73']
for patch, c in zip(bp['boxes'], h_colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(['0', '1', '2', '3', '≥4'])
ax.set_xlabel('Number of H atoms'); ax.set_ylabel('Atom confidence')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (c) Formal charge
ax = axes[2]; _panel_label(ax, '(c)')
charge_data = atom_df.groupby('formal_charge')['confidence'].agg(['mean', 'std', 'count'])
charge_data = charge_data[charge_data['count'] > 20]
ax.bar(range(len(charge_data)), charge_data['mean'].values,
       yerr=charge_data['std'].values, color='#0072B2', alpha=0.8,
       capsize=3, edgecolor='white', linewidth=0.3)
ax.set_xticks(range(len(charge_data)))
ax.set_xticklabels([str(int(c)) for c in charge_data.index])
ax.set_xlabel('Formal charge'); ax.set_ylabel('Mean atom confidence')
ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Confidence by formal charge')

fig.suptitle('Atom Confidence: Substituent Depth & Electronic Effects', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_07_substituent_detail.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 8: BRICS fragment context
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 8] BRICS fragment context")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) BRICS context boxplot
ax = axes[0, 0]; _panel_label(ax, '(a)')
brics_order = ['fragment_interior', 'junction', 'no_cut']
data = [atom_df[atom_df['brics_ctx'] == c]['confidence'].values for c in brics_order]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
for patch, c in zip(bp['boxes'], [BRICS_COLORS[x] for x in brics_order]):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(['Fragment\ninterior', 'Junction\n(BRICS cut)', 'No BRICS\ncut'], fontsize=9)
ax.set_ylabel('Atom confidence'); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('BRICS context vs confidence')

# (b) Fragment size vs confidence
ax = axes[0, 1]; _panel_label(ax, '(b)')
fs_data = [atom_df[atom_df['frag_size_bin'] == s]['confidence'].values for s in ['1-5', '6-10', '11-15', '16-20', '>20']]
fs_labels = [f'{s} (n={len(d):,})' for s, d in zip(['1-5', '6-10', '11-15', '16-20', '>20'], fs_data)]
bp = ax.boxplot(fs_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
fs_colors = ['#D55E00', '#E69F00', '#56B4E9', '#0072B2', '#009E73']
for patch, c in zip(bp['boxes'], fs_colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(fs_labels, fontsize=8)
ax.set_xlabel('Fragment size (atoms)'); ax.set_ylabel('Atom confidence')
ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Fragment size vs confidence')

# (c) Junction atoms by element
ax = axes[1, 0]; _panel_label(ax, '(c)')
junction_df = atom_df[atom_df['brics_ctx'] == 'junction']
interior_df = atom_df[atom_df['brics_ctx'] == 'fragment_interior']
for e in ['C', 'N', 'O', 'S']:
    j_vals = junction_df[junction_df['element'] == e]['confidence']
    i_vals = interior_df[interior_df['element'] == e]['confidence']
    if len(j_vals) > 10 and len(i_vals) > 10:
        j_mean = j_vals.mean(); i_mean = i_vals.mean()
        print(f"  {e}: junction mean={j_mean:.4f} (n={len(j_vals):,}), interior mean={i_mean:.4f} (n={len(i_vals):,}), "
              f"Δ={j_mean-i_mean:+.4f}")

# Compare junction vs interior by element
junc_by_elem = []
int_by_elem = []
elem_list = []
for e in ['C', 'N', 'O']:
    jv = junction_df[junction_df['element'] == e]['confidence'].values
    iv = interior_df[interior_df['element'] == e]['confidence'].values
    if len(jv) > 10:
        junc_by_elem.append(jv); int_by_elem.append(iv); elem_list.append(e)

x = np.arange(len(elem_list))
width = 0.35
j_means = [np.mean(v) for v in junc_by_elem]
j_stds = [np.std(v) for v in junc_by_elem]
i_means = [np.mean(v) for v in int_by_elem]
i_stds = [np.std(v) for v in int_by_elem]
ax.bar(x - width/2, j_means, width, yerr=j_stds, color='#D55E00', alpha=0.85,
       label='Junction (BRICS cut)', capsize=3)
ax.bar(x + width/2, i_means, width, yerr=i_stds, color='#0072B2', alpha=0.85,
       label='Fragment interior', capsize=3)
ax.set_xticks(x); ax.set_xticklabels(elem_list)
ax.set_ylabel('Mean atom confidence'); ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Junction vs interior by element')

# (d) Number of fragments per molecule vs confidence
ax = axes[1, 1]; _panel_label(ax, '(d)')
n_frag_dist = atom_df.groupby('mol_idx')['n_frags_in_mol'].first()
frag_bins = pd.cut(n_frag_dist, bins=[0, 1, 2, 4, 8, 50],
                    labels=['1', '2', '3-4', '5-8', '>8'])
atom_df_mol = atom_df.merge(n_frag_dist.reset_index().rename(columns={'n_frags_in_mol': 'mol_n_frags'}),
                            on='mol_idx', how='left')
atom_df_mol['mol_frag_bin'] = pd.cut(atom_df_mol['mol_n_frags'], bins=[0, 1, 2, 4, 8, 50],
                                      labels=['1', '2', '3-4', '5-8', '>8'])
frag_mol_data = [atom_df_mol[atom_df_mol['mol_frag_bin'] == b]['confidence'].values for b in ['1', '2', '3-4', '5-8', '>8']]
frag_mol_labels = [f'{b} (n={len(d):,})' for b, d in zip(['1', '2', '3-4', '5-8', '>8'], frag_mol_data)]
bp = ax.boxplot(frag_mol_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
for patch, c in zip(bp['boxes'], fs_colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(frag_mol_labels, fontsize=8)
ax.set_xlabel('BRICS fragments in molecule'); ax.set_ylabel('Atom confidence')
ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Molecule fragment count vs confidence')

fig.suptitle('Atom Confidence: BRICS Fragment Context', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_08_brics.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 9: Comprehensive feature correlation heatmap
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 9] Feature correlation heatmap")
# One-hot encode categorical features
cat_cols = ['element', 'hybridization', 'ring_aro_label', 'scaffold_ctx', 'brics_ctx', 'ring_size_bin']
num_cols = ['confidence', 'is_aromatic', 'is_in_ring', 'is_fused', 'n_rings', 'min_ring_size',
            'degree', 'total_degree', 'num_Hs', 'formal_charge', 'mean_nbr_dist', 'mean_bond_dev',
            'n_nbrs', 'is_scaffold']

# Build feature matrix
feat_df = atom_df[num_cols].copy()
# Add one-hot for key categories
for e in ['C', 'N', 'O', 'S']:
    feat_df[f'elem_{e}'] = (atom_df['element'] == e).astype(int)
feat_df['hyb_SP3'] = (atom_df['hybridization'] == 'SP3').astype(int)
feat_df['hyb_SP2'] = (atom_df['hybridization'] == 'SP2').astype(int)
feat_df['in_ring_aro'] = ((atom_df['is_in_ring']) & (atom_df['is_aromatic'])).astype(int)
feat_df['is_scaffold_ring'] = (atom_df['scaffold_ctx'] == 'scaffold_ring').astype(int)
feat_df['is_substituent'] = (atom_df['scaffold_ctx'] == 'substituent').astype(int)
feat_df['brics_junction'] = (atom_df['brics_ctx'] == 'junction').astype(int)

all_feat_cols = ['confidence', 'is_aromatic', 'is_in_ring', 'is_fused', 'n_rings',
                 'degree', 'total_degree', 'num_Hs', 'mean_nbr_dist', 'mean_bond_dev',
                 'elem_C', 'elem_N', 'elem_O', 'elem_S',
                 'hyb_SP3', 'hyb_SP2', 'in_ring_aro', 'is_scaffold_ring', 'is_substituent',
                 'brics_junction']
feat_subset = feat_df[all_feat_cols].dropna()

corr = feat_subset.corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            vmin=-0.3, vmax=0.3, center=0, square=True,
            linewidths=0.3, linecolor='white',
            xticklabels=all_feat_cols, yticklabels=all_feat_cols,
            annot_kws={'fontsize': 7}, ax=ax,
            cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})
ax.set_title('Correlation: Atom Features vs Confidence', fontsize=13, fontweight='bold')
plt.yticks(rotation=0)
savefig('atom_conf_fine_09_correlation.png')


# ══════════════════════════════════════════════════════════════════════════
# Fig 10: Inverse distance formula — dist ≈ sqrt(-2·ln(conf))
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 10] Inverse distance formula analysis")
# Confidence → estimated positional error (Å)
# formula: dist ≈ sqrt(-2 * ln(conf))
# This comes from inverting conf = exp(-dist²/2), the Gaussian likelihood
atom_df['est_dist_err'] = np.sqrt(-2 * np.log(np.clip(atom_df['confidence'], 0.001, 0.999)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (a) Confidence → estimated distance error mapping
ax = axes[0, 0]; _panel_label(ax, '(a)')
conf_range = np.linspace(0.01, 0.99, 500)
dist_range = np.sqrt(-2 * np.log(conf_range))
ax.plot(conf_range, dist_range, color='#0072B2', linewidth=2.5)
ax.fill_between(conf_range, 0, dist_range, alpha=0.1, color='#0072B2')
# Mark key confidence values
for c_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    d_val = np.sqrt(-2 * np.log(c_val))
    ax.plot(c_val, d_val, 'o', color='#D55E00', markersize=6)
    ax.annotate(f'conf={c_val}\nd={d_val:.2f}Å', (c_val, d_val), fontsize=8,
                xytext=(10, 10), textcoords='offset points')
ax.set_xlabel('Confidence'); ax.set_ylabel('Estimated distance error (Å)')
ax.set_title('Inverse formula: dist = sqrt(−2·ln(conf))')
ax.grid(alpha=0.15, linestyle='--')

# (b) Estimated distance error distribution
ax = axes[0, 1]; _panel_label(ax, '(b)')
ax.hist(atom_df['est_dist_err'].clip(0, 5), bins=100, color='#D55E00', alpha=0.8,
        edgecolor='white', linewidth=0.2)
ax.set_xlabel('Estimated positional error (Å)'); ax.set_ylabel('Count')
ax.set_title(f'Mean est. error: {atom_df["est_dist_err"].mean():.2f} Å')
ax.axvline(atom_df['est_dist_err'].median(), color='#333333', linestyle='--', linewidth=1.5,
           label=f'Median: {atom_df["est_dist_err"].median():.2f} Å')
ax.legend(fontsize=9)

# (c) Estimated error by element
ax = axes[0, 2]; _panel_label(ax, '(c)')
main_elems_for_err = ['C', 'N', 'O', 'S', 'F', 'Br']
err_data = [atom_df[atom_df['element'] == e]['est_dist_err'].values for e in main_elems_for_err
            if len(atom_df[atom_df['element'] == e]) > 20]
err_labels = [f'{e}\n(n={len(d):,})' for e, d in zip(main_elems_for_err, err_data)]
bp = ax.boxplot(err_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
err_colors = ['#555555', '#0072B2', '#D55E00', '#F0E442', '#56B4E9', '#999999']
for patch, c in zip(bp['boxes'], err_colors[:len(err_data)]):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(err_labels, fontsize=8)
ax.set_ylabel('Estimated error (Å)'); ax.set_title('Estimated error by element')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (d) Estimated error vs actual bond distance deviation
ax = axes[1, 0]; _panel_label(ax, '(d)')
sample = atom_df.dropna(subset=['mean_bond_dev', 'est_dist_err']).sample(min(30000, len(atom_df)), random_state=42)
ax.hist2d(sample['mean_bond_dev'], sample['est_dist_err'].clip(0, 5), bins=60, cmap='Blues',
          vmax=300, cmin=1)
ax.set_xlabel('Mean bond deviation (Å)'); ax.set_ylabel('Estimated error (Å)')
ax.axvline(x=0, color='#D55E00', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('Estimated error vs bond distance deviation')

# (e) Estimated error by scaffold context
ax = axes[1, 1]; _panel_label(ax, '(e)')
ctx_order = ['scaffold_ring', 'scaffold_linker', 'substituent']
err_ctx_data = [atom_df[atom_df['scaffold_ctx'] == c]['est_dist_err'].values for c in ctx_order]
bp = ax.boxplot(err_ctx_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops={'color': '#333333', 'linewidth': 1.2})
for patch, c in zip(bp['boxes'], ['#0072B2', '#56B4E9', '#E69F00']):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(['Scaffold\nring', 'Scaffold\nlinker', 'Substituent'], fontsize=9)
ax.set_ylabel('Estimated error (Å)'); ax.set_title('Estimated error by scaffold context')
ax.grid(axis='y', alpha=0.15, linestyle='--')

# (f) Estimated error vs conf_std (per molecule)
ax = axes[1, 2]; _panel_label(ax, '(f)')
mol_err = atom_df.groupby('mol_idx').agg(
    mean_est_err=('est_dist_err', 'mean'),
    conf_mean=('confidence', 'mean'),
    conf_std=('confidence', 'std'),
).reset_index()
sc = ax.scatter(mol_err['conf_std'], mol_err['mean_est_err'], c=mol_err['conf_mean'],
                cmap='coolwarm_r', alpha=0.3, s=8, edgecolors='none')
ax.set_xlabel('Per-molecule conf std'); ax.set_ylabel('Mean est. error (Å)')
cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('Mean conf')
# Non-linear relationship annotation
ax.text(0.95, 0.95, 'Higher conf_std →\nlarger error range', transform=ax.transAxes,
        ha='right', va='top', fontsize=8, style='italic')

fig.suptitle('Inverse Distance Formula: Confidence → Positional Error (dist ≈ sqrt(−2·ln(conf)))',
             fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_fine_10_inverse_distance.png')

# Summary statistics
print(f"\n  --- Inverse Distance Formula Summary ---")
print(f"  Mean estimated error: {atom_df['est_dist_err'].mean():.3f} Å")
print(f"  Median estimated error: {atom_df['est_dist_err'].median():.3f} Å")
print(f"  Confidence 0.1 → error = {np.sqrt(-2*np.log(0.1)):.2f} Å")
print(f"  Confidence 0.3 → error = {np.sqrt(-2*np.log(0.3)):.2f} Å")
print(f"  Confidence 0.5 → error = {np.sqrt(-2*np.log(0.5)):.2f} Å")
print(f"  Confidence 0.7 → error = {np.sqrt(-2*np.log(0.7)):.2f} Å")
print(f"  Confidence 0.9 → error = {np.sqrt(-2*np.log(0.9)):.2f} Å")
for ctx in ['scaffold_ring', 'substituent', 'scaffold_linker']:
    sub = atom_df[atom_df['scaffold_ctx'] == ctx]
    print(f"  {ctx:<20s}: mean err={sub['est_dist_err'].mean():.3f} Å, "
          f"median err={sub['est_dist_err'].median():.3f} Å")


# ══════════════════════════════════════════════════════════════════════════
print(f"\nAll fine-grained analysis figures saved to: {OUT_DIR}")
