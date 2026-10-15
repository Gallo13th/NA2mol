#!/usr/bin/env python
"""Analyze relationship between atom-level confidence and molecular properties."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

# ── Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'output'
OUT_DIR.mkdir(exist_ok=True)
SDF_PATH = SCRIPT_DIR.parent / 'downstream_project' / 'fluomolgen' / '7kvv_sp1' / 'genmols_raw.sdf'
FIXED_SDF_PATH = SDF_PATH.parent / 'genmols_fixed.sdf'

plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
})

def _panel_label(ax, label, x=-0.06, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left')

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Data extraction
# ══════════════════════════════════════════════════════════════════════════
print("Loading SDF...")
supplier = Chem.SDMolSupplier(str(SDF_PATH))
rows = []
conf_lists = []
n_skipped = 0

for mol in supplier:
    if mol is None:
        n_skipped += 1
        continue
    try:
        row = {
            '_MolConfidence': float(mol.GetProp('_MolConfidence')),
            'affinity': float(mol.GetProp('minimizedAffinity')),
            'rmsd': float(mol.GetProp('minimizedRMSD')),
            'qed': float(mol.GetProp('QED')),
            'sa_score': float(mol.GetProp('SAscore')),
            'logp': float(mol.GetProp('LogP')),
            'mw': float(mol.GetProp('MolWt')),
            'rotb': int(mol.GetProp('RotB')),
            'hbd': int(mol.GetProp('HBD')),
            'hba': int(mol.GetProp('HBA')),
            'rings': int(mol.GetProp('Rings')),
            'n_atoms': mol.GetNumAtoms(),
        }
        conf_str = mol.GetProp('_AtomConfidenceList')
        conf_arr = np.array([float(x) for x in conf_str.split(',')])
        row['conf_mean'] = conf_arr.mean()
        row['conf_median'] = np.median(conf_arr)
        row['conf_min'] = conf_arr.min()
        row['conf_max'] = conf_arr.max()
        row['conf_std'] = conf_arr.std()
        row['conf_q25'] = np.percentile(conf_arr, 25)
        row['conf_q75'] = np.percentile(conf_arr, 75)
        row['conf_iqr'] = row['conf_q75'] - row['conf_q25']
        row['conf_cv'] = conf_arr.std() / conf_arr.mean() if conf_arr.mean() > 0 else 0
        rows.append(row)
        conf_lists.append(conf_arr)
    except Exception:
        n_skipped += 1

df = pd.DataFrame(rows)
all_atom_confs = np.concatenate(conf_lists)

print(f"  Molecules loaded: {len(df):,}")
print(f"  Skipped: {n_skipped}")
print(f"  Total atom confidence values: {len(all_atom_confs):,}")
print(f"  _MolConfidence range: [{df['_MolConfidence'].min():.4f}, {df['_MolConfidence'].max():.4f}]")
print(f"  Atom confidence range: [{all_atom_confs.min():.4f}, {all_atom_confs.max():.4f}]")

# ── Confidence quartile bins ──
df['conf_quartile'] = pd.qcut(df['conf_mean'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Correlation analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS: Atom Confidence vs Molecular Properties")
print("=" * 70)

prop_cols = ['affinity', 'rmsd', 'qed', 'sa_score', 'logp', 'mw', 'rotb']
conf_cols = ['_MolConfidence', 'conf_mean', 'conf_median', 'conf_min', 'conf_max', 'conf_std']

print(f"\n{'Stat':<12s}", end='')
for p in prop_cols:
    print(f" {p:>10s}", end='')
print()

for conf_col in conf_cols:
    # Pearson
    print(f"{conf_col:<12s}", end='')
    for p in prop_cols:
        valid = df[[conf_col, p]].dropna()
        r, pv = pearsonr(valid[conf_col], valid[p])
        sig = '*' if pv < 0.01 else ('·' if pv < 0.05 else '')
        print(f" {r:+.3f}{sig}  ", end='')
    print()
    # Spearman
    print(f"{'  (Spearman)':<12s}", end='')
    for p in prop_cols:
        valid = df[[conf_col, p]].dropna()
        r, pv = spearmanr(valid[conf_col], valid[p])
        sig = '*' if pv < 0.01 else ('·' if pv < 0.05 else '')
        print(f" {r:+.3f}{sig}  ", end='')
    print()
    print()

# Quartile summary
print("─" * 70)
print("Property means by confidence quartile:")
print(f"  {'Quartile':<12s} {'N':>6s} {'Affinity':>10s} {'RMSD':>8s} {'QED':>8s} {'SA':>8s} {'logP':>8s} {'MW':>8s}")
for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
    sub = df[df['conf_quartile'] == q]
    print(f"  {q:<12s} {len(sub):6d} {sub['affinity'].mean():10.2f} {sub['rmsd'].mean():8.3f} "
          f"{sub['qed'].mean():8.3f} {sub['sa_score'].mean():8.3f} {sub['logp'].mean():8.3f} {sub['mw'].mean():8.1f}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Confidence distribution histograms
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Confidence distributions")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
_panel_label(ax, '(a)')
ax.hist(all_atom_confs, bins=80, color='#0072B2', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Per-atom confidence'); ax.set_ylabel('Count')
ax.set_title('All atom confidence values')

ax = axes[1]
_panel_label(ax, '(b)')
ax.hist(df['conf_mean'], bins=60, color='#D55E00', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Per-molecule mean confidence'); ax.set_ylabel('Count')
ax.set_title('Per-molecule mean atom confidence')

ax = axes[2]
_panel_label(ax, '(c)')
ax.hist(df['_MolConfidence'], bins=60, color='#009E73', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.set_xlabel('_MolConfidence'); ax.set_ylabel('Count')
ax.set_title('Molecule-level confidence')

savefig('atom_conf_01_distributions.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Confidence vs Affinity scatter
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 2] Confidence vs Affinity")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# (a) _MolConfidence vs affinity
ax = axes[0]; _panel_label(ax, '(a)')
sc = ax.scatter(df['_MolConfidence'], df['affinity'], c=df['qed'], cmap='RdYlGn',
                alpha=0.4, s=8, edgecolors='none')
ax.set_xlabel('_MolConfidence'); ax.set_ylabel('Affinity (kcal/mol)')
cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('QED')
r, pv = pearsonr(df['_MolConfidence'], df['affinity'])
ax.text(0.95, 0.05, f'r={r:.3f} (p={pv:.2e})', transform=ax.transAxes,
        ha='right', fontsize=9, style='italic')

# (b) mean atom confidence vs affinity
ax = axes[1]; _panel_label(ax, '(b)')
sc = ax.scatter(df['conf_mean'], df['affinity'], c=df['qed'], cmap='RdYlGn',
                alpha=0.4, s=8, edgecolors='none')
ax.set_xlabel('Mean atom confidence'); ax.set_ylabel('Affinity (kcal/mol)')
cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('QED')
r, pv = pearsonr(df['conf_mean'], df['affinity'])
ax.text(0.95, 0.05, f'r={r:.3f} (p={pv:.2e})', transform=ax.transAxes,
        ha='right', fontsize=9, style='italic')

# (c) mean confidence vs RMSD
ax = axes[2]; _panel_label(ax, '(c)')
sc = ax.scatter(df['conf_mean'], df['rmsd'], c=df['affinity'], cmap='coolwarm_r',
                alpha=0.4, s=8, edgecolors='none')
ax.set_xlabel('Mean atom confidence'); ax.set_ylabel('RMSD (Å)')
cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('Affinity')
r, pv = pearsonr(df['conf_mean'], df['rmsd'])
ax.text(0.95, 0.05, f'r={r:.3f} (p={pv:.2e})', transform=ax.transAxes,
        ha='right', fontsize=9, style='italic')

fig.suptitle('Atom Confidence vs Binding Affinity & RMSD', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_02_affinity_scatter.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Confidence vs drug-likeness scatter matrix
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 3] Confidence vs drug-likeness")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

panels = [
    ('conf_mean', 'qed', 'Mean atom conf', 'QED', '(a)'),
    ('conf_mean', 'sa_score', 'Mean atom conf', 'SA Score', '(b)'),
    ('conf_mean', 'logp', 'Mean atom conf', 'logP', '(c)'),
    ('conf_mean', 'mw', 'Mean atom conf', 'Molecular Weight', '(d)'),
]
for ax, (xcol, ycol, xlab, ylab, label) in zip(axes.flat, panels):
    _panel_label(ax, label)
    sc = ax.scatter(df[xcol], df[ycol], c=df['affinity'], cmap='coolwarm_r',
                    alpha=0.35, s=8, edgecolors='none')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    r, pv = pearsonr(df[xcol].dropna(), df[ycol].dropna())
    ax.text(0.95, 0.05, f'r={r:.3f}', transform=ax.transAxes,
            ha='right', fontsize=9, style='italic')
    if ycol == 'qed':
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('Affinity')

fig.suptitle('Atom Confidence vs Drug-likeness Properties', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_03_druglikeness.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Binned confidence boxplots
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 4] Binned confidence boxplots")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

metrics = [
    ('affinity', 'Affinity (kcal/mol)', axes[0, 0], '(a)'),
    ('rmsd', 'RMSD (Å)', axes[0, 1], '(b)'),
    ('qed', 'QED', axes[0, 2], '(c)'),
    ('sa_score', 'SA Score', axes[1, 0], '(d)'),
    ('logp', 'logP', axes[1, 1], '(e)'),
    ('mw', 'Molecular Weight', axes[1, 2], '(f)'),
]
colors_q = ['#E69F00', '#56B4E9', '#009E73', '#0072B2']

for col, ylab, ax, label in metrics:
    _panel_label(ax, label)
    data_by_q = [df[df['conf_quartile'] == q][col].dropna().values for q in
                 ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']]
    bp = ax.boxplot(data_by_q, patch_artist=True, showfliers=False,
                    medianprops={'color': '#333333', 'linewidth': 1.2},
                    whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8})
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_q[i])
        patch.set_alpha(0.75)
    ax.set_xticklabels(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], fontsize=9)
    ax.set_ylabel(ylab)
    ax.grid(axis='y', alpha=0.15, linestyle='--')
    # Q4 vs Q1 mean difference
    q1_mean = df[df['conf_quartile'] == 'Q1 (low)'][col].mean()
    q4_mean = df[df['conf_quartile'] == 'Q4 (high)'][col].mean()
    ax.text(0.98, 0.95, f'Δ(Q4−Q1)={q4_mean-q1_mean:+.3f}', transform=ax.transAxes,
            ha='right', fontsize=8, style='italic')

fig.suptitle('Property Distributions by Mean Atom Confidence Quartile', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_04_quartile_boxplots.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Low vs High confidence violin plots
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 5] Low vs High confidence comparison")
lo = df[df['conf_quartile'] == 'Q1 (low)']
hi = df[df['conf_quartile'] == 'Q4 (high)']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, (col, ylab, label) in zip(axes.flat, [
    ('affinity', 'Affinity (kcal/mol)', '(a)'),
    ('rmsd', 'RMSD (Å)', '(b)'),
    ('qed', 'QED', '(c)'),
    ('sa_score', 'SA Score', '(d)'),
    ('logp', 'logP', '(e)'),
    ('mw', 'Molecular Weight', '(f)'),
]):
    _panel_label(ax, label)
    data = [lo[col].dropna().values, hi[col].dropna().values]
    vp = ax.violinplot(data, positions=[0, 1], showmedians=True,
                       showextrema=True, widths=0.7)
    vp['bodies'][0].set_facecolor('#E69F00'); vp['bodies'][0].set_alpha(0.7)
    vp['bodies'][1].set_facecolor('#0072B2'); vp['bodies'][1].set_alpha(0.7)
    for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if part in vp:
            vp[part].set_color('#333333')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Low conf\n(Q1)', 'High conf\n(Q4)'], fontsize=9)
    ax.set_ylabel(ylab)
    ax.grid(axis='y', alpha=0.15, linestyle='--')

fig.suptitle('Low vs High Confidence Molecules: Property Distributions', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_05_low_vs_high.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Per-atom confidence profile (normalized position)
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 6] Per-atom confidence profile")
MAX_POS = 65
pos_conf = {i: [] for i in range(MAX_POS)}
for conf_arr in conf_lists:
    n = len(conf_arr)
    if n == 0:
        continue
    for i in range(n):
        pos_conf[i].append(conf_arr[i])

positions = []
means = []
stds = []
for pos in sorted(pos_conf.keys()):
    vals = pos_conf[pos]
    if len(vals) >= 5:
        positions.append(pos + 1)
        means.append(np.mean(vals))
        stds.append(np.std(vals))

positions = np.array(positions); means = np.array(means); stds = np.array(stds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]; _panel_label(ax, '(a)')
ax.fill_between(positions, means - stds, means + stds, alpha=0.25, color='#0072B2')
ax.plot(positions, means, color='#0072B2', linewidth=1.8)
ax.set_xlabel('Atom position (index)'); ax.set_ylabel('Mean confidence ± std')
ax.set_title('Confidence by atom position')
ax.grid(alpha=0.15, linestyle='--')

# (b) Normalized position (0-1 scale) — bin into 20 bins
ax = axes[1]; _panel_label(ax, '(b)')
norm_conf_bins = {i: [] for i in range(20)}
for conf_arr in conf_lists:
    n = len(conf_arr)
    if n == 0:
        continue
    for i, c in enumerate(conf_arr):
        bin_idx = min(int((i / n) * 20), 19)
        norm_conf_bins[bin_idx].append(c)

bin_centers = np.linspace(0, 1, 20) + 0.025
bin_means = [np.mean(norm_conf_bins[i]) for i in range(20)]
bin_stds = [np.std(norm_conf_bins[i]) for i in range(20)]

ax.fill_between(bin_centers, np.array(bin_means) - np.array(bin_stds),
                np.array(bin_means) + np.array(bin_stds), alpha=0.25, color='#D55E00')
ax.plot(bin_centers, bin_means, color='#D55E00', linewidth=1.8)
ax.set_xlabel('Normalized atom position (0=start, 1=end)')
ax.set_ylabel('Mean confidence ± std')
ax.set_title('Confidence by normalized position')
ax.grid(alpha=0.15, linestyle='--')

fig.suptitle('Atom Confidence vs Position in Molecule', fontsize=14, fontweight='bold', y=1.01)
savefig('atom_conf_06_position_profile.png')


# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Fixed vs unfixed molecule confidence comparison
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 7] Fixed vs unfixed comparison")
# Identify which molecules were fixed (SMILES changed)
if FIXED_SDF_PATH.exists():
    raw_supplier = Chem.SDMolSupplier(str(SDF_PATH))
    fixed_supplier = Chem.SDMolSupplier(str(FIXED_SDF_PATH))

    n_mols = min(len(raw_supplier), len(fixed_supplier))
    fixed_flags = []
    for i in range(n_mols):
        r_mol = raw_supplier[i]; f_mol = fixed_supplier[i]
        if r_mol is None or f_mol is None:
            fixed_flags.append(False)
            continue
        try:
            r_smi = Chem.MolToSmiles(r_mol)
            f_smi = Chem.MolToSmiles(f_mol)
            fixed_flags.append(r_smi != f_smi)
        except Exception:
            fixed_flags.append(False)

    df['was_fixed'] = fixed_flags[:len(df)]
    n_fixed = df['was_fixed'].sum()
    n_unfixed = len(df) - n_fixed
    print(f"  Fixed molecules: {n_fixed}, Unfixed: {n_unfixed}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    comp_cols = [
        ('conf_mean', 'Mean atom confidence', axes[0, 0], '(a)'),
        ('_MolConfidence', '_MolConfidence', axes[0, 1], '(b)'),
        ('affinity', 'Affinity (kcal/mol)', axes[0, 2], '(c)'),
        ('rmsd', 'RMSD (Å)', axes[1, 0], '(d)'),
        ('qed', 'QED', axes[1, 1], '(e)'),
        ('sa_score', 'SA Score', axes[1, 2], '(f)'),
    ]

    for col, xlab, ax, label in comp_cols:
        _panel_label(ax, label)
        unfixed_data = df[~df['was_fixed']][col].dropna().values
        fixed_data = df[df['was_fixed']][col].dropna().values
        ax.hist(unfixed_data, bins=50, color='#0072B2', alpha=0.6, label=f'Unfixed (n={len(unfixed_data):,})',
                edgecolor='white', linewidth=0.2)
        ax.hist(fixed_data, bins=50, color='#D55E00', alpha=0.6, label=f'Fixed (n={len(fixed_data)})',
                edgecolor='white', linewidth=0.2)
        ax.set_xlabel(xlab); ax.set_ylabel('Count')
        ax.legend(fontsize=8, frameon=True, edgecolor='#cccccc')
        # Mean comparison
        ax.axvline(unfixed_data.mean(), color='#0072B2', linestyle='--', linewidth=1.5)
        ax.axvline(fixed_data.mean(), color='#D55E00', linestyle='--', linewidth=1.5)

    fig.suptitle('Fixed vs Unfixed Molecules: Confidence & Property Distributions',
                 fontsize=14, fontweight='bold', y=1.01)
    savefig('atom_conf_07_fixed_vs_unfixed.png')

    # Print comparison stats
    print(f"\n  Fixed vs Unfixed comparison:")
    print(f"  {'Metric':<20s} {'Unfixed':>10s} {'Fixed':>10s} {'Δ':>10s}")
    for col, name in [('conf_mean', 'Mean atom conf'), ('_MolConfidence', '_MolConfidence'),
                       ('affinity', 'Affinity'), ('rmsd', 'RMSD'), ('qed', 'QED'), ('sa_score', 'SA')]:
        u = df[~df['was_fixed']][col].mean()
        f = df[df['was_fixed']][col].mean()
        print(f"  {name:<20s} {u:10.4f} {f:10.4f} {f-u:+10.4f}")
else:
    print("  genmols_fixed.sdf not found — skipping fixed vs unfixed comparison")


# ══════════════════════════════════════════════════════════════════════════
# Figure 8: Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════
print("[Fig 8] Correlation heatmap")
heat_cols = ['_MolConfidence', 'conf_mean', 'conf_median', 'conf_min', 'conf_max',
             'conf_std', 'conf_cv',
             'affinity', 'rmsd', 'qed', 'sa_score', 'logp', 'mw', 'rotb']
heat_labels = ['MolConf', 'MeanConf', 'MedConf', 'MinConf', 'MaxConf',
               'StdConf', 'CVConf',
               'Affinity', 'RMSD', 'QED', 'SA', 'logP', 'MW', 'RotB']

corr = df[heat_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

fig, ax = plt.subplots(figsize=(12, 10))
import seaborn as sns
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            vmin=-0.4, vmax=0.4, center=0, square=True,
            linewidths=0.5, linecolor='white',
            xticklabels=heat_labels, yticklabels=heat_labels,
            annot_kws={'fontsize': 8}, ax=ax, cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
ax.set_title('Correlation Matrix: Atom Confidence vs Molecular Properties',
             fontsize=13, fontweight='bold')
plt.yticks(rotation=0)
savefig('atom_conf_08_correlation_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to: {OUT_DIR}")
