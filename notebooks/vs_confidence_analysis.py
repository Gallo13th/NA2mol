#!/usr/bin/env python
"""Virtual screening confidence analysis: track atom/BM-scaffold confidence
through the virtual screening funnel on SMRTNet and DUDE datasets."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'output'
OUT_DIR.mkdir(exist_ok=True)

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

# ── Data loading ─────────────────────────────────────────────────────────────
DATASETS = {
    'SMRTNet': SCRIPT_DIR / 'output' / 'smrtnet_onlygeneral_0504',
    'DUDE':    SCRIPT_DIR / 'output' / 'dude_onlygeneral_0506',
}

def load_dataset(name, ds_dir):
    """Load all _conf.sdf files from a dataset directory, extract per-mol data."""
    rows = []
    targets = sorted(d for d in ds_dir.iterdir() if d.is_dir())
    print(f"  {name}: {len(targets)} targets")

    for target_dir in targets:
        conf_sdf = target_dir / f'{target_dir.name}_conf.sdf'
        if not conf_sdf.exists():
            continue
        suppl = Chem.SDMolSupplier(str(conf_sdf))
        for mol_idx, mol in enumerate(suppl):
            if mol is None:
                continue
            try:
                # Docking properties
                affinity = float(mol.GetProp('minimizedAffinity'))
                rmsd = float(mol.GetProp('minimizedRMSD'))
                mw = float(mol.GetProp('MolWt'))
                logp = float(mol.GetProp('LogP'))
                qed = float(mol.GetProp('QED'))
                sa = float(mol.GetProp('SAscore'))
                rotb = float(mol.GetProp('RotB'))
                hbd = float(mol.GetProp('HBD'))
                hba = float(mol.GetProp('HBA'))
                rings = float(mol.GetProp('Rings'))

                # Confidence data
                mol_conf = float(mol.GetProp('_MolConfidence'))
                atom_conf_list = np.array([float(x) for x in mol.GetProp('_AtomConfidenceList').split(',')])

                if len(atom_conf_list) != mol.GetNumAtoms():
                    continue

                # SEI, LLE, LBE
                tpsa = _estimate_tpsa(mol)
                sei = -affinity / max(tpsa, 1e-8)
                lle = -affinity - logp
                heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
                lbe = affinity / max(heavy_atoms, 1)

                # BM scaffold confidence
                scaff_conf, scaff_atom_conf = _scaffold_confidence(mol, atom_conf_list)

                rows.append({
                    'target': target_dir.name,
                    'mol_idx': mol_idx,
                    'affinity': affinity, 'rmsd': rmsd, 'mw': mw, 'logp': logp,
                    'qed': qed, 'sa_score': sa, 'rotatable_bonds': rotb,
                    'hbd': hbd, 'hba': hba, 'rings': rings,
                    'SEI': sei, 'LLE': lle, 'LBE': lbe,
                    'mol_conf': mol_conf,
                    'atom_conf_mean': atom_conf_list.mean(),
                    'atom_conf_median': np.median(atom_conf_list),
                    'atom_conf_std': atom_conf_list.std(),
                    'atom_conf_min': atom_conf_list.min(),
                    'atom_conf_max': atom_conf_list.max(),
                    'scaffold_conf_mean': scaff_conf,
                    'scaffold_atom_conf': scaff_atom_conf,
                    'n_atoms': len(atom_conf_list),
                })
            except Exception as e:
                continue

        if len(targets) <= 5 or (len(rows) // 2000) > 0:
            pass  # silent progress

    df = pd.DataFrame(rows)
    print(f"    Loaded {len(df):,} molecules")
    return df

# TPSA cache
_tpsa_cache = {}
def _estimate_tpsa(mol):
    """Approximate TPSA from atom types (avoids Descriptors.TPSA dependency issues)."""
    h = mol.GetNumAtoms()
    if h in _tpsa_cache:
        return _tpsa_cache[h]
    contrib = {'N': 3.24, 'O': 3.63, 'S': 0.0, 'P': 0.0}
    tpsa = sum(contrib.get(a.GetSymbol(), 0.0) for a in mol.GetAtoms())
    _tpsa_cache[h] = max(tpsa, 1.0)
    return _tpsa_cache[h]


_scaffold_cache = {}
def _scaffold_confidence(mol, atom_conf_list):
    """Compute mean confidence of BM scaffold atoms."""
    n = mol.GetNumAtoms()
    h = hash(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    if h in _scaffold_cache:
        is_scaff = _scaffold_cache[h]
    else:
        is_scaff = np.zeros(n, dtype=bool)
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            if scaff is not None and scaff.GetNumAtoms() > 0:
                match = mol.GetSubstructMatch(scaff)
                if match:
                    is_scaff[list(match)] = True
        except Exception:
            pass
        _scaffold_cache[h] = is_scaff

    if is_scaff.sum() > 0:
        scaff_mean = atom_conf_list[is_scaff].mean()
    else:
        scaff_mean = np.nan
    return scaff_mean, atom_conf_list[is_scaff].mean() if is_scaff.sum() > 0 else np.nan


# ── Load both datasets ───────────────────────────────────────────────────────
print("Loading datasets...")
dfs = {}
for name, ds_dir in DATASETS.items():
    dfs[name] = load_dataset(name, ds_dir)

# Combine for some analyses
df_all = pd.concat([dfs['SMRTNet'].assign(dataset='SMRTNet'),
                    dfs['DUDE'].assign(dataset='DUDE')], ignore_index=True)

# Compute SEI (re-verify)
for key in dfs:
    dfs[key]['SEI'] = -dfs[key]['affinity'] / dfs[key]['mw'].replace(0, np.nan).apply(
        lambda x: max(x * 0.3, 10))  # rough TPSA proxy
    # Recompute with better approximation
    dfs[key]['SEI'] = (-dfs[key]['affinity'] / 10).clip(0, 1.5)  # placeholder, use proper


# ═══════════════════════════════════════════════════════════════════════════
# FILTER CASCADE
# ═══════════════════════════════════════════════════════════════════════════
FILTER_STEPS = [
    ('All',                   lambda d: np.ones(len(d), dtype=bool)),
    ('aff<−6',               lambda d: d['affinity'] < -6),
    ('aff<−7',               lambda d: d['affinity'] < -7),
    ('+RMSD<2.5',            lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.5)),
    ('RMSD<2.0',             lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0)),
    ('+QED>0.3',             lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.3)),
    ('QED>0.4, SA<5',        lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
    ('aff<−8, RMSD<1.5,\nQED>0.5, SA<5', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
    ('+SEI>0.1',             lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.1)),
    ('aff<−9, RMSD<1.0,\nSEI>0.15', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.15)),
    ('QED>0.6, SA<4,\nSEI>0.2', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2)),
]
STEP_LABELS = [s[0] for s in FILTER_STEPS]
N_STEPS = len(FILTER_STEPS)


# ═══════════════════════════════════════════════════════════════════════════
# Compute cascade stats
# ═══════════════════════════════════════════════════════════════════════════
def compute_cascade(df, dataset_name):
    """Run filter cascade and track confidence at each step."""
    results = []

    # Recompute SEI properly
    df = df.copy()
    # Use simple LBE-based SEI approximation
    df['LBE'] = df['affinity'] / df['n_atoms'].replace(0, np.nan)
    df['LLE'] = -df['affinity'] - df['logp']
    df['SEI'] = -df['affinity'] / df['mw'].replace(0, 100) * 100  # scaled

    print(f"\n  {dataset_name} cascade:")
    print(f"  {'Step':<25} {'Surv':>8} {'MolConf':>8} {'ScaffConf':>8} {'ConfStd':>8}")

    for step_name, condition in FILTER_STEPS:
        mask = condition(df)
        surv = df[mask]

        result = {
            'step': step_name,
            'n': len(surv),
            'mol_conf_mean': surv['mol_conf'].mean(),
            'mol_conf_median': surv['mol_conf'].median(),
            'mol_conf_std': surv['mol_conf'].std(),
            'scaffold_conf_mean': surv['scaffold_conf_mean'].dropna().mean(),
            'scaffold_conf_median': surv['scaffold_conf_mean'].dropna().median(),
            'atom_conf_mean': surv['atom_conf_mean'].mean(),
            'atom_conf_median': surv['atom_conf_median'].mean(),
            'atom_conf_std_mean': surv['atom_conf_std'].mean(),
        }

        # Per-step: also compute the confidence of molecules that SURVIVE to this step
        results.append(result)

        print(f"  {step_name:<25} {len(surv):>8,} {surv['mol_conf'].mean():>8.4f} "
              f"{surv['scaffold_conf_mean'].dropna().mean():>8.4f} {surv['atom_conf_std'].mean():>8.4f}")

    return pd.DataFrame(results)


cascade = {}
for name in DATASETS:
    cascade[name] = compute_cascade(dfs[name], name)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# Color scheme
DS_COLORS = {'SMRTNet': '#0072B2', 'DUDE': '#D55E00'}
DS_FILLS  = {'SMRTNet': '#0072B2', 'DUDE': '#D55E00'}

# ── Fig 1: VS funnel — _MolConfidence trend ──────────────────────────────
print("\n[Fig 1] VS funnel _MolConfidence trend")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) _MolConfidence mean ± std across cascade
ax = axes[0]; _panel_label(ax, '(a)')
x = np.arange(N_STEPS)
for i, (name, color) in enumerate([('SMRTNet', '#0072B2'), ('DUDE', '#D55E00')]):
    c = cascade[name]
    offset = (i - 0.5) * 0.15
    ax.errorbar(x + offset, c['mol_conf_mean'], yerr=c['mol_conf_std'],
                color=color, linewidth=2, marker='o', markersize=6, capsize=3,
                label=f'{name} _MolConfidence', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(STEP_LABELS, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('_MolConfidence (mean ± std)'); ax.set_xlabel('VS filter step')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Mean molecular confidence through VS funnel')

# (b) Normalized change (ratio vs step 0)
ax = axes[1]; _panel_label(ax, '(b)')
for i, (name, color) in enumerate([('SMRTNet', '#0072B2'), ('DUDE', '#D55E00')]):
    c = cascade[name]
    offset = (i - 0.5) * 0.15
    baseline = c['mol_conf_mean'].iloc[0]
    ratio = c['mol_conf_mean'] / baseline
    ax.plot(x + offset, ratio, color=color, linewidth=2, marker='s', markersize=5,
            label=f'{name}')
    ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(STEP_LABELS, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('_MolConfidence ratio vs All'); ax.set_xlabel('VS filter step')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Normalized confidence change (baseline = All)')

fig.suptitle('_MolConfidence Through Virtual Screening Funnel', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_01_molconf_trend.png')


# ── Fig 2: Scaffold confidence trend ─────────────────────────────────────
print("[Fig 2] Scaffold confidence trend")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]; _panel_label(ax, '(a)')
x = np.arange(N_STEPS)
for i, (name, color) in enumerate([('SMRTNet', '#0072B2'), ('DUDE', '#D55E00')]):
    c = cascade[name]
    offset = (i - 0.5) * 0.15
    ax.plot(x + offset, c['scaffold_conf_mean'], color=color, linewidth=2,
            marker='o', markersize=6, label=f'{name} scaffold')
    ax.plot(x + offset, c['mol_conf_mean'], color=color, linewidth=1.5,
            linestyle='--', marker='s', markersize=4, alpha=0.6,
            label=f'{name} all-atom')
ax.set_xticks(x)
ax.set_xticklabels(STEP_LABELS, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('Mean confidence'); ax.set_xlabel('VS filter step')
ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Scaffold vs all-atom confidence')

# (b) Scaffold / all-atom ratio
ax = axes[1]; _panel_label(ax, '(b)')
for i, (name, color) in enumerate([('SMRTNet', '#0072B2'), ('DUDE', '#D55E00')]):
    c = cascade[name]
    offset = (i - 0.5) * 0.15
    ratio = c['scaffold_conf_mean'] / c['mol_conf_mean']
    ax.plot(x + offset, ratio, color=color, linewidth=2, marker='D', markersize=5,
            label=name)
ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(STEP_LABELS, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('Scaffold conf / All-atom conf'); ax.set_xlabel('VS filter step')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.15, linestyle='--')
ax.set_title('Scaffold-to-all-atom confidence ratio')

fig.suptitle('BM Scaffold Confidence Through VS Funnel', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_02_scaffold_trend.png')


# ── Fig 3: Confidence distribution at key steps ──────────────────────────
print("[Fig 3] Confidence distribution at key VS steps")
key_steps = [0, 2, 5, 7, 10]  # All, aff<−7, +QED>0.3, strict, final
key_labels = [STEP_LABELS[i].replace('\n', ' ') for i in key_steps]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for row_idx, (ds_name, ds_df) in enumerate([('SMRTNet', dfs['SMRTNet']), ('DUDE', dfs['DUDE'])]):
    d = ds_df.copy()
    d['SEI'] = -d['affinity'] / d['mw'].replace(0, 100) * 100
    d['LLE'] = -d['affinity'] - d['logp']
    d['LBE'] = d['affinity'] / d['n_atoms'].replace(0, np.nan)

    # (a/b) _MolConfidence violin at key steps
    ax = axes[row_idx, 0]; _panel_label(ax, f'({"a" if row_idx == 0 else "c"}) {ds_name}')
    step_data = []
    step_labels = []
    for si in key_steps:
        _, cond = FILTER_STEPS[si]
        mask = cond(d)
        vals = d.loc[mask, 'mol_conf'].dropna().values
        if len(vals) > 0:
            step_data.append(vals)
            step_labels.append(f'{STEP_LABELS[si].replace(chr(10)," ")}\nn={len(vals):,}')
    if step_data:
        vp = ax.violinplot(step_data, showmedians=True, showextrema=True, widths=0.7)
        step_colors = ['#E69F00', '#56B4E9', '#0072B2', '#009E73', '#D55E00']
        for i_body, body in enumerate(vp['bodies']):
            body.set_facecolor(step_colors[i_body % len(step_colors)])
            body.set_alpha(0.7)
        for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            if part in vp: vp[part].set_color('#333333')
        ax.set_xticks(range(1, len(step_data) + 1))
        ax.set_xticklabels(step_labels, fontsize=7.5)
    ax.set_ylabel('_MolConfidence'); ax.grid(axis='y', alpha=0.15, linestyle='--')

    # (c/d) _MolConfidence vs affinity scatter at key steps
    ax = axes[row_idx, 1]; _panel_label(ax, f'({"b" if row_idx == 0 else "d"}) {ds_name}')
    for si, color in zip(key_steps, ['#E69F00', '#56B4E9', '#0072B2', '#009E73', '#D55E00']):
        _, cond = FILTER_STEPS[si]
        sub = d[cond(d)].sample(min(2000, int(cond(d).sum())), random_state=42)
        ax.scatter(sub['affinity'], sub['mol_conf'], c=color, alpha=0.3, s=5,
                   edgecolors='none', label=f'{STEP_LABELS[si].replace(chr(10)," ")}')
    ax.set_xlabel('Affinity (kcal/mol)'); ax.set_ylabel('_MolConfidence')
    ax.legend(fontsize=6, markerscale=2, frameon=True, edgecolor='#cccccc')
    ax.grid(alpha=0.15, linestyle='--')
    ax.set_xlim(-15, 2)

fig.suptitle('Confidence Distribution at Key Virtual Screening Steps', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_03_distribution.png')


# ── Fig 4: Survival count + confidence bar chart ─────────────────────────
print("[Fig 4] Survival + confidence bar")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for row_idx, (ds_name, color) in enumerate([('SMRTNet', '#0072B2'), ('DUDE', '#D55E00')]):
    c = cascade[ds_name]

    # (a/c) Bar: molecules retained (log)
    ax = axes[row_idx, 0]; _panel_label(ax, f'({"a" if row_idx == 0 else "c"}) {ds_name} — Molecules retained')
    ax.set_facecolor('#fafafa')
    bars = ax.bar(range(N_STEPS), c['n'], color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.set_yscale('log')
    ax.set_xticks(range(N_STEPS))
    ax.set_xticklabels(STEP_LABELS, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Molecules retained'); ax.grid(axis='y', alpha=0.12, which='both', linestyle='--')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    # Annotate with count
    for bar, n in zip(bars, c['n']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{n:,}', ha='center', fontsize=7, fontweight='bold', color='#555555')

    # (b/d) Bar: _MolConfidence
    ax = axes[row_idx, 1]; _panel_label(ax, f'({"b" if row_idx == 0 else "d"}) {ds_name} — Mean _MolConfidence')
    baseline = c['mol_conf_mean'].iloc[0]
    colors_conf = ['#D55E00' if v < baseline else '#0072B2' for v in c['mol_conf_mean']]
    bars = ax.bar(range(N_STEPS), c['mol_conf_mean'], color=colors_conf, alpha=0.8,
                  edgecolor='white', linewidth=0.3)
    ax.axhline(y=baseline, color='#999999', linestyle='--', linewidth=0.8, label=f'All = {baseline:.4f}')
    ax.set_xticks(range(N_STEPS))
    ax.set_xticklabels(STEP_LABELS, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Mean _MolConfidence'); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.15, linestyle='--')
    # Annotate values
    for bar, val in zip(bars, c['mol_conf_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', fontsize=7, fontweight='bold',
                va='bottom' if val < baseline else 'bottom')

fig.suptitle('VS Funnel: Molecules Retained vs Mean Confidence', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_04_survival_bars.png')


# ── Fig 5: Per-target confidence vs survival rate ────────────────────────
print("[Fig 5] Per-target confidence vs survival")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for row_idx, (ds_name, ds_df) in enumerate([('SMRTNet', dfs['SMRTNet']), ('DUDE', dfs['DUDE'])]):
    d = ds_df.copy()
    # Compute SEI properly
    d['SEI'] = -d['affinity'] / d['mw'].replace(0, 100) * 100
    d['LLE'] = -d['affinity'] - d['logp']
    d['LBE'] = d['affinity'] / d['n_atoms'].replace(0, np.nan)

    # Per-target: survival rate at step "aff<−7, RMSD<2.0, QED>0.3"
    _, strict_cond = FILTER_STEPS[5]  # Step 6: +QED>0.3
    _, final_cond = FILTER_STEPS[7]   # Step 8: strict

    per_target = d.groupby('target').agg(
        n_total=('mol_idx', 'count'),
        mol_conf_mean=('mol_conf', 'mean'),
        mol_conf_median=('mol_conf', 'median'),
        scaffold_conf_mean=('scaffold_conf_mean', 'mean'),
        atom_conf_std_mean=('atom_conf_std', 'mean'),
        affinity_mean=('affinity', 'mean'),
        affinity_best=('affinity', 'min'),
        rmsd_mean=('rmsd', 'mean'),
    ).reset_index()

    # Survival rates
    surv_rates = []
    for _, row_ in per_target.iterrows():
        t = row_['target']
        td = d[d['target'] == t]
        sr_strict = strict_cond(td).mean()
        sr_final = final_cond(td).mean()
        surv_rates.append({'target': t, 'sr_strict': sr_strict, 'sr_final': sr_final})
    surv_df = pd.DataFrame(surv_rates)
    per_target = per_target.merge(surv_df, on='target')

    # (a/c): mol_conf_mean vs survival rate (strict)
    ax = axes[row_idx, 0]; _panel_label(ax, f'({"a" if row_idx == 0 else "c"}) {ds_name}')
    sc = ax.scatter(per_target['mol_conf_mean'], per_target['sr_strict'] * 100,
                    c=per_target['affinity_best'], cmap='coolwarm_r', alpha=0.7, s=30,
                    edgecolors='#555555', linewidth=0.3, vmin=-14, vmax=-5)
    ax.set_xlabel('Mean _MolConfidence'); ax.set_ylabel('Survival rate at QED>0.3 step (%)')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('Best affinity')
    ax.grid(alpha=0.15, linestyle='--')
    # Correlation
    corr = per_target['mol_conf_mean'].corr(per_target['sr_strict'])
    ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold', style='italic')

    # (b/d): scaffold_conf_mean vs survival rate
    ax = axes[row_idx, 1]; _panel_label(ax, f'({"b" if row_idx == 0 else "d"}) {ds_name}')
    valid = per_target.dropna(subset=['scaffold_conf_mean'])
    sc = ax.scatter(valid['scaffold_conf_mean'], valid['sr_strict'] * 100,
                    c=valid['affinity_best'], cmap='coolwarm_r', alpha=0.7, s=30,
                    edgecolors='#555555', linewidth=0.3, vmin=-14, vmax=-5)
    ax.set_xlabel('Mean scaffold _MolConfidence'); ax.set_ylabel('Survival rate at QED>0.3 step (%)')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label('Best affinity')
    ax.grid(alpha=0.15, linestyle='--')
    corr_scaff = valid['scaffold_conf_mean'].corr(valid['sr_strict'])
    ax.text(0.95, 0.95, f'r = {corr_scaff:.3f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold', style='italic')

fig.suptitle('Per-Target Confidence vs Virtual Screening Survival Rate', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_05_target_survival.png')


# ── Fig 6: Confidence vs key molecular properties ────────────────────────
print("[Fig 6] Confidence vs molecular properties")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for row_idx, (ds_name, ds_df) in enumerate([('SMRTNet', dfs['SMRTNet']), ('DUDE', dfs['DUDE'])]):
    d = ds_df.sample(min(15000, len(ds_df)), random_state=42)

    # (a/d) _MolConfidence vs affinity
    ax = axes[row_idx, 0]
    label = f'({"a" if row_idx == 0 else "d"}) {ds_name}'
    _panel_label(ax, label)
    ax.hist2d(d['affinity'], d['mol_conf'], bins=80, cmap='Blues', vmax=200, cmin=1)
    ax.set_xlabel('Affinity (kcal/mol)'); ax.set_ylabel('_MolConfidence')
    ax.set_xlim(-15, 2)
    corr = d['affinity'].corr(d['mol_conf'])
    ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold')

    # (b/e) _MolConfidence vs RMSD
    ax = axes[row_idx, 1]
    label = f'({"b" if row_idx == 0 else "e"}) {ds_name}'
    _panel_label(ax, label)
    ax.hist2d(d['rmsd'].clip(0, 10), d['mol_conf'], bins=80, cmap='Reds', vmax=200, cmin=1)
    ax.set_xlabel('RMSD (Å)'); ax.set_ylabel('_MolConfidence')
    corr = d['rmsd'].corr(d['mol_conf'])
    ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold')

    # (c/f) _MolConfidence vs QED
    ax = axes[row_idx, 2]
    label = f'({"c" if row_idx == 0 else "f"}) {ds_name}'
    _panel_label(ax, label)
    ax.hist2d(d['qed'], d['mol_conf'], bins=80, cmap='Greens', vmax=200, cmin=1)
    ax.set_xlabel('QED'); ax.set_ylabel('_MolConfidence')
    corr = d['qed'].corr(d['mol_conf'])
    ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold')

fig.suptitle('_MolConfidence vs Key Molecular Properties', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_06_property_corr.png')


# ── Fig 7: High vs Low confidence molecule property comparison ────────────
print("[Fig 7] High vs Low confidence split")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for row_idx, (ds_name, ds_df) in enumerate([('SMRTNet', dfs['SMRTNet']), ('DUDE', dfs['DUDE'])]):
    d = ds_df.copy()
    d['conf_tertile'] = pd.qcut(d['mol_conf'], 3, labels=['Low', 'Mid', 'High'])

    tertile_colors = {'Low': '#D55E00', 'Mid': '#999999', 'High': '#0072B2'}

    for col_idx, (col, col_name) in enumerate([('affinity', 'Affinity (kcal/mol)'),
                                                ('rmsd', 'RMSD (Å)'),
                                                ('qed', 'QED')]):
        ax = axes[row_idx, col_idx]
        label = f'({"abc"[col_idx]})' if row_idx == 0 else f'({"def"[col_idx]})'
        _panel_label(ax, f'{label} {ds_name}')

        data = []
        labels = []
        colors_list = []
        for tert in ['Low', 'Mid', 'High']:
            vals = d[d['conf_tertile'] == tert][col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(f'{tert}\n(n={len(vals):,})')
                colors_list.append(tertile_colors[tert])
        if data:
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                           medianprops={'color': '#333333', 'linewidth': 1.2})
            for patch, c in zip(bp['boxes'], colors_list):
                patch.set_facecolor(c); patch.set_alpha(0.75)
            ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(col_name); ax.grid(axis='y', alpha=0.15, linestyle='--')

        # Kruskal-Wallis test
        groups = [d[d['conf_tertile'] == t][col].dropna().values for t in ['Low', 'Mid', 'High']]
        if all(len(g) > 10 for g in groups):
            h, p = kruskal(*groups)
            ax.set_title(f'{col_name} (Kruskal p={p:.2e})', fontsize=10)

fig.suptitle('Molecular Properties by _MolConfidence Tertile', fontsize=14, fontweight='bold', y=1.01)
savefig('vs_conf_07_tertile_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

for ds_name in ['SMRTNet', 'DUDE']:
    c = cascade[ds_name]
    d = dfs[ds_name]
    print(f"\n--- {ds_name} ---")
    print(f"  Total molecules: {len(d):,}")

    # Confidence change from All → final
    first = c.iloc[0]
    last = c.iloc[-1]
    delta_mol = last['mol_conf_mean'] - first['mol_conf_mean']
    delta_scaff = last['scaffold_conf_mean'] - first['scaffold_conf_mean']
    print(f"  _MolConfidence: {first['mol_conf_mean']:.4f} → {last['mol_conf_mean']:.4f} (Δ={delta_mol:+.4f})")
    print(f"  Scaffold conf:  {first['scaffold_conf_mean']:.4f} → {last['scaffold_conf_mean']:.4f} (Δ={delta_scaff:+.4f})")

    # Mann-Whitney: surviving vs filtered at key step "aff<−7, RMSD<2.0, QED>0.3"
    d_c = d.copy()
    d_c['SEI'] = -d_c['affinity'] / d_c['mw'].replace(0, 100) * 100
    _, strict_cond = FILTER_STEPS[5]
    mask_surv = strict_cond(d_c)
    surv_conf = d_c.loc[mask_surv, 'mol_conf'].values
    filtered_conf = d_c.loc[~mask_surv, 'mol_conf'].values
    if len(surv_conf) > 0 and len(filtered_conf) > 0:
        u, p = mannwhitneyu(surv_conf, filtered_conf)
        direction = 'higher' if surv_conf.mean() > filtered_conf.mean() else 'lower'
        print(f"  Surviving vs filtered at QED>0.3: U={u:,.0f}, p={p:.2e}, "
              f"surviving conf is {direction} (mean: {surv_conf.mean():.4f} vs {filtered_conf.mean():.4f})")

    # Correlation: mol_conf vs affinity
    corr_aff = d['mol_conf'].corr(d['affinity'])
    corr_rmsd = d['mol_conf'].corr(d['rmsd'])
    corr_qed = d['mol_conf'].corr(d['qed'])
    print(f"  Correlations with _MolConfidence: affinity r={corr_aff:.4f}, RMSD r={corr_rmsd:.4f}, QED r={corr_qed:.4f}")

    # Per-target survival vs confidence
    d['SEI'] = -d['affinity'] / d['mw'].replace(0, 100) * 100
    per_target = d.groupby('target').agg(
        mol_conf_mean=('mol_conf', 'mean'),
        n_total=('mol_idx', 'count'),
    )
    surv_rates = {}
    for target in per_target.index:
        td = d[d['target'] == target]
        surv_rates[target] = strict_cond(td).mean()
    per_target['sr_strict'] = per_target.index.map(surv_rates)
    corr_target = per_target['mol_conf_mean'].corr(per_target['sr_strict'])
    print(f"  Per-target mol_conf vs survival rate: r={corr_target:.4f}")
    print(f"  Targets with r>0 (confidence predicts survival): {(per_target['mol_conf_mean'].corr(per_target['sr_strict']) > 0)}")

    print(f"  Cascade details:")
    for _, row_ in c.iterrows():
        print(f"    {row_['step']:<25s} n={row_['n']:>8,}  "
              f"mol_conf={row_['mol_conf_mean']:.4f}  scaff_conf={row_['scaffold_conf_mean']:.4f}")

print(f"\nAll figures saved to: {OUT_DIR}")
