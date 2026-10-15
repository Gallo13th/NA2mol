"""Phase 5 only: Virtual screening cascade figure."""
import sys, gc, time, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, QED
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

OUT_DIR = Path(__file__).resolve().parent / 'output'
CACHE_DIR = OUT_DIR / 'cache_chemspace'

FONT_PATH = Path(__file__).resolve().parent.parent / 'NotoSansCJKsc-Regular.otf'
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'grid.alpha': 0.15, 'grid.linestyle': '--', 'grid.linewidth': 0.4,
})
if FONT_PATH.exists():
    from matplotlib.font_manager import fontManager
    fontManager.addfont(str(FONT_PATH))
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']

def _panel_label(ax, label, x=-0.06, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='left')

def savefig(name, tight=True):
    path = OUT_DIR / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {path}")

MAIN_METHODS = [
    'RNA2smol (Baseline)', 'RNA2smol (Only General Feat)',
    'MolCRAFT', 'Pocket2Mol', 'ED2Mol',
]
METHOD_DISPLAY = {
    'RNA2smol (Baseline)':          'RNA2smol (Baseline)',
    'RNA2smol (Only General Feat)': 'RNA2smol',
    'MolCRAFT':                     'MolCRAFT',
    'Pocket2Mol':                   'Pocket2Mol',
    'ED2Mol':                       'ED2Mol',
}
def _short_name(m):
    return METHOD_DISPLAY.get(m, m)

METHOD_COLORS = {
    'RNA2smol (Baseline)':          '#D55E00',
    'RNA2smol (Only General Feat)': '#E69F00',
    'MolCRAFT':                     '#0072B2',
    'Pocket2Mol':                   '#009E73',
    'ED2Mol':                       '#CC79A7',
}
BKG_PROT_COLOR = '#8C8C8C'
BKG_RNA_COLOR  = '#AA4466'

# ── Load minimal data ──
print("Loading data...")
gen = pd.read_csv(OUT_DIR / 'smrtbench_combined_results.csv')
gen = gen[gen['mode'].isin(MAIN_METHODS)]

bkg = pd.read_csv(Path(__file__).resolve().parent / 'bkg_smiles.csv', sep='\t')

def is_druglike(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        lp = Crippen.MolLogP(mol)
        return 150 <= mw <= 700 and -5 <= lp <= 10 and QED.qed(mol) >= 0.2
    except Exception:
        return False

bkg_prot_raw = bkg[bkg['TARGET_CATEGORY'] == 'Protein']['SMILES'].dropna().unique()
bkg_rna_raw  = bkg[bkg['TARGET_CATEGORY'] == 'RNA']['SMILES'].dropna().unique()
bkg_prot = np.array([s for s in bkg_prot_raw if is_druglike(s)])
bkg_rna  = np.array([s for s in bkg_rna_raw  if is_druglike(s)])
print(f"  BindingDB Protein: {len(bkg_prot):,}  NA: {len(bkg_rna):,}")

# Load cached fingerprints (the big ones)
print("Loading cached fingerprints...")
prot_fps = np.load(CACHE_DIR / 'bkg_protein_fps.npy')
rna_fps  = np.load(CACHE_DIR / 'bkg_na_fps.npy')

method_fps_dict = {}
method_smiles = {}
for m in MAIN_METHODS:
    cache_name = m.replace(' ', '_').replace('(', '').replace(')', '') + '_fps.npy'
    method_fps_dict[m] = np.load(CACHE_DIR / cache_name)
    raw = gen[gen['mode'] == m]['SMILES'].dropna().unique()
    method_smiles[m] = np.array([s for s in raw if is_druglike(s)])
    print(f"  {_short_name(m)}: {len(method_smiles[m]):,} mols, {len(method_fps_dict[m]):,} fps")

# ── Jaccard functions ──
def jaccard_distance_matrix(A, B):
    inter = A.astype(np.float32) @ B.astype(np.float32).T
    sumA = A.sum(axis=1, keepdims=True)
    sumB = B.sum(axis=1, keepdims=True).T
    union = sumA + sumB - inter
    union = np.maximum(union, 1)
    return 1.0 - inter / union

def cov_mmd(gen_fps, ref_fps, thresholds):
    D = jaccard_distance_matrix(ref_fps, gen_fps)
    min_dists = D.min(axis=1)
    cov = np.array([(min_dists <= (1.0 - t)).mean() for t in thresholds])
    mmd = float(min_dists.mean())
    return cov, mmd

# ── Build SMILES → fingerprint index ──
smiles_to_fp = {}
for m in MAIN_METHODS:
    smi_arr = method_smiles[m]
    fps_arr = method_fps_dict[m]
    smiles_to_fp[m] = dict(zip(smi_arr, range(len(fps_arr))))

# ── Filter cascade ──
filter_steps = [
    ('① All',                   lambda d: np.ones(len(d), dtype=bool)),
    ('② aff<−6',               lambda d: d['affinity'] < -6),
    ('③ aff<−7',               lambda d: d['affinity'] < -7),
    ('④ +RMSD<2.5',            lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.5)),
    ('⑤ RMSD<2.0',             lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0)),
    ('⑥ +QED>0.3',             lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.3)),
    ('⑦ QED>0.4, +SA<5',       lambda d: (d['affinity'] < -7) & (d['rmsd'] < 2.0) & (d['qed'] > 0.4) & (d['sa_score'] < 5)),
    ('⑧ aff<−8, RMSD<1.5,\nQED>0.5, SA<5', lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5)),
    ('⑨ +SEI>0.1',             lambda d: (d['affinity'] < -8) & (d['rmsd'] < 1.5) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.1)),
    ('⑩ aff<−9, RMSD<1.0,\nQED>0.5, SEI>0.15', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.5) & (d['sa_score'] < 5) & (d['SEI'] > 0.15)),
    ('⑪ QED>0.6, SA<4,\nSEI>0.2', lambda d: (d['affinity'] < -9) & (d['rmsd'] < 1.0) & (d['qed'] > 0.6) & (d['sa_score'] < 4) & (d['SEI'] > 0.2)),
]
step_labels = [s[0] for s in filter_steps]

N_EVAL = 5000
thresholds = np.linspace(0, 1.0, 51)
tau_idx = 15  # τ=0.3

np.random.seed(42)
prot_ref_idx = np.random.choice(len(prot_fps), min(N_EVAL, len(prot_fps)), replace=False)
rna_ref_idx  = np.random.choice(len(rna_fps),  min(N_EVAL, len(rna_fps)),  replace=False)
prot_ref = prot_fps[prot_ref_idx]
rna_ref  = rna_fps[rna_ref_idx]

vs_results = {m: [] for m in MAIN_METHODS}

print("\nPhase 5: Virtual screening cascade")
print(f"  {'Step':<20} {'surv':>7}  {'COV_NA':>8} {'COV_Prot':>8} {'ratio':>8}")
print(f"  {'-'*60}")

for m in MAIN_METHODS:
    method_data = gen[gen['mode'] == m].copy()
    smi_to_idx = smiles_to_fp[m]
    fps_all = method_fps_dict[m]

    for step_name, condition in filter_steps:
        mask = condition(method_data)
        surv = int(mask.sum())
        surv_smiles = method_data.loc[mask, 'SMILES'].unique()

        surv_indices = [smi_to_idx[s] for s in surv_smiles if s in smi_to_idx]
        if len(surv_indices) >= 2:
            surv_fps = fps_all[np.array(surv_indices)]
            cov_prot_curve, mmd_prot = cov_mmd(surv_fps, prot_ref, thresholds)
            cov_rna_curve,  mmd_rna  = cov_mmd(surv_fps, rna_ref,  thresholds)
            cov_p, cov_n = float(cov_prot_curve[tau_idx]), float(cov_rna_curve[tau_idx])
            mmd_prot, mmd_rna = float(mmd_prot), float(mmd_rna)
        else:
            cov_p, cov_n, mmd_prot, mmd_rna = np.nan, np.nan, np.nan, np.nan

        vs_results[m].append({
            'step': step_name, 'surv': surv,
            'cov_prot': cov_p, 'cov_rna': cov_n,
            'mmd_prot': mmd_prot, 'mmd_rna': mmd_rna,
        })

    print(f"\n  {_short_name(m)}:")
    for r in [vs_results[m][0], vs_results[m][2], vs_results[m][5], vs_results[m][7],
              vs_results[m][9], vs_results[m][10]]:
        if not np.isnan(r['cov_rna']):
            ratio = r['cov_rna'] / r['cov_prot'] if r['cov_prot'] > 1e-10 else np.nan
            print(f"    {r['step']:<20} surv={r['surv']:>7,}  COV_NA={r['cov_rna']:.4f}"
                  f"  COV_Prot={r['cov_prot']:.4f}  ratio={ratio:.2f}")

# ── Self COV reference (compute fresh) ──
n_self = min(N_EVAL // 2, len(prot_fps) // 2)
prot_a = prot_fps[np.random.choice(len(prot_fps), n_self, replace=False)]
prot_b = prot_fps[np.random.choice(len(prot_fps), n_self, replace=False)]
prot_self_cov, _ = cov_mmd(prot_a, prot_b, thresholds)

n_self_r = min(N_EVAL // 2, len(rna_fps) // 2)
rna_a = rna_fps[np.random.choice(len(rna_fps), n_self_r, replace=False)]
rna_b = rna_fps[np.random.choice(len(rna_fps), n_self_r, replace=False)]
rna_self_cov, _ = cov_mmd(rna_a, rna_b, thresholds)
prot_self_cov_03 = prot_self_cov[tau_idx]
rna_self_cov_03  = rna_self_cov[tau_idx]

# ── Free fingerprint arrays before plotting ──
del prot_fps, rna_fps, prot_ref, rna_ref
gc.collect()

# ── Fig 5: Virtual screening ──
print("\n[Fig 5] Virtual screening cascade")
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
fig.subplots_adjust(bottom=0.22)

x = np.arange(len(step_labels))

# Panel (a): Survival funnel
ax = axes[0]
_panel_label(ax, '(a)')
ax.set_facecolor('#fafafa')
for m in MAIN_METHODS:
    surv_counts = [r['surv'] for r in vs_results[m]]
    ax.plot(x, surv_counts, color=METHOD_COLORS[m], linewidth=2.5,
            marker='o', markersize=7, label=_short_name(m), zorder=5)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(step_labels, rotation=25, ha='right', fontsize=8.5)
ax.set_ylabel('Molecules retained')
ax.grid(alpha=0.12, which='both', linestyle='--')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))

# Panel (b): COV(τ=0.3) — solid=NA, dashed=Protein
ax = axes[1]
_panel_label(ax, '(b)')
ax.set_facecolor('#fafafa')

for m in MAIN_METHODS:
    cov_na = [r['cov_rna'] for r in vs_results[m]]
    cov_pr = [r['cov_prot'] for r in vs_results[m]]
    ax.plot(x, cov_na, color=METHOD_COLORS[m], linewidth=2.0, marker='o',
            markersize=6, label=f'{_short_name(m)} vs NA', zorder=5)
    ax.plot(x, cov_pr, color=METHOD_COLORS[m], linewidth=1.2, marker='s',
            markersize=4, linestyle='--', alpha=0.55, zorder=4)

ax.axhline(y=rna_self_cov_03, color=BKG_RNA_COLOR, linewidth=2.0, linestyle='-', alpha=0.6)
ax.axhline(y=prot_self_cov_03, color=BKG_PROT_COLOR, linewidth=2.0, linestyle='--', alpha=0.6)

legend_handles = [
    Line2D([0], [0], color='#333333', linewidth=2.2, marker='o', markersize=5, label='vs BindingDB NA (solid)'),
    Line2D([0], [0], color='#333333', linewidth=1.5, marker='s', markersize=4,
           linestyle='--', alpha=0.55, label='vs BindingDB Prot (dashed)'),
    Line2D([0], [0], color=BKG_RNA_COLOR, linewidth=2.0, label='NA self (τ=0.3)'),
    Line2D([0], [0], color=BKG_PROT_COLOR, linewidth=2.0, linestyle='--', label='Prot self (τ=0.3)'),
]
for m in MAIN_METHODS:
    legend_handles.append(Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.5, label=_short_name(m)))

ax.set_xticks(x)
ax.set_xticklabels(step_labels, rotation=25, ha='right', fontsize=8.5)
ax.set_ylabel('COV(τ=0.3)')
ax.set_title('NA vs. Protein ligand space coverage', fontsize=12, weight='bold')
ax.legend(handles=legend_handles, fontsize=6.5, frameon=True, edgecolor='#cccccc',
          ncol=2, loc='upper right')
ax.grid(alpha=0.12, linestyle='--')
ax.set_ylim(0, 1.05)

savefig('fig_chemspace_virtual_screening.png')

# ── Ratio summary ──
print(f"\n  NA vs Protein COV(τ=0.3) ratio:")
print(f"  (ratio > 1 → more NA-like; ratio < 1 → more Prot-like)")
print(f"  {'Method':<35} {'All':>8} {'aff<−7':>8} {'+QED>0.3':>8} {'Strict':>8}")
print(f"  {'-'*75}")
for m in MAIN_METHODS:
    steps = vs_results[m]
    ratios = []
    for i in [0, 2, 5, 7]:
        if i < len(steps) and steps[i]['cov_prot'] > 1e-10:
            ratios.append(steps[i]['cov_rna'] / steps[i]['cov_prot'])
        else:
            ratios.append(np.nan)
    print(f"  {_short_name(m):<35} {ratios[0]:>8.2f} {ratios[1]:>8.2f} {ratios[2]:>8.2f} {ratios[3]:>8.2f}")

print("\nDone.")
