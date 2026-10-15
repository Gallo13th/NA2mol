"""
Chemical space evaluation: SMRTBench generated molecules vs. BindingDB ligands.

Pipeline (each phase cached to disk, incremental):
  Phase 1 — Compute ECFP4 fingerprints for all molecules → cache/*_fps.npy
  Phase 2 — Fit PCA on BindingDB reference, transform all → cache/*_pca.npy
  Phase 3 — UMAP on a subsample for 2D visualization
  Phase 4 — Metrics (COV, MMD, 1-NNA) via vectorized Jaccard
  Phase 5 — Figures

Design: BindingDB defines the reference chemical space.
All generated molecules are projected INTO this space (PCA fit on ref, transform gen).
"""
import sys
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = OUT_DIR / 'cache_chemspace'
CACHE_DIR.mkdir(exist_ok=True)

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

def log(msg):
    print(msg)
    sys.stdout.flush()

# ── Methods & colors ────────────────────────────────────────────────────
MAIN_METHODS = [
    'RNA2smol (Baseline)', 'RNA2smol (Only General Feat)',
    'MolCRAFT', 'Pocket2Mol', 'ED2Mol',
]
# Display names: Only General Feat = complete model → just "RNA2smol"
METHOD_DISPLAY = {
    'RNA2smol (Baseline)':          'RNA2smol (Baseline)',
    'RNA2smol (Only General Feat)': 'RNA2smol',
    'MolCRAFT':                     'MolCRAFT',
    'Pocket2Mol':                   'Pocket2Mol',
    'ED2Mol':                       'ED2Mol',
}
def _short_name(m):
    return METHOD_DISPLAY.get(m, m.split('(')[0].strip())
METHOD_COLORS = {
    'RNA2smol (Baseline)':          '#D55E00',
    'RNA2smol (Only General Feat)': '#E69F00',
    'MolCRAFT':                     '#0072B2',
    'Pocket2Mol':                   '#009E73',
    'ED2Mol':                       '#CC79A7',
    'BindingDB Protein':            '#8C8C8C',
    'BindingDB NA':                 '#AA4466',
}
BKG_PROT_COLOR = '#8C8C8C'   # neutral gray — distinct from all 5 method colors
BKG_RNA_COLOR  = '#AA4466'   # rose/crimson — distinct from orange/yellow & purple

# ── RDKit ───────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load & pre-filter
# ══════════════════════════════════════════════════════════════════════════
log("Loading data...")
gen = pd.read_csv(OUT_DIR / 'smrtbench_combined_results.csv')
gen = gen[gen['mode'].isin(MAIN_METHODS)]
bkg = pd.read_csv(Path(__file__).resolve().parent / 'bkg_smiles.csv', sep='\t')

def is_druglike(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        if mw < 150 or mw > 700:
            return False
        lp = Crippen.MolLogP(mol)
        if lp < -5 or lp > 10:
            return False
        if QED.qed(mol) < 0.2:
            return False
        return True
    except Exception:
        return False

log("Pre-filtering (MW 150–700, QED>0.2, logP −5..10)...")
bkg_prot_raw = bkg[bkg['TARGET_CATEGORY'] == 'Protein']['SMILES'].dropna().unique()
bkg_rna_raw  = bkg[bkg['TARGET_CATEGORY'] == 'RNA']['SMILES'].dropna().unique()
bkg_prot = np.array([s for s in bkg_prot_raw if is_druglike(s)])
bkg_rna  = np.array([s for s in bkg_rna_raw  if is_druglike(s)])
log(f"  BindingDB Protein: {len(bkg_prot_raw):,} → {len(bkg_prot):,}")
log(f"  BindingDB NA:      {len(bkg_rna_raw):,}  → {len(bkg_rna):,}")

method_smiles = {}
for m in MAIN_METHODS:
    raw = gen[gen['mode'] == m]['SMILES'].dropna().unique()
    filtered = np.array([s for s in raw if is_druglike(s)])
    method_smiles[m] = filtered
    log(f"  {m}: {len(raw):,} → {len(filtered):,}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Phase 1 — Compute & cache ECFP4 fingerprints
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Phase 1: ECFP4 fingerprints (cached)")
log("=" * 60)

def compute_fps(smiles_list, cache_name):
    """Return [N × 2048] float32 numpy array of ECFP4 fingerprints."""
    cache_path = CACHE_DIR / f'{cache_name}_fps.npy'
    if cache_path.exists():
        arr = np.load(cache_path)
        log(f"  Loaded cached: {cache_path.name}  ({len(arr):,} mols)")
        return arr

    t0 = time.time()
    fps_list = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), arr)
            fps_list.append(arr)
        if (i + 1) % 20000 == 0:
            log(f"    ... {i+1}/{len(smiles_list)}")
    fps_arr = np.array(fps_list, dtype=np.float32)
    np.save(cache_path, fps_arr)
    log(f"  Computed & cached: {cache_path.name}  ({len(fps_arr):,} mols, {time.time()-t0:.0f}s)")
    return fps_arr

prot_fps = compute_fps(bkg_prot, 'bkg_protein')
rna_fps  = compute_fps(bkg_rna,  'bkg_na')

bkg_ref_fps = np.vstack([prot_fps, rna_fps])
bkg_ref_labels = np.array(['Protein'] * len(prot_fps) + ['NA'] * len(rna_fps))

method_fps_dict = {}
for m in MAIN_METHODS:
    method_fps_dict[m] = compute_fps(method_smiles[m], m.replace(' ', '_').replace('(', '').replace(')', ''))

# ══════════════════════════════════════════════════════════════════════════
# 3. Phase 2 — Fit PCA on BindingDB reference, transform all
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Phase 2: PCA (fit on BindingDB reference → transform all)")
log("=" * 60)

pca_path = CACHE_DIR / 'pca_model.pkl'
pca_ref_path = CACHE_DIR / 'pca_reference.npy'

if pca_path.exists() and pca_ref_path.exists():
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    bkg_ref_pca = np.load(pca_ref_path)
    log(f"  Loaded cached PCA model + reference coords")
else:
    t0 = time.time()
    log(f"  Fitting PCA on {len(bkg_ref_fps):,} BindingDB reference fingerprints...")
    pca = PCA(n_components=50, random_state=42)
    bkg_ref_pca = pca.fit_transform(bkg_ref_fps)
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    np.save(pca_ref_path, bkg_ref_pca)
    log(f"  Saved PCA model  (explained var: {pca.explained_variance_ratio_.sum():.3f}, {time.time()-t0:.0f}s)")

prot_pca = bkg_ref_pca[bkg_ref_labels == 'Protein']
rna_pca  = bkg_ref_pca[bkg_ref_labels == 'NA']

# Transform each method
method_pca = {}
for m in MAIN_METHODS:
    cache_path = CACHE_DIR / f'{m.replace(" ", "_").replace("(", "").replace(")", "")}_pca.npy'
    if cache_path.exists():
        method_pca[m] = np.load(cache_path)
        log(f"  Loaded cached: {cache_path.name}")
    else:
        t0 = time.time()
        log(f"  Transforming {m} ({len(method_fps_dict[m]):,} mols)...")
        method_pca[m] = pca.transform(method_fps_dict[m])
        np.save(cache_path, method_pca[m])
        log(f"  Saved: {cache_path.name}  ({time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════
# 4. Phase 3 — UMAP on FULL BindingDB PCA → transform generated molecules
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Phase 3: UMAP 2D (full BindingDB → transform generated)")
log("=" * 60)

import umap

umap_2d_path = CACHE_DIR / 'umap_2d.npz'
umap_model_path = CACHE_DIR / 'umap_model.pkl'

if umap_2d_path.exists() and umap_model_path.exists():
    t0 = time.time()
    data = np.load(umap_2d_path, allow_pickle=True)
    umap_prot = data['prot']
    umap_rna = data['rna']
    umap_methods = {m: data[f'm_{i}'] for i, m in enumerate(MAIN_METHODS)}
    with open(umap_model_path, 'rb') as f:
        reducer = pickle.load(f)
    log(f"  Loaded cached UMAP model + full BindingDB 2D coords ({time.time()-t0:.0f}s)")
else:
    t0 = time.time()
    log(f"  Fitting UMAP on full {len(bkg_ref_pca):,} BindingDB PCA coords "
        f"(n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1,
                        min_dist=0.1, n_neighbors=15, metric='euclidean',
                        low_memory=True, verbose=True)
    bkg_umap_2d = reducer.fit_transform(bkg_ref_pca)
    umap_prot = bkg_umap_2d[bkg_ref_labels == 'Protein']
    umap_rna  = bkg_umap_2d[bkg_ref_labels == 'NA']
    log(f"  UMAP fit: {time.time()-t0:.0f}s  (Prot: {len(umap_prot):,}, NA: {len(umap_rna):,})")

    # Transform generated molecules through the learned UMAP
    umap_methods = {}
    for m in MAIN_METHODS:
        t1 = time.time()
        log(f"    Transforming {_short_name(m)} ({len(method_pca[m]):,} mols)...")
        umap_methods[m] = reducer.transform(method_pca[m])
        log(f"      done ({time.time()-t1:.0f}s)")

    np.savez_compressed(umap_2d_path,
                        prot=umap_prot, rna=umap_rna,
                        **{f'm_{i}': umap_methods[m] for i, m in enumerate(MAIN_METHODS)})
    with open(umap_model_path, 'wb') as f:
        pickle.dump(reducer, f)
    log(f"  Saved UMAP model + full coords (total {time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════
# 5. Phase 4 — Standard metrics (vectorized Jaccard)
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Phase 4: Standard metrics (ECFP4, vectorized)")
log("=" * 60)

def jaccard_distance_matrix(A, B):
    """Vectorized Jaccard distance: D[i,j] = 1 − |A_i ∩ B_j| / |A_i ∪ B_j|."""
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

def internal_diversity(fps_arr):
    if len(fps_arr) < 2:
        return np.nan
    n = min(len(fps_arr), 2000)
    idx = np.random.choice(len(fps_arr), n, replace=False)
    D = jaccard_distance_matrix(fps_arr[idx], fps_arr[idx])
    triu = np.triu_indices_from(D, k=1)
    return float(D[triu].mean())

def one_nna(gen_fps, ref_fps, n_subsample=2000):
    n_g = min(len(gen_fps), n_subsample)
    n_r = min(len(ref_fps), n_subsample)
    idx_g = np.random.choice(len(gen_fps), n_g, replace=False)
    idx_r = np.random.choice(len(ref_fps), n_r, replace=False)
    comb = np.vstack([gen_fps[idx_g], ref_fps[idx_r]])
    labels = np.array([0] * n_g + [1] * n_r)
    D = jaccard_distance_matrix(comb, comb)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    return float((labels[nn] == labels).mean())

N_EVAL = 5000
thresholds = np.linspace(0, 1.0, 51)

prot_eval = prot_fps[np.random.choice(len(prot_fps), min(N_EVAL, len(prot_fps)), replace=False)]
rna_eval  = rna_fps[np.random.choice(len(rna_fps),  min(N_EVAL, len(rna_fps)),  replace=False)]
method_eval = {m: method_fps_dict[m][np.random.choice(len(method_fps_dict[m]),
                min(N_EVAL, len(method_fps_dict[m])), replace=False)] for m in MAIN_METHODS}

log(f"  {'Set':<35} {'Internal Div':>12}")
log(f"  {'-'*47}")
log(f"  {'BindingDB Protein':<35} {internal_diversity(prot_eval):>12.4f}")
log(f"  {'BindingDB NA':<35}     {internal_diversity(rna_eval):>12.4f}")
for m in MAIN_METHODS:
    log(f"  {m:<35} {internal_diversity(method_eval[m]):>12.4f}")

# Self metrics
n_self = min(N_EVAL // 2, len(prot_fps) // 2)
prot_a = prot_fps[np.random.choice(len(prot_fps), n_self, replace=False)]
prot_b = prot_fps[np.random.choice(len(prot_fps), n_self, replace=False)]
prot_self_cov, prot_self_mmd = cov_mmd(prot_a, prot_b, thresholds)
prot_self_1nna = one_nna(prot_a, prot_b)

n_self_r = min(N_EVAL // 2, len(rna_fps) // 2)
rna_a = rna_fps[np.random.choice(len(rna_fps), n_self_r, replace=False)]
rna_b = rna_fps[np.random.choice(len(rna_fps), n_self_r, replace=False)]
rna_self_cov, rna_self_mmd = cov_mmd(rna_a, rna_b, thresholds)
rna_self_1nna = one_nna(rna_a, rna_b)

for label, ref_fps, self_cov, self_mmd, self_1nna in [
    ('Protein', prot_eval, prot_self_cov, prot_self_mmd, prot_self_1nna),
    ('NA',      rna_eval,  rna_self_cov,  rna_self_mmd,  rna_self_1nna),
]:
    log(f"\n  COV — Coverage of BindingDB {label} space:")
    log(f"  {'Method':<35} {'τ=0.2':>8} {'τ=0.3':>8} {'τ=0.4':>8} {'τ=0.5':>8} {'MMD':>8} {'1-NNA':>8}")
    log(f"  {'-'*83}")
    log(f"  {label + ' self':<35} {self_cov[10]:>8.4f} {self_cov[15]:>8.4f} {self_cov[20]:>8.4f} {self_cov[25]:>8.4f} {self_mmd:>8.4f} {self_1nna:>8.4f}")
    for m in MAIN_METHODS:
        cov, mmd_v = cov_mmd(method_eval[m], ref_fps, thresholds)
        nna = one_nna(method_eval[m], ref_fps)
        log(f"  {m:<35} {cov[10]:>8.4f} {cov[15]:>8.4f} {cov[20]:>8.4f} {cov[25]:>8.4f} {mmd_v:>8.4f} {nna:>8.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. Phase 4b — Coverage of displayed reference molecules (sampled for efficiency)
# ══════════════════════════════════════════════════════════════════════════
log("\n[Coverage] Computing per-method coverage for UMAP display sample (τ=0.3)...")

N_DISPLAY = 12000
np.random.seed(42)

# Sample reference fingerprints for coverage computation
n_prot_disp = min(N_DISPLAY // 2, len(prot_fps))
n_na_disp   = min(N_DISPLAY // 2, len(rna_fps))
idx_prot_disp = np.random.choice(len(prot_fps), n_prot_disp, replace=False)
idx_na_disp   = np.random.choice(len(rna_fps),  n_na_disp,  replace=False)

# Fingerprints of displayed reference points
umap_disp_prot_fps = prot_fps[idx_prot_disp]
umap_disp_na_fps   = rna_fps[idx_na_disp]
umap_disp_fps = np.vstack([umap_disp_prot_fps, umap_disp_na_fps])
umap_disp_is_na = np.array([False] * n_prot_disp + [True] * n_na_disp)

# UMAP 2D coords of displayed reference points
umap_disp_prot = umap_prot[idx_prot_disp]
umap_disp_na   = umap_rna[idx_na_disp]
umap_disp_all = np.vstack([umap_disp_prot, umap_disp_na])

def batch_jaccard_min(ref_fps, gen_fps, batch_size=4000):
    """Min Jaccard distance from each ref row to any gen row, batched over gen."""
    min_dist = np.full(len(ref_fps), np.inf, dtype=np.float32)
    for i in range(0, len(gen_fps), batch_size):
        D = jaccard_distance_matrix(ref_fps, gen_fps[i:i+batch_size])
        min_dist = np.minimum(min_dist, D.min(axis=1))
    return min_dist

tau_cover = 0.3
method_cover = {}
for m in MAIN_METHODS:
    t0 = time.time()
    min_dist = batch_jaccard_min(umap_disp_fps, method_fps_dict[m])
    method_cover[m] = min_dist <= (1.0 - tau_cover)
    cov_prot = method_cover[m][~umap_disp_is_na].mean()
    cov_na = method_cover[m][umap_disp_is_na].mean()
    log(f"  {_short_name(m)}: COV_NA(τ=0.3)={cov_na:.4f}  COV_Prot(τ=0.3)={cov_prot:.4f}  ({time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════
# 5. Phase 4c — Per-molecule RNA bias: T_NA − T_Prot
# ══════════════════════════════════════════════════════════════════════════
log("\n[RNA bias] Computing per-molecule RNA-likeness (max Tanimoto to NA vs Protein)...")

# Use eval subsets (5000 each) as reference for efficiency
rna_bias = {}
for m in MAIN_METHODS:
    t0 = time.time()
    fps = method_eval[m]
    # Max Tanimoto to NA refs
    D_na = batch_jaccard_min(fps, rna_eval[:4000])
    T_na = 1.0 - D_na
    # Max Tanimoto to Prot refs
    D_prot = batch_jaccard_min(fps, prot_eval[:4000])
    T_prot = 1.0 - D_prot
    bias = T_na - T_prot
    rna_bias[m] = bias

    n_na = (bias > 0.05).mean() * 100
    n_mid = (np.abs(bias) <= 0.05).mean() * 100
    n_prot = (bias < -0.05).mean() * 100
    log(f"  {m}: RNA-leaning={n_na:.1f}%  intermediate={n_mid:.1f}%  Prot-leaning={n_prot:.1f}%  (mean bias={bias.mean():.4f}, {time.time()-t0:.0f}s)")

# BindingDB reference bias: how NA-like are Protein ligands themselves, and vice versa?
log("  Computing BindingDB reference bias...")
t0 = time.time()

# Protein ligands: T_NA − T_Prot (split Prot set to avoid self-comparison)
n_pb = min(3000, len(prot_eval))
prot_self_ref = prot_eval[n_pb:min(n_pb + 3000, len(prot_eval))]
D_prot_prot = batch_jaccard_min(prot_eval[:n_pb], prot_self_ref)
D_prot_na   = batch_jaccard_min(prot_eval[:n_pb], rna_eval[:4000])
prot_bias = (1.0 - D_prot_na) - (1.0 - D_prot_prot)

# NA ligands: T_NA − T_Prot (split NA set to avoid self-comparison)
n_nb = min(900, len(rna_eval))
na_self_ref = rna_eval[n_nb:min(n_nb + 900, len(rna_eval))]
D_na_na   = batch_jaccard_min(rna_eval[:n_nb], na_self_ref)
D_na_prot = batch_jaccard_min(rna_eval[:n_nb], prot_eval[:4000])
na_bias_ref = (1.0 - D_na_na) - (1.0 - D_na_prot)

log(f"  BindingDB Protein (N={len(prot_eval[:n_pb]):,}): mean bias={prot_bias.mean():.4f}  "
    f"NA-lean={(prot_bias > 0.05).mean()*100:.1f}%  "
    f"inter={(np.abs(prot_bias) <= 0.05).mean()*100:.1f}%  "
    f"Prot-lean={(prot_bias < -0.05).mean()*100:.1f}%")
log(f"  BindingDB NA (N={len(rna_eval[:n_nb]):,}):      mean bias={na_bias_ref.mean():.4f}  "
    f"NA-lean={(na_bias_ref > 0.05).mean()*100:.1f}%  "
    f"inter={(np.abs(na_bias_ref) <= 0.05).mean()*100:.1f}%  "
    f"Prot-lean={(na_bias_ref < -0.05).mean()*100:.1f}%  ({time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════
# 6. Phase 5 — Figures
# ══════════════════════════════════════════════════════════════════════════

# ── Fig 1: UMAP coverage map (2×3 grid) ──
log("\n[Fig 1] UMAP coverage map — which reference molecules does each method cover?")

# UMAP-specific colors: Protein=blue, NA=red, covered dots=uniform green
UMAP_PROT_COLOR = '#4477AA'
UMAP_NA_COLOR   = '#CC3311'
UMAP_COVER_COLOR = '#228833'  # uniform green for all methods

umap_ref_all = np.vstack([umap_prot, umap_rna])

# Shared KDE grid for background density
x_all_for_kde = np.vstack([umap_prot, umap_rna])
x_range = np.linspace(x_all_for_kde[:, 0].min(), x_all_for_kde[:, 0].max(), 80)
y_range = np.linspace(x_all_for_kde[:, 1].min(), x_all_for_kde[:, 1].max(), 80)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.vstack([xx.ravel(), yy.ravel()])

# Reference KDEs
kde_prot = gaussian_kde(umap_prot.T) if len(umap_prot) > 10 else None
kde_na  = gaussian_kde(umap_rna.T)  if len(umap_rna) > 10 else None
zz_prot = kde_prot(grid_points).reshape(xx.shape) if kde_prot else None
zz_na  = kde_na(grid_points).reshape(xx.shape)   if kde_na  else None

if zz_prot is not None:
    lv_prot = np.linspace(zz_prot.max() * 0.08, zz_prot.max(), 8)
if zz_na is not None:
    lv_na = np.linspace(zz_na.max() * 0.08, zz_na.max(), 8)

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
axes = axes.flatten()

# Panel (a): Reference space only (filled KDE)
ax = axes[0]
_panel_label(ax, '(a) BindingDB reference', x=-0.05, y=1.02)
if zz_prot is not None:
    ax.contourf(xx, yy, zz_prot, levels=lv_prot, cmap='Blues', alpha=0.25, zorder=0)
if zz_na is not None:
    ax.contourf(xx, yy, zz_na, levels=lv_na, cmap='Reds', alpha=0.25, zorder=1)
leg_ref = [
    plt.scatter([], [], color=UMAP_PROT_COLOR, s=30, label='BindingDB Protein'),
    plt.scatter([], [], color=UMAP_NA_COLOR,   s=30, label='BindingDB NA'),
]
ax.legend(handles=leg_ref, loc='lower left', fontsize=8, frameon=True, edgecolor='#cccccc')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'{len(umap_ref_all):,} reference ligands', fontsize=10, color='#555555')

# Panels (b)–(f): Coverage per method
for i, m in enumerate(MAIN_METHODS):
    ax = axes[i + 1]
    _panel_label(ax, f'({"bcdef"[i]}) {_short_name(m)}', x=-0.05, y=1.02)

    # Reference KDE background (same as panel a)
    if zz_prot is not None:
        ax.contourf(xx, yy, zz_prot, levels=lv_prot, cmap='Blues', alpha=0.25, zorder=0)
    if zz_na is not None:
        ax.contourf(xx, yy, zz_na, levels=lv_na, cmap='Reds', alpha=0.25, zorder=1)

    covered = method_cover[m]
    uncovered = ~covered

    # Uncovered: very faint
    if uncovered.sum() > 0:
        ax.scatter(umap_disp_all[uncovered, 0], umap_disp_all[uncovered, 1],
                   s=0.4, c='#cccccc', alpha=0.10, rasterized=True, zorder=2)

    # Covered: uniform green dots
    if covered.sum() > 0:
        ax.scatter(umap_disp_all[covered, 0], umap_disp_all[covered, 1],
                   s=1.0, c=UMAP_COVER_COLOR, alpha=0.55, rasterized=True, zorder=3,
                   edgecolors='none')

    ax.set_xticks([]); ax.set_yticks([])

    # Stats annotation
    cov_p_val = covered[~umap_disp_is_na].mean()
    cov_n_val = covered[umap_disp_is_na].mean()
    ax.text(0.97, 0.97, f'COV$_{{\\rm NA}}$ = {cov_n_val:.3f}\nCOV$_{{\\rm Prot}}$ = {cov_p_val:.3f}',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.85))

fig.subplots_adjust(wspace=0.05, hspace=0.15)
savefig('fig_chemspace_umap.png', tight=False)

# ── Fig 1b: RNA bias distribution ──
log("[Fig 1b] Per-molecule RNA bias: max Tanimoto to NA − max Tanimoto to Protein")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(bottom=0.20)

# Panel (a): KDE of RNA bias per method + BindingDB reference distributions
ax = axes[0]
_panel_label(ax, '(a)')
bias_range = np.linspace(-0.5, 0.5, 300)

# Plot BindingDB reference distributions as thick dashed lines
try:
    kde_pb = gaussian_kde(prot_bias)
    ax.plot(bias_range, kde_pb(bias_range), color=UMAP_PROT_COLOR, linewidth=2.5,
            linestyle='--', label='BindingDB Protein', zorder=10)
except Exception:
    pass
try:
    kde_nb = gaussian_kde(na_bias_ref)
    ax.plot(bias_range, kde_nb(bias_range), color=UMAP_NA_COLOR, linewidth=2.5,
            linestyle='--', label='BindingDB NA', zorder=10)
except Exception:
    pass

# Plot method distributions
for m in MAIN_METHODS:
    bias = rna_bias[m]
    try:
        kde = gaussian_kde(bias)
        ax.plot(bias_range, kde(bias_range), color=METHOD_COLORS[m], linewidth=2.0, label=_short_name(m))
        ax.fill_between(bias_range, kde(bias_range), color=METHOD_COLORS[m], alpha=0.06)
    except Exception:
        pass

# Shade intermediate zone
ax.axvspan(-0.05, 0.05, color='#cccccc', alpha=0.20, zorder=0)
ax.axvline(x=0, color='#333333', linewidth=0.8, linestyle='--')
ax.set_xlabel('RNA bias  (T$_{NA}$ − T$_{Prot}$)')
ax.set_ylabel('Density')
ax.legend(fontsize=7, frameon=True, edgecolor='#cccccc', ncol=2)
ax.grid(alpha=0.12, linestyle='--')
ax.text(0.02, 0.95, '← Prot-leaning', transform=ax.transAxes, fontsize=8,
        va='top', ha='left', color='#888888')
ax.text(0.98, 0.95, 'NA-leaning →', transform=ax.transAxes, fontsize=8,
        va='top', ha='right', color='#888888')

# Panel (b): Stacked bar — fraction in each zone + BindingDB reference
ax = axes[1]
_panel_label(ax, '(b)')
x_bars = np.arange(len(MAIN_METHODS) + 2)  # +2 for BindingDB ref
w = 0.55
na_fracs = [(rna_bias[m] > 0.05).mean() * 100 for m in MAIN_METHODS]
mid_fracs = [(np.abs(rna_bias[m]) <= 0.05).mean() * 100 for m in MAIN_METHODS]
prot_fracs = [(rna_bias[m] < -0.05).mean() * 100 for m in MAIN_METHODS]
# Add BindingDB reference
all_na = na_fracs + [(na_bias_ref > 0.05).mean() * 100, (prot_bias > 0.05).mean() * 100]
all_mid = mid_fracs + [(np.abs(na_bias_ref) <= 0.05).mean() * 100, (np.abs(prot_bias) <= 0.05).mean() * 100]
all_prot = prot_fracs + [(na_bias_ref < -0.05).mean() * 100, (prot_bias < -0.05).mean() * 100]

ax.bar(x_bars, all_prot, w, color='#4477AA', alpha=0.80, label='Prot-leaning  (bias < −0.05)')
ax.bar(x_bars, all_mid, w, bottom=all_prot, color='#BBBBBB', alpha=0.80,
       label='Intermediate  (|bias| ≤ 0.05)')
ax.bar(x_bars, all_na, w, bottom=np.array(all_prot) + np.array(all_mid),
       color='#CC3311', alpha=0.80, label='NA-leaning  (bias > 0.05)')

for i, (p, m, n) in enumerate(zip(all_prot, all_mid, all_na)):
    ax.text(i, p / 2, f'{p:.1f}%', ha='center', va='center', fontsize=7.5)
    ax.text(i, p + m / 2, f'{m:.1f}%', ha='center', va='center', fontsize=7.5)
    ax.text(i, p + m + n / 2, f'{n:.1f}%', ha='center', va='center', fontsize=7.5, color='white')

all_labels = [_short_name(m) for m in MAIN_METHODS] + ['BindingDB\nNA', 'BindingDB\nProtein']
ax.set_xticks(x_bars)
ax.set_xticklabels(all_labels, fontsize=8.5)
ax.set_ylabel('Fraction of molecules (%)')
ax.legend(fontsize=7, frameon=True, edgecolor='#cccccc', ncol=1, loc='upper right')
ax.grid(axis='y', alpha=0.12, linestyle='--')
ax.set_ylim(0, 100)

savefig('fig_chemspace_rna_bias.png')

# ── Fig 2: COV curves ──
log("[Fig 2] Coverage curves")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax_idx, (bkg_label, bkg_color, self_cov, ref_fps) in enumerate([
    ('Protein', BKG_PROT_COLOR, prot_self_cov, prot_eval),
    ('NA',      BKG_RNA_COLOR,  rna_self_cov,  rna_eval),
]):
    ax = axes[ax_idx]
    _panel_label(ax, f'({"ab"[ax_idx]})')
    ax.plot(thresholds, self_cov, color=bkg_color, linewidth=2.5, linestyle='--',
            label=f'{bkg_label} self', zorder=1)
    for m in MAIN_METHODS:
        cov, _ = cov_mmd(method_eval[m], ref_fps, thresholds)
        ax.plot(thresholds, cov, color=METHOD_COLORS[m], linewidth=1.8, label=_short_name(m), zorder=5)
    ax.set_xlabel('Tanimoto threshold τ')
    ax.set_ylabel(f'Fraction of {bkg_label} ligands covered')
    ax.set_title(f'Coverage of BindingDB {bkg_label} space', fontsize=12, weight='bold')
    ax.legend(fontsize=7.5, frameon=True, edgecolor='#cccccc')
    ax.grid(alpha=0.15, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

savefig('fig_chemspace_coverage.png')

# ── Fig 3: MMD & 1-NNA bars ──
log("[Fig 3] MMD & 1-NNA bar charts")

all_metrics = {}
for m in MAIN_METHODS:
    cov_prot, mmd_prot = cov_mmd(method_eval[m], prot_eval, thresholds)
    cov_rna, mmd_rna = cov_mmd(method_eval[m], rna_eval, thresholds)
    all_metrics[m] = {
        'mmd_prot': mmd_prot, 'mmd_rna': mmd_rna,
        'nna_prot': one_nna(method_eval[m], prot_eval),
        'nna_rna':  one_nna(method_eval[m], rna_eval),
    }

fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
x_bars = np.arange(len(MAIN_METHODS))
w = 0.35
short_names = [_short_name(m) for m in MAIN_METHODS]

ax = axes[0]; _panel_label(ax, '(a)')
ax.bar(x_bars - w/2, [all_metrics[m]['mmd_prot'] for m in MAIN_METHODS],
       w, color=BKG_PROT_COLOR, alpha=0.85, label='vs Protein ligands')
ax.bar(x_bars + w/2, [all_metrics[m]['mmd_rna'] for m in MAIN_METHODS],
       w, color=BKG_RNA_COLOR, alpha=0.85, label='vs NA ligands')
ax.axhline(y=prot_self_mmd, color=BKG_PROT_COLOR, linewidth=1.2, linestyle='--', alpha=0.7)
ax.axhline(y=rna_self_mmd, color=BKG_RNA_COLOR, linewidth=1.2, linestyle='--', alpha=0.7)
ax.set_xticks(x_bars); ax.set_xticklabels(short_names, fontsize=8.5)
ax.set_ylabel('MMD (mean min Jaccard dist.)')
ax.set_title('Minimum Matching Distance', fontsize=12, weight='bold')
ax.legend(fontsize=8, frameon=True, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.15, linestyle='--')

ax = axes[1]; _panel_label(ax, '(b)')
ax.bar(x_bars - w/2, [all_metrics[m]['nna_prot'] for m in MAIN_METHODS],
       w, color=BKG_PROT_COLOR, alpha=0.85, label='vs Protein ligands')
ax.bar(x_bars + w/2, [all_metrics[m]['nna_rna'] for m in MAIN_METHODS],
       w, color=BKG_RNA_COLOR, alpha=0.85, label='vs NA ligands')
ax.axhline(y=0.5, color='#333333', linewidth=1.2, linestyle=':', alpha=0.7)
ax.axhline(y=prot_self_1nna, color=BKG_PROT_COLOR, linewidth=1.2, linestyle='--', alpha=0.7)
ax.axhline(y=rna_self_1nna, color=BKG_RNA_COLOR, linewidth=1.2, linestyle='--', alpha=0.7)
ax.set_xticks(x_bars); ax.set_xticklabels(short_names, fontsize=8.5)
ax.set_ylabel('1-NNA accuracy')
ax.set_title('1-Nearest Neighbor Accuracy', fontsize=12, weight='bold')
ax.legend(fontsize=8, frameon=True, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.15, linestyle='--')

savefig('fig_chemspace_metrics.png')

# ── Fig 4: Property distributions ──
log("[Fig 4] Property distributions")
palette_all = {**METHOD_COLORS}

def compute_props(smiles_list, label, max_n=None):
    if max_n and len(smiles_list) > max_n:
        smiles_list = np.random.choice(smiles_list, max_n, replace=False)
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        records.append({
            'mw': Descriptors.MolWt(mol), 'logp': Crippen.MolLogP(mol),
            'qed': QED.qed(mol), 'tpsa': Descriptors.TPSA(mol),
            'hba': rdMolDescriptors.CalcNumHBA(mol),
            'hbd': rdMolDescriptors.CalcNumHBD(mol),
            'rot_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'rings': rdMolDescriptors.CalcNumRings(mol),
        })
    return pd.DataFrame(records).assign(source=label)

plot_props_sel = [
    ('mw', 'Molecular weight'), ('qed', 'QED'), ('logp', 'logP'),
    ('tpsa', 'TPSA (Å²)'), ('hba', 'HBA'), ('hbd', 'HBD'),
    ('rot_bonds', 'Rotatable bonds'), ('rings', 'Rings'),
]
props_parts = []
for m in MAIN_METHODS:
    props_parts.append(compute_props(method_smiles[m], m, max_n=6000))
props_parts.append(compute_props(bkg_prot, 'BindingDB Protein', max_n=8000))
props_parts.append(compute_props(bkg_rna,  'BindingDB NA',     max_n=8000))
props_all = pd.concat(props_parts, ignore_index=True)
order = ['BindingDB Protein', 'BindingDB NA'] + MAIN_METHODS

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for ax_idx, (col, label) in enumerate(plot_props_sel):
    ax = axes[ax_idx // 4][ax_idx % 4]
    _panel_label(ax, f'({"abcdefgh"[ax_idx]})')
    for group in order:
        sub = props_all[props_all['source'] == group][col].dropna()
        if len(sub) < 5:
            continue
        is_bkg = 'BindingDB' in group
        try:
            kde = gaussian_kde(sub.values)
            x_r = np.linspace(sub.quantile(0.005), sub.quantile(0.995), 200)
            ax.plot(x_r, kde(x_r), color=palette_all[group],
                    linewidth=1.0 if is_bkg else 1.5, alpha=0.35 if is_bkg else 0.8)
            if not is_bkg:
                ax.fill_between(x_r, kde(x_r), color=palette_all[group], alpha=0.06)
        except Exception:
            pass
    ax.set_xlabel(label); ax.set_ylabel('Density')
    ax.grid(alpha=0.15, linestyle='--')

handles = [Line2D([0], [0], color=palette_all[g], linewidth=2, label=_short_name(g) if g in MAIN_METHODS else g) for g in order]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=True, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.04))
savefig('fig_chemspace_properties.png')

# Free memory before Phase 5 (virtual screening is memory-heavy)
del props_all, props_parts, palette_all, umap_disp_fps, umap_disp_prot_fps, umap_disp_na_fps
del method_cover
import gc; gc.collect()

# ══════════════════════════════════════════════════════════════════════════
# 6. Phase 5 — Virtual screening: survival + NA vs Protein COV (aligned with visualize_methods.py)
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Phase 5: Virtual screening cascade")
log("=" * 60)

# Build SMILES → fingerprint index
smiles_to_fp = {}
for m in MAIN_METHODS:
    smi_arr = method_smiles[m]
    fps_arr = method_fps_dict[m]
    smiles_to_fp[m] = dict(zip(smi_arr, range(len(fps_arr))))

# Filter cascade — aligned with visualize_methods.py fig6
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

# Reference sets for MMD/COV computation
np.random.seed(42)
prot_ref = prot_fps[np.random.choice(len(prot_fps), min(N_EVAL, len(prot_fps)), replace=False)]
rna_ref  = rna_fps[np.random.choice(len(rna_fps),  min(N_EVAL, len(rna_fps)),  replace=False)]

tau_idx = 15  # τ=0.3
vs_results = {m: [] for m in MAIN_METHODS}

log(f"\n  COV(τ=0.3) across virtual screening steps:")
log(f"  {'Step':<20} {'surv':>7}  {'COV_NA':>8} {'COV_Prot':>8} {'ratio':>8}")
log(f"  {'-'*60}")

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

    # Log first and last valid steps for this method
    first = vs_results[m][0]
    last_vals = [(r['cov_rna'], r['cov_prot']) for r in vs_results[m]
                 if not np.isnan(r['cov_rna'])]
    log(f"\n  {m}:")
    for r in [vs_results[m][0], vs_results[m][2], vs_results[m][5], vs_results[m][7],
              vs_results[m][9], vs_results[m][10]]:
        if not np.isnan(r['cov_rna']):
            ratio = r['cov_rna'] / r['cov_prot'] if r['cov_prot'] > 1e-10 else np.nan
            log(f"    {r['step']:<20} surv={r['surv']:>7,}  COV_NA={r['cov_rna']:.4f}"
                f"  COV_Prot={r['cov_prot']:.4f}  ratio={ratio:.2f}")

prot_self_cov_03 = prot_self_cov[tau_idx]
rna_self_cov_03  = rna_self_cov[tau_idx]

# ── Fig 5: Virtual screening — survival + NA/Prot COV comparison ──
log("\n[Fig 5] Virtual screening cascade (aligned with visualize_methods.py style)")

import matplotlib.ticker as mticker

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

# Panel (b): COV(τ=0.3) — solid=NA, dashed=Protein (direct comparison per method)
ax = axes[1]
_panel_label(ax, '(b)')
ax.set_facecolor('#fafafa')

for m in MAIN_METHODS:
    cov_na = [r['cov_rna'] for r in vs_results[m]]
    cov_pr = [r['cov_prot'] for r in vs_results[m]]
    # Solid for NA, dashed for Protein
    ax.plot(x, cov_na, color=METHOD_COLORS[m], linewidth=2.0, marker='o',
            markersize=6, label=f'{m} vs NA', zorder=5)
    ax.plot(x, cov_pr, color=METHOD_COLORS[m], linewidth=1.2, marker='s',
            markersize=4, linestyle='--', alpha=0.55, zorder=4)

# Reference lines
ax.axhline(y=rna_self_cov_03, color=BKG_RNA_COLOR, linewidth=2.0, linestyle='-',
           alpha=0.6)
ax.axhline(y=prot_self_cov_03, color=BKG_PROT_COLOR, linewidth=2.0, linestyle='--',
           alpha=0.6)

# Legend: NA/Prot line style + self lines + methods (compact)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='#333333', linewidth=2.2, marker='o', markersize=5, label='vs BindingDB NA (solid)'),
    Line2D([0], [0], color='#333333', linewidth=1.5, marker='s', markersize=4,
           linestyle='--', alpha=0.55, label='vs BindingDB Prot (dashed)'),
    Line2D([0], [0], color=BKG_RNA_COLOR, linewidth=2.0, label='NA self (τ=0.3)'),
    Line2D([0], [0], color=BKG_PROT_COLOR, linewidth=2.0, linestyle='--', label='Prot self (τ=0.3)'),
]
# Add method color markers
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

# ══════════════════════════════════════════════════════════════════════════
# 7. Phase 6 — DUDE (Protein targets) vs SMRTBench (RNA targets) comparison
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Phase 6: RNA-target vs Protein-target chemical space comparison")
log("=" * 60)

# ── 6a. Method mapping ──
# SMRTBench (RNA targets) key → common display name
SMRT_KEY = {
    'RNA2smol (Only General Feat)': 'RNA2smol',
    'MolCRAFT':                     'MolCRAFT',
    'Pocket2Mol':                   'Pocket2Mol',
    'ED2Mol':                       'ED2Mol',
}
# DUDE (Protein targets) mode → common display name
DUDE_MODE_MAP = {
    'RNA2smol':  'RNA2smol',
    'MolCRAFT':  'MolCRAFT',
    'ed2mol':    'ED2Mol',
    'Pocket2Mol':'Pocket2Mol',
}
COMPARE_METHODS = ['RNA2smol', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']
# Reverse: common → SMRTBench key
COMMON_TO_SMRT = {v: k for k, v in SMRT_KEY.items()}
# Reverse: common → DUDE mode
COMMON_TO_DUDE = {v: k for k, v in DUDE_MODE_MAP.items()}

# ── 6b. Load & filter DUDE data ──
log("Loading DUDE data...")
dude_raw = pd.read_csv(Path(__file__).resolve().parent / 'dude_stat_all.csv',
                       usecols=['SMILES', 'mode', 'mw', 'qed', 'logp'])
# Apply same drug-likeness filter using precomputed columns
dude = dude_raw[(dude_raw['mw'] >= 150) & (dude_raw['mw'] <= 700) &
                (dude_raw['qed'] > 0.2) &
                (dude_raw['logp'] >= -5) & (dude_raw['logp'] <= 10)]
dude = dude[dude['mode'].isin(DUDE_MODE_MAP)]
log(f"  DUDE raw: {len(dude_raw):,}  →  filtered: {len(dude):,}")

dude_smiles = {}
dude_props = {}
for mode, common in DUDE_MODE_MAP.items():
    subset = dude[dude['mode'] == mode]
    # Deduplicate SMILES
    uniq = subset['SMILES'].dropna().unique()
    dude_smiles[common] = uniq
    # Store properties for matched SMILES
    smi_to_row = subset.drop_duplicates('SMILES').set_index('SMILES')
    dude_props[common] = smi_to_row.reindex(uniq)
    log(f"  DUDE {common} ({mode}): {len(subset):,} → {len(uniq):,} unique SMILES")
del dude_raw, dude

# ── 6c. Compute ECFP4 fingerprints (cached) ──
log("\n  DUDE fingerprint computation...")
dude_fps = {}
for m in COMPARE_METHODS:
    cache_name = f'dude_{m.replace(" ", "_")}'
    dude_fps[m] = compute_fps(dude_smiles[m], cache_name)

# ── 6d. PCA transform (use existing model) ──
log("\n  DUDE PCA transform...")
dude_pca = {}
for m in COMPARE_METHODS:
    dude_pca[m] = pca.transform(dude_fps[m])
    log(f"  DUDE {m}: {len(dude_fps[m]):,} mols → {dude_pca[m].shape[1]} PCs")

# ── 6e. UMAP transform (use existing model) ──
log("\n  DUDE UMAP transform...")
dude_umap = {}
for m in COMPARE_METHODS:
    t0 = time.time()
    dude_umap[m] = reducer.transform(dude_pca[m])
    log(f"  DUDE {m}: {len(dude_umap[m]):,} mols → 2D ({time.time()-t0:.0f}s)")

# ── 6f. RNA bias for DUDE molecules ──
log("\n  DUDE RNA bias computation (max Tanimoto to NA vs Protein refs)...")
dude_bias = {}
N_BIAS_EVAL = 5000
for m in COMPARE_METHODS:
    t0 = time.time()
    n_eval = min(N_BIAS_EVAL, len(dude_fps[m]))
    idx = np.random.choice(len(dude_fps[m]), n_eval, replace=False)
    fps = dude_fps[m][idx]
    D_na = batch_jaccard_min(fps, rna_eval[:4000])
    D_prot = batch_jaccard_min(fps, prot_eval[:4000])
    dude_bias[m] = (1.0 - D_na) - (1.0 - D_prot)

    n_na = (dude_bias[m] > 0.05).mean() * 100
    n_mid = (np.abs(dude_bias[m]) <= 0.05).mean() * 100
    n_prot = (dude_bias[m] < -0.05).mean() * 100
    log(f"  DUDE {m}: NA-lean={n_na:.1f}%  inter={n_mid:.1f}%  Prot-lean={n_prot:.1f}%  "
        f"(mean bias={dude_bias[m].mean():.4f}, {time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════
# Phase 6 Figures
# ══════════════════════════════════════════════════════════════════════════

# ── Fig 6a: UMAP density comparison (2 rows × 4 columns) ──
log("\n[Fig 6a] UMAP density: RNA targets (SMRTBench) vs Protein targets (DUDE)")

N_UMAP_DISP = 6000  # molecules to display per method per target type
# Sample SMRTBench molecules for display
smrt_umap_disp = {}
for smrt_key, common in SMRT_KEY.items():
    n = min(N_UMAP_DISP, len(umap_methods[smrt_key]))
    idx = np.random.choice(len(umap_methods[smrt_key]), n, replace=False)
    smrt_umap_disp[common] = umap_methods[smrt_key][idx]
# Sample DUDE molecules for display
dude_umap_disp = {}
for m in COMPARE_METHODS:
    n = min(N_UMAP_DISP, len(dude_umap[m]))
    idx = np.random.choice(len(dude_umap[m]), n, replace=False)
    dude_umap_disp[m] = dude_umap[m][idx]

fig, axes = plt.subplots(2, 4, figsize=(24, 10.5))
ROW_LABELS = ['RNA targets\n(SMRTBench)', 'Protein targets\n(DUDE)']

for row, (label, data_dict) in enumerate(zip(ROW_LABELS, [smrt_umap_disp, dude_umap_disp])):
    for col, m in enumerate(COMPARE_METHODS):
        ax = axes[row, col]
        # BindingDB KDE background
        if zz_prot is not None:
            ax.contourf(xx, yy, zz_prot, levels=lv_prot, cmap='Blues', alpha=0.18, zorder=0)
        if zz_na is not None:
            ax.contourf(xx, yy, zz_na, levels=lv_na, cmap='Reds', alpha=0.18, zorder=1)

        pts = data_dict[m]
        n_disp = min(3000, len(pts))  # cap scatter points for visual clarity
        if len(pts) > n_disp:
            pts = pts[np.random.choice(len(pts), n_disp, replace=False)]

        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c=METHOD_COLORS[COMMON_TO_SMRT[m]],
                   alpha=0.35, rasterized=True, zorder=2, edgecolors='none')

        ax.set_xticks([]); ax.set_yticks([])

        # Column title (method name) on top row only
        if row == 0:
            ax.set_title(_short_name(COMMON_TO_SMRT[m]), fontsize=12, weight='bold', color='#333333')
        # Row label on first column
        if col == 0:
            ax.set_ylabel(label, fontsize=10, weight='bold', color='#555555', labelpad=8)

        # Molecule count annotation
        ax.text(0.97, 0.04, f'n={len(data_dict[m]):,}', transform=ax.transAxes,
                fontsize=7.5, ha='right', va='bottom', color='#666666')

fig.subplots_adjust(wspace=0.03, hspace=0.06)
savefig('fig_chemspace_dude_umap_comparison.png')

# ── Fig 6b: RNA bias shift (violin: RNA vs Protein targets per method) ──
log("[Fig 6b] RNA bias comparison: RNA vs Protein targets")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Prepare SMRTBench RNA bias (recompute from method_eval for consistency)
smrt_bias_for_plot = {}
for smrt_key, common in SMRT_KEY.items():
    smrt_bias_for_plot[common] = rna_bias[smrt_key]

for i, m in enumerate(COMPARE_METHODS):
    ax = axes[i]
    _panel_label(ax, f'({"abcd"[i]}) {m}', x=-0.05, y=1.02)

    smrt_b = smrt_bias_for_plot[m]
    dude_b = dude_bias[m]

    # Violin plots
    parts = ax.violinplot([smrt_b, dude_b], positions=[0, 1], widths=0.5,
                          showmeans=True, showmedians=True)

    # Style violins: RNA targets (left) solid, Protein targets (right) hatched
    colors = [METHOD_COLORS[COMMON_TO_SMRT[m]], '#666666']
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[j])
        pc.set_alpha(0.55 if j == 0 else 0.65)
        pc.set_edgecolor('#333333')
        pc.set_linewidth(0.6)
    for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if key in parts:
            parts[key].set_color('#333333')
            parts[key].set_linewidth(0.8)

    # Reference lines
    ax.axhline(y=prot_bias.mean(), color=BKG_PROT_COLOR, linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(y=na_bias_ref.mean(), color=BKG_RNA_COLOR, linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='#999999', linewidth=0.6, linestyle=':')

    # Annotation
    ax.text(0.02, 0.96, f'RNA: μ={smrt_b.mean():.3f}\nProt: μ={dude_b.mean():.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.8))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['RNA\ntargets', 'Protein\ntargets'], fontsize=9)
    ax.set_ylabel('RNA bias')
    ax.set_ylim(-0.55, 0.55)
    ax.grid(alpha=0.12, linestyle='--')

# Shared legend for reference lines
from matplotlib.lines import Line2D
legend_ref = [
    Line2D([0], [0], color=BKG_PROT_COLOR, linewidth=1.2, linestyle='--', label=f'BindingDB Prot (μ={prot_bias.mean():.3f})'),
    Line2D([0], [0], color=BKG_RNA_COLOR, linewidth=1.2, linestyle='--', label=f'BindingDB NA (μ={na_bias_ref.mean():.3f})'),
]
fig.legend(handles=legend_ref, fontsize=8.5, frameon=True, edgecolor='#cccccc',
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(wspace=0.12, hspace=0.25, bottom=0.10)
savefig('fig_chemspace_dude_rna_bias.png')

# ── Fig 6c: Property distribution comparison ──
log("[Fig 6c] Property distributions: RNA vs Protein targets")

# Compute properties for SMRTBench molecules (cached via existing method_smiles)
smrt_props = {}
for smrt_key, common in SMRT_KEY.items():
    from rdkit.Chem import Descriptors, Crippen, QED
    mw_vals, qed_vals, logp_vals = [], [], []
    for smi in method_smiles[smrt_key]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mw_vals.append(Descriptors.MolWt(mol))
            qed_vals.append(QED.qed(mol))
            logp_vals.append(Crippen.MolLogP(mol))
    smrt_props[common] = {'MW': np.array(mw_vals), 'QED': np.array(qed_vals), 'logP': np.array(logp_vals)}

fig, axes = plt.subplots(3, 4, figsize=(22, 12))
PROP_NAMES = ['MW', 'QED', 'logP']
PROP_LABELS = ['Molecular Weight', 'QED', 'logP']
PROP_COLORS = {'MW': '#4477AA', 'QED': '#EE7733', 'logP': '#009988'}

for row, (pname, plabel) in enumerate(zip(PROP_NAMES, PROP_LABELS)):
    for col, m in enumerate(COMPARE_METHODS):
        ax = axes[row, col]

        smrt_p = smrt_props[m][pname]
        dude_p = dude_props[m][pname.lower()].values.astype(float)

        # KDE comparison
        x_min = min(smrt_p.min(), dude_p.min())
        x_max = max(smrt_p.max(), dude_p.max())
        pad = (x_max - x_min) * 0.1
        x_range = np.linspace(x_min - pad, x_max + pad, 200)

        try:
            kde_smrt = gaussian_kde(smrt_p)
            kde_dude = gaussian_kde(dude_p)
            ax.fill_between(x_range, kde_smrt(x_range), alpha=0.4,
                            color=METHOD_COLORS[COMMON_TO_SMRT[m]], label='RNA targets')
            ax.fill_between(x_range, kde_dude(x_range), alpha=0.35,
                            color='#666666', label='Protein targets')
        except Exception:
            pass  # skip KDE if fails

        # Mean lines
        ax.axvline(smrt_p.mean(), color=METHOD_COLORS[COMMON_TO_SMRT[m]],
                   linewidth=1.2, linestyle='-', alpha=0.8)
        ax.axvline(dude_p.mean(), color='#666666',
                   linewidth=1.2, linestyle='--', alpha=0.8)

        # Title row (method names on top panel)
        if row == 0:
            ax.set_title(m, fontsize=11, weight='bold', color='#333333')
        # Y-axis label row
        if col == 0:
            ax.set_ylabel(plabel, fontsize=10)

        # Stats annotation
        ax.text(0.97, 0.95, f'RNA μ={smrt_p.mean():.1f}\nProt μ={dude_p.mean():.1f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.75))
        ax.grid(alpha=0.10, linestyle='--')

        if row == 2:
            ax.set_xlabel(plabel, fontsize=9)

fig.subplots_adjust(wspace=0.18, hspace=0.22)
# Legend
legend_props = [
    plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS[COMMON_TO_SMRT[COMPARE_METHODS[0]]], alpha=0.4, label='RNA targets'),
    plt.Rectangle((0, 0), 1, 1, fc='#666666', alpha=0.35, label='Protein targets'),
]
fig.legend(handles=legend_props, fontsize=9, frameon=True, edgecolor='#cccccc',
           loc='upper right', bbox_to_anchor=(0.98, 0.98))
savefig('fig_chemspace_dude_properties.png')

# Cleanup
del dude_fps, dude_pca, dude_umap, smrt_umap_disp, dude_umap_disp, smrt_props

# ══════════════════════════════════════════════════════════════════════════
# 8. Summary
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Summary")
log("=" * 60)
log(f"  BindingDB Protein: {len(prot_fps):,}  NA: {len(rna_fps):,}")
for m in MAIN_METHODS:
    log(f"  {m}: {len(method_fps_dict[m]):,}")
log(f"\n  PCA: {pca.explained_variance_ratio_.sum():.3f} explained variance (50 PCs)")
log(f"  ECFP4 metrics:")
log(f"  {'Method':<35} {'MMD(Prot)':>10} {'1-NNA(Prot)':>11} {'MMD(NA)':>10} {'1-NNA(NA)':>11}")
log(f"  {'-'*77}")
for m in MAIN_METHODS:
    log(f"  {m:<35} {all_metrics[m]['mmd_prot']:>10.4f} {all_metrics[m]['nna_prot']:>11.4f} {all_metrics[m]['mmd_rna']:>10.4f} {all_metrics[m]['nna_rna']:>11.4f}")
log(f"  {'Protein self':<35} {prot_self_mmd:>10.4f} {prot_self_1nna:>11.4f}")
log(f"  {'NA self':<35}     {rna_self_mmd:>10.4f} {rna_self_1nna:>11.4f}")

log(f"\n  Virtual Screening — NA vs Protein COV(τ=0.3) ratio:")
log(f"  (ratio > 1 → more NA-like; ratio < 1 → more Prot-like)")
log(f"  {'Method':<35} {'All':>8} {'aff<−7':>8} {'+QED>0.3':>8} {'Strict':>8}")
log(f"  {'-'*75}")
for m in MAIN_METHODS:
    steps = vs_results[m]
    ratios = []
    for i in [0, 2, 5, 7]:  # All, aff<-7, +QED>0.3, QED>0.4+SA<5
        if i < len(steps) and not np.isnan(steps[i]['cov_rna']) and steps[i]['cov_prot'] > 1e-10:
            ratios.append(steps[i]['cov_rna'] / steps[i]['cov_prot'])
        else:
            ratios.append(np.nan)
    log(f"  {m:<35} {ratios[0]:>8.2f} {ratios[1]:>8.2f} {ratios[2]:>8.2f} {ratios[3]:>8.2f}")

log(f"\n  Cache: {CACHE_DIR}/")
log(f"  Figures: {OUT_DIR}/fig_chemspace_*.png")
log("=" * 60)
