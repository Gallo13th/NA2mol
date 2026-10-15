"""
Compute strain energy for RNA2smol (Baseline) using PoseCheck.
Samples 300 mols/target, skips >60 heavy atoms, batch=25.
Retries failed batches individually.
"""
from pathlib import Path
import pandas as pd, numpy as np, pickle, os, re, glob
from rdkit import Chem, RDLogger
import posecheck
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

SDF_DIR = str(ROOT / 'notebooks/newbsl/output/baseline_merged')
CSV_PATH = str(ROOT / 'notebooks/output/smrtbench_combined_results.csv')
CACHE_PATH = str(ROOT / 'notebooks/output/cache_strain_energy.pkl')
MODE_NAME = 'RNA2smol (Baseline)'
MAX_PER_TARGET = 200
MAX_HEAVY = 50
BATCH_SIZE = 5  # small batch, each mol individually try/except inside compute_batch

def parse_pdb(name):
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand\.sdf$', name.lower())
    return m.group(1) if m else name[:4].lower()

def compute_batch(mols):
    """Try batch, fall back to individual on failure."""
    try:
        pc = posecheck.PoseCheck()
        pc.load_ligands_from_mols(mols)
        return pc.calculate_strain_energy()
    except Exception:
        results = []
        for mol in mols:
            try:
                pc = posecheck.PoseCheck()
                pc.load_ligands_from_mols([mol])
                results.append(pc.calculate_strain_energy()[0])
            except Exception:
                results.append(np.nan)
        return results

# Load cache
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)
if MODE_NAME in cache:
    n_old = len(cache[MODE_NAME])
    print(f'Existing cache: {n_old} entries, reusing...')
    strain_lookup = cache[MODE_NAME]
    # Find already-completed targets
    completed_pdbs = set(k[0] for k in strain_lookup)
else:
    strain_lookup = {}
    completed_pdbs = set()

sdf_files = sorted(glob.glob(os.path.join(SDF_DIR, '*.sdf')))
print(f'{len(sdf_files)} SDF files, {len(completed_pdbs)} targets already cached')

for sdf_path in tqdm(sdf_files, desc='Baseline'):
    basename = os.path.basename(sdf_path)
    pdb_id = parse_pdb(basename)
    if pdb_id in completed_pdbs:
        continue

    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    except:
        continue

    batch_mols, batch_keys = [], []
    mol_count = 0

    for mol in supplier:
        if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
            continue
        if mol.GetNumHeavyAtoms() > MAX_HEAVY:
            continue
        if mol_count >= MAX_PER_TARGET:
            break
        try:
            smi = Chem.MolToSmiles(mol)
        except:
            continue
        key = (pdb_id, smi)
        if key in strain_lookup:
            continue
        batch_mols.append(mol)
        batch_keys.append(key)
        mol_count += 1

        if len(batch_mols) >= BATCH_SIZE:
            strains = compute_batch(batch_mols)
            for k, s in zip(batch_keys, strains):
                strain_lookup[k] = s
            batch_mols, batch_keys = [], []

    if batch_mols:
        strains = compute_batch(batch_mols)
        for k, s in zip(batch_keys, strains):
            strain_lookup[k] = s

    # Incremental save
    cache[MODE_NAME] = strain_lookup
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

# Merge
valid = sum(1 for v in strain_lookup.values() if not np.isnan(v))
vals = [v for v in strain_lookup.values() if not np.isnan(v)]
print(f'\nStrain: {valid}/{len(strain_lookup)} valid, median={np.median(vals):.1f}, mean={np.mean(vals):.1f}')

df = pd.read_csv(CSV_PATH)
if 'strain' not in df.columns:
    df['strain'] = np.nan
matched = 0
mask = df['mode'] == MODE_NAME
for idx in df[mask].index:
    pdb_id = str(df.at[idx, 'pdb_id']).lower()
    smi = str(df.at[idx, 'SMILES'])
    key = (pdb_id, smi)
    if key in strain_lookup and not np.isnan(strain_lookup[key]):
        df.at[idx, 'strain'] = strain_lookup[key]
        matched += 1

df.to_csv(CSV_PATH, index=False)
print(f'Matched {matched}/{mask.sum()} ({matched/mask.sum()*100:.1f}%)')
