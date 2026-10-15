"""
Compute strain energy for RNA2smol (Baseline) SDFs.
Samples up to 300 molecules per target, skips molecules >60 heavy atoms.
"""
import pandas as pd, numpy as np, pickle, os, re, glob
from rdkit import Chem, RDLogger
import posecheck
from tqdm import tqdm
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

SDF_DIR = str(ROOT / 'notebooks/newbsl/output/baseline_merged')
CSV_PATH = str(ROOT / 'notebooks/output/smrtbench_combined_results.csv')
CACHE_PATH = str(ROOT / 'notebooks/output/cache_strain_energy.pkl')
MODE_NAME = 'RNA2smol (Baseline)'
MAX_PER_TARGET = 300
MAX_HEAVY = 60

def parse_pdb(name):
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand\.sdf$', name.lower())
    return m.group(1) if m else name[:4].lower()

# Load existing cache
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

if MODE_NAME in cache:
    print(f'Removing old cache for {MODE_NAME}')
    del cache[MODE_NAME]

strain_lookup = {}
sdf_files = sorted(glob.glob(os.path.join(SDF_DIR, '*.sdf')))
print(f'Found {len(sdf_files)} SDF files')

for sdf_path in tqdm(sdf_files, desc='Baseline targets'):
    basename = os.path.basename(sdf_path)
    pdb_id = parse_pdb(basename)

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

        if len(batch_mols) >= 100:
            try:
                pc = posecheck.PoseCheck()
                pc.load_ligands_from_mols(batch_mols)
                strains = pc.calculate_strain_energy()
                for k, s in zip(batch_keys, strains):
                    strain_lookup[k] = s
            except Exception as e:
                for k in batch_keys:
                    strain_lookup[k] = np.nan
            batch_mols, batch_keys = [], []

    if batch_mols:
        try:
            pc = posecheck.PoseCheck()
            pc.load_ligands_from_mols(batch_mols)
            strains = pc.calculate_strain_energy()
            for k, s in zip(batch_keys, strains):
                strain_lookup[k] = s
        except Exception as e:
            for k in batch_keys:
                strain_lookup[k] = np.nan

    # Incremental save after each target
    cache[MODE_NAME] = strain_lookup
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

# Merge into CSV
print(f'\nComputed strain for {len(strain_lookup)} unique (pdb, SMILES)')
df = pd.read_csv(CSV_PATH)
if 'strain' not in df.columns:
    df['strain'] = np.nan

matched = 0
mask = df['mode'] == MODE_NAME
for idx in df[mask].index:
    pdb_id = str(df.at[idx, 'pdb_id']).lower()
    smi = str(df.at[idx, 'SMILES'])
    key = (pdb_id, smi)
    if key in strain_lookup:
        df.at[idx, 'strain'] = strain_lookup[key]
        matched += 1

df.to_csv(CSV_PATH, index=False)
print(f'Matched {matched}/{mask.sum()} ({matched/mask.sum()*100:.1f}%)')
