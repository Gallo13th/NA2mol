"""
Compute strain energy for RNA2smol (Baseline) using RDKit native MMFF.
Strain = energy difference between input conformer and MMFF-optimized conformer.
Samples up to 500 molecules per target, skips molecules >80 heavy atoms.
Avoids PoseCheck entirely for speed and reliability.
"""
import pandas as pd, numpy as np, pickle, os, re, glob
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

SDF_DIR = str(ROOT / 'notebooks/newbsl/output/baseline_merged')
CSV_PATH = str(ROOT / 'notebooks/output/smrtbench_combined_results.csv')
CACHE_PATH = str(ROOT / 'notebooks/output/cache_strain_energy.pkl')
MODE_NAME = 'RNA2smol (Baseline)'
MAX_PER_TARGET = 500
MAX_HEAVY = 80
MAX_MMFF_STEPS = 200

def parse_pdb(name):
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand\.sdf$', name.lower())
    return m.group(1) if m else name[:4].lower()

def compute_strain(mol):
    """Strain per heavy atom = |E_input - E_opt| / N_heavy via MMFF. Returns NaN on failure."""
    try:
        mol_h = Chem.AddHs(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        mp = AllChem.MMFFGetMoleculeProperties(mol_h)
        if mp is None:
            return np.nan
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp)
        if ff is None:
            return np.nan
        e_input = ff.CalcEnergy()
        ff.Minimize(maxIts=MAX_MMFF_STEPS)
        e_opt = ff.CalcEnergy()
        strain = abs(e_input - e_opt) / max(n_heavy, 1)
        return strain
    except Exception:
        return np.nan

# Load existing cache
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

if MODE_NAME in cache:
    del cache[MODE_NAME]

strain_lookup = {}
sdf_files = sorted(glob.glob(os.path.join(SDF_DIR, '*.sdf')))
print(f'{len(sdf_files)} SDF files')

for sdf_path in tqdm(sdf_files, desc='Baseline'):
    basename = os.path.basename(sdf_path)
    pdb_id = parse_pdb(basename)

    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    except:
        continue

    mol_count = 0
    for mol in supplier:
        if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
            continue
        if mol.GetNumHeavyAtoms() > MAX_HEAVY:
            continue
        if mol_count >= MAX_PER_TARGET:
            break

        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except:
            continue

        key = (pdb_id, smi)
        if key in strain_lookup:
            continue

        strain_lookup[key] = compute_strain(mol)
        mol_count += 1

    # Incremental save
    cache[MODE_NAME] = strain_lookup
    if mol_count % 10 == 0:  # save every ~10 targets
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)

# Final cache save
cache[MODE_NAME] = strain_lookup
with open(CACHE_PATH, 'wb') as f:
    pickle.dump(cache, f)

# Merge into CSV
valid = sum(1 for v in strain_lookup.values() if not np.isnan(v))
print(f'\nStrain computed: {valid}/{len(strain_lookup)} valid')
if valid > 0:
    vals = [v for v in strain_lookup.values() if not np.isnan(v)]
    print(f'Median: {np.median(vals):.1f}, Mean: {np.mean(vals):.1f}, Range: [{np.min(vals):.1f}, {np.max(vals):.1f}]')

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
