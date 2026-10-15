"""Baseline strain via PoseCheck with RDKit MMFF pre-validation."""
from pathlib import Path
import pandas as pd, numpy as np, pickle, os, re, glob
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import posecheck
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

SDF_DIR = str(ROOT / 'notebooks/newbsl/output/baseline_merged')
CSV_PATH = str(ROOT / 'notebooks/output/smrtbench_combined_results.csv')
CACHE_PATH = str(ROOT / 'notebooks/output/cache_strain_energy.pkl')
MODE_NAME = 'RNA2smol (Baseline)'
MAX_PER_TARGET = 300
MAX_HEAVY = 50

def parse_pdb(name):
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand\.sdf$', name.lower())
    return m.group(1) if m else name[:4].lower()

def prevalidate(mol):
    """MMFF pre-check. Returns True if molecule passes."""
    try:
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() == 0 or mol.GetNumHeavyAtoms() > MAX_HEAVY:
            return False
        mol_h = Chem.AddHs(mol)
        mp = AllChem.MMFFGetMoleculeProperties(mol_h)
        if mp is None: return False
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp)
        if ff is None: return False
        return ff.Minimize(maxIts=50) == 0
    except:
        return False

def compute_posecheck(mol):
    try:
        pc = posecheck.PoseCheck()
        pc.load_ligands_from_mols([mol])
        return pc.calculate_strain_energy()[0]
    except:
        return np.nan

# Load cache
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)
if MODE_NAME in cache:
    strain_lookup = cache[MODE_NAME]
    done_pdbs = {k[0] for k in strain_lookup}
    print(f'Reusing {len(strain_lookup)} cached entries from {len(done_pdbs)} targets')
else:
    strain_lookup = {}
    done_pdbs = set()

sdf_files = sorted(glob.glob(os.path.join(SDF_DIR, '*.sdf')))
for sdf_path in tqdm(sdf_files, desc='Baseline'):
    pdb_id = parse_pdb(os.path.basename(sdf_path))
    if pdb_id in done_pdbs:
        continue

    try:
        supp = Chem.SDMolSupplier(sdf_path, removeHs=True)
    except:
        continue

    count = 0
    for mol in supp:
        if mol is None: continue
        if count >= MAX_PER_TARGET: break
        if not prevalidate(mol): continue
        try:
            smi = Chem.MolToSmiles(mol)
        except:
            continue
        key = (pdb_id, smi)
        if key in strain_lookup: continue
        strain_lookup[key] = compute_posecheck(mol)
        count += 1

    # Save incrementally
    cache[MODE_NAME] = strain_lookup
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

# Merge
valid = sum(1 for v in strain_lookup.values() if not np.isnan(v))
vals = [v for v in strain_lookup.values() if not np.isnan(v)]
print(f'\nStrain: {valid}/{len(strain_lookup)} valid')
if vals:
    print(f'Median: {np.median(vals):.1f}, Mean: {np.mean(vals):.1f}')

df = pd.read_csv(CSV_PATH)
if 'strain' not in df.columns: df['strain'] = np.nan
matched = 0
mask = df['mode'] == MODE_NAME
for idx in df[mask].index:
    key = (str(df.at[idx, 'pdb_id']).lower(), str(df.at[idx, 'SMILES']))
    if key in strain_lookup and not np.isnan(strain_lookup[key]):
        df.at[idx, 'strain'] = strain_lookup[key]
        matched += 1
df.to_csv(CSV_PATH, index=False)
print(f'Matched {matched}/{mask.sum()} ({matched/mask.sum()*100:.1f}%)')
