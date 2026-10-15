"""Compute PoseCheck strain for RNA2smol on DUDE benchmark."""
from pathlib import Path
import pandas as pd, numpy as np, pickle, os, re, glob
from rdkit import Chem, RDLogger
import posecheck
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

SDF_DIR = str(ROOT / 'notebooks/output/dude_onlygeneral_0506')
CSV_PATH = str(ROOT / 'notebooks/dude_stat_all.csv')
MODE_NAME = 'RNA2smol'
MAX_HEAVY = 80
BATCH_SIZE = 100

def parse_pdb(dname):
    return dname.replace('dude_h_', '')

# Discover SDFs
target_dirs = sorted([d for d in os.listdir(SDF_DIR) if os.path.isdir(os.path.join(SDF_DIR, d))])
print(f'{len(target_dirs)} target dirs')

strain_lookup = {}

for td in tqdm(target_dirs, desc='DUDE RNA2smol'):
    pdb_id = parse_pdb(td)
    sdf = os.path.join(SDF_DIR, td, f'{td}.sdf')
    if not os.path.exists(sdf): continue

    try:
        supp = Chem.SDMolSupplier(sdf, removeHs=True)
    except: continue

    batch_mols, batch_keys = [], []
    for mol in supp:
        if mol is None or mol.GetNumAtoms()==0 or mol.GetNumConformers()==0: continue
        if mol.GetNumHeavyAtoms() > MAX_HEAVY: continue
        try: smi = Chem.MolToSmiles(mol)
        except: continue
        key = (pdb_id, smi)
        if key in strain_lookup: continue
        batch_mols.append(mol); batch_keys.append(key)
        if len(batch_mols) >= BATCH_SIZE:
            try:
                pc = posecheck.PoseCheck(); pc.load_ligands_from_mols(batch_mols)
                for k,s in zip(batch_keys, pc.calculate_strain_energy()): strain_lookup[k]=s
            except Exception as e:
                for k in batch_keys: strain_lookup[k]=np.nan
            batch_mols=[]; batch_keys=[]
    if batch_mols:
        try:
            pc = posecheck.PoseCheck(); pc.load_ligands_from_mols(batch_mols)
            for k,s in zip(batch_keys, pc.calculate_strain_energy()): strain_lookup[k]=s
        except:
            for k in batch_keys: strain_lookup[k]=np.nan

valid = sum(1 for v in strain_lookup.values() if not np.isnan(v))
vals = [v for v in strain_lookup.values() if not np.isnan(v)]
print(f'\nStrain: {valid}/{len(strain_lookup)} valid')
if vals: print(f'Median: {np.median(vals):.1f}, Mean: {np.mean(vals):.1f}')

# Merge into CSV
df = pd.read_csv(CSV_PATH)
if 'strain' not in df.columns: df['strain'] = np.nan
m = 0; mask = df['mode'] == MODE_NAME
for idx in df[mask].index:
    k = (str(df.at[idx,'pdb_id']), str(df.at[idx,'SMILES']))
    if k in strain_lookup and not np.isnan(strain_lookup[k]):
        df.at[idx,'strain'] = strain_lookup[k]; m += 1
df.to_csv(CSV_PATH, index=False)
print(f'Matched {m}/{mask.sum()} ({m/mask.sum()*100:.1f}%)')
