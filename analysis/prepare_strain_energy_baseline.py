"""Baseline strain via PoseCheck, batch=50, no pre-filter, sample=500/target."""
from pathlib import Path
import pandas as pd, numpy as np, pickle, os, re, glob
from rdkit import Chem, RDLogger
import posecheck
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

SDF_DIR = str(ROOT / 'data/benchmark/newbsl/output/baseline_merged')
CSV_PATH = str(ROOT / 'data/benchmark/output/smrtbench_combined_results.csv')
CACHE_PATH = str(ROOT / 'data/benchmark/output/cache_strain_energy.pkl')
MODE_NAME = 'RNA2smol (Baseline)'
MAX_PER_TARGET = 500
MAX_HEAVY = 80

def parse_pdb(n):
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand\.sdf$', n.lower())
    return m.group(1) if m else n[:4].lower()

with open(CACHE_PATH, 'rb') as f: cache = pickle.load(f)
if MODE_NAME in cache:
    lookup = cache[MODE_NAME]; done = {k[0] for k in lookup}
    print(f'Reusing {len(lookup)} cached, {len(done)} targets')
else:
    lookup = {}; done = set()

sdf_files = sorted(glob.glob(os.path.join(SDF_DIR, '*.sdf')))
for sdf_path in tqdm(sdf_files, desc='Baseline'):
    pdb_id = parse_pdb(os.path.basename(sdf_path))
    if pdb_id in done: continue
    try:
        supp = Chem.SDMolSupplier(sdf_path, removeHs=True)
    except: continue
    batch_mols, batch_keys = [], []
    count = 0
    for mol in supp:
        if mol is None or mol.GetNumAtoms()==0 or mol.GetNumConformers()==0: continue
        if mol.GetNumHeavyAtoms() > MAX_HEAVY: continue
        if count >= MAX_PER_TARGET: break
        try: smi = Chem.MolToSmiles(mol)
        except: continue
        key = (pdb_id, smi)
        if key in lookup: continue
        batch_mols.append(mol); batch_keys.append(key); count += 1
        if len(batch_mols) >= 50:
            try:
                pc = posecheck.PoseCheck(); pc.load_ligands_from_mols(batch_mols)
                for k,s in zip(batch_keys, pc.calculate_strain_energy()): lookup[k]=s
            except Exception as e:
                for k in batch_keys: lookup[k]=np.nan
            batch_mols=[]; batch_keys=[]
    if batch_mols:
        try:
            pc = posecheck.PoseCheck(); pc.load_ligands_from_mols(batch_mols)
            for k,s in zip(batch_keys, pc.calculate_strain_energy()): lookup[k]=s
        except: 
            for k in batch_keys: lookup[k]=np.nan
    cache[MODE_NAME]=lookup
    with open(CACHE_PATH,'wb') as f: pickle.dump(cache,f)

valid = sum(1 for v in lookup.values() if not np.isnan(v))
vals = [v for v in lookup.values() if not np.isnan(v)]
print(f'\nStrain: {valid}/{len(lookup)} valid')
if vals: print(f'Median: {np.median(vals):.1f}, Mean: {np.mean(vals):.1f}')

df=pd.read_csv(CSV_PATH)
if 'strain' not in df.columns: df['strain']=np.nan
m=0; mask=df['mode']==MODE_NAME
for idx in df[mask].index:
    k=(str(df.at[idx,'pdb_id']).lower(), str(df.at[idx,'SMILES']))
    if k in lookup and not np.isnan(lookup[k]): df.at[idx,'strain']=lookup[k]; m+=1
df.to_csv(CSV_PATH,index=False)
print(f'Matched {m}/{mask.sum()} ({m/mask.sum()*100:.1f}%)')
