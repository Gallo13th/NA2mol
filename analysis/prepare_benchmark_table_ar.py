"""
Evaluate legacy autoregressive molecules from zqf_res_pocket_new_epoch113_smina.
Mirrors cells 0+2 of prop_eval_plot.ipynb.
"""
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import glob
import os
from tqdm import tqdm
from rdkit.Chem import Descriptors, Crippen, QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit import RDLogger
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')
mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def calculate_descriptors(mol):
    descriptor_dict = {}
    try:
        descriptor_dict['mw'] = Descriptors.MolWt(mol)
        descriptor_dict['heavy_atoms'] = mol.GetNumHeavyAtoms()
        descriptor_dict['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        descriptor_dict["SMILES"] = Chem.MolToSmiles(mol)
        descriptor_dict['qed'] = QED.qed(mol)
        descriptor_dict['logp'] = Crippen.MolLogP(mol)
        descriptor_dict['tpsa'] = Descriptors.TPSA(mol)
        try:
            descriptor_dict['sa_score'] = sascorer.calculateScore(mol)
        except Exception:
            descriptor_dict['sa_score'] = 3.0
        try:
            descriptor_dict['affinity'] = float(mol.GetProp('minimizedAffinity'))
        except Exception:
            pass
        try:
            descriptor_dict['rmsd'] = float(mol.GetProp('minimizedRMSD'))
        except Exception:
            pass
    except Exception as e:
        print(f"Error calculating descriptors: {e}")
    return descriptor_dict


def assign_mw_bin(mw):
    if mw < 120:
        return '0–120'
    elif mw < 180:
        return '120-180'
    elif mw < 300:
        return '180–300'
    elif mw < 420:
        return '300-420'
    else:
        return '≥420'


DATA_DIR = str(ROOT / 'data/benchmark/zqf_res_pocket_new_epoch113_smina')
OUT_DIR = str(ROOT / 'data/benchmark/output')
MODE_NAME = 'Legacy autoregressive model'

pocket_dirs = sorted(os.listdir(DATA_DIR))
print(f"Found {len(pocket_dirs)} pocket directories.")

all_res = []

for pocket_dir in pocket_dirs:
    # gt_1yit_0_VIR_pocket.pdb -> pocket = 1yit_0_VIR
    pocket = pocket_dir.replace('gt_', '').replace('_pocket.pdb', '')
    pdb_id = pocket[:4]

    active_smi_file = fstr(ROOT / "data/benchmark/RNALigBench/activesmi/{pdb_id}_pos.smi")
    active_smis = [line.rstrip('\n') for line in open(active_smi_file).readlines()]

    # Compute ECFP4 fingerprints for actives
    active_fps = []
    invalid_count = 0
    for smi in tqdm(active_smis, desc=f"ECFP4 for {pdb_id} actives", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_count += 1
            continue
        try:
            fp = mgen.GetFingerprint(mol)
            active_fps.append(fp)
        except Exception:
            invalid_count += 1

    print(f"[{pocket}] Loaded {len(active_fps)} active fingerprints ({invalid_count} invalid)")

    if len(active_fps) == 0:
        print(f"  Skipping {pocket}: no valid active fingerprints")
        continue

    # Load all SDF files in this pocket directory
    sdf_dir = os.path.join(DATA_DIR, pocket_dir)
    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, '*.sdf')))
    print(f"  Loading {len(sdf_files)} SDF files...")

    results = []
    mol_idx = 0
    for sdf_file in tqdm(sdf_files, desc=f"Processing {pocket}", leave=False):
        try:
            supplier = Chem.SDMolSupplier(sdf_file, removeHs=True)
        except Exception as e:
            print(f"  Error loading {sdf_file}: {e}")
            continue
        for mol in supplier:
            if mol is None:
                continue
            try:
                result = calculate_descriptors(mol)
            except Exception as e:
                continue

            gen_fp = mgen.GetFingerprint(mol)
            sims = DataStructs.BulkTanimotoSimilarity(gen_fp, active_fps)
            max_sim = max(sims) if sims else 0.0

            result['mol_index'] = mol_idx
            mol_idx += 1
            result['max_ecfp4_tanimoto'] = max_sim
            result['active_smi'] = active_smis[sims.index(max_sim)]
            result['pdb_id'] = pdb_id
            result['pocket'] = pocket
            result['mode'] = MODE_NAME
            results.append(result)

    if results:
        result_df = pd.DataFrame(results)
        all_res.append(result_df)
        print(f"  Processed {len(results)} molecules for {pocket}")
    else:
        print(f"  No valid molecules for {pocket}")

# Combine all results
print(f"\nCombining results from {len(all_res)} pockets...")
combined = pd.concat(all_res, ignore_index=True)

# Filter out molecules with no affinity (docking failed)
combined = combined[combined['affinity'] < 0].copy()

# Compute derived metrics
combined['LBE'] = combined['affinity'] / combined['heavy_atoms']
combined['SEI'] = -combined['affinity'] / combined['tpsa'].replace(0, np.nan)
combined['SEI'] = combined['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
combined['LLE'] = -combined['affinity'] - combined['logp']
combined['mw_bin'] = combined['mw'].apply(assign_mw_bin)

print(f"Total molecules: {len(combined)}")
print(f"Unique targets: {combined['pdb_id'].nunique()}")
print(f"Unique pockets: {combined['pocket'].nunique()}")

# Save
out_path = os.path.join(OUT_DIR, 'rna2smol_ar_smrtbench_results.csv')
combined.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
