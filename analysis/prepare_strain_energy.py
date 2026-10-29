"""
Compute PoseCheck strain energy for all paper methods in SMRT-Net benchmark.
Matches molecules by (mode, pdb_id, SMILES) to the combined CSV.
Skips old ablation methods (Allfeat Cst/Nocst, Generalfeat Cst/Nocst).
"""
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
import posecheck
import os, glob, pickle, re
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

# Data lives under data/, not next to this script.
OUT_DIR = ROOT / 'data' / 'benchmark' / 'output'
CSV_PATH = OUT_DIR / 'smrtbench_combined_results.csv'
CACHE_PATH = OUT_DIR / 'cache_strain_energy.pkl'

# Method → SDF directory mapping
# For each method, we need a function: pdb_id → list of (sdf_path, mode_name)
SDF_SOURCES = {
    'RNA2smol (Only General Feat)': {
        'dir': OUT_DIR / 'smrtnet_onlygeneral_0504',
        'pattern': 'subdir',  # {dir}/{target}_ligand/{target}_ligand.sdf
    },
    'RNA2smol (Baseline)': {
        'dir': (ROOT / 'data' / 'benchmark' / 'newbsl' / 'output' / 'baseline_merged'),
        'pattern': 'flat',
    },
    'MolCRAFT': {
        'dir': OUT_DIR / 'molcraft_minimized',
        'pattern': 'flat',  # {dir}/{pocket}.sdf
    },
    'Pocket2Mol': {
        'dir': OUT_DIR / 'pkt2mol',
        'pattern': 'flat_pkt',  # {dir}/{pocket}_ligands.sdf (contains many mols)
    },
    'ED2Mol': {
        'dir': OUT_DIR / 'ed2mol',
        'pattern': 'subdir',  # {dir}/{target}_ligand/{target}_ligand.sdf
    },
    'Pretrain Only': {
        'dir': OUT_DIR / 'ablation_pretrain_only',
        'pattern': 'subdir',
    },
    'Pretrain+Constrain': {
        'dir': OUT_DIR / 'ablation_pretrain_constrain_only',
        'pattern': 'subdir',
    },
}

BATCH_SIZE = 100  # molecules per PoseCheck batch (reduced for stability)


def parse_pdb_from_target(target_name):
    """Extract pdb_id from various target name formats."""
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand$', target_name.lower())
    if m:
        return m.group(1)
    m = re.match(r'^([a-z0-9]{4})_.+', target_name.lower())
    if m:
        return m.group(1)
    return target_name[:4].lower()


def discover_sdf_files(method_cfg, base_dir):
    """Discover SDF files for a method, grouped by pdb_id."""
    pattern = method_cfg['pattern']
    pdb_sdfs = defaultdict(list)

    if pattern == 'subdir':
        for target_dir in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, target_dir)
            if not os.path.isdir(full):
                continue
            # Try {target_dir}/{target_dir}.sdf first, fall back to any .sdf
            sdf = os.path.join(full, f'{target_dir}.sdf')
            if os.path.exists(sdf):
                pdb_id = parse_pdb_from_target(target_dir)
                pdb_sdfs[pdb_id].append(sdf)
            else:
                # Fallback for dirs with differently-named SDFs (e.g. output.sdf)
                any_sdf = sorted(glob.glob(os.path.join(full, '*.sdf')))
                if any_sdf:
                    pdb_id = parse_pdb_from_target(target_dir)
                    for s in any_sdf:
                        pdb_sdfs[pdb_id].append(s)

    elif pattern == 'flat':
        for sdf_file in sorted(glob.glob(os.path.join(base_dir, '*.sdf'))):
            basename = os.path.basename(sdf_file).replace('.sdf', '')
            pdb_id = parse_pdb_from_target(basename)
            pdb_sdfs[pdb_id].append(sdf_file)

    elif pattern == 'flat_pkt':
        for sdf_file in sorted(glob.glob(os.path.join(base_dir, '*_ligands.sdf'))):
            basename = os.path.basename(sdf_file).replace('_ligands.sdf', '')
            pdb_id = parse_pdb_from_target(basename)
            pdb_sdfs[pdb_id].append(sdf_file)

    return pdb_sdfs


def compute_strain_batch(mols):
    """Compute strain energy for a batch of RDKit mols."""
    valid_mols = []
    valid_idx = []
    for i, mol in enumerate(mols):
        if mol is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0:
            valid_mols.append(mol)
            valid_idx.append(i)
    if not valid_mols:
        return [np.nan] * len(mols)

    try:
        pc = posecheck.PoseCheck()
        pc.load_ligands_from_mols(valid_mols)
        strains = pc.calculate_strain_energy()
    except Exception as e:
        print(f"  PoseCheck batch error (n={len(valid_mols)}): {e}")
        return [np.nan] * len(mols)

    # Map back
    result = [np.nan] * len(mols)
    for idx, s in zip(valid_idx, strains):
        result[idx] = s
    return result


def process_method(mode_name, method_cfg):
    """Process all SDFs for one method. Returns dict: (pdb_id, SMILES) → strain."""
    base_dir = str(method_cfg['dir'])
    if not os.path.exists(base_dir):
        print(f"  SKIP: directory not found: {base_dir}")
        return {}

    pdb_sdfs = discover_sdf_files(method_cfg, base_dir)
    all_pdbs = sum(len(v) for v in pdb_sdfs.values())
    print(f"  {len(pdb_sdfs)} targets, {all_pdbs} SDF files")

    strain_lookup = {}

    for pdb_id, sdf_files in tqdm(sorted(pdb_sdfs.items()), desc=f"  {mode_name}", leave=False):
        # Collect all molecules from these SDFs, preserving SMILES
        batch_mols = []
        batch_keys = []
        for sdf_path in sdf_files:
            try:
                supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
            except Exception:
                continue
            for mol in supplier:
                if mol is None or mol.GetNumConformers() == 0:
                    continue
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    continue
                key = (pdb_id, smi)
                if key not in strain_lookup:
                    batch_mols.append(mol)
                    batch_keys.append(key)

                # Process in batches
                if len(batch_mols) >= BATCH_SIZE:
                    strains = compute_strain_batch(batch_mols)
                    for k, s in zip(batch_keys, strains):
                        strain_lookup[k] = s
                    batch_mols = []
                    batch_keys = []

        # Process remaining
        if batch_mols:
            strains = compute_strain_batch(batch_mols)
            for k, s in zip(batch_keys, strains):
                strain_lookup[k] = s

    return strain_lookup


def main():
    # Load or compute strain cache
    if CACHE_PATH.exists():
        print(f"Loading cached strain data from {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            all_strains = pickle.load(f)
    else:
        all_strains = {}

    for mode_name, cfg in SDF_SOURCES.items():
        if mode_name in all_strains:
            print(f"\n{mode_name} — already cached ({len(all_strains[mode_name])} entries), skipping")
            continue
        print(f"\n{mode_name}")
        strains = process_method(mode_name, cfg)
        all_strains[mode_name] = strains
        n = len(strains)
        valid = sum(1 for v in strains.values() if not np.isnan(v))
        print(f"  → {valid}/{n} unique (pdb, SMILES) with strain computed")
        # Incremental save after each method
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(all_strains, f)

    # Merge into combined CSV
    print("\nMerging strain into combined CSV...")
    df = pd.read_csv(CSV_PATH)
    df['strain'] = np.nan

    matched = 0
    for mode_name, strains in all_strains.items():
        mask = df['mode'] == mode_name
        if not mask.any():
            continue
        for idx in df[mask].index:
            pdb_id = str(df.at[idx, 'pdb_id']).lower()
            smi = str(df.at[idx, 'SMILES'])
            key = (pdb_id, smi)
            if key in strains:
                df.at[idx, 'strain'] = strains[key]
                matched += 1
        n_mode = mask.sum()
        n_strain = df.loc[mask, 'strain'].notna().sum()
        print(f"  {mode_name}: {n_strain}/{n_mode} matched ({n_strain/n_mode*100:.1f}%)")

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved updated CSV with strain column ({matched} total matched)")
    print(f"Overall strain coverage: {df['strain'].notna().sum()}/{len(df)} "
          f"({df['strain'].notna().sum()/len(df)*100:.1f}%)")


if __name__ == '__main__':
    main()
