"""
Evaluate two new ablation experiments:
  - ablation_pretrain_only       (pretraining only, no constrain, no contrastive)
  - ablation_pretrain_constrain_only (pretraining + constrain, no contrastive)

Mirrors prepare_benchmark_table_ar.py to ensure consistent metrics.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, QED, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer
import glob, os, re
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')
mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ACTIVE_SMI_DIR = str(ROOT / "data/benchmark/RNALigBench/activesmi")
OUTPUT_DIR = str(ROOT / "data/benchmark/output")

ABLATION_CONFIGS = {
    "Pretrain Only": str(ROOT / "data/benchmark/output/ablation_pretrain_only"),
    "Pretrain+Constrain": str(ROOT / "data/benchmark/output/ablation_pretrain_constrain_only"),
}

COMBINED_CSV = os.path.join(OUTPUT_DIR, "smrtbench_combined_results.csv")
MAX_MOLS_PER_TARGET = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def calculate_descriptors(mol):
    d = {}
    try:
        d['mw'] = Descriptors.MolWt(mol)
        d['heavy_atoms'] = mol.GetNumHeavyAtoms()
        d['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        d['SMILES'] = Chem.MolToSmiles(mol)
        d['qed'] = QED.qed(mol)
        d['logp'] = Crippen.MolLogP(mol)
        d['tpsa'] = Descriptors.TPSA(mol)
        try:
            d['sa_score'] = sascorer.calculateScore(mol)
        except Exception:
            d['sa_score'] = 3.0
        try:
            d['affinity'] = float(mol.GetProp('minimizedAffinity'))
        except Exception:
            d['affinity'] = np.nan
        try:
            d['rmsd'] = float(mol.GetProp('minimizedRMSD'))
        except Exception:
            d['rmsd'] = np.nan
    except Exception as e:
        pass
    return d


def load_active_fps(active_smi_file):
    """Load fingerprints from active SMILES file."""
    active_smis = []
    with open(active_smi_file) as f:
        for line in f:
            smi = line.strip()
            if smi:
                active_smis.append(smi)

    fps = []
    valid_smis = []
    for smi in active_smis:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                fps.append(mgen.GetFingerprint(mol))
                valid_smis.append(smi)
            except Exception:
                continue
    return valid_smis, fps


def get_active_fps_for_pdb(pdb_id):
    """Try active SMILES file with multiple naming patterns."""
    patterns = [
        os.path.join(ACTIVE_SMI_DIR, f"{pdb_id}_pos.smi"),
        os.path.join(ACTIVE_SMI_DIR, f"{pdb_id}.smi"),
    ]
    for pat in patterns:
        if os.path.exists(pat):
            return load_active_fps(pat)
    return [], []


def parse_pocket_from_dirname(dirname):
    """Extract pdb_id and pocket from directory name like '2ku0_A_ISI_ligand'."""
    m = re.match(r'^([a-z0-9]{4})_(.+)_ligand$', dirname.lower())
    if m:
        return m.group(1), m.group(2)
    return dirname[:4], dirname


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_ablation(ablation_name, base_dir):
    """Process one ablation experiment directory."""
    target_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    all_res = []
    print(f"\n{'='*60}")
    print(f"Processing: {ablation_name} ({len(target_dirs)} targets)")
    print(f"{'='*60}")

    for tdir in tqdm(target_dirs, desc=ablation_name):
        pdb_id, pocket_tag = parse_pocket_from_dirname(tdir)
        pocket_full = f"{pdb_id}_{pocket_tag}"

        # Find SDF file
        sdf_candidates = [
            os.path.join(base_dir, tdir, f"{tdir}.sdf"),
        ] + sorted(glob.glob(os.path.join(base_dir, tdir, "*.sdf")))

        sdf_file = None
        for c in sdf_candidates:
            if os.path.exists(c):
                sdf_file = c
                break

        if sdf_file is None:
            continue

        # Load active FPs
        active_smis, active_fps = get_active_fps_for_pdb(pdb_id)

        # Read molecules
        results = []
        mol_idx = 0
        try:
            supplier = Chem.SDMolSupplier(sdf_file, removeHs=True)
        except Exception:
            continue

        for mol in supplier:
            if mol is None or mol_idx >= MAX_MOLS_PER_TARGET:
                continue

            result = calculate_descriptors(mol)

            # ECFP4 Tanimoto
            if active_fps:
                try:
                    gen_fp = mgen.GetFingerprint(mol)
                    sims = DataStructs.BulkTanimotoSimilarity(gen_fp, active_fps)
                    max_sim = max(sims) if sims else 0.0
                    result['max_ecfp4_tanimoto'] = max_sim
                    result['active_smi'] = active_smis[sims.index(max_sim)]
                except Exception:
                    result['max_ecfp4_tanimoto'] = None
                    result['active_smi'] = None
            else:
                result['max_ecfp4_tanimoto'] = None
                result['active_smi'] = None

            result['mol_index'] = mol_idx
            mol_idx += 1
            result['pdb_id'] = pdb_id
            result['pocket'] = pocket_full
            result['mode'] = ablation_name
            results.append(result)

        if results:
            df = pd.DataFrame(results)
            all_res.append(df)

    if not all_res:
        print(f"  No data for {ablation_name}")
        return pd.DataFrame()

    combined = pd.concat(all_res, ignore_index=True)

    # Filter: keep only negative affinity (docked)
    combined = combined[combined['affinity'] < 0].copy()

    # Derived metrics
    combined['LBE'] = combined['affinity'] / combined['heavy_atoms'].replace(0, np.nan)
    combined['SEI'] = -combined['affinity'] / combined['tpsa'].replace(0, np.nan)
    combined['SEI'] = combined['SEI'].replace([float('inf'), float('-inf')], np.nan).fillna(0)
    combined['LLE'] = -combined['affinity'] - combined['logp']
    combined['mw_bin'] = combined['mw'].apply(assign_mw_bin)

    print(f"  Molecules: {len(combined)}, Targets: {combined['pdb_id'].nunique()}")
    print(f"  Mean affinity: {combined['affinity'].mean():.3f}")
    print(f"  Mean QED: {combined['qed'].mean():.3f}")
    print(f"  Mean SA: {combined['sa_score'].mean():.3f}")

    return combined


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    all_abl_dfs = []

    for name, path in ABLATION_CONFIGS.items():
        df = process_ablation(name, path)
        if not df.empty:
            # Save individual CSV
            safe_name = name.lower().replace(' ', '_').replace('+', '')
            csv_path = os.path.join(OUTPUT_DIR, f"rna2smol_abl_{safe_name}_smrtbench_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")
            all_abl_dfs.append(df)

    # Merge into combined results
    if all_abl_dfs and os.path.exists(COMBINED_CSV):
        print(f"\n{'='*60}")
        print("Merging into combined results...")
        existing = pd.read_csv(COMBINED_CSV)
        print(f"  Existing rows: {len(existing)}, modes: {existing['mode'].unique().tolist()}")

        # Remove old entries for these two ablation modes if they exist
        new_mode_names = list(ABLATION_CONFIGS.keys())
        existing = existing[~existing['mode'].isin(new_mode_names)]

        # Append new data
        new_abl = pd.concat(all_abl_dfs, ignore_index=True)
        merged = pd.concat([existing, new_abl], ignore_index=True)
        merged.to_csv(COMBINED_CSV, index=False)
        print(f"  Merged rows: {len(merged)}, modes: {merged['mode'].unique().tolist()}")
        print(f"  Saved: {COMBINED_CSV}")
    else:
        print("No existing combined CSV or no ablation data — skipping merge.")
