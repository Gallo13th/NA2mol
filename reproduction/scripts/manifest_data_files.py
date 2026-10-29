#!/usr/bin/env python3
"""Record the bulk data files that must ship with the release.

These are excluded from git by design. The manifest gives path, size and (for
load-bearing files) SHA256 so a reviewer can verify a deposited copy.
"""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

# Files the reproduction entry points read. Changing these changes paper numbers.
LOAD_BEARING = [
    'data/benchmark/dude_stat_all.csv',
    'data/benchmark/smrtnet_stat_all.csv',
    'data/benchmark/DUDE_active_compounds.csv',
    'data/benchmark/output/smrtbench_combined_results.csv',
    'training_ligands_leakage_table.csv',
    'pretrain_smiles.txt.gz',
    'paperdata/smrtbench.tar.gz',
    'paperdata/smrtnet_ablation_rna2smol.tar.gz',
    'paperdata/dudue_onlygeneral_0506.tar.gz',
    'case_study/kras_g4/ber102_gen10k/valid_mols.sdf',
]

def sha256(path, limit_mb=400):
    if path.stat().st_size > limit_mb * 1024 * 1024:
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    rows = []
    for rel in LOAD_BEARING:
        p = ROOT / rel
        rows.append({'path': rel, 'present': p.is_file(),
                     'bytes': p.stat().st_size if p.is_file() else None,
                     'sha256': sha256(p) if p.is_file() else None})
        print(('PRESENT' if p.is_file() else 'MISSING'), rel)
    out = ROOT / 'reproduction/manifests/data_files.json'
    out.write_text(json.dumps({
        'note': 'Bulk inputs excluded from git; deposit these with the release.',
        'excluded_by_design': ['pretraining LMDB (author to upload separately)'],
        'files': rows}, indent=2) + '\n')
    print('\nwrote', out.relative_to(ROOT))

if __name__ == '__main__':
    main()
