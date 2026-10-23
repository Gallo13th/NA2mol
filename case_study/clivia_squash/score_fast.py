#!/usr/bin/env python3
"""Fast, correct scoring on the RNA2smol stack.

Generated molecules carry the sampler's own binding pose in the pocket frame (verified:
centroid within ~1 A of the pocket centroid), so they are scored with `--score_only` on that
pose. Re-embedding from SMILES would discard the model's conformer and score a different
molecule, which is what produced the earlier near-zero nonsense.

The references have no pose in these structures, so they are placed by global docking inside a
box around the pocket (`--autobox_ligand pocket --autobox_add 4 --exhaustiveness 8`) and the
best affinity is reported.

`--minimize` is deliberately not used: it needs an explicit scoring box, and without one it
returns unphysical near-zero or positive values.
"""
import glob
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path('/home/data/users/gaoletian/rna2smol')
HERE = ROOT / 'case_study/clivia_squash'
SMINA = ROOT / 'util' / 'smina.static'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs, RDLogger                              # noqa: E402
from rdkit.Chem import rdFingerprintGenerator                              # noqa: E402

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
POCKETS = {'clivia': str(HERE / 'clivia_8HZE_pocket.pdb'),
           'squash': str(HERE / 'squash_7KVU_pocket.pdb')}
REFERENCES = {
    'mol1': 'COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1',
    'mol2': 'COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1',
    'mol3': 'COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1',
    'mol4': 'COc1ccc(-c2cc(-c3ccncc3)nc3ccc(OC)cc23)cc1',
    'mol5': 'COc1ccc2c(c1)C(-c1ccc(O)cc1)C=C(-c1cccc(C(=O)O)c1)N2',
}
NAN = float('nan')


def _repair(mol):
    import importlib.util
    spec = importlib.util.spec_from_file_location('rb', str(HERE / 'repair_bond_orders.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['rb'] = m
    spec.loader.exec_module(m)
    out, _ = m.repair_bond_orders(mol)
    return out


def native(task):
    smi, block, pocket = task
    d = tempfile.mkdtemp(prefix='s_')
    f = os.path.join(d, 'm.sdf')
    with open(f, 'w') as fh:
        fh.write(block)
    try:
        r = subprocess.run([str(SMINA), '-r', pocket, '-l', f, '--score_only', '--cpu', '1'],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return (smi, NAN)
    for line in (r.stdout or '').splitlines():
        if line.startswith('Affinity:'):
            try:
                return (smi, float(line.split()[1]))
            except (ValueError, IndexError):
                return (smi, NAN)
    return (smi, NAN)


def docked(task):
    smi, pocket = task
    from rdkit.Chem import AllChem
    m0 = Chem.MolFromSmiles(smi)
    if m0 is None:
        return (smi, NAN)
    m = Chem.AddHs(m0)
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
        return (smi, NAN)
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    d = tempfile.mkdtemp(prefix='d_')
    f = os.path.join(d, 'm.sdf')
    w = Chem.SDWriter(f); w.SetKekulize(False)
    w.write(Chem.RemoveHs(m)); w.close()
    try:
        r = subprocess.run([str(SMINA), '-r', pocket, '-l', f,
                            '--autobox_ligand', pocket, '--autobox_add', '4',
                            '--exhaustiveness', '8', '--num_modes', '1', '--cpu', '2'],
                           capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return (smi, NAN)
    for line in (r.stdout or '').splitlines():
        p = line.split()
        if len(p) == 4 and p[0] == '1':
            try:
                return (smi, float(p[1]))
            except ValueError:
                break
    return (smi, NAN)


def main() -> int:
    gen = {}
    for f in sorted(HERE.glob('output_traj/st*/molecules.sdf')):
        if f.stat().st_size == 0:
            continue
        for m in Chem.SDMolSupplier(str(f), sanitize=False):
            if m is None:
                continue
            try:
                Chem.SanitizeMol(m)
            except Exception:
                continue
            m = _repair(m)
            if m is None:
                continue
            smi = Chem.MolToSmiles(m)
            if smi not in gen:
                gen[smi] = Chem.MolToMolBlock(m, kekulize=False)
    print(f'generated {len(gen)}', flush=True)

    import pandas as pd
    ref_fps = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v)) for k, v in REFERENCES.items()}
    rows = []
    for pname, ppath in POCKETS.items():
        if not os.path.exists(ppath):
            print(f'missing {ppath}'); continue
        with Pool(64) as pool:
            for smi, aff in pool.imap_unordered(native, [(s, b, ppath) for s, b in gen.items()],
                                                chunksize=8):
                fp = GEN.GetFingerprint(Chem.MolFromSmiles(smi))
                sims = {k: DataStructs.TanimotoSimilarity(fp, v) for k, v in ref_fps.items()}
                b = max(sims, key=sims.get)
                rows.append(dict(kind='generated', pocket=pname, smiles=smi, affinity=aff,
                                 ecfp4=round(sims[b], 4), nearest=b))
        with Pool(10) as pool:
            for smi, aff in pool.map(docked, [(v, ppath) for v in REFERENCES.values()]):
                k = [kk for kk, vv in REFERENCES.items() if vv == smi][0]
                rows.append(dict(kind='reference', pocket=pname, smiles=smi, affinity=aff,
                                 ecfp4=1.0, nearest=k))
        print(f'  {pname} done', flush=True)

    df = pd.DataFrame(rows)
    out = HERE / 'reports' / 'rna2smol_scores.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('wrote', out)
    for pname in df.pocket.unique():
        sub = df[df.pocket == pname]
        g = sub[sub.kind == 'generated'].dropna(subset=['affinity'])
        r = sub[sub.kind == 'reference'].dropna(subset=['affinity'])
        print(f'\n===== {pname} =====')
        for _, x in r.iterrows():
            print(f'  ref {x.nearest:5s} global dock {x.affinity:8.2f}')
        if len(g):
            print(f'  generated own pose: n={len(g)}  best {g.affinity.min():.2f}  '
                  f'median {g.affinity.median():.2f}')
            if len(r):
                best = r.affinity.min()
                n = int((g.affinity < best).sum())
                print(f'  beats best reference ({best:.2f}): {n}/{len(g)} '
                      f'({100*n/len(g):.1f}%)')
            print('  best generated (own pose):')
            for _, t in g.nsmallest(5, 'affinity').iterrows():
                print(f'    {t.affinity:8.2f}  ECFP4 {t.ecfp4:.3f}  {t.smiles[:58]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
