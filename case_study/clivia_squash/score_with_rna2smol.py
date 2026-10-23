#!/usr/bin/env python3
"""Score generated and reference molecules on the RNA2smol stack, correctly.

Two different questions need two different smina modes, and mixing them up produced the
earlier nonsense of everything being ~0:

* **Generated molecules already have a binding pose.** The sampler emits coordinates in the
  pocket frame (verified: centroid within ~1 A of the pocket centroid), so their pose is
  scored with `--score_only`. Re-embedding from SMILES would throw the model's conformer
  away and score a different molecule.
* **The references have no experimental pose in these structures.** They are placed by
  global docking, `--autobox_ligand <pocket> --autobox_add 4 --exhaustiveness N`, and the
  best affinity is reported.

`--minimize` is not used: it minimises the input pose under a scoring box that must be
supplied, and with the box omitted it returns unphysical near-zero or positive values.

Output columns: kind, pocket, smiles, affinity_native (score of the given pose, generated
only), affinity_docked (best global pose), ecfp4_to_ref, nearest_ref.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

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
    m = importlib.util.module_from_spec(spec); sys.modules['rb'] = m
    spec.loader.exec_module(m)
    out, _ = m.repair_bond_orders(mol)
    return out


def _score_only(sdf, pocket):
    try:
        r = subprocess.run([str(SMINA), '-r', pocket, '-l', sdf, '--score_only', '--cpu', '1'],
                           capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return NAN
    for line in (r.stdout or '').splitlines():
        if line.startswith('Affinity:'):
            try:
                return float(line.split()[1])
            except (ValueError, IndexError):
                return NAN
    return NAN


def _dock(sdf, pocket, exhaustiveness):
    try:
        r = subprocess.run([str(SMINA), '-r', pocket, '-l', sdf,
                            '--autobox_ligand', pocket, '--autobox_add', '4',
                            '--exhaustiveness', str(exhaustiveness), '--num_modes', '1',
                            '--cpu', '1'],
                           capture_output=True, text=True, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        return NAN
    best = NAN
    for line in (r.stdout or '').splitlines():
        # "1       -12.5      0.000      0.000"
        parts = line.split()
        if len(parts) == 4 and parts[0] == '1':
            try:
                best = float(parts[1])
            except ValueError:
                pass
            break
    return best


def _worker(task):
    kind, smi_or_sdf, sdf_text, pocket, exhaustiveness = task
    d = tempfile.mkdtemp(prefix='smn_')
    sdf = os.path.join(d, 'm.sdf')
    if sdf_text is not None:
        with open(sdf, 'w') as f:
            f.write(sdf_text)
    else:
        mol = Chem.MolFromSmiles(smi_or_sdf)
        if mol is None:
            return (kind, smi_or_sdf, NAN, NAN)
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return (kind, smi_or_sdf, NAN, NAN)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        w = Chem.SDWriter(sdf); w.SetKekulize(False)
        w.write(Chem.RemoveHs(mol)); w.close()
    native = _score_only(sdf, pocket) if kind == 'generated' else NAN
    docked = _dock(sdf, pocket, exhaustiveness)
    return (kind, smi_or_sdf, native, docked)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--exhaustiveness', type=int, default=8)
    ap.add_argument('--out', default=str(HERE / 'reports' / 'rna2smol_scores.csv'))
    a = ap.parse_args()

    # generated: keep the model's own 3D pose, repair bond orders, keep as SDF text
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
            if smi in gen:
                continue
            gen[smi] = Chem.MolToMolBlock(m, kekulize=False)
    items = list(gen.items())
    if a.limit:
        items = items[:a.limit]
    print(f'generated (unique, repaired, own pose): {len(items)}   references: {len(REFERENCES)}')

    ref_fps = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v)) for k, v in REFERENCES.items()}
    rows = []
    with Pool(a.workers) as pool:
        for pname, ppath in POCKETS.items():
            if not os.path.exists(ppath):
                print(f'missing {ppath}'); continue
            tasks = [('reference', smi, None, ppath, a.exhaustiveness) for smi in REFERENCES.values()]
            tasks += [('generated', smi, block, ppath, a.exhaustiveness) for smi, block in items]
            print(f'  {pname}: scoring {len(tasks)} ...', flush=True)
            for kind, key, native, docked in pool.imap_unordered(_worker, tasks, chunksize=2):
                smi = key if kind == 'generated' else key
                fp = GEN.GetFingerprint(Chem.MolFromSmiles(smi))
                sims = {k: DataStructs.TanimotoSimilarity(fp, v) for k, v in ref_fps.items()}
                b = max(sims, key=sims.get)
                rows.append(dict(kind=kind, pocket=pname, smiles=smi,
                                 affinity_native=native, affinity_docked=docked,
                                 ecfp4_to_ref=round(sims[b], 4), nearest_ref=b))

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nwrote {out} ({len(df)} rows)')

    for pname in df.pocket.unique():
        sub = df[df.pocket == pname]
        ref = sub[sub.kind == 'reference']
        g = sub[sub.kind == 'generated']
        print(f'\n================ {pname} ================')
        print('  references (global dock, best pose):')
        for _, r in ref.iterrows():
            print(f'    {r.nearest_ref:5s} docked {r.affinity_docked:8.2f}')
        gn = g.dropna(subset=['affinity_native'])
        gd = g.dropna(subset=['affinity_docked'])
        if len(gn):
            print(f'  generated, OWN model pose (score_only): n={len(gn)}  '
                  f'best {gn.affinity_native.min():.2f}  median {gn.affinity_native.median():.2f}')
        if len(gd):
            print(f'  generated, global docked:               n={len(gd)}  '
                  f'best {gd.affinity_docked.min():.2f}  median {gd.affinity_docked.median():.2f}')
        best_ref = ref.affinity_docked.min()
        if len(gn):
            better = int((gn.affinity_native < best_ref).sum())
            print(f'  native pose beats the best reference ({best_ref:.2f}): '
                  f'{better}/{len(gn)} ({100*better/len(gn):.1f}%)')
        top = gn.nsmallest(5, 'affinity_native') if len(gn) else gn
        if len(top):
            print('  best generated (native pose):')
            for _, t in top.iterrows():
                print(f'    {t.affinity_native:8.2f}  ECFP4 {t.ecfp4_to_ref:.3f}  {t.smiles[:60]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
