#!/usr/bin/env python3
"""Score generated and reference molecules on the RNA2smol stack, in parallel.

The stack is what generation itself uses: smina minimisation against the pocket

    ./util/smina.static -r <pocket.pdb> -l <mol.sdf> -o <out.sdf> --minimize --minimize_iters 1000

and the reported score is `minimizedAffinity`. The five reference compounds go through the
identical call, so generated vs reference affinities are directly comparable.

Reports per pocket: reference affinities, generated distribution, how many generated
molecules beat each reference, and the relation between ECFP4 similarity and affinity.
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


def _repair(mol):
    import importlib.util
    spec = importlib.util.spec_from_file_location('rb', str(HERE / 'repair_bond_orders.py'))
    m = importlib.util.module_from_spec(spec); sys.modules['rb'] = m
    spec.loader.exec_module(m)
    out, _ = m.repair_bond_orders(mol)
    return out


_POCKET_CENTRE = {}


def pocket_centre(pocket):
    """Centroid of the pocket atoms, cached. The embedded ligand must be moved here:
    a conformer left at the origin contacts nothing and scores the same in any pocket."""
    if pocket in _POCKET_CENTRE:
        return _POCKET_CENTRE[pocket]
    import numpy as np
    xyz = []
    with open(pocket) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    continue
    c = np.asarray(xyz).mean(axis=0)
    _POCKET_CENTRE[pocket] = c
    return c


def _dock(task):
    """task = (smiles, pocket_path) -> affinity (nan on failure)."""
    smi, pocket = task
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float('nan')
    # smina scores 0 for a molecule without 3D coordinates; embed and optimise first so
    # every molecule (reference and generated alike) enters the stack the same way
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return float('nan')
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    # move the conformer into the pocket so the receptor actually contributes
    import numpy as np
    conf = mol.GetConformer()
    pts = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    shift = pocket_centre(pocket) - pts.mean(axis=0)
    from rdkit.Geometry import Point3D
    for i in range(mol.GetNumAtoms()):
        pp = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pp.x + shift[0], pp.y + shift[1], pp.z + shift[2]))
    mol = Chem.RemoveHs(mol)
    d = tempfile.mkdtemp(prefix='smn_')
    mf = os.path.join(d, 'mol.sdf')
    out = os.path.join(d, 'out.sdf')
    w = Chem.SDWriter(mf)
    w.write(mol); w.close()
    try:
        res = subprocess.run([str(SMINA), '-r', pocket, '-l', mf, '-o', out,
                              '--minimize', '--minimize_iters', '1000', '--cpu', '1'],
                             capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return float('nan')
    # The output SDF's minimizedAffinity property is unreliable here (it comes back 0.00000
    # regardless of the pose), so take the post-minimisation value from stdout:
    #   "Affinity: <input> <minimized> (kcal/mol)"
    for line in (res.stdout or '').splitlines():
        if line.startswith('Affinity:'):
            parts = line.split()
            try:
                return float(parts[-2])
            except (ValueError, IndexError):
                return float('nan')
    return float('nan')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--out', default=str(HERE / 'reports' / 'rna2smol_scores.csv'))
    a = ap.parse_args()

    # unique, sanitizable, repaired generated molecules
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
            gen.setdefault(Chem.MolToSmiles(m), None)
    gen_smis = sorted(gen)
    if a.limit:
        gen_smis = gen_smis[:a.limit]
    print(f'generated (unique, repaired): {len(gen_smis)}   references: {len(REFERENCES)}')

    ref_fps = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v)) for k, v in REFERENCES.items()}
    rows = []
    with Pool(a.workers) as pool:
        for pname, ppath in POCKETS.items():
            if not os.path.exists(ppath):
                print(f'missing {ppath}'); continue
            work = [(smi, ppath) for smi in REFERENCES.values()] + \
                   [(smi, ppath) for smi in gen_smis]
            print(f'  docking {len(work)} molecules against {pname} ...', flush=True)
            affs = pool.map(_dock, work, chunksize=4)
            for (smi, _), aff in zip(work, affs):
                fp = GEN.GetFingerprint(Chem.MolFromSmiles(smi))
                sims = {k: DataStructs.TanimotoSimilarity(fp, v) for k, v in ref_fps.items()}
                b = max(sims, key=sims.get)
                kind = 'reference' if smi in REFERENCES.values() else 'generated'
                rows.append(dict(smiles=smi, kind=kind, pocket=pname, affinity=aff,
                                 ecfp4_to_ref=round(sims[b], 4), nearest_ref=b))

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nwrote {out} ({len(df)} rows)')

    for pname in df.pocket.unique():
        sub = df[df.pocket == pname]
        ref = sub[sub.kind == 'reference']
        g = sub[sub.kind == 'generated'].dropna(subset=['affinity'])
        print(f'\n================ {pname} ================')
        print('  reference affinities:')
        for _, r in ref.iterrows():
            print(f'    {r.nearest_ref}  {r.affinity:8.2f}')
        if len(g):
            print(f'  generated: n={len(g)}  best {g.affinity.min():.2f}  '
                  f'median {g.affinity.median():.2f}  worst {g.affinity.max():.2f}')
            for k, r in zip(ref.nearest_ref, ref.affinity):
                better = int((g.affinity < r).sum())
                print(f'    better than {k} ({r:.2f}): {better}/{len(g)} '
                      f'({100*better/len(g):.1f}%)')
            top = g.nsmallest(5, 'affinity')
            print('  best generated:')
            for _, t in top.iterrows():
                print(f'    {t.affinity:8.2f}  ECFP4 {t.ecfp4_to_ref:.3f}  {t.smiles[:64]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
