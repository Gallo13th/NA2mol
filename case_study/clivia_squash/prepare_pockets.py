#!/usr/bin/env python3
"""Prepare Clivia and Squash aptamer pockets for cold-start generation.

Mirrors the BER102 case study convention exactly:
  - pocket = standard nucleotides/ions within `cutoff` A of the bound fluorophore,
    single state (MODEL 1)
  - dummy ligand = a single carbon at the fluorophore's geometric centre, which is
    what genbypkt uses to place the pocket (cold start, seed_used=False)

Structures: Clivia 8HZE (1.59 A, fluorophore NI4) and Squash 7KVU (2.68 A, 2ZY).
Both are the highest-resolution entries of their aptamer in the PDB.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # rna2smol/
WORK = Path(__file__).resolve().parent
PDB = WORK / 'pdb'

TARGETS = {
    'clivia': {'pdb': '8HZE', 'lig': 'NI4', 'chain': 'A', 'num': 109, 'cutoff': 14.0},
    'squash': {'pdb': '7KVU', 'lig': '2ZY', 'chain': 'G', 'num': 106, 'cutoff': 14.0},
}

# extract_pocket.py expects a 1-indexed MODEL number and skips every line when the
# file has no MODEL records (single-state X-ray structures), so pass model 0 for those.
def model_index(pdb: Path) -> int:
    return 1 if any(l.startswith('MODEL') for l in pdb.read_text().splitlines()) else 0


def ligand_centre(pdb: Path, resname: str, chain: str, num: int):
    """Geometric centre of the named ligand's atoms."""
    xyz = []
    for line in pdb.read_text().splitlines():
        if not line.startswith(('HETATM', 'ATOM')):
            continue
        if line[17:20].strip() != resname:
            continue
        if line[21].strip() != chain:
            continue
        if int(line[22:26]) != num:
            continue
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not xyz:
        raise SystemExit(f'ligand {resname} {chain}{num} not found in {pdb}')
    return np.array(xyz)


def write_dummy(out: Path, centre: np.ndarray) -> None:
    """Single-carbon SDF at the ligand centre, written by RDKit.

    Hand-written SDF text is not reliably parseable by RDKit's SDMolSupplier, and
    genbypkt reads this file via parse_sdf_file, so build it with RDKit itself.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles('[C]')
    m = Chem.AddHs(m)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(centre[0]), float(centre[1]), float(centre[2])))
    m.AddConformer(conf, assignId=True)
    w = Chem.SDWriter(str(out))
    w.write(m)
    w.close()


def write_dummy_pdb(out: Path, centre) -> None:
    """genbypkt.read_sdf_ligand_coords reads <stem>.pdb next to the dummy SDF, so the
    same single-carbon placeholder has to exist in PDB format too."""
    x, y, z = centre
    out.write_text(
        f"HETATM    1  C   LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
        "END\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', choices=list(TARGETS) + ['all'], default='all')
    a = ap.parse_args()
    names = list(TARGETS) if a.target == 'all' else [a.target]

    extractor = ROOT / 'case_study/kras_g4/rdock_protocol/extract_pocket.py'
    if not extractor.is_file():
        raise SystemExit(f'missing {extractor}')

    for name in names:
        cfg = TARGETS[name]
        pdb = PDB / f"{cfg['pdb']}.pdb"
        if not pdb.is_file():
            raise SystemExit(f'missing {pdb}')
        stem = f"{name}_{cfg['pdb']}"
        pocket = WORK / f'{stem}_pocket.pdb'
        dummy = WORK / f'{stem}_dummy.sdf'

        subprocess.run([sys.executable, str(extractor),
                        '--pdb', str(pdb),
                        '--ligand-name', cfg['lig'],
                        '--ligand-num', str(cfg['num']),
                        '--chain', cfg['chain'],
                        '--cutoff', str(cfg['cutoff']),
                        '--model', str(model_index(pdb)),
                        '--out', str(pocket)], check=True)

        centre = ligand_centre(pdb, cfg['lig'], cfg['chain'], cfg['num']).mean(axis=0)
        write_dummy(dummy, centre)
        write_dummy_pdb(dummy.with_suffix('.pdb'), centre)

        n_res = sum(1 for l in pocket.read_text().splitlines() if l.startswith('ATOM'))
        n_nt = len({(l[21], l[22:26]) for l in pocket.read_text().splitlines() if l.startswith('ATOM')})
        print(f'{name}: {cfg["pdb"]} ligand {cfg["lig"]} -> '
              f'{pocket.name} ({n_nt} residues, {n_res} atoms); dummy at '
              f'({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
