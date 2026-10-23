#!/usr/bin/env python3
"""Extract every ligand-centred pocket from all known Clivia and Squash structures.

Each non-solvent, non-ion hetero group is a ligand instance, and each instance defines a
candidate pocket: standard nucleotides/ions whose geometric centre lies within CUTOFF of
the ligand centre (the BER102 convention used in the paper).

Writes one PDB per instance plus a JSON summary with the residue composition, so the
pockets can be compared before any generation is attempted.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PDB = HERE / 'pdb_all'
OUT = HERE / 'pockets'
CUTOFF = 14.0

# ions / crystallisation additives that are not the fluorophore
NOT_LIGAND = {'HOH', 'DOD', 'MG', 'K', 'NA', 'MN', 'NI', 'ZN', 'CA', 'CL', 'IRI',
              'GTP', 'GDP', 'PO4', 'SO4', 'PEG', 'EDO', 'GOL', 'ACT', 'MES', 'TRS'}
NUC = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'RA', 'RC', 'RG', 'RU'}
IONS_OK = {'MG', 'K', 'NA', 'MN', 'CA', 'ZN'}

# Clivia: 8HZD O2I, 8HZE/8HZK/8HZL/8HZM NI4, 8HZF NJL, 8HZJ O00
# Squash: 7KVT/7KVU 2ZY, 7KVV 747
ENTRIES = {
    'clivia': ['8HZD', '8HZE', '8HZF', '8HZJ', '8HZK', '8HZL', '8HZM'],
    'squash': ['7KVT', '7KVV', '7KVU'],
}


def parse(pdb):
    atoms = []
    for line in pdb.read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        atoms.append(dict(
            rec=line[:6], serial=line[6:11], name=line[12:16], alt=line[16],
            res=line[17:20].strip(), chain=line[21], num=line[22:26].strip(),
            xyz=xyz, el=line[76:78].strip() or line[12:16].strip()[:1], line=line))
    return atoms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cutoff', type=float, default=CUTOFF)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    summary = []
    for aptamer, ids in ENTRIES.items():
        for pid in ids:
            pdb = PDB / f'{pid}.pdb'
            if not pdb.is_file():
                print(f'missing {pdb}')
                continue
            atoms = parse(pdb)
            # group by (resname, chain, resnum) to get ligand instances and residues
            groups = {}
            for at in atoms:
                groups.setdefault((at['res'], at['chain'], at['num']), []).append(at)
            lig_keys = [k for k in groups
                        if k[0] not in NOT_LIGAND and k[0] not in NUC
                        and len(groups[k]) >= 5]           # a real ligand has several atoms
            if not lig_keys:
                print(f'{pid}: no ligand found')
                continue
            res_keys = [k for k in groups if k[0] in NUC or k[0] in IONS_OK]
            for k in lig_keys:
                latoms = [x for x in groups[k] if x['el'] != 'H']
                centre = np.array([x['xyz'] for x in latoms]).mean(axis=0)
                kept = []
                for rk in res_keys:
                    ratoms = groups[rk]
                    rc = np.array([x['xyz'] for x in ratoms]).mean(axis=0)
                    if np.linalg.norm(rc - centre) < a.cutoff:
                        kept.append(rk)
                comp = Counter(rk[0] for rk in kept)
                name = f'{pid}_{k[0]}{k[2]}_{k[1]}'
                outp = OUT / f'{name}.pdb'
                with outp.open('w') as f:
                    for rk in kept:
                        for x in groups[rk]:
                            f.write(x['line'].rstrip() + '\n')
                    f.write('END\n')
                summary.append(dict(
                    aptamer=aptamer, pdb=pid, ligand=k[0], ligand_chain=k[1],
                    ligand_num=k[2], n_ligand_atoms=len(latoms),
                    centre=[round(float(v), 2) for v in centre],
                    n_residues=len(kept),
                    n_nucleotides=sum(v for kk, v in comp.items() if kk in NUC),
                    n_ions=sum(v for kk, v in comp.items() if kk in IONS_OK),
                    composition=dict(sorted(comp.items())),
                    file=outp.name))
                print(f'{name}: {len(kept)} residues  {dict(sorted(comp.items()))}')

    (HERE / 'pockets_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\n{len(summary)} pockets -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
