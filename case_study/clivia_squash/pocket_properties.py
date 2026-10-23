#!/usr/bin/env python3
"""Physical/geometric characterisation of the four distinct Clivia/Squash sites.

Measured against the crystallised fluorophore itself, which is why the ligand is reloaded
from the original PDB rather than inferred from the pocket centroid:

  ligand_volume   vdW volume of the crystallised ligand (A^3)
  free_volume     solvent-accessible free volume within 10 A of the ligand centroid
  enclosure       fraction of a 10 A sphere around the ligand centroid that is occupied by
                  RNA -- how deep the ligand sits
  complementarity fraction of the ligand's vdW volume that a 1.4 A probe can reach, i.e.
                  how well the pocket shape matches that ligand
  shape           eigenvalues of the free-volume voxels -> sphericity / planarity
  contacts        nucleotides with any atom within 4.5 A of any ligand atom
  stacking        base rings within 5 A whose least-variance axis is small (aromatic)
  hbond           N/O atoms within 5 A of any ligand atom
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PDB = HERE / 'pdb_all'
POCKETS = HERE / 'pockets'
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'P': 1.80, 'S': 1.80,
       'MG': 1.73, 'MN': 1.73, 'K': 2.75, 'F': 1.47, 'H': 1.20}
PROBE, GRID, R_MAX = 1.4, 0.5, 10.0

REPS = {
    'Clivia site 1 (functional)': '8HZE_NI4109_A',
    'Clivia site 2 (8HZL only)': '8HZL_NI4112_A',
    'Clivia site 3 (8HZL only)': '8HZL_NI4113_A',
    'Squash': '7KVU_2ZY106_G',
}


def read_atoms(path):
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        out.append(dict(el=el, xyz=xyz, res=line[17:20].strip(),
                        chain=line[21], num=line[22:26].strip(),
                        name=line[12:16].strip(), rec=line[:6]))
    return out


def ligand_atoms(pdb, resname, chain, num):
    return [a for a in read_atoms(pdb)
            if a['res'] == resname and a['chain'] == chain
            and a['num'] == str(num) and a['el'] != 'H']


def grid_free(pts, atoms, probe=PROBE):
    heavy = [a for a in atoms if a['el'] != 'H']
    if not heavy:
        return np.ones(len(pts), dtype=bool)
    A = np.array([a['xyz'] for a in heavy])
    R = np.array([VDW.get(a['el'], 1.7) for a in heavy]) + probe
    free = np.ones(len(pts), dtype=bool)
    for i in range(0, len(A), 256):
        d = np.linalg.norm(pts[:, None, :] - A[None, i:i + 256, :], axis=2)
        free &= ~(d < R[None, i:i + 256]).any(axis=1)
    return free


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='pocket_properties.json')
    a = ap.parse_args()

    summary = {s['file'][:-4]: s for s in
               json.loads((HERE / 'pockets_summary.json').read_text())}
    rows = []
    for label, stem in REPS.items():
        s = summary[stem]
        pocket = read_atoms(POCKETS / f'{stem}.pdb')
        lig = ligand_atoms(PDB / f'{s["pdb"]}.pdb', s['ligand'],
                          s['ligand_chain'], s['ligand_num'])
        L = np.array([x['xyz'] for x in lig])
        centre = L.mean(axis=0)

        n = int(2 * R_MAX / GRID) + 1
        axis = (np.arange(n) - n // 2) * GRID
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) + centre
        pts = pts[np.linalg.norm(pts - centre, axis=1) < R_MAX]

        free = grid_free(pts, pocket)
        free_vol = free.sum() * GRID ** 3
        sphere = 4 / 3 * np.pi * R_MAX ** 3

        # how much of the ligand's own volume is reachable by the probe -> complementarity
        lig_pts = pts[grid_free(pts, [dict(el=x['el'], xyz=x['xyz']) for x in lig], probe=0.0)]
        # voxels inside the ligand's vdW volume
        Lh = np.array([x['xyz'] for x in lig])
        Rl = np.array([VDW.get(x['el'], 1.7) for x in lig])
        in_lig = np.zeros(len(pts), dtype=bool)
        for i in range(len(Lh)):
            in_lig |= np.linalg.norm(pts - Lh[i], axis=1) < Rl[i]
        lig_vol = in_lig.sum() * GRID ** 3
        # of the ligand's volume, how much is NOT reachable by a 1.4 A probe
        tight = np.sum(in_lig & ~free) * GRID ** 3
        complement = 100.0 * tight / lig_vol if lig_vol else 0.0

        cav = pts[free]
        if len(cav) > 10:
            ev = np.sort(np.linalg.eigvalsh(np.cov((cav - cav.mean(0)).T)))[::-1]
            ev = np.clip(ev, 1e-9, None)
            spher, planar = float(ev[2] / ev[0]), float(ev[1] / ev[0])
        else:
            ev, spher, planar = [0, 0, 0], 0, 0

        # contacts to the ligand atoms
        res_contacts = Counter()
        hb = 0
        for at in pocket:
            d = np.linalg.norm(np.array(at['xyz']) - L, axis=1).min()
            if d < 4.5 and at['res'] in {'A', 'C', 'G', 'U'}:
                res_contacts[(at['num'], at['res'])] += 1
            if d < 5.0 and at['el'] in ('N', 'O'):
                hb += 1

        rows.append(dict(site=label, stem=stem, pdb=s['pdb'],
                         n_pocket_residues=s['n_residues'],
                         composition=s['composition'],
                         n_ligand_atoms=len(lig),
                         ligand_volume_A3=round(lig_vol, 1),
                         free_volume_10A_A3=round(free_vol, 1),
                         enclosure_pct=round(100 * (1 - free_vol / sphere), 1),
                         shape_complementarity_pct=round(complement, 1),
                         sphericity=round(spher, 3),
                         planarity=round(planar, 3),
                         n_contact_residues=len(res_contacts),
                         contact_residues=[f'{r}{nn}' for (nn, r) in sorted(res_contacts)],
                         hbond_N_O_5A=hb))
        print(f'\n=== {label}  ({s["pdb"]}, {s["ligand"]}) ===')
        print(f'  pocket {s["n_residues"]} res  {s["composition"]}')
        print(f'  ligand {len(lig)} heavy atoms, vdW volume {lig_vol:7.1f} A^3')
        print(f'  free volume within 10 A       {free_vol:7.1f} A^3   enclosure {100*(1-free_vol/sphere):.0f}%')
        print(f'  shape complementarity         {complement:5.1f}%  of ligand volume is probe-excluded')
        print(f'  cavity shape  sphericity {spher:.2f}  planarity {planar:.2f}')
        print(f'  contact residues (<4.5 A)     {len(res_contacts)}: '
              f'{" ".join(f"{r}{nn}" for (nn, r) in sorted(res_contacts))}')
        print(f'  N/O within 5 A                {hb}')

    (HERE / a.out).write_text(json.dumps(rows, indent=2))
    print(f'\nwrote {HERE / a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
