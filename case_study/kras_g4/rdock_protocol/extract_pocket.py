#!/usr/bin/env python3
"""Extract a single-state pocket PDB from a (possibly multi-MODEL) PDB.

Self-contained (no project imports). Follows the BER101-era convention:
  - MODEL 1 only (NMR ensembles -> first model), MODEL/ENDMDL stripped
  - ligand selected by (resname, resnum, chain); its geometric center computed
  - residues kept if (residue center - ligand center) < cutoff  [genbypkt logic]
  - whitelist: standard amino acids / nucleotides / ions; water & other
    het ligands excluded; all atoms (incl. NMR H) of kept residues are written

Usage:
  python extract_pocket.py --pdb 7X8M.pdb --ligand-name BER --ligand-num 102 \
      --chain X --cutoff 14.0 --out ber102_pocket.pdb
"""
import argparse
import numpy as np

AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'HID', 'HIE', 'HIP', 'CYX',
}
NUC_RESNAMES = {
    'DA', 'DC', 'DG', 'DT', 'RA', 'RC', 'RG', 'RU', 'A', 'C', 'G', 'U',
    'THM', 'DG5', 'DG3', 'DA5', 'DA3', 'DT5', 'DT3', 'DC5', 'DC3',
}
ION_RESNAMES = {'K', 'NA', 'CL', 'MG', 'CA', 'ZN', 'MN', 'BR', 'IOD'}
WATER_RESNAMES = {'HOH', 'WAT', 'DOD', 'TIP3', 'SOL'}
ALLOWED = AA_RESNAMES | NUC_RESNAMES | ION_RESNAMES


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', required=True)
    ap.add_argument('--ligand-name', required=True)
    ap.add_argument('--ligand-num', type=int, required=True)
    ap.add_argument('--chain', default=None,
                    help='ligand chain id (default: any)')
    ap.add_argument('--cutoff', type=float, default=14.0)
    ap.add_argument('--model', type=int, default=1, help='1-indexed MODEL')
    ap.add_argument('--extra-resnames', default='',
                    help='comma-separated extra allowed residue names')
    ap.add_argument('--out', required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    allowed = ALLOWED | set(s.strip().upper() for s in args.extra_resnames.split(',') if s.strip())

    model_idx = 0
    lines = []          # ATOM/HETATM lines of selected model
    lig_lines = []
    with open(args.pdb) as f:
        for raw in f:
            rec = raw.strip()
            if rec.startswith('MODEL'):
                model_idx += 1
                continue
            if rec.startswith('ENDMDL'):
                if model_idx == args.model:
                    break            # selected model finished
                continue
            if model_idx != args.model:
                continue
            if rec.startswith(('ATOM', 'HETATM')):
                lines.append(raw.rstrip('\n'))
                resname = rec[17:20].strip().upper()
                try:
                    resnum = int(rec[22:26])
                except ValueError:
                    continue
                ch = rec[21]
                if (resname == args.ligand_name.upper() and resnum == args.ligand_num
                        and (args.chain is None or ch == args.chain)):
                    lig_lines.append(raw.rstrip('\n'))

    if not lig_lines:
        raise SystemExit(f'[FATAL] ligand {args.ligand_name}{args.ligand_num}'
                         f'{"" if args.chain is None else " chain " + args.chain} not found in MODEL {args.model}')
    lig_heavy = [l for l in lig_lines if l[76:78].strip().upper() != 'H']
    if not lig_heavy:   # very old files without element column
        lig_heavy = [l for l in lig_lines if not l[12:16].strip().startswith('H')]
    lig_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lig_heavy])
    center = lig_xyz.mean(axis=0)
    print(f'[ligand] {args.ligand_name}{args.ligand_num} chain '
          f'{lig_lines[0][21]}: {len(lig_lines)} atoms ({len(lig_heavy)} heavy), '
          f'center = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})')

    # residue -> (center, lines)
    order, res = [], {}
    for l in lines:
        key = (l[21], l[22:26], l[26], l[17:20].strip().upper())
        if key not in res:
            res[key] = {'atoms': [], 'el': []}
            order.append(key)
        xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
        res[key]['atoms'].append(xyz)
        res[key]['el'].append(l[76:78].strip().upper())

    kept, summary = [], {}
    for key in order:
        resname = key[3]
        if resname in WATER_RESNAMES or resname not in allowed:
            continue
        rc = np.array(res[key]['atoms']).mean(axis=0)
        if np.linalg.norm(rc - center) < args.cutoff:
            kept.append(key)
            summary[resname] = summary.get(resname, 0) + 1

    kept_set = set(kept)
    out_lines, serial = [], 1
    for l in lines:
        key = (l[21], l[22:26], l[26], l[17:20].strip().upper())
        if key in kept_set:
            out_lines.append(f'{l[:6]}{serial:5d}{l[11:]}')
            serial += 1
    out_lines.append('TER')
    out_lines.append('END')
    with open(args.out, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')

    n_atoms = sum(len(res[k]['atoms']) for k in kept)
    n_heavy = sum(1 for k in kept for e in res[k]['el'] if e != 'H')
    print(f'[pocket ] {len(kept)} residues within {args.cutoff} A '
          f'(center-center): {summary}')
    print(f'[pocket ] {n_atoms} atoms ({n_heavy} heavy) -> {args.out}')


if __name__ == '__main__':
    main()
