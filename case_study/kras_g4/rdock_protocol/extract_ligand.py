#!/usr/bin/env python3
"""Extract a co-crystal ligand as a correct SD file (ground-truth reference).

Method (fixes the BER101_ref.sdf orphan-atom lesson):
  1. pull the ligand residue ATOM/HETATM lines from the requested MODEL
  2. build an RDKit mol from the PDB block (CONECT-aware + proximity bonding),
     heavy atoms only
  3. build the template mol from a trusted SMILES
  4. full unique substructure match  SMILES-atom <-> PDB-atom  (validates the
     bond graph; no orphan atoms possible), copy PDB coordinates per atom
  5. sanitize, check formula/charge/coord-range, write SDF

Usage:
  python extract_ligand.py --pdb 7X8M.pdb --ligand-name BER --ligand-num 102 \
      --chain X --smiles 'COc1ccc2cc3[n+](cc2c1)CCc1cc2c(cc1-3)OCO2' \
      --out ber102_ref.sdf --name BER102
"""
import argparse
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry import Point3D

RDLogger.DisableLog('rdApp.*')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', required=True)
    ap.add_argument('--ligand-name', required=True)
    ap.add_argument('--ligand-num', type=int, required=True)
    ap.add_argument('--chain', default=None)
    ap.add_argument('--model', type=int, default=1)
    ap.add_argument('--smiles', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='ref')
    return ap.parse_args()


def ligand_block(pdb, resname, resnum, chain, model):
    model_idx, keep, conect = 0, [], []
    serials = set()
    with open(pdb) as f:
        for raw in f:
            rec = raw.rstrip('\n')
            if rec.startswith('MODEL'):
                model_idx += 1
                continue
            if rec.startswith('ENDMDL'):
                if model_idx == model:
                    break
                continue
            if model_idx != model:
                continue
            if rec.startswith(('ATOM', 'HETATM')):
                rn = rec[17:20].strip().upper()
                try:
                    num = int(rec[22:26])
                except ValueError:
                    continue
                ch = rec[21]
                if rn == resname.upper() and num == resnum and (chain is None or ch == chain):
                    keep.append(rec)
                    serials.add(int(rec[6:11]))
    if not keep:
        raise SystemExit(f'[FATAL] {resname}{resnum} not found in MODEL {model}')
    with open(pdb) as f:
        for raw in f:
            rec = raw.rstrip('\n')
            if rec.startswith('CONECT') and serials & {int(x) for x in rec.split()[1:] if x.isdigit()}:
                conect.append(rec)
    return keep, conect


def main():
    args = parse_args()
    resname = args.ligand_name.upper()

    atom_lines, conect = ligand_block(args.pdb, resname, args.ligand_num,
                                      args.chain, args.model)
    heavy = [l for l in atom_lines if l[76:78].strip().upper() != 'H']
    print(f'[ligand ] {len(atom_lines)} atoms ({len(heavy)} heavy) in block; '
          f'{len(conect)} related CONECT lines')

    # --- RDKit mols ---
    block = '\n'.join(heavy) + '\nEND\n'
    pdb_mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False,
                                   proximityBonding=True)
    if pdb_mol is None:
        raise SystemExit('[FATAL] RDKit could not parse ligand PDB block')
    try:
        Chem.SanitizeMol(pdb_mol)
    except Exception as e:
        print(f'[warn  ] PDB mol strict sanitize failed ({e}); continuing with match')

    tmpl = Chem.MolFromSmiles(args.smiles)
    if tmpl is None:
        raise SystemExit('[FATAL] template SMILES not parseable')
    n_t, n_p = tmpl.GetNumAtoms(), pdb_mol.GetNumAtoms()
    if n_t != n_p:
        raise SystemExit(f'[FATAL] atom count mismatch: SMILES {n_t} vs PDB heavy {n_p}')

    # graph-isomorphism match: reduce both to element-labelled single-bond
    # graphs (PDB-side bonds are proximity singles; aromatic template would
    # never match otherwise). Degrees are compared separately below.
    def to_graph(mol):
        rw = Chem.RWMol(mol)
        for b in rw.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in rw.GetAtoms():
            a.SetIsAromatic(False)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetFormalCharge(0)
        g = rw.GetMol()
        Chem.SanitizeMol(g, Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                         Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        return g

    matches = to_graph(pdb_mol).GetSubstructMatches(to_graph(tmpl), uniquify=True)
    if not matches:
        raise SystemExit('[FATAL] no substructure match even on plain graphs')
    print(f'[match  ] graph-isomorphism matches: {len(matches)}')
    match = matches[0]  # validated further by per-atom degree check
    if len(set(match)) != n_t:
        raise SystemExit('[FATAL] match not injective')

    # degree check (orphan-atom lesson): every PDB atom must carry the same
    # bond degree as its template partner
    bad = []
    for ti, pi in enumerate(match):
        dt = tmpl.GetAtomWithIdx(ti).GetDegree()
        dp = pdb_mol.GetAtomWithIdx(pi).GetDegree()
        if dt != dp:
            bad.append((ti, dt, dp))
    if bad:
        raise SystemExit(f'[FATAL] bond degree mismatches (tmpl_idx, d_tmpl, d_pdb): {bad}')
    if len(matches) > 1:
        ok = [m for m in matches
              if all(tmpl.GetAtomWithIdx(t).GetDegree() == pdb_mol.GetAtomWithIdx(p).GetDegree()
                     for t, p in enumerate(m))]
        if len(ok) != 1:
            raise SystemExit(f'[FATAL] ambiguous graph mapping ({len(ok)} pass degree check)')
        match = ok[0]
    print(f'[match  ] unique validated mapping template->PDB: {match}')

    # --- copy coordinates onto template-ordered molecule ---
    mol = Chem.Mol(tmpl)
    conf = Chem.Conformer(n_t)
    coords = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in heavy])
    for ti, pi in enumerate(match):
        x, y, z = coords[pi]
        conf.SetAtomPosition(ti, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    # --- validation (heavy-atom graph, before H addition) ---
    Chem.SanitizeMol(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    charge = Chem.GetFormalCharge(mol)
    frags = len(Chem.GetMolFrags(mol))
    bond_lengths = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = np.linalg.norm(coords[match[i]] - coords[match[j]])
        bond_lengths.append(d)
    bl = np.array(bond_lengths)
    orphan = [ti for ti in range(n_t) if mol.GetAtomWithIdx(ti).GetDegree() == 0]
    print(f'[valid  ] sanitize OK | formula {formula} | formal charge {charge} | '
          f'fragments {frags} | orphan atoms {orphan}')
    print(f'[valid  ] coord bbox x[{coords[:,0].min():.1f},{coords[:,0].max():.1f}] '
          f'y[{coords[:,1].min():.1f},{coords[:,1].max():.1f}] '
          f'z[{coords[:,2].min():.1f},{coords[:,2].max():.1f}]')
    print(f'[valid  ] mapped bond lengths: min {bl.min():.2f} max {bl.max():.2f} '
          f'(flag if > 2.0 A)')
    if bl.max() > 2.0:
        raise SystemExit('[FATAL] implausible bond length - mapping wrong')
    if frags != 1 or orphan:
        raise SystemExit('[FATAL] fragmented/orphan atoms')

    # --- explicit POLAR hydrogens only (rDock REQUIREMENT: polar H must be
    # explicit, rDock never adds them). Nonpolar H must NOT be present:
    # rDock's own RemoveNonPolarHydrogens() segfaults on ligand SDs the same
    # way as on receptor MOL2 (erase-during-iteration), so we hand rDock the
    # extended-carbon model directly. ---
    polar = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() in ('N', 'O', 'S', 'P') and a.GetTotalNumHs() > 0]
    mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=polar)
    n_h = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'H')
    Chem.SanitizeMol(mol)

    mol.SetProp('_Name', args.name)
    w = Chem.SDWriter(args.out)
    w.write(mol)
    w.close()
    print(f'[saved  ] {args.out} ({n_t} heavy + {n_h} explicit H, template atom order)')


if __name__ == '__main__':
    main()
