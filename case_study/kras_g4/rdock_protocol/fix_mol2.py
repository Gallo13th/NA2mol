#!/usr/bin/env python3
"""Post-process an obabel-generated receptor MOL2 for rDock.

Fixes (all required for correct rDock electrostatics / parsing):
  1. remove ALL non-polar hydrogens ourselves (H bonded to carbon).
     rDock 2013.1's own RemoveNonPolarHydrogens() segfaults on
     erase-during-iteration; feeding the extended-carbon model directly
     is exactly the input contract stated in the User Guide (polar H
     explicit, non-polar H implicit). Phosphate O-H wrongly added by
     obabel -h (PDB has no bond orders) are dropped as well.
  2. substructure names: obabel concatenates resname+resnum ("DG11");
     strip the trailing residue number so RbtIonicAtoms.prm lookup works.
     DNA names (DG/DA/DC/DT) are additionally mapped to their RNA
     equivalents (G/A/C/U) because the stock RbtIonicAtoms.prm only has
     RNA nucleotide sections. Protein names are restored verbatim (charges
     for ASP/GLU/LYS/ARG/HIS/...).
  3. phosphate atom names OP1/OP2 -> O1P/O2P (stock table naming)
  4. rewrite in the fixed-width column layout rDock's MOL2 parser
     expects + consistent MOLECULE counts + @<TRIPOS>SUBSTRUCTURE
     records (whitespace-joined lines or stale counts segfault it too)

Usage: fix_mol2.py <in.mol2> <out.mol2>
"""
import re
import sys

DNA2RNA = {'DG': 'G', 'DA': 'A', 'DC': 'C', 'DT': 'U'}
# substructure name = alphabetic residue name + residue number ("DG11")
RES_STRIP = re.compile(r'^([A-Z]+)(\d+)$')
CARBON_TYPES = {'C.3', 'C.2', 'C.1', 'C.ar', 'C.CAT', 'C.O2'}


def fix_subst(name):
    m = RES_STRIP.match(name)
    base = m.group(1) if m else name
    return DNA2RNA.get(base, base)


def main(src, dst):
    lines = open(src).read().splitlines()
    i0 = lines.index('@<TRIPOS>ATOM')
    i1 = next(i for i, l in enumerate(lines) if l.startswith('@<TRIPOS>BOND'))

    atoms = [l.split() for l in lines[i0 + 1:i1] if l.strip()]
    bonds = [l.split() for l in lines[i1 + 1:] if l.strip() and not l.startswith('@')]

    # 1. rename + find H to drop
    for a in atoms:
        a[1] = a[1].replace("'", 'p')
        if a[1] == 'OP1':
            a[1] = 'O1P'
        elif a[1] == 'OP2':
            a[1] = 'O2P'
        a[7] = fix_subst(a[7])

    idx2atom = {int(a[0]): a for a in atoms}
    phos_o = {'O1P', 'O2P', 'OP1', 'OP2'}
    drop = set()
    n_np, n_pph = 0, 0
    for b in bonds:
        a1, a2 = int(b[1]), int(b[2])
        n1, n2 = idx2atom[a1][1], idx2atom[a2][1]
        t1, t2 = idx2atom[a1][5], idx2atom[a2][5]
        # H on carbon -> non-polar, remove (rDock's own removal segfaults)
        if t1 == 'H' and t2 in CARBON_TYPES:
            drop.add(a1); n_np += 1
        elif t2 == 'H' and t1 in CARBON_TYPES:
            drop.add(a2); n_np += 1
        # H on phosphate O -> obabel artifact, remove (phosphate stays anionic)
        elif n1.startswith('H') and n2 in phos_o:
            drop.add(a1); n_pph += 1
        elif n2.startswith('H') and n1 in phos_o:
            drop.add(a2); n_pph += 1

    remap = {}
    new_atoms = []
    for a in atoms:
        if int(a[0]) in drop:
            continue
        remap[int(a[0])] = len(new_atoms) + 1
        a[0] = str(len(new_atoms) + 1)
        new_atoms.append(a)
    new_bonds = []
    for b in bonds:
        a1, a2 = int(b[1]), int(b[2])
        if a1 in drop or a2 in drop:
            continue
        new_bonds.append([b[0], str(remap[a1]), str(remap[a2])] + b[3:])

    def fmt_atom(a):
        s = (f'{int(a[0]):>7} {a[1]:<8} {float(a[2]):>9.4f} {float(a[3]):>9.4f} '
             f'{float(a[4]):>9.4f} {a[5]:<6} {int(a[6]):>4} {a[7]:<8}')
        if len(a) > 8:
            s += f' {float(a[8]):>9.4f}'
        return s

    def fmt_bond(b):
        parts = [b[0], b[1], b[2]] + ([b[3]] if len(b) > 3 else ['1'])
        return f'{int(parts[0]):>6} {int(parts[1]):>6} {int(parts[2]):>6} {parts[3]:>2}'

    out = []
    substs = sorted({(int(a[6]), a[7]) for a in new_atoms})
    for l in lines[:i0]:
        parts = l.split()
        if (len(parts) >= 2 and all(p.replace('.', '').replace('-', '').isdigit()
                                    for p in parts)):
            parts[0] = str(len(new_atoms))
            parts[1] = str(len(new_bonds))
            if len(parts) > 2:
                parts[2] = str(len(substs))
            out.append(' '.join(parts))
        else:
            out.append(l)
    out.append('@<TRIPOS>ATOM')
    out += [fmt_atom(a) for a in new_atoms]
    out.append('@<TRIPOS>BOND')
    out += [fmt_bond(b) for b in new_bonds]
    out.append('@<TRIPOS>SUBSTRUCTURE')
    for sid, nm in substs:
        out.append(f'{sid:>7} {nm:<8} 1 RESIDUE {sid:>4}')
    for l in lines[i1 + 1:]:
        if l.strip() and not l.startswith('@'):
            continue  # old bond records, replaced above
        parts = l.split()
        if len(parts) > 1 and RES_STRIP.match(parts[1]):
            parts[1] = fix_subst(parts[1])
            out.append(' '.join(parts))
            continue
        out.append(l)

    with open(dst, 'w') as f:
        f.write('\n'.join(out) + '\n')
    n_polar = sum(1 for a in new_atoms if a[5] == 'H')
    print(f'[fix_mol2] {len(atoms)} -> {len(new_atoms)} atoms '
          f'(dropped {n_np} nonpolar H + {n_pph} phosphate H, '
          f'kept {n_polar} polar H), {len(bonds)} -> {len(new_bonds)} bonds')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
