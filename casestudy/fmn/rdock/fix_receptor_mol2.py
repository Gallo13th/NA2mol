#!/usr/bin/env python
"""
Fix an obabel-produced mol2 so rDock can parse it:
  1. Replaces apostrophes in atom names ("'" -> "p") (rDock segfault cause #1)
  2. Adds the missing @<TRIPOS>SUBSTRUCTURE section by grouping atoms on
     their (subst_id, subst_name) fields already present in the ATOM records,
     and updates the MOLECULE header counts (rDock segfault cause #2)
  3. Kekulizes aromatic bonds ("ar" -> alternating 1/2) so the receptor has
     only integer bond orders, which rDock prefers.

Usage: python fix_receptor_mol2.py receptor.mol2
"""
import sys
from collections import OrderedDict

def kekulize_ar(ar_pairs):
    """ar_pairs: list of (a1, a2) edges that were aromatic.
    Assign 1/2 alternately per connected aromatic ring component via DFS."""
    # build adjacency among aromatic edges
    adj = {}
    for a, b in ar_pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    order = {}
    for start in adj:
        if start in order:
            continue
        # BFS bipartition: group 0 -> 'first', group 1 -> 'second'
        stack = [start]
        order[start] = 1
        comp = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in order:
                    order[v] = 2 if order[u] == 1 else 1
                    comp.append(v)
                    stack.append(v)
        if len(comp) % 2 != 0:
            # odd ring; bipartition impossible -> keep all single
            for n in comp:
                order[n] = 1
    result = {}
    for a, b in ar_pairs:
        result[(a, b)] = 1 if order.get(a, 1) == order.get(b, 1) else 2
    return result

def fix(path, kekulize=False):
    lines = open(path).readlines()
    sections = {}
    cur = None
    for ln in lines:
        if ln.startswith("@<TRIPOS>"):
            cur = ln.strip()
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(ln)

    # --- 1. sanitize atom names + collect residue groups ---
    atom_lines = sections.get("@<TRIPOS>ATOM", [])
    subst_order = OrderedDict()  # subst_id -> subst_name
    root_atom = {}               # subst_id -> first atom id
    new_atom_lines = []
    for ln in atom_lines:
        if not ln.strip():
            continue
        flds = ln.split()
        atom_id = int(flds[0])
        name = flds[1].replace("'", "p")
        x, y, z, atype = flds[2], flds[3], flds[4], flds[5]
        subst_id = flds[6]
        subst_name = flds[7] if len(flds) > 7 else "RES"
        charge = flds[8] if len(flds) > 8 else "0.0"
        sid = int(subst_id)
        if sid not in subst_order:
            subst_order[sid] = subst_name
            root_atom[sid] = atom_id
        new_atom_lines.append(
            f"{atom_id:>7} {name:<6} {x:>11.4f} {y:>11.4f} {z:>11.4f} "
            f"{atype:<5} {subst_id:>3} {subst_name:<6} {charge}\n"
            if False else
            # rebuild robustly without reformatting floats
            "%7d %-6s %s %s %s %-5s %3d %-6s %s\n" % (
                atom_id, name, x, y, z, atype, sid, subst_name, charge)
        )

    # --- 2. kekulize aromatic bonds (optional) ---
    bond_lines = sections.get("@<TRIPOS>BOND", [])
    ar_pairs = []
    new_bond_lines = []
    for ln in bond_lines:
        if not ln.strip():
            continue
        flds = ln.split()
        bid, a1, a2, bo = flds[0], flds[1], flds[2], flds[3]
        if bo == "ar":
            if kekulize:
                ar_pairs.append((int(a1), int(a2)))
            else:
                new_bond_lines.append("%7s %5s %5s ar\n" % (bid, a1, a2))
        else:
            new_bond_lines.append("%7s %5s %5s %s\n" % (bid, a1, a2, bo))
    if kekulize:
        ar_orders = kekulize_ar(ar_pairs)
        for i, (a1, a2) in enumerate(ar_pairs, start=len(new_bond_lines) + 1):
            bo = ar_orders[(a1, a2)]
            new_bond_lines.append("%7d %5d %5d %d\n" % (i, a1, a2, bo))

    # --- 3. rebuild molecule ---
    n_atoms = len(new_atom_lines)
    n_bonds = len(new_bond_lines)
    n_subst = len(subst_order)

    out = []
    out.append("@<TRIPOS>MOLECULE\n")
    out.append("receptor\n")
    out.append(f" {n_atoms} {n_bonds} {n_subst} 0 0\n")
    out.append("SMALL\n")
    out.append("GASTEIGER\n\n")
    out.append("@<TRIPOS>ATOM\n")
    out.extend(new_atom_lines)
    out.append("@<TRIPOS>BOND\n")
    out.extend(new_bond_lines)
    out.append("@<TRIPOS>SUBSTRUCTURE\n")
    for sid, sname in subst_order.items():
        out.append("%6d %-6s %6d ROOT RESIDUE\n" % (sid, sname, root_atom[sid]))

    with open(path, "w") as fh:
        fh.writelines(out)
    print(f"[fix] atoms={n_atoms} bonds={n_bonds} substructures={n_subst} "
          f"(aromatic bonds {'kekulized' if kekulize else 'kept as ar'}: {len(ar_pairs)})")

if __name__ == "__main__":
    # second arg optional: if "kek", kekulize aromatic bonds; else keep "ar"
    kek = (len(sys.argv) > 2 and sys.argv[2] == "kek")
    fix(sys.argv[1], kekulize=kek)
