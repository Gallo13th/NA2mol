#!/usr/bin/env python
"""SPEC 04 — Build D / A / pi single-role functional fragment libraries.

Method (per task prompt): template matching + sp3 extension.
- D library: donor templates, extend sp3 substituents via BFS, stop at conjugated bond.
- A library: acceptor templates, +1 layer sp3 light extension.
- pi library: bridge templates, match atoms only.

Data source: fluor_util_2/FluoDB-Full.csv (does NOT read known_mols).
"""
import os
import sys
import pickle
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = "/home/data/users/gaoletian/rna2smol"
CSV_PATH = os.path.join(ROOT, "fluor_util_2", "FluoDB-Full.csv")
OUT_DIR = os.path.join(ROOT, "fragment_library", "functional_blocks")

# ---------- Step 1: functional unit templates (complete structure SMARTS) ----------
D_TEMPLATES = {
    "tert_amine": "[NX3;H0]([#6])([#6])[#6]",   # tertiary amine
    "sec_amine_ar": "[NX3;H1][c]",               # aryl secondary amine
    "carbazole": "c1ccc2c(c1)[nH]c1ccccc12",
    "indole": "c1ccc2[nH]ccc2c1",
    "triphenylamine": "N(c1ccccc1)(c1ccccc1)c1ccccc1",
    "phenothiazine": "N1c2ccccc2Sc2ccccc21",
    "phenoxazine": "N1c2ccccc2Oc2ccccc21",
    "dibenzofuran": "c1ccc2c(c1)oc1ccccc12",
    "dibenzothiophene": "c1ccc2c(c1)sc1ccccc12",
    "pyrrole": "c1cc[nH]c1",
}
A_TEMPLATES = {
    "rhodanine": "O=C1NC(=S)SC1",            # whole rhodanine ring
    "cyano_ethylene": "C=C(C#N)",            # cyanoethylene (mol4 acceptor)
    "cyanoacrylic": "C=C(C#N)C(=O)O",        # cyanoacrylic acid
    "barbituric": "O=C1NC(=O)NC(=O)C1",
    "cyano": "[#6]#[#7]",                     # cyano (incl alpha carbon)
    "nitro": "[N+](=O)[O-]",
    "thiocarbonyl": "C=S",
}
PI_TEMPLATES = {
    "stilbene": "C(=Cc1ccccc1)c1ccccc1",     # stilbene (mol4 bridge)
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "thiophene": "c1ccsc1",
    "benzene": "c1ccccc1",
    "fluorene": "c1ccc2c(c1)Cc1ccccc1-2",
    "phenylene_vinylene": "C=Cc1ccccc1",
}


# ---------- Step 2: unit extraction ----------
def is_conjugated_bond(bond):
    """aromatic / double / triple bond == conjugated"""
    bt = bond.GetBondType()
    return bond.GetIsAromatic() or bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)


def prune_dangling_aromatic(mol, include):
    """Drop aromatic atoms whose aromatic-bond neighbours are not all in `include`.

    Realises the stated rule "aromatic ring does not enter D" (and analogously for A):
    a partially-matched aromatic atom (e.g. the ipso-carbon of a phenyl substituent
    pulled in by a generic [#6] in an anchor SMARTS) is a boundary marker, not part
    of the functional unit, and must be removed. Fully-captured aromatic systems
    (carbazole / TPA / benzene / stilbene ...) keep every aromatic atom because all
    of their aromatic-bond neighbours are within `include`.
    """
    include = set(include)
    changed = True
    while changed:
        changed = False
        for idx in list(include):
            atom = mol.GetAtomWithIdx(idx)
            if not atom.GetIsAromatic():
                continue
            for nb in atom.GetNeighbors():
                ni = nb.GetIdx()
                if ni in include:
                    continue
                b = mol.GetBondBetweenAtoms(idx, ni)
                if b is not None and b.GetIsAromatic():
                    # an aromatic neighbour is missing -> this atom is dangling
                    include.discard(idx)
                    changed = True
                    break
    return include


def extract_unit(mol, match_atoms, role):
    """Extract match_atoms as a substructure fragment with modest sp3 extension.

    role='donor' (='D'): BFS collect sp3 neighbours (alkyl/alkoxy); stop at sp2/aromatic boundary.
    role='acceptor' (='A'): match atoms + direct sp3 small substituent (1 layer).
    role='bridge' (='pi'): match atoms only.
    Returns RDKit Mol (substructure) or None.
    """
    # normalise role aliases used between Step 2 (D/A/pi) and Step 3 (donor/acceptor/bridge)
    role_norm = {"D": "donor", "A": "acceptor", "pi": "bridge"}.get(role, role)
    match_set = set(match_atoms)
    include = set(match_atoms)

    if role_norm == "donor":
        frontier = list(match_atoms)
        visited = set(match_atoms)
        while frontier:
            a = frontier.pop()
            atom = mol.GetAtomWithIdx(a)
            for nb in atom.GetNeighbors():
                ni = nb.GetIdx()
                if ni in visited:
                    continue
                b = mol.GetBondBetweenAtoms(a, ni)
                if b is None:
                    continue
                # D boundary: cross conjugated bond -> stop, do not include
                if is_conjugated_bond(b):
                    continue
                if nb.GetAtomicNum() == 1:
                    continue
                visited.add(ni)
                include.add(ni)
                frontier.append(ni)
    elif role_norm == "acceptor":
        for a in match_atoms:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nb.GetIdx()
                b = mol.GetBondBetweenAtoms(a, ni)
                if b is None:
                    continue
                if not is_conjugated_bond(b) and nb.GetAtomicNum() > 1:
                    include.add(ni)
    # role_norm == "bridge": no extension

    # Realise "aromatic ring does not enter the unit unless fully captured":
    # drop aromatic atoms whose aromatic-bond neighbours are missing.
    include = prune_dangling_aromatic(mol, include)

    # Extract substructure via RWMol: remove all atoms not in `include`
    try:
        rw = Chem.RWMol(mol)
        keep_set = set(include)
        # remove from highest idx downward so indices stay valid
        for idx in sorted(
            [a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep_set],
            reverse=True,
        ):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            pass
        return out
    except Exception:
        return None


# ---------- Step 3: main pipeline ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"[load] {len(df)} rows from {CSV_PATH}")

    # compile patterns once
    compiled = {}
    for tpl_dict, role in [
        (D_TEMPLATES, "donor"),
        (A_TEMPLATES, "acceptor"),
        (PI_TEMPLATES, "bridge"),
    ]:
        compiled[role] = []
        for name, smarts in tpl_dict.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                print(f"  [warn] bad SMARTS for {role}/{name}: {smarts}")
                continue
            compiled[role].append((name, pat))

    buckets = {"donor": Counter(), "acceptor": Counter(), "bridge": Counter()}
    # also track source template name per smiles for debugging
    src_name = {"donor": {}, "acceptor": {}, "bridge": {}}

    n_mol_ok = 0
    n_mol_bad = 0
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            n_mol_bad += 1
            continue
        n_mol_ok += 1
        for role, pats in compiled.items():
            for name, pat in pats:
                for m in mol.GetSubstructMatches(pat):
                    unit = extract_unit(mol, m, role)
                    if unit is None or unit.GetNumAtoms() < 2:
                        continue
                    try:
                        usmi = Chem.MolToSmiles(unit)
                    except Exception:
                        continue
                    if not usmi:
                        continue
                    buckets[role][usmi] += 1
                    src_name[role].setdefault(usmi, name)

    print(f"[parse] ok={n_mol_ok} bad={n_mol_bad}")
    for role in ("donor", "acceptor", "bridge"):
        print(f"[count] {role}: {len(buckets[role])} unique fragments, "
              f"total occurrences={sum(buckets[role].values())}")

    # ---------- Step 4: dedup + save library ----------
    role_to_file = {
        "donor": "donors.pkl",
        "acceptor": "acceptors.pkl",
        "bridge": "bridges.pkl",
    }
    summary = {}
    for role, fname in role_to_file.items():
        items = []
        for usmi, freq in buckets[role].most_common():
            smarts = None
            m = Chem.MolFromSmiles(usmi)
            if m is not None:
                try:
                    smarts = Chem.MolToSmarts(m)
                except Exception:
                    smarts = None
            items.append({
                "smiles": usmi,
                "smarts": smarts,
                "role": role,
                "freq": freq,
                "src_template": src_name[role].get(usmi, "?"),
            })
        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "wb") as f:
            pickle.dump(items, f)
        summary[role] = items
        print(f"[save] {out_path}  ({len(items)} fragments)")

    # ---------- build_report.md ----------
    write_report(summary)
    print("[done]")


def write_report(summary):
    role_title = {
        "donor": "Donor (D) library",
        "acceptor": "Acceptor (A) library",
        "bridge": "Bridge (pi) library",
    }
    lines = []
    lines.append("# Functional Blocks Build Report\n")
    lines.append("Built by `build_functional_blocks.py` (template matching + sp3 extension).\n")
    lines.append("Data source: `fluor_util_2/FluoDB-Full.csv`. Not using BRICS.\n")
    lines.append("")
    for role in ("donor", "acceptor", "bridge"):
        items = summary[role]
        lines.append(f"## {role_title[role]}")
        lines.append("")
        lines.append(f"- unique fragments: **{len(items)}**")
        lines.append(f"- total occurrences: **{sum(it['freq'] for it in items)}**")
        lines.append("")
        lines.append("### top-10")
        lines.append("")
        lines.append("| rank | smiles | freq | src_template |")
        lines.append("|------|--------|------|--------------|")
        for i, it in enumerate(items[:10], 1):
            lines.append(f"| {i} | `{it['smiles']}` | {it['freq']} | {it['src_template']} |")
        lines.append("")
    out = os.path.join(OUT_DIR, "build_report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
