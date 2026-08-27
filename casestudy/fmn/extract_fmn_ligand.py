#!/usr/bin/env python
"""
Step0: Extract the FMN ligand (HETATM resname FMN) from MODEL 1 of 1FMN.pdb
into an SDF (with 3D) for use as:
  (1) rDock cavity reference + self-docking validation ligand
  (2) RNAsmol scoring validation ligand (SMILES derived from the graph)
Also dumps the FMN SMILES to FMN_ligand.smi for downstream use.
"""
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "1FMN.pdb")
OUT_SDF = os.path.join(HERE, "FMN_ligand.sdf")
OUT_SMI = os.path.join(HERE, "FMN_ligand.smi")
OUT_PDB = os.path.join(HERE, "FMN_crystal.pdb")

RESNAME = "FMN"

# Canonical isomeric SMILES for FMN (flavin mononucleotide)
FMN_SMILES = "Cc1cc2nc3c(=O)[nH]c(=O)nc3n(C[C@H](O)[C@H](O)[C@H](O)COP(O)(O)=O)c2cc1C"

def extract_model1_residue(pdb_path, resname):
    """Return (atom_lines, conect_lines, atom_count) for HETATM atoms of
    `resname` in MODEL 1, plus the CONECT records that only reference those
    atoms. Atom serials are preserved so CONECT records stay valid."""
    lines = []
    in_model1 = False
    seen_serials = set()
    n = 0
    with open(pdb_path) as fh:
        for ln in fh:
            rec = ln[0:6].strip()
            if rec == "MODEL":
                in_model1 = (ln.split()[1] == "1")
                continue
            if rec == "ENDMDL":
                if in_model1:
                    break
            if not in_model1:
                continue
            if rec == "HETATM" and ln[17:20].strip() == resname:
                lines.append(ln)
                try:
                    seen_serials.add(int(ln[6:11]))
                except ValueError:
                    pass
                n += 1
    # collect CONECT records referencing only our serials (global scan)
    conect = []
    with open(pdb_path) as fh:
        for ln in fh:
            if not ln.startswith("CONECT"):
                continue
            flds = ln.split()
            serials = [int(x) for x in flds[1:] if x.lstrip("-").isdigit()]
            if serials and all(s in seen_serials for s in serials):
                conect.append(ln)
    return lines, conect, n

def main():
    atoms, conect, n = extract_model1_residue(PDB, RESNAME)
    print(f"[extract] {n} HETATM atoms for {RESNAME} in MODEL 1 ; "
          f"{len(conect)} CONECT records")

    # Write the ligand PDB keeping original serials (so CONECT stays valid)
    # and appending the matching CONECT records.
    with open(OUT_PDB, "w") as fh:
        fh.write("MODEL        1\n")
        for ln in atoms:
            fh.write(ln if ln.endswith("\n") else ln + "\n")
        fh.write("ENDMDL\n")
        for ln in conect:
            fh.write(ln if ln.endswith("\n") else ln + "\n")
    print(f"[write] {OUT_PDB}")

    # Build FMN from the canonical SMILES (guaranteed-correct chemistry /
    # bond orders) rather than fighting RDKit's PDB bond-order perception of
    # the isoalloxazine ring. Then embed 3D and translate the centroid onto
    # the crystal FMN centroid so the ligand sits inside the 1FMN pocket
    # (used as the rDock cavity reference + self-docking seed).
    mol = Chem.MolFromSmiles(FMN_SMILES)
    if mol is None:
        print("[error] FMN template SMILES is invalid", file=sys.stderr)
        sys.exit(1)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    # crystal centroid of the FMN heavy atoms
    cryst_conf = None
    cryst = Chem.MolFromPDBFile(OUT_PDB, removeHs=True, sanitize=False)
    if cryst is not None and cryst.GetNumConformers():
        import numpy as np
        cpos = cryst.GetConformer().GetPositions()
        cen_cryst = cpos.mean(axis=0)
        mpos = mol.GetConformer().GetPositions()
        cen_mol = mpos.mean(axis=0)
        shift = cen_cryst - cen_mol
        mol_conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = mol_conf.GetAtomPosition(i)
            mol_conf.SetAtomPosition(i, (p.x + shift[0],
                                         p.y + shift[1],
                                         p.z + shift[2]))
        print(f"[align] translated FMN centroid -> crystal centroid "
              f"({cen_cryst[0]:.2f},{cen_cryst[1]:.2f},{cen_cryst[2]:.2f})")

    mol_noH = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol_noH)
    print(f"[rdkit] heavy atoms={mol_noH.GetNumHeavyAtoms()}  "
          f"total atoms={mol.GetNumAtoms()}")
    print(f"[smiles] {smi}")
    with open(OUT_SMI, "w") as fh:
        fh.write(smi + "\tfmn\n")

    # Write SDF with 3D (all-H) and the heavy-atom reference separately
    w = Chem.SDWriter(OUT_SDF)
    mol.SetProp("_Name", "FMN")
    mol.SetProp("SMILES", smi)
    w.write(mol)
    w.close()
    print(f"[write] {OUT_SDF}  ({mol.GetNumAtoms()} atoms, with 3D + H)")
    print(f"[write] {OUT_SMI}")

if __name__ == "__main__":
    main()
