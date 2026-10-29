"""Extract FMN binding pocket from 1FMN MODEL 1 using genbypkt.extract_pocket logic."""
import sys
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

# Constants copied from genbypkt.py
WATER_RESNAMES = {'HOH', 'WAT', 'H2O', 'DOD', 'TIP', 'TIP3', 'TIP4', 'SOL'}
AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'SEC', 'PYL', 'MSE', 'CAS', 'CSD', 'CSO', 'CSS', 'CSW', 'CSX', 'CYM',
    'CYX', 'HYP', 'KCX', 'LLP', 'MLY', 'NLE', 'OCS', 'PCA', 'PTR', 'SEP',
    'TPO', 'TPQ', 'TYS', 'UNK',
}
NUC_RESNAMES = {
    'A', 'G', 'C', 'T', 'U', 'DA', 'DG', 'DC', 'DT', 'DU',
    'RA', 'RG', 'RC', 'RU', 'DI', 'ADE', 'GUA', 'CYT', 'THY', 'URA', 'URI',
}
ION_RESNAMES = {
    'NA', 'CL', 'CA', 'MG', 'ZN', 'K', 'FE', 'CU', 'MN', 'CO',
    'NI', 'CD', 'HG', 'PB', 'AG', 'AU', 'AL', 'BA', 'BE', 'CS',
    'LI', 'RB', 'SR', 'V', 'YB', 'CR', 'MO', 'W', 'SE', 'BR',
    'F', 'I', 'XE', 'KR', 'AR', 'HE', 'NE',
}
ALLOWED_RESNAMES = AA_RESNAMES | NUC_RESNAMES | ION_RESNAMES


def extract_pocket(residues, ligand_coords, distance_cutoff=20.0):
    pocket_residues = []
    ligand_center = np.mean(ligand_coords, axis=0)
    for residue in residues:
        resname = residue.resname.strip().upper()
        if resname in WATER_RESNAMES:
            continue
        if resname not in ALLOWED_RESNAMES:
            continue
        atom_coords = np.array([atom.coord for atom in residue])
        if len(atom_coords) == 0:
            continue
        residue_center = np.mean(atom_coords, axis=0)
        dist = np.linalg.norm(ligand_center - residue_center)
        if dist < distance_cutoff:
            pocket_residues.append(residue)
    return pocket_residues


class PocketSelect(Select):
    def __init__(self, residues):
        self.residue_set = set(residues)

    def accept_residue(self, residue):
        return residue in self.residue_set


def save_pocket(pocket_residues, structure, output_pdb):
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, PocketSelect(pocket_residues))
    print(f"Pocket saved to: {output_pdb}")


def main():
    pdb_file = "/home/data/users/gaoletian/rna2smol/casestudy/fmn/1FMN.pdb"
    out_pocket = "/home/data/users/gaoletian/rna2smol/casestudy/fmn/1FMN_pocket_extracted.pdb"
    cutoff = 14.0

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1FMN", pdb_file)

    models = list(structure.get_models())
    print(f"Total models: {len(models)}")
    model1 = models[0]
    print(f"Using MODEL 1 (id={model1.id})")

    # Collect all residues in MODEL 1
    all_residues = list(model1.get_residues())
    print(f"Total residues in MODEL 1: {len(all_residues)}")

    # Separate FMN ligand and RNA residues
    fmn_atoms_coords = []
    rna_residues = []
    for res in all_residues:
        rn = res.resname.strip().upper()
        if rn == 'FMN':
            for atom in res:
                fmn_atoms_coords.append(atom.coord)
        else:
            rna_residues.append(res)

    fmn_coords = np.array(fmn_atoms_coords)
    print(f"FMN atoms in MODEL 1: {len(fmn_coords)}")
    if len(fmn_coords) == 0:
        print("ERROR: no FMN found in MODEL 1")
        sys.exit(1)
    print(f"FMN coord min: {fmn_coords.min(axis=0)}")
    print(f"FMN coord max: {fmn_coords.max(axis=0)}")
    print(f"FMN center: {fmn_coords.mean(axis=0)}")

    print(f"RNA/other residues: {len(rna_residues)}")

    # Extract pocket
    pocket = extract_pocket(rna_residues, fmn_coords, distance_cutoff=cutoff)
    print(f"Pocket residues (cutoff={cutoff}A): {len(pocket)}")
    resnames = [r.resname.strip() for r in pocket]
    from collections import Counter
    print(f"Pocket residue name distribution: {dict(Counter(resnames))}")

    # Count atoms
    n_atoms = sum(1 for r in pocket for _ in r.get_atoms())
    print(f"Pocket atom count: {n_atoms}")

    save_pocket(pocket, structure, out_pocket)
    print("DONE")


if __name__ == "__main__":
    main()
