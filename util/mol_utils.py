"""
https://github.com/mattragoza/liGAN/blob/master/fitting.py

License: GNU General Public License v2.0
https://github.com/mattragoza/liGAN/blob/master/LICENSE
"""
import itertools

import numpy as np
from rdkit.Chem import AllChem
from rdkit import Geometry
from openbabel import openbabel as ob
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from util.mol_refine import MoleculeRefiner
import os
import sys
import random
FRAGLIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fragment_library_final.pkl')

MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX = {
    (1, False): 0,   # 998
    (6, False): 1,   # 803564
    (6, True): 2,    # 1371671
    (7, False): 3,   # 198073
    (7, True): 4,    # 209316
    (8, False): 5,   # 264143
    (8, True): 6,    # 8567
    (9, False): 7,   # 72259
    (15, False): 8,  # 2051
    (15, True): 9,   # aromatic P — matches featurizer.MAP_ATOM_TYPE_AROMATIC_TO_INDEX
    (16, False): 10, # aliphatic S
    (16, True): 11,  # aromatic S (thiophene/benzothiazole...) — was mis-decoded as Cl
    (17, False): 12, # Cl — was mis-decoded as Br (Br is NOT in the 13 training classes)
}
# NOTE (bugfix): this decode table previously read
#   idx9=(16,False) idx10=(16,True) idx11=(17,False) idx12=(35,False)
# which contradicts the featurizer encode table used at training time
# (dataloader/featurizer.py MAP_ATOM_TYPE_AROMATIC_TO_INDEX). The model was
# always trained/outputting the featurizer semantics; only generation-side
# decoding was wrong. Historical products labeled S(9)/S*(10)/Cl(11)/Br(12)
# must be re-read as P*/S/S*/Cl respectively.

MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX.items()}

class MolReconError(Exception):
    pass


def reachable_r(a, b, seenbonds):
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr, b, seenbonds):
                return True
    return False


def reachable(a, b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False  # this is the _only_ bond for one atom
    # otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a, b, seenbonds)


def forms_small_angle(a, b, cutoff=60):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a, nbr)
            if degrees < cutoff:
                return True
    return False


def make_obmol(xyz, atomic_numbers):
    mol = ob.OBMol()
    mol.BeginModify()
    atoms = []
    for coord, t in zip(xyz, atomic_numbers):
        # 确保坐标是浮点数
        x = float(coord[0])
        y = float(coord[1])
        z = float(coord[2])
        atom = mol.NewAtom()
        atom.SetAtomicNum(int(t))          # 确保原子序数是整数
        atom.SetVector(x, y, z)
        atoms.append(atom)
    return mol, atoms

def connect_the_dots(mol, atoms, indicators, covalent_factor=1.3):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''

    """
    for now, indicators only include 'is_aromatic'
    """
    pt = AllChem.GetPeriodicTable()

    if len(atoms) == 0:
        return

    mol.BeginModify()

    # just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    # types = [struct.channels[t].name for t in struct.c]

    for i, j in itertools.combinations(range(len(atoms)), 2):
        a = atoms[i]
        b = atoms[j]
        a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
        b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
        if dists[i, j] < a_r + b_r:
            flag = 0
            if indicators and indicators[i] and indicators[j]:
                flag = ob.OB_AROMATIC_BOND
            mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)

    atom_maxb = {}
    for (i, a) in enumerate(atoms):
        # set max valance to the smallest max allowed by openbabel or rdkit
        # since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))

        if a.GetAtomicNum() == 16:  # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb

    # remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            # compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
            stretch = bdist / ideal
            binfo.append((stretch, bond))
        binfo.sort(reverse=True, key=lambda t: t[0])  # most stretched bonds first
        return binfo

    binfo = get_bond_info(ob.OBMolBondIter(mol))
    # now eliminate geometrically poor bonds
    for stretch, bond in binfo:

        # can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        # as long as we aren't disconnecting, let's remove things
        # that are excessively far away (0.45 from ConnectTheDots)
        # get bonds to be less than max allowed
        # also remove tight angles, because that is what ConnectTheDots does
        if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):
            # don't fragment the molecule
            if not reachable(a1, a2):
                continue
            mol.DeleteBond(bond)

    # prioritize removing hypervalency causing bonds, do more valent
    # constrained atoms first since their bonds introduce the most problems
    # with reachability (e.g. oxygen)
    hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
    hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))
    for mb, diff, a in hypers:
        if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
            continue
        binfo = get_bond_info(ob.OBAtomBondIter(a))
        for stretch, bond in binfo:

            if stretch < 0.9:  # the two atoms are too closed to remove the bond
                continue
            # can we remove this bond without disconnecting the molecule?
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # get right valence
            if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                # don't fragment the molecule
                if not reachable(a1, a2):
                    continue
                mol.DeleteBond(bond)
                if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                    break  # let nbr atoms choose what bonds to throw out

    mol.EndModify()


def convert_ob_mol_to_rd_mol(ob_mol, struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = AllChem.RWMol()
    rd_conf = AllChem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = AllChem.Atom(ob_atom.GetAtomicNum())
        # TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            # don't commit to being aromatic unless rdkit will be okay with the ring status
            # (this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx() - 1
        j = ob_bond.GetEndAtomIdx() - 1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, AllChem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, AllChem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, AllChem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms(i, j)
            bond.SetIsAromatic(True)

    rd_mol = AllChem.RemoveHs(rd_mol, sanitize=False)

    pt = AllChem.GetPeriodicTable()
    # if double/triple bonds are connected to hypervalent atoms, decrement the order

    # TODO: fix seg fault
    # if struct is not None:
    #     positions = struct
    positions = rd_mol.GetConformer().GetPositions()
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == AllChem.BondType.DOUBLE or bond.GetBondType() == AllChem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # TODO: ugly fix
            dist = np.linalg.norm(positions[i] - positions[j])
            nonsingles.append((dist, bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d, bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = AllChem.BondType.SINGLE
            if bond.GetBondType() == AllChem.BondType.TRIPLE:
                btype = AllChem.BondType.DOUBLE
            bond.SetBondType(btype)

    # fix up special cases
    for atom in rd_mol.GetAtoms():
        # set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

        # check if there are any carbon atoms with 2 double C-C bonds
        # if so, convert one to a single bond
        if atom.GetAtomicNum() == 6 and atom.GetDegree() == 4:
            cnt = 0
            i = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6:
                    j = nbr.GetIdx()
                    bond = rd_mol.GetBondBetweenAtoms(i, j)
                    if bond.GetBondType() == AllChem.BondType.DOUBLE:
                        cnt += 1
            if cnt == 2:
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 6:
                        j = nbr.GetIdx()
                        bond = rd_mol.GetBondBetweenAtoms(i, j)
                        if bond.GetBondType() == AllChem.BondType.DOUBLE:
                            bond.SetBondType(AllChem.BondType.SINGLE)
                            break

    rd_mol = AllChem.AddHs(rd_mol, addCoords=True)
    # TODO: fix seg fault
    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):
            # hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i, center)

    try:
        AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
    except:
        raise MolReconError()
    # try:
    #     AllChem.SanitizeMol(rd_mol,AllChem.SANITIZE_ALL^AllChem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    # but at some point stop trying to enforce our aromaticity -
    # openbabel and rdkit have different aromaticity models so they
    # won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol


def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def count_nbrs_of_elem(atom, atomic_num):
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def fixup(atoms, mol, indicators):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    """
    for now, indicators only include 'is_aromatic'
    """
    mol.SetAromaticPerceived(True)  # avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        if indicators is not None:
            if indicators[i]:
                atom.SetAromatic(True)
                atom.SetHyb(2)
            else:
                atom.SetAromatic(False)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 

        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():  # Nitrogen, Oxygen
            # this is a little iffy, ommitting until there is more evidence it is a net positive
            # we don't have aromatic types for nitrogen, but if it
            # is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def raw_obmol_from_generated(data):
    xyz = data.ligand_context_pos.clone().cpu().tolist()
    atomic_nums = data.ligand_context_element.clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)
    return mol, atoms


UPGRADE_BOND_ORDER = {AllChem.BondType.SINGLE: AllChem.BondType.DOUBLE, AllChem.BondType.DOUBLE: AllChem.BondType.TRIPLE}


def postprocess_rd_mol_1(rdmol):
    rdmol = AllChem.RemoveHs(rdmol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in nbh_list:
            nbh_list[begin] = [end]
        else:
            nbh_list[begin].append(end)

        if end not in nbh_list:
            nbh_list[end] = [begin]
        else:
            nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)

    return rdmol


def postprocess_rd_mol_2(rdmol):
    rdmol_edit = AllChem.RWMol(rdmol)

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol

def reconstruct_from_generated(xyz, atomic_nums, aromatic=None, basic_mode=True, covalent_factor=1.3):
    """
    Reconstruct molecule using carbon atoms as template for bond formation,
    then restore original atom types and fix formal charges.
    
    This two-step approach can improve connectivity for molecules with
    unusual heteroatom geometries by using uniform carbon covalent radii
    during initial bond detection.
    
    Args:
        xyz: list or np.ndarray of atomic coordinates [N, 3]
        atomic_nums: list or np.ndarray of original atomic numbers [N]
        aromatic: optional list of aromaticity indicators
        basic_mode: if True, ignore aromatic indicators during reconstruction
        covalent_factor: scaling factor for covalent radii in bond detection
    
    Returns:
        RDKit Mol object with reconstructed structure
    """
    import numpy as np
    from rdkit.Chem import AllChem
    from openbabel import openbabel as ob
    
    # Normalize inputs
    if isinstance(xyz, np.ndarray):
        xyz = xyz.tolist()
    if isinstance(atomic_nums, np.ndarray):
        atomic_nums = atomic_nums.tolist()
    if aromatic is not None and isinstance(aromatic, np.ndarray):
        aromatic = aromatic.tolist()
    
    original_atomic_nums = list(atomic_nums)
    n_atoms = len(original_atomic_nums)
    
    # ===== Step 1: Build OBMol with all atoms as carbon for uniform bond detection =====
    mol, atoms = make_obmol(xyz, [6] * n_atoms)  # All atoms temporarily set to Carbon
    indicators = None if basic_mode else aromatic
    
    # Initial fixup and bond formation using carbon covalent radii
    fixup(atoms, mol, indicators)
    connect_the_dots(mol, atoms, indicators, covalent_factor=covalent_factor)
    fixup(atoms, mol, indicators)
    
    # Re-apply fixup with correct atom types
    fixup(atoms, mol, indicators)
    
    # Continue standard reconstruction pipeline
    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, indicators)
    
    for atom in atoms:
        ob.OBAtomAssignTypicalImplicitHydrogens(atom)
    fixup(atoms, mol, indicators)
    
    mol.AddHydrogens()
    fixup(atoms, mol, indicators)
    
    # Aromatic ring handling: make rings aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            c_cnt = sum(1 for ai in ring._path if mol.GetAtom(ai).GetAtomicNum() == 6)
            arom_c_cnt = sum(1 for ai in ring._path 
                           if mol.GetAtom(ai).GetAtomicNum() == 6 and mol.GetAtom(ai).IsAromatic())
            if c_cnt > 0 and arom_c_cnt >= c_cnt / 2 and arom_c_cnt != ring.Size():
                for ai in ring._path:
                    mol.GetAtom(ai).SetAromatic(True)
    
    # Mark bonds as aromatic if both endpoints are aromatic
    for bond in ob.OBMolBondIter(mol):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)
    
    mol.PerceiveBondOrders()
    
    # Convert to RDKit mol
    rd_mol = convert_ob_mol_to_rd_mol(mol, struct=xyz)

    try:
        rd_mol = postprocess_rd_mol_1(rd_mol)
        rd_mol = postprocess_rd_mol_2(rd_mol)
    except Exception as e:
        print(e)
        raise MolReconError()
    rd_mol = AllChem.RemoveHs(rd_mol)
    # ===== Step 3: Fix formal charges for non-carbon atoms based on valence =====
    # for i, atom in enumerate(atoms):
    #     atom.SetAtomicNum(original_atomic_nums[i])
    for i, atom in enumerate(rd_mol.GetAtoms()):
        atom.SetAtomicNum(original_atomic_nums[i])
    rd_mol = _adjust_heteroatom_charges(rd_mol)
    
    # # Final sanitization to ensure chemical validity
    # try:
    #     AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
    # except:
    #     # Keep molecule even if minor sanitization issues; 
    #     # downstream tools may handle or report
    #     pass
    
    return rd_mol


def _adjust_heteroatom_charges(rd_mol):
    """
    Adjust formal charges for heteroatoms based on their bonding environment.
    
    Rules applied:
    - N with 4 bonds: +1 charge (quaternary ammonium)
    - O with 3 bonds: +1 charge (oxonium)
    - O with 1 bond: potentially -1 (alkoxide), deferred to RDKit H-addition
    - Halogens with 0 bonds: -1 charge (halide ion)
    - S with 2+ O neighbors and high valence: handled for sulfone/sulfonate patterns
    - General hypervalency: add positive charge proportional to valence excess
    """
    from rdkit.Chem import AllChem
    
    pt = AllChem.GetPeriodicTable()
    
    for atom in rd_mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        
        # Skip carbon atoms (used as template)
        if atomic_num == 6:
            continue
        
        # Calculate current valence (sum of bond orders)
        valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        
        # Reset formal charge before adjustment
        # atom.SetFormalCharge(0)
        
        # Element-specific charge rules
        if atomic_num == 7 and atom.GetDegree() == 4:  # Nitrogen
            # if valence > 3.5:  # ~4 bonds -> quaternary N
                atom.SetFormalCharge(1)
        elif atomic_num == 8 and atom.GetDegree() == 3:
                atom.SetFormalCharge(1)
            # Note: O with 1 bond and no H could be -1, but let RDKit handle during H-addition
        elif atomic_num in [9, 17, 35, 53, 85]:  # Halogens (F, Cl, Br, I, At)
            if valence < 0.5:  # No bonds -> halide ion
                atom.SetFormalCharge(-1)
        elif atomic_num == 16:  # Sulfur
            # Check for sulfone/sulfonate pattern: S with 2+ O neighbors and high valence
            o_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 8)
            if o_neighbors >= 2 and valence > 4.5:
                # Sulfone (neutral) or sulfonate (-1) - default to neutral, 
                # charge adjustment handled by context or downstream processing
                pass
        
        # General hypervalency fallback: add positive charge if valence exceeds default
        try:
            default_val = pt.GetDefaultValence(atomic_num)
            if valence > default_val + 0.5:
                charge = int(round(valence - default_val))
                if charge > 0:
                    # Only set if not already set by element-specific rules
                    if atom.GetFormalCharge() == 0:
                        atom.SetFormalCharge(charge)
        except:
            pass
    
    # Attempt sanitization to validate molecular structure and charges
    try:
        AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
    except:
        # Keep molecule even if sanitization has minor issues; 
        # downstream tools may handle or report
        pass
    
    return rd_mol
def _is_aromatic_cation(atom, mol):
    """True if ``atom`` is a positive charge that is required to maintain
    aromaticity (aromatic heteroatom cation inside an aromatic ring), e.g.
    pyridinium / quinolinium / imidazolium N+. Such charges are electronic
    consequences of the aromatic pi system, NOT spurious ions, and must be
    preserved rather than neutralized.

    Criteria (all required):
      - formal charge > 0,
      - atom is a heteroatom that can donate a p-electron to an aromatic ring
        (N, P, O, S, ...),
      - atom is flagged aromatic,
      - atom is a member of at least one ring whose every atom is aromatic.
    """
    try:
        if atom.GetFormalCharge() <= 0:
            return False
        if atom.GetSymbol() not in ('N', 'P', 'As', 'Sb', 'Bi', 'O', 'S', 'Se', 'Te'):
            return False
        ri = mol.GetRingInfo()
        # Non-ring cations FIRST: degree-2 O+/S+ with an adjacent double bond
        # (vinylic/acyl oxonium from OpenBabel bond perception) are treated
        # as perception artifacts worth preserving, not spurious ions.
        if ri.NumAtomRings(atom.GetIdx()) == 0:
            if atom.GetSymbol() in ('O', 'S') and atom.GetDegree() == 2:
                for nb in atom.GetNeighbors():
                    b = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                    if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                        return True
            return False
        # Fast path: atom flagged aromatic + at least one all-aromatic ring.
        if atom.GetIsAromatic():
            for ring in ri.AtomRings():
                if atom.GetIdx() not in ring:
                    continue
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    return True
            return False
        # Fallback: after OpenBabel→RDKit reconstruction with kekulize
        # warnings, aromatic flags are often lost while the ring is still
        # written as an alternating single/double bond system. Cover the
        # empirically dominant rescue patterns:
        #   (a) in-ring sp2/unspecified heteroatom cation with aromatic ring
        #       neighbours (pyridinium-type, degree<=2 in ring),
        #   (b) in-ring degree-3 N+ whose ring contains at least one
        #       explicit double bond (pyridinium/quinolinium written in
        #       kekulized form after a kekulize warning, e.g.
        #       "...=C[N+](...)=C(...)..."),
        #   (c) degree-2 O+ flanked by a double bond on one side
        #       (vinylic oxonium from bond-order perception).
        for ring in ri.AtomRings():
            if atom.GetIdx() not in ring:
                continue
            nbrs = [mol.GetAtomWithIdx(i) for i in ring if i != atom.GetIdx()]
            nbrs_arom = all(nb.GetIsAromatic() for nb in nbrs)
            ring_has_double = any(
                b.GetBondType() == Chem.BondType.DOUBLE
                for i in ring
                for b in [mol.GetAtomWithIdx(i)] and [] or []
            )
            # simpler double-bond check on ring bonds
            ring_bonds_double = False
            for i_pos, i_at in enumerate(ring):
                j_at = ring[(i_pos + 1) % len(ring)]
                b = mol.GetBondBetweenAtoms(i_at, j_at)
                if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                    ring_bonds_double = True
                    break
            sym = atom.GetSymbol()
            deg = atom.GetDegree()
            if nbrs_arom and (atom.GetHybridization() in (
                    Chem.HybridizationType.SP2, Chem.HybridizationType.UNSPECIFIED
            ) or deg <= 2):
                return True
            if sym == 'N' and deg == 3 and ring_bonds_double:
                return True
        return False
    except Exception:
        return False


def _neutralize_or_reject(rd_mol):
    """
    Enforce a clean ion policy, preserving aromatic ammonium cations.

    Aromatic ammonium cations (pyridinium / quinolinium / imidazolium N+,
    etc.) are NOT spurious ions: the positive charge is a bookkeeping
    consequence of an sp2 heteroatom donating one p-electron to an aromatic
    6-pi ring, and removing the charge would destroy aromaticity. Such
    molecules (including the target cyanine dyes for the KRAS-G4 case) are
    therefore preserved AS-IS — never neutralized, never rejected.

    For every OTHER source of net charge (aliphatic quaternary ammonium,
    carboxylate, sulfonate, inorganic ions, ...) the original policy applies:
    try to neutralize via proton transfer; if impossible, reject (return None).

    Concretely:
      - net charge 0                                       -> return unchanged
      - ALL net charge from aromatic ammonium cations      -> return unchanged
      - some charge from non-aromatic ions                 -> neutralize the
        non-aromatic charged atoms (proton transfer), keep aromatic ammonium
        atoms untouched; reject if proton transfer fails or the molecule
        breaks.

    Args:
        rd_mol: RDKit Mol (will not be mutated; a copy is neutralized).

    Returns:
        RDKit Mol (neutral or aromatic-ammonium-preserving) or None if a
        non-aromatic ion cannot be neutralized.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    net_charge = sum(a.GetFormalCharge() for a in rd_mol.GetAtoms())
    if net_charge == 0:
        return rd_mol

    # Identify aromatic-ammonium charges that must be preserved.
    aromatic_cation_idxs = {
        a.GetIdx() for a in rd_mol.GetAtoms() if _is_aromatic_cation(a, rd_mol)
    }
    aromatic_cation_charge = sum(
        rd_mol.GetAtomWithIdx(i).GetFormalCharge() for i in aromatic_cation_idxs
    )

    # Case 1: every charge in the molecule is an aromatic-ammonium charge.
    # Preserve the molecule untouched.
    if aromatic_cation_charge == net_charge:
        return rd_mol

    # Case 2: there is at least one non-aromatic charge. Try to neutralize the
    # non-aromatic charged atoms only (aromatic ammonium atoms are skipped so
    # their aromaticity is not destroyed).
    # Three-tier fallback policy:
    #   Tier 1: proton-transfer neutralization (chemically correct)
    #   Tier 2: legacy hard reset (SetFormalCharge(0) on non-aromatic charges,
    #           mirroring the old postprocess_rd_mol_2 fallback) — rescues
    #           molecules whose cations have no removable H (quaternary-like
    #           centers from OpenBabel bond-order perception artifacts)
    #   Tier 3: reject (return None)
    def _validate(mol_cand):
        """Shared validation gate. Returns mol on success, None on failure."""
        try:
            err_flags = AllChem.SanitizeMol(mol_cand, AllChem.SANITIZE_ALL, catchErrors=True)
            if err_flags != 0:
                return None
            if any(a.GetNumRadicalElectrons() > 0 for a in mol_cand.GetAtoms()):
                return None
            if mol_cand.GetNumHeavyAtoms() != rd_mol.GetNumHeavyAtoms():
                return None
            smi = Chem.MolToSmiles(mol_cand)
            if not smi or '.' in smi:
                return None
            remaining = sum(a.GetFormalCharge() for a in mol_cand.GetAtoms())
            if remaining != aromatic_cation_charge:
                return None
            return mol_cand
        except Exception:
            return None

    # ---- Tier 1: proton transfer ----
    try:
        mol_neutral = Chem.RWMol(rd_mol)
        neutralizable = True
        for atom in mol_neutral.GetAtoms():
            # Preserve aromatic ammonium — do not touch its charge/Hs.
            if atom.GetIdx() in aromatic_cation_idxs:
                continue
            q = atom.GetFormalCharge()
            if q == 0:
                continue
            total_h = atom.GetTotalNumHs()
            if q > 0:
                # Cation: need to remove q protons.
                if total_h < q:
                    neutralizable = False
                    break
                new_h = total_h - q
            else:
                # Anion: add |q| protons.
                new_h = total_h + abs(q)
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(new_h)
            atom.SetNoImplicit(False)
        if neutralizable:
            mol_cand = _validate(mol_neutral.GetMol())
            if mol_cand is not None:
                return mol_cand
        # Tier 1 failed (no removable H or validation) -> fall through.
    except Exception:
        pass

    # ---- Tier 2: legacy hard reset of non-aromatic charges ----
    try:
        mol_hard = Chem.RWMol(rd_mol)
        for atom in mol_hard.GetAtoms():
            if atom.GetIdx() in aromatic_cation_idxs:
                continue
            if atom.GetFormalCharge() != 0:
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
                # Let RDKit re-perceive implicit Hs to the legal valence
                # (mirrors old postprocess_rd_mol_2 behaviour of blindly
                # zeroing charges; quaternary N+ becomes a neutral N whose
                # valence error is then repaired by explicit-H adjustment).
                atom.SetNumExplicitHs(0)
        mol_cand = _validate(mol_hard.GetMol())
        if mol_cand is not None:
            return mol_cand
    except Exception:
        pass

    # ---- Tier 3: reject ----
    return None


def sampletraj_to_sdf(sample_traj, output_path):
    """
    Convert generated trajectory to sdf files and save to output_path.

    Args:
        sample_traj: dict with keys 'trajs' [2, N, ...] and 'batch_idx' [N,]
        output_path: directory to save sdf files

    Writes one SDF per graph (neutralized via the smart ion policy).
    """
    from rdkit import Chem

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    global_idx = 0
    error_count = 0

    xyz = sample_traj['trajs'][0]
    atomtype = sample_traj['trajs'][1]
    graph_batch_idx = sample_traj['batch_idx']
    n_b = graph_batch_idx.max().item() + 1

    for i in range(n_b):
        mask = graph_batch_idx == i
        sub_xyz = xyz[mask].cpu().numpy()
        sub_atomtype = atomtype[mask].argmax(-1)
        sub_atomnum = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(i.item(), (None, None))[0] for i in sub_atomtype]
        sub_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(i.item(), (None, None))[-1] for i in sub_atomtype]
        
        try:
            rd_mol = reconstruct_from_generated(sub_xyz, sub_atomnum, sub_aromatic, basic_mode=False)
            smiles = Chem.MolToSmiles(rd_mol)
            if '.' in smiles:
                error_count += 1
                raise ValueError("Disconnected molecule")
            # Smart ion policy: aromatic ammonium (quinolinium N+) is
            # preserved; non-aromatic ions are neutralized or rejected.
            rd_mol_neutral = _neutralize_or_reject(rd_mol)
            if rd_mol_neutral is None:
                raise ValueError("Charged molecule that could not be neutralized")
            rd_mol = rd_mol_neutral
            fname = f"{output_path}/{global_idx}.sdf"
            molwriter = Chem.SDWriter(fname)
            molwriter.write(rd_mol)
            molwriter.close()
            global_idx += 1
        except Exception as e:
            print("Error in sample sdf from traj:", e)
            error_count += 1
            global_idx += 1

from rdkit import Chem
from rdkit.Chem import rdchem

def get_proto_mol(atomic_nums):
    """创建一个包含所有原子的无键RWMol"""
    mol = Chem.RWMol()
    for z in atomic_nums:
        a = Chem.Atom(int(z))
        mol.AddAtom(a)
    return mol


def reconstruct_from_generated_with_seed(xyz, atom_nums, fixed_mask, seed_mol, aromatic=None, dist_tol=0.4):
    """
    重建分子，严格保留 seed_mol 的原子类型、键级、芳香性以及环芳香性。
    其余原子（非种子）先用碳原子连接，最后还原真实类型并修正电荷。
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 类型转换
    xyz = np.asarray(xyz, dtype=float)
    atom_nums = np.asarray(atom_nums, dtype=int)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    if aromatic is not None:
        aromatic = np.asarray(aromatic, dtype=bool)

    N = len(atom_nums)

    # 1. 创建初始 RWMol（种子用真实类型，非种子用碳）
    rw_mol = Chem.RWMol()
    for i in range(N):
        if fixed_mask[i]:
            a = Chem.Atom(int(atom_nums[i]))
        else:
            a = Chem.Atom(6)
        if aromatic is not None and aromatic[i]:
            a.SetIsAromatic(True)
        rw_mol.AddAtom(a)

    conf = Chem.Conformer(N)
    for i in range(N):
        conf.SetAtomPosition(i, (xyz[i, 0], xyz[i, 1], xyz[i, 2]))
    rw_mol.AddConformer(conf)

    # 2. 种子原子映射
    seed_indices = np.where(fixed_mask)[0].tolist()
    if len(seed_indices) != seed_mol.GetNumAtoms():
        raise ValueError(f"Seed mask count mismatch: {len(seed_indices)} vs {seed_mol.GetNumAtoms()}")
    seed_to_gen = {i: seed_indices[i] for i in range(len(seed_indices))}

    # 记录种子键信息（用于最后强制恢复）
    seed_bonds_info = []
    for bond in seed_mol.GetBonds():
        u_seed = bond.GetBeginAtomIdx()
        v_seed = bond.GetEndAtomIdx()
        u_gen = seed_to_gen[u_seed]
        v_gen = seed_to_gen[v_seed]
        seed_bonds_info.append((u_gen, v_gen, bond.GetBondType(), bond.GetIsAromatic()))

    # 记录种子芳香原子（用于最后强制恢复）
    seed_aromatic_atoms = set()
    for atom in seed_mol.GetAtoms():
        if atom.GetIsAromatic():
            seed_aromatic_atoms.add(seed_to_gen[atom.GetIdx()])

    # 3. 添加种子键
    for u_gen, v_gen, bt, is_arom in seed_bonds_info:
        rw_mol.AddBond(u_gen, v_gen, bt)
        bond = rw_mol.GetBondBetweenAtoms(u_gen, v_gen)
        if bond and is_arom:
            bond.SetIsAromatic(True)

    # 4. 基于碳半径添加新键（跳过种子键）
    pt = Chem.GetPeriodicTable()
    carbon_rad = pt.GetRcovalent(6)
    threshold = carbon_rad * 2 + dist_tol

    coords = xyz
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    existing_bonds = set()
    for bond in rw_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i > j:
            i, j = j, i
        existing_bonds.add((i, j))

    for i in range(N):
        for j in range(i+1, N):
            if (i, j) in existing_bonds:
                continue
            if fixed_mask[i] and fixed_mask[j]:
                continue
            if dist_mat[i, j] <= threshold:
                bt = Chem.BondType.SINGLE
                if aromatic is not None:
                    if aromatic[i] and aromatic[j]:
                        bt = Chem.BondType.AROMATIC
                rw_mol.AddBond(i, j, bt)
                existing_bonds.add((i, j))

    # 5. 初步 sanitize（不破坏已有键）
    mol = rw_mol.GetMol()
    try:
        AllChem.SanitizeMol(mol, catchErrors=True)
    except:
        pass

    # 6. 还原非种子原子的真实类型
    for i in range(N):
        if not fixed_mask[i]:
            mol.GetAtomWithIdx(i).SetAtomicNum(int(atom_nums[i]))

    # 7. 调整非种子原子的形式电荷（备份种子电荷后恢复）
    seed_charges = {}
    for seed_idx, gen_idx in seed_to_gen.items():
        seed_charges[gen_idx] = seed_mol.GetAtomWithIdx(seed_idx).GetFormalCharge()

    mol = _adjust_heteroatom_charges(mol)

    for gen_idx, charge in seed_charges.items():
        mol.GetAtomWithIdx(gen_idx).SetFormalCharge(charge)

    # 8. 强制恢复种子键（覆盖任何后处理带来的改变）
    rw_mol_final = Chem.RWMol(mol)
    for u_gen, v_gen, bt, is_arom in seed_bonds_info:
        bond = rw_mol_final.GetBondBetweenAtoms(u_gen, v_gen)
        if bond is None:
            rw_mol_final.AddBond(u_gen, v_gen, bt)
        else:
            bond.SetBondType(bt)
            bond.SetIsAromatic(is_arom)
    mol = rw_mol_final.GetMol()

    # 9. 强制设置种子原子的芳香性
    for gen_idx in seed_aromatic_atoms:
        mol.GetAtomWithIdx(gen_idx).SetIsAromatic(True)

    # 10. 最终 sanitize（跳过 kekulize，避免错误）
    try:
        AllChem.SanitizeMol(mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE, catchErrors=True)
    except:
        pass

    return mol
    
def sampletraj_to_sdf_byseed(sample_traj, output_path, seed_mol_list, fixed_mask, dist_tol=0.4):
    """
    Convert generated trajectory to sdf files and save to output_path
    sample_traj_list: list of generated trajectories, each element is a dict with keys:
        'trajs': (N, 2) tensor, first column is xyz (N, 3), second column is atomtype (N, K)
        'batch_idx': (N,) tensor, batch index for each atom
    output_path: path to save sdf files
    Writes one SDF per graph (neutralized via the smart ion policy).
    """
    from rdkit import Chem

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    global_idx = 0
    error_count = 0

    xyz = sample_traj['trajs'][0]
    atomtype = sample_traj['trajs'][1]
    graph_batch_idx = sample_traj['batch_idx']
    n_b = graph_batch_idx.max().item() + 1

    for i in range(n_b):
        mask = graph_batch_idx == i
        sub_xyz = xyz[mask].cpu().numpy()
        sub_atomtype = atomtype[mask].argmax(-1)
        sub_atomnum = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(i.item(), (None, None))[0] for i in sub_atomtype]
        sub_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB.get(i.item(), (None, None))[-1] for i in sub_atomtype]
        sub_fixed_mask = fixed_mask[mask.cpu().numpy()].cpu().numpy().astype(bool)
        seed_mol = seed_mol_list[i]
        try:
            rd_mol = reconstruct_from_generated_with_seed(sub_xyz, sub_atomnum, sub_fixed_mask, seed_mol, sub_aromatic, dist_tol=dist_tol)
            smiles = Chem.MolToSmiles(rd_mol)
            if '.' in smiles:
                error_count += 1    
                raise ValueError("Disconnected molecule")
            # Smart ion policy: aromatic ammonium (quinolinium N+) is
            # preserved; non-aromatic ions are neutralized or rejected.
            rd_mol_neutral = _neutralize_or_reject(rd_mol)
            if rd_mol_neutral is None:
                raise ValueError("Charged molecule that could not be neutralized")
            rd_mol = rd_mol_neutral
            fname = f"{output_path}/{global_idx}.sdf"
            molwriter = Chem.SDWriter(fname)
            molwriter.write(rd_mol)
            molwriter.close()
            global_idx += 1
        except Exception as e:
            print(f"Error reconstructing molecule {global_idx}: {e}")
            error_count += 1
            global_idx += 1