#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name :    tools/ligand_code_util.py
Description :   
Author :       Zhouwenbiao
Create Time :   2025-04-05
-------------------------------------------------
"""

import os
from io import StringIO
from rdkit.Chem import ChemicalFeatures, AllChem, rdPartialCharges
from Bio.PDB import PDBParser
import os
import warnings
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdGeometry
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit import RDLogger
import sys, os
from Bio.Data import PDBData
RDLogger.DisableLog('rdApp.*')
RDLogger.logger().setLevel(RDLogger.ERROR)
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_c_stderr():
    """Suppress C-level stderr (e.g., from RDKit C++ code)."""
    fd_stderr = sys.stderr.fileno()
    fd_null = os.open(os.devnull, os.O_WRONLY)
    # Save original stderr
    saved_stderr = os.dup(fd_stderr)
    try:
        os.dup2(fd_null, fd_stderr)
        yield
    finally:
        os.dup2(saved_stderr, fd_stderr)
        os.close(fd_null)
        os.close(saved_stderr)
ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
'''
Parser from MolCRAFT
@article{qu2024molcraft,
  title={MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space},
  author={Qu, Yanru and Qiu, Keyue and Song, Yuxuan and Gong, Jingjing and Han, Jiawei and Zheng, Mingyue and Zhou, Hao and Ma, Wei-Ying},
  journal={ICML 2024},
  year={2024}
}

@article{song2024unified,
  title={Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks},
  author={Song, Yuxuan and Gong, Jingjing and Qu, Yanru and Zhou, Hao and Zheng, Mingyue and Liu, Jingjing and Ma, Wei-Ying},
  journal={ICLR 2024},
  year={2024}
}

git repo: https://github.com/AlgoMole/MolCRAFT/tree/master/MolCRAFT
'''

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")
from math import isnan,isinf

def parse_sdf_file(path=None,block=None):
    from rdkit import RDLogger
    # allow_atom_index = {6,7,8,9,15,16,17}
    RDLogger.DisableLog('rdApp.*')
    # fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    # factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    if isinstance(path, str):
        # read mol
        if path.endswith('.sdf'):
            rdmol = Chem.MolFromMolFile(path, sanitize=False)
        elif path.endswith('.mol2'):
            rdmol = Chem.MolFromMol2File(path, sanitize=False)
        else:
            raise ValueError
    elif isinstance(block, str):
        rdmol = Chem.MolFromMolBlock(block, sanitize=False)
    else:
        raise ValueError('Unknown type of path: %s' % type(path))
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    if rd_num_atoms >= 100:
        raise ValueError
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.int8)
    with suppress_c_stderr():
        try:
            for feat in factory.GetFeaturesForMol(rdmol):
                feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
        except Exception:
            # 可选：记录或忽略
            pass

    # Get hybridization in the order of atom idx.
    # calculate partial charge
    rdPartialCharges.ComputeGasteigerCharges(rdmol)
    hybridization = []
    partialcharges = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        # charge = atom.GetProp("_GasteigerCharge")
        hybridization.append((idx, hybr))
        # partialcharges.append((idx, float(charge)))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]
    partialcharges = sorted(partialcharges)
    partialcharges = [v[1] for v in partialcharges]
    
    # ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    # accum_pos = 0
    # accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        # assert atom_num in allow_atom_index, print(f"Atom Num :{atom_num}")
        element.append(atom_num)
        # atom_weight = ptable.GetAtomicWeight(atom_num)
        # accum_pos += pos[atom_idx] * atom_weight
        # accum_mass += atom_weight
    # center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int32)

    
    
    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.int32)
    edge_type = np.array(edge_type, dtype=np.int32)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'partialcharges': partialcharges
    }
    return data

class FastLigandCodeUtil():
    
    def __init__(self):
        self.atom_list = "C|N|O|S|F|Cl|Br|I|P|H|B|Unknown"
        self.atom_vocab_c2i = {atom: i for i, atom in enumerate(self.atom_list.split("|"))}
        self.atom_vocab_i2c = {i: atom for atom, i in self.atom_vocab_c2i.items()}
        self.resolution = 0.1
        self.grid_size = 240
        
    def encode(self,sdf_path):
        mol = Chem.SDMolSupplier(sdf_path)[0]
        # get all atom pos and type
        atom_pos = []
        atom_types = []
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_pos.append([pos.x, pos.y, pos.z])
            atom_types.append(self.atom_vocab_c2i.get(atom.GetSymbol(), self.atom_vocab_c2i["Unknown"]))
        atom_pos = np.array(atom_pos)
        atom_types = np.array(atom_types)

        return atom_pos, atom_types

def parse_pdb_file(path=None, block=None):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    # Parse with RDKit first
    if isinstance(path, str):
        if path.endswith('.pdb'):
            rdmol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=False)
        else:
            raise ValueError("Unsupported file format for PDB parser.")
    elif isinstance(block, str):
        rdmol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    else:
        raise ValueError('Unknown type of path: %s' % type(path))

    if rdmol is None:
        raise ValueError("Failed to parse the PDB file.")

    try:
        # Try standard sanitization
        Chem.SanitizeMol(rdmol)
    except Exception as e:
        # If standard fails, try more lenient approach
        try:
            Chem.SanitizeMol(rdmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                          Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                          Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                          Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                          Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                          Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=False)
        except Exception as e:
            # print(f"Error in sanitization: {e}")
            return None

    if not rdmol.GetConformers():
        return None

    # Now remove Hs
    try:
        rdmol = Chem.RemoveHs(rdmol, sanitize=False)
    except Exception as e:
        # print(f"RemoveHs failed: {e}")
        return None

    rd_num_atoms = rdmol.GetNumAtoms()
    if rd_num_atoms == 0:
        return None
    
    # 获取残基信息
    residues = []
    residue_ids = []
    residue_names = []
    chain_ids = []
    
    for atom in rdmol.GetAtoms():
        # 获取原子信息
        info = atom.GetPDBResidueInfo()
        if info:
            # 残基名称 (如 LYS, ALA, HEM等)
            res_name = info.GetResidueName().strip()
            res_name = res_name + ' ' * (3 - len(res_name))  # 补齐到3个字符
            if res_name in PDBData.protein_letters_3to1_extended:
                res_name = 'PR' + PDBData.protein_letters_3to1_extended[res_name]
            elif res_name in PDBData.nucleic_letters_3to1_extended:
                res_name = 'NT' + PDBData.nucleic_letters_3to1_extended[res_name]
            else:
                res_name = "UNK"
            # 残基编号
            res_id = info.GetResidueNumber()
            # 链标识符
            chain_id = info.GetChainId()
        else:
            # 如果没有PDB信息，使用默认值
            res_name = "UNK"
            res_id = 0
            chain_id = "A"
        
        residue_names.append(res_name)
        residue_ids.append(res_id)
        chain_ids.append(chain_id)
        
        # 创建唯一的残基标识符
        residue_key = f"{chain_id}:{res_id}:{res_name}"
        residues.append(residue_key)
    
    # 转换为numpy数组
    residue_names = np.array(residue_names, dtype='U3')  # U3表示最多3个字符的Unicode字符串
    residue_ids = np.array(residue_ids, dtype=np.int32)
    chain_ids = np.array(chain_ids, dtype='U1')  # U1表示1个字符的Unicode字符串
    residues = np.array(residues, dtype='U50')  # U50表示最多50个字符的Unicode字符串
    
    # Feature matrix - add backbone as an additional feature
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES) + 1], dtype=np.int8)  # +1 for backbone
    with suppress_c_stderr():
        try:
            for feat in factory.GetFeaturesForMol(rdmol):
                feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
        except Exception as e:
            # print(f"Feature generation warning: {str(e)}")
            pass

    # Hybridization state
    hybridization = []
    partialcharges = []
    try:
        rdPartialCharges.ComputeGasteigerCharges(rdmol)
    except Exception as e:
        # print(f"Charge calculation warning: {str(e)}")
        pass
    
    for atom in rdmol.GetAtoms():
        try:
            hybr = str(atom.GetHybridization())
        except:
            hybr = 'UNSPECIFIED'
        try:
            charge = atom.GetProp("_GasteigerCharge")
            if isnan(float(charge)) or isinf(float(charge)):
                charge = 0.0
            else:
                charge = float(charge)
        except:
            charge = 0
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
        partialcharges.append((idx, float(charge)))
    
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]
    partialcharges = sorted(partialcharges)
    partialcharges = np.array([v[1] for v in partialcharges], dtype=np.float16)

    # Atomic elements and centroid
    try:
        pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float16)
    except:
        return None
        
    element = []
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)

    element = np.array(element, dtype=np.int16)

    data = {
        'element': element,
        'pos': pos,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'partialcharges': partialcharges,
        'residue_name': residue_names,  # 残基名称 (3字母代码)
        'residue_id': residue_ids,      # 残基编号
        'chain_id': chain_ids,          # 链标识符
        'residue': residues,            # 唯一残基标识符
    }
    
    # 添加额外的残基信息统计
    unique_residues = np.unique(residues)
    data['num_residues'] = len(unique_residues)
    
    if len(pos) != len(hybridization):
        return None
    return data

if __name__ == "__main__":
    import copy

    ori_smiles='COC1CC(C(=O)N(c2ccc(S(F)(F)(F)(F)F)cc2)C(C(=O)N(CCN2CCC2)C2CC2)c2cccnc2)N(C#N)C1'
    mol = Chem.MolFromSmiles('../EDmol/data/mol_sdf/2ano_ligand.sdf')
    print(parse_sdf_file('../EDmol/data/mol_sdf/2ano_ligand.sdf'))
    print(parse_pdb_file('../EDmol/data/pocket_pdb/2ano.pdb'))
