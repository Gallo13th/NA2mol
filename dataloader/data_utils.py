#!/usr/bin/env python
# coding=utf-8
import os
import pickle
from rdkit import Chem


def convert_to_file(pickle_path, out_dir, max_rows = -1, file_list=None):
    with open(pickle_path, "rb") as f:
        rows_list = pickle.load(f)
    if max_rows > 0:
        max_num = min(len(rows_list),max_rows)
    else:
        max_num = len(rows_list)
    acc_num = 0
    for row in rows_list:
        if acc_num >= max_num: break
        mol_name = row["mol_smiles"].replace('/','_')
        with open(os.path.join(out_dir, f"{mol_name}.sdf"),"w") as f:
            f.write(row["mol_conf_sdf"])
        for idx, resolution in enumerate(row["mol_ed_res"]):
            with open(os.path.join(out_dir, f"{mol_name}_res{resolution}.pdb"), "w") as f:
                f.write(row["mol_ed_pdb"][idx])
        file_list.append(mol_name + '\n')

        acc_num += 1
                
# def ed_pdb_read(ed_pdb_path):
#     mol = Chem.MolFromPDBFile(ed_pdb_path, sanitize=False, removeHs=False)
#     if mol is None:
#         print("错误：无法解析 PDB 文件")
#         return
#     # 获取第一个构象的坐标（PDB通常只有一个构象）
#     conf = mol.GetConformer()
#     # 遍历所有原子
#     cnt = 0
#     elements = []
#     poss = []
#     for atom in mol.GetAtoms():
#         # 获取原子类型（元素符号）
#         element = atom.GetSymbol()
        
#         # 获取坐标
#         pos = conf.GetAtomPosition(atom.GetIdx())
#         x, y, z = pos.x, pos.y, pos.z
#         cnt += 1
#         elements.append(element)
#         poss.append([x, y, z])
#     return elements, poss
#     #     print(f"原子类型：{element}，坐标：({x:.3f}, {y:.3f}, {z:.3f})")
#     # print(cnt)

import numpy as np
from Bio.PDB.PDBParser import PDBParser

def ed_pdb_read(ed_pdb_path):
    """
    Read a PDB file using Bio.PDB and extract atom elements and coordinates efficiently.
    
    Args:
        ed_pdb_path (str): Path to the PDB file.
    
    Returns:
        tuple: (elements, positions) where elements is a list of atom symbols (str)
               and positions is a NumPy array of shape (n_atoms, 3) with x, y, z coordinates.
    
    Raises:
        ValueError: If the PDB file cannot be parsed or contains no atoms.
    """
    # Initialize parser
    parser = PDBParser(QUIET=True)
    
    try:
        # Parse PDB file
        structure = parser.get_structure('pdb', ed_pdb_path)
    except Exception as e:
        raise ValueError(f"Cannot parse PDB file {ed_pdb_path}: {str(e)}")
    
    # Extract atoms from the first model
    atoms = list(structure.get_atoms())  # Includes ATOM and HETATM
    n_atoms = len(atoms)
    
    if n_atoms == 0:
        raise ValueError(f"No atoms found in PDB file: {ed_pdb_path}")
    
    # Pre-allocate arrays
    elements = []
    positions = np.empty((n_atoms, 3), dtype=np.float32)
    
    # Extract element symbols and coordinates
    for i, atom in enumerate(atoms):
        elements.append(atom.element)
        positions[i] = atom.get_coord()
    
    return elements, positions

if __name__ == '__main__':
    file_list = []
    for i in range(99,100):
        p = f'code/Lingo3DMol-main/datasets/pretrain_10w_ed/ed_mol_chunk{i}.pkl'
        convert_to_file(p, "code/Lingo3DMol-main/datasets/valset", max_rows=-1, file_list=file_list)
    # with open('/data00/jiahao/Lingo3DMol-main/datasets/smiles_id.txt', 'w') as f:
    #     f.writelines(file_list)
    # ed_pdb_read("/data00/jiahao/Lingo3DMol-main/datasets/read_pickle_output/Cc1cc(C)c(NS(C)(=O)=O)c(C)c1NC(=O)C1Cc2ccccc2CN1C(=O)C(C)(C)C_res1.5.pdb")