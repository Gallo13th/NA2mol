import pickle
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, AllChem, rdMolAlign
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Geometry import Point3D
import os
import gzip
import prolif as plf
from rdkit import Chem

def custom_break_brics_bonds(mol):
    return BRICS.BreakBRICSBonds(mol)
    # Custom atom types from RDKit BRICS (SMARTS for environments)
    atom_types = {
        1: '[C;D3]([#0,#6,#7,#8])(=O)',
        3: '[O;D2]-;!@[#0,#6,#1]',
        4: '[C;!D1;!$(C=*)]-;!@[#6]',
        5: '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',
        6: '[C;D3;!R](=O)-;!@[#0,#6,#7,#8]',
        7: '[C;D2,D3]-[#6]',
        8: '[C;!R;!D1;!$(C!-*)]',
        9: '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]',
        10: '[N;R;$(N(@C(=O))@[C,N,O,S])]',
        11: '[S;D2](-;!@[#0,#6])',
        12: '[S;D4]([#6,#0])(=O)(=O)',
        13: '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
        14: '[c;$(c(:[c,n,o,s]):[n,o,s])]',
        15: '[C;$(C(-;@C)-;@C)]',
        16: '[c;$(c(:c):c)]',
    }

    # Custom bond pairs (label1, label2, bond_symbol) - subset excluding L7 (alkenes), L16, L14 (14-14,16-16), and parts of L8/L10 for fewer breaks
    custom_bond_pairs = [
        # L1
        (1, 3, '-;!@'),
        (1, 5, '-;!@'),
        (1, 10, '-;!@'),
        # L3 (partial, exclude some)
        (3, 4, '-;!@'),
        (3, 13, '-;!@'),
        (3, 14, '-;!@'),
        (3, 15, '-;!@'),
        # (3, 16, '-;!@'),  # excluded for larger fragments
        # L4
        (4, 5, '-;!@'),
        (4, 11, '-;!@'),
        # L5
        (5, 12, '-;!@'),
        (5, 14, '-;!@'),
        # (5, 16, '-;!@'),  # excluded
        (5, 13, '-;!@'),
        (5, 15, '-;!@'),
        # L6
        (6, 13, '-;!@'),
        (6, 14, '-;!@'),
        (6, 15, '-;!@'),
        # (6, 16, '-;!@'),  # excluded
        # L8 (partial)
        (8, 9, '-;!@'),
        (8, 10, '-;!@'),
        (8, 13, '-;!@'),
        (8, 14, '-;!@'),
        # (8, 15, '-;!@'),  # excluded
        # (8, 16, '-;!@'),  # excluded
        # L9
        (9, 13, '-;!@'),
        (9, 14, '-;!@'),
        (9, 15, '-;!@'),
        # (9, 16, '-;!@'),  # excluded
        # L10 (partial)
        (10, 13, '-;!@'),
        (10, 14, '-;!@'),
        # (10, 15, '-;!@'),  # excluded
        # (10, 16, '-;!@'),  # excluded
        # L11
        (11, 13, '-;!@'),
        (11, 14, '-;!@'),
        (11, 15, '-;!@'),
        # (11, 16, '-;!@'),  # excluded
        # L13
        (13, 14, '-;!@'),
        (13, 15, '-;!@'),
        # (13, 16, '-;!@'),  # excluded
        # L14 (partial)
        # (14, 14, '-;!@'),  # excluded
        (14, 15, '-;!@'),
        # (14, 16, '-;!@'),  # excluded
        # L15
        # (15, 16, '-;!@'),  # excluded
        # L16 excluded entirely
    ]

    bonds = []
    for l1, l2, bond_sym in custom_bond_pairs:
        smarts = f'[$({atom_types[l1]})]{bond_sym}[$({atom_types[l2]})]'
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            a1, a2 = match
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            bonds.append(((a1, a2), (str(l1), str(l2)))) 

    if not bonds:
        return mol  # No breaks, return original

    # Break with custom bonds (symmetric labels)
    # dummy_labels = [(l1, l2) for l1, l2, _ in custom_bond_pairs] + [(l2, l1) for l1, l2, _ in custom_bond_pairs]
    try:
        return BRICS.BreakBRICSBonds(mol, bonds=bonds)
    except Exception as e:
        assert 0, (mol, bonds, e)

def get_nci_info(rd_protein, rd_ligand):
    # prot = plf.Molecule.from_rdkit(Chem.MolFromPDBFile(protein, removeHs=False))
    # rd_protein = Chem.AddHs(rd_protein, addCoords=True)
    prot = plf.Molecule.from_rdkit(rd_protein)
    # prot_h = Chem.AddHs(prot, addCoords=True)
    # lig = plf.Molecule.from_rdkit(Chem.SDMolSupplier(ligand, removeHs=False)[0])
    rd_ligand = Chem.AddHs(rd_ligand, addCoords=True)
    lig = plf.Molecule.from_rdkit(rd_ligand)
    # lig_h = Chem.AddHs(lig, addCoords=True)
    fp = plf.Fingerprint()
    fp.run_from_iterable([lig], prot)

    # 输出 DataFrame
    df = fp.to_dataframe()

    # 提取第一帧（也是唯一一帧）
    df_frame = df.iloc[0]  # Series with MultiIndex (residue, interaction)

    # 将 MultiIndex 展开为两层：residue 和 interaction
    df_long = df_frame.unstack(level=1)  # residue 为行，interaction 为列

    # 替换 NaN 为 False（因为 unstack 后缺失组合会变成 NaN）
    df_long = df_long.fillna(False).astype(bool)
    return df_long

def get_nci_info_batch(rd_protein, rd_ligands):
    prot = plf.Molecule.from_rdkit(rd_protein)
    ligs = []
    for rd_ligand in rd_ligands:
        rd_ligand = Chem.AddHs(rd_ligand, addCoords=True)
        lig = plf.Molecule.from_rdkit(rd_ligand)
        ligs.append(lig)
    fp = plf.Fingerprint()
    fp.run_from_iterable(ligs, prot)

    # 输出 DataFrame
    df = fp.to_dataframe()
    dfs = []
    # 提取第一帧（也是唯一一帧）
    for i in range(len(rd_ligands)):
        df_frame = df.iloc[i]  # Series with MultiIndex (residue, interaction)
    
        # 将 MultiIndex 展开为两层：residue 和 interaction
        df_long = df_frame.unstack(level=1)  # residue 为行，interaction 为列
    
        # 替换 NaN 为 False（因为 unstack 后缺失组合会变成 NaN）
        df_long = df_long.fillna(False).astype(bool)
        dfs.append(df_long)
        
    return dfs

# ==========================================
# 工具函数（保持不变或小幅优化）
# ==========================================

def get_skeleton_key(mol):
    """生成拓扑骨架Key（与构建库时一致）"""
    if mol is None:
        return None
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in m.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)


def extract_submol(mol, atom_indices):
    """提取子结构"""
    rw = Chem.RWMol(mol)
    atoms_to_remove = [i for i in range(rw.GetNumAtoms()) if i not in atom_indices]
    for idx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def remove_dummy_atoms(mol):
    """移除虚拟原子（*）"""
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    clean = rw.GetMol()
    try:
        Chem.SanitizeMol(clean)
    except:
        pass
    return clean


def generate_fingerprint(mol, radius=3, nbits=2048):
    """使用 radius=3 提高对环大小的区分度"""
    if mol is None:
        return None
    Chem.GetSymmSSSR(mol)
    # fpgen = AllChem.GetMorganFingerprintGenerator(radius=radius, fpSize=nbits)
    fp = GetMorganFingerprintAsBitVect(mol,2)
    return fp

def prepare_seed_fragments(mol):
    """
    从完整分子中提取片段，筛选条件：
    1. 可旋转键 ≤ 3
    2. 重原子数在3-15之间
    
    返回符合条件的片段列表（每个片段都是独立的分子对象）
    
    参数:
        mol: RDKit分子对象
    
    返回:
        list: 符合条件的片段分子对象列表
        list: 每个片段的详细信息列表
    """
    if mol is None:
        return None, []
    
    # 辅助函数：删除虚拟原子
    def remove_dummy_atoms(fragment_mol):
        """删除片段中的虚拟原子（原子序号为0的原子）"""
        if fragment_mol is None:
            return None
        
        rw_mol = Chem.RWMol(fragment_mol)
        atoms_to_remove = []
        
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # 虚拟原子
                atoms_to_remove.append(atom.GetIdx())
        
        for idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.RemoveAtom(idx)
        
        clean_mol = rw_mol.GetMol()
        
        try:
            Chem.SanitizeMol(clean_mol)
        except:
            pass
        
        return clean_mol
    
    # 辅助函数：计算可旋转键数量
    def count_rotatable_bonds(fragment_mol):
        """计算片段的可旋转键数量（排除末端旋转和酰胺键）"""
        return AllChem.CalcNumRotatableBonds(fragment_mol)
    
    # 辅助函数：计算重原子数量
    def count_heavy_atoms(fragment_mol):
        """计算重原子数量（非氢原子）"""
        count = 0
        for atom in fragment_mol.GetAtoms():
            if atom.GetAtomicNum() > 1:  # 排除氢和虚拟原子
                count += 1
        return count
    
    # 尝试BRICS切分
    try:
        frags_mols = BRICS.BreakBRICSBonds(mol)
        # frags_mols = custom_break_brics_bonds(mol)
        fragments_indices = Chem.GetMolFrags(frags_mols, asMols=False)
    except Exception as e:
        print(f"Error in BRICS fragmentation in PREPARE SEED: {e}")
        print("Current frags mols:",frags_mols)
        return None, []
    
    qualified_fragments = []
    fragment_details = []
    
    # 处理每个片段
    for frag_id, atom_indices in enumerate(fragments_indices):
        # 提取片段分子
        fragment_mol = extract_submol(frags_mols, atom_indices)
        
        if fragment_mol is None or fragment_mol.GetNumAtoms() == 0:
            continue
        
        # 删除虚拟原子
        clean_fragment = remove_dummy_atoms(fragment_mol)
        
        if clean_fragment is None or clean_fragment.GetNumAtoms() == 0:
            continue
        
        # 计算片段属性
        num_rotatable_bonds = count_rotatable_bonds(clean_fragment)
        num_heavy_atoms = count_heavy_atoms(clean_fragment)
        
        # 检查筛选条件 - 注意：文档说3-15，但代码中是5-15，这里保持代码逻辑
        if num_rotatable_bonds <= num_heavy_atoms // 4 and 5 <= num_heavy_atoms <= 15:
            fragment_info = {
                'fragment_id': frag_id,
                'fragment_mol': clean_fragment,
                'num_rotatable_bonds': num_rotatable_bonds,
                'num_heavy_atoms': num_heavy_atoms,
                'smiles': Chem.MolToSmiles(clean_fragment),
                'atom_indices': list(atom_indices)
            }
            
            # 将清洁的片段分子对象添加到列表中
            qualified_fragments.append(clean_fragment)
            fragment_details.append(fragment_info)
            
            # print(f"✓ Fragment {frag_id}: {num_heavy_atoms} heavy atoms, {num_rotatable_bonds} rotatable bonds")
            # print(f"  SMILES: {fragment_info['smiles']}")
        else:
            # print(f"✗ Fragment {frag_id} filtered out: {num_heavy_atoms} heavy atoms, {num_rotatable_bonds} rotatable bonds")
            pass
    
    # 如果没有符合条件的片段，返回None
    if not qualified_fragments:
        # print("No fragments meet the criteria")
        return None, []
    
    # 返回片段分子对象列表（限制最多3个，如原代码所示）
    # max_fragments = min(3, len(qualified_fragments))
    
    
    return qualified_fragments, fragment_details

def extract_and_clean_fragments(mol):
    """
    输入完整分子，返回处理后的片段集合（去掉连接点虚拟原子，仅保留重原子）
    
    参数:
        mol: RDKit分子对象
    
    返回:
        list: 每个元素是一个字典，包含清理后的片段信息
            {
                'fragment_id': 片段ID,
                'clean_mol': 清理后的RDKit分子对象（不含虚拟原子）,
                'clean_molblock': 清理后的molblock字符串,
                'original_smiles': 原始片段SMILES（含虚拟原子）,
                'clean_smiles': 清理后的SMILES（不含虚拟原子）,
                'atom_indices': 在原分子中的原子索引列表
            }
    """
    if mol is None:
        return []
    
    result_fragments = []
    
    try:
        # 1. BRICS 切分
        frags_mols = BRICS.BreakBRICSBonds(mol)
        # frags_mols = custom_break_brics_bonds(mol)
        fragments_indices = Chem.GetMolFrags(frags_mols, asMols=False)
    except Exception as e:
        print(f"Error in BRICS fragmentation in EXTRACT AND CLEAN FRAGEMENT: {e}")
        return []
    
    # 辅助函数：删除虚拟原子
    def remove_dummy_atoms(fragment_mol):
        """删除片段中的虚拟原子（原子序号为0的原子）"""
        if fragment_mol is None:
            return None
            
        # 创建分子的可编辑版本
        rw_mol = Chem.RWMol(fragment_mol)
        
        # 查找并标记要删除的原子
        atoms_to_remove = []
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # 虚拟原子
                atoms_to_remove.append(atom.GetIdx())
        
        # 从后往前删除原子，避免索引变化
        for idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.RemoveAtom(idx)
        
        # 获取清理后的分子
        clean_mol = rw_mol.GetMol()
        
        # 尝试sanitize分子
        try:
            Chem.SanitizeMol(clean_mol)
        except:
            # 如果sanitization失败，仍然返回分子
            pass
        
        return clean_mol
    
    # 2. 处理每个片段
    for frag_id, atom_indices in enumerate(fragments_indices):
        # 提取片段分子
        fragment_mol = extract_submol(frags_mols, atom_indices)
        
        if fragment_mol is None or fragment_mol.GetNumAtoms() == 0:
            continue
        
        # 3. 删除虚拟原子
        clean_fragment = remove_dummy_atoms(fragment_mol)
        
        if clean_fragment is None or clean_fragment.GetNumAtoms() == 0:
            continue
        
        # 4. 准备片段信息
        fragment_info = {
            'fragment_id': frag_id,
            'clean_mol': clean_fragment,
            'clean_molblock': Chem.MolToMolBlock(clean_fragment),
            'original_smiles': Chem.MolToSmiles(fragment_mol),
            'clean_smiles': Chem.MolToSmiles(clean_fragment),
            'atom_indices': list(atom_indices),  # 原分子中的原子索引
            'num_atoms': clean_fragment.GetNumAtoms(),
            'num_dummy_atoms_removed': fragment_mol.GetNumAtoms() - clean_fragment.GetNumAtoms()
        }
        
        result_fragments.append(fragment_info)
    
    return result_fragments


_smiles_to_mol_cache = {}

def _get_cached_target_mol(smiles):
    if smiles not in _smiles_to_mol_cache:
        _smiles_to_mol_cache[smiles] = Chem.MolFromSmiles(smiles)
    return _smiles_to_mol_cache[smiles]


def extract_substructure_with_conformation(full_mol, target_smiles, add_hs=False):
    """
    从完整分子中提取与给定SMILES完全匹配的子结构，并保留构象。

    参数:
        full_mol: 完整的有构象的RDKit分子对象
        target_smiles: 要匹配的SMILES字符串
        add_hs: 是否在提取后添加氢原子

    返回:
        tuple: (子结构分子对象, 详细信息字典)
            - 子结构分子对象（带3D构象）
            - 详细信息包含: 'smiles', 'atom_indices', 'heavy_atoms', 'num_atoms'
            - 如果没有匹配返回 (None, {})
    """
    if full_mol is None or not target_smiles:
        return None, {}

    if full_mol.GetNumAtoms() == 0:
        return None, {}

    num_conformers = full_mol.GetNumConformers()
    if num_conformers == 0:
        return None, {}

    target_mol = _get_cached_target_mol(target_smiles)
    if target_mol is None:
        return None, {}

    matches = full_mol.GetSubstructMatches(target_mol)
    if not matches:
        return None, {}

    match = matches[0]
    atom_indices = list(match)
    n_atoms = len(atom_indices)

    e_mol = Chem.EditableMol(full_mol)
    atoms_to_remove = [i for i in range(full_mol.GetNumAtoms()) if i not in atom_indices]
    for idx in sorted(atoms_to_remove, reverse=True):
        e_mol.RemoveAtom(idx)

    submol = e_mol.GetMol()

    if add_hs:
        submol = Chem.AddHs(submol, addCoords=True)

    if num_conformers > 0:
        source_conf = full_mol.GetConformer(0)
        submol_conf = Chem.Conformer(n_atoms)
        for i, idx in enumerate(atom_indices):
            pos = source_conf.GetAtomPosition(idx)
            submol_conf.SetAtomPosition(i, (pos.x, pos.y, pos.z))
        submol.AddConformer(submol_conf)

    try:
        Chem.SanitizeMol(submol)
    except:
        pass

    details = {
        'smiles': target_smiles,
        'atom_indices': atom_indices,
        'heavy_atoms': target_mol.GetNumHeavyAtoms(),
        'num_atoms': target_mol.GetNumAtoms()
    }

    return submol, details


def extract_matching_substructures_with_conformation(full_mol, target_smiles_list, max_results=None, add_hs=False):
    """
    从完整分子中提取与给定SMILES列表中某个分子完全匹配的子结构，并保留构象。

    参数:
        full_mol: 完整的有构象的RDKit分子对象
        target_smiles_list: 一组SMILES列表
        max_results: 最大返回数量，None表示返回所有匹配
        add_hs: 是否在提取后添加氢原子

    返回:
        tuple: (片段分子对象列表, 片段详细信息列表)
            - 片段分子对象列表（带3D构象）
            - 详细信息列表，每个字典包含:
                - 'fragment_id': 片段ID
                - 'fragment_mol': 片段分子对象
                - 'num_rotatable_bonds': 可旋转键数量
                - 'num_heavy_atoms': 重原子数量
                - 'smiles': SMILES字符串
                - 'atom_indices': 原子索引列表
            - 如果没有匹配的片段，返回 (None, [])
    """
    if full_mol is None or not target_smiles_list:
        return None, []

    if full_mol.GetNumAtoms() == 0:
        return None, []

    num_conformers = full_mol.GetNumConformers()
    if num_conformers == 0:
        return None, []

    qualified_fragments = []
    fragment_details = []

    for frag_id, target_smiles in enumerate(target_smiles_list):
        submol, detail = extract_substructure_with_conformation(full_mol, target_smiles, add_hs)
        if submol is not None:
            num_rotatable_bonds = AllChem.CalcNumRotatableBonds(submol)
            num_heavy_atoms = submol.GetNumHeavyAtoms()

            fragment_info = {
                'fragment_id': frag_id,
                'fragment_mol': submol,
                'num_rotatable_bonds': num_rotatable_bonds,
                'num_heavy_atoms': num_heavy_atoms,
                'smiles': detail['smiles'],
                'atom_indices': detail['atom_indices']
            }

            qualified_fragments.append(submol)
            fragment_details.append(fragment_info)

            if max_results is not None and len(qualified_fragments) >= max_results:
                break

    if not qualified_fragments:
        return None, []

    return qualified_fragments, fragment_details


import stat
import subprocess

def Minimize(input_sdf, pdb, workdir):
    smina = "./util/smina.static"
    if not os.access(smina, os.X_OK):
        os.chmod(smina, os.stat(smina).st_mode | stat.S_IXUSR)
    # tmpdir = os.path.join(workdir, 'tmp')
    output_sdf = os.path.join(workdir, input_sdf.split("/")[-1].split(".sdf")[0]+"_lig_score.sdf")
    command = f"{smina} -r {pdb} -l {input_sdf} -o {output_sdf} --minimize --minimize_iters 1000 --cpu 1" #  >> {workdir}/smina.log 2>&1"
    # print("Running minimization command:", command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    #print(result)
    return output_sdf

# ==========================================
# 新版 MoleculeRefiner（适配 index.pkl）
# ==========================================

class MoleculeRefiner:
    def __init__(self, index_path="fragment_library/index.pkl", top_n_skeletons=None):
        print("Loading fragment index...")
        if index_path.endswith('.gz'):
            with gzip.open(index_path, 'rb') as f:
                self.full_index = pickle.load(f)
        else:
            with open(index_path, 'rb') as f:
                self.full_index = pickle.load(f)
    
        # Step 1: 获取原始数据
        skel_freq_list = self.full_index['skel_freq']  # 已按 freq 降序排序的 [(skel, freq), ...]
        
        # Step 2: 限制骨架数量（如果指定）
        if top_n_skeletons is not None and top_n_skeletons > 0:
            top_skel_items = skel_freq_list[:top_n_skeletons]
            allowed_skel_keys = set(skel for skel, _ in top_skel_items)
            print(f"Restricting to top {top_n_skeletons} skeletons (min freq: {top_skel_items[-1][1]})")
        else:
            allowed_skel_keys = set(skel for skel, _ in skel_freq_list)
            print("Using all skeletons in index.")
    
        # Step 3: 构建过滤后的 skel_freq 字典
        self.skel_freq = {skel: freq for skel, freq in skel_freq_list if skel in allowed_skel_keys}
    
        # Step 4: 过滤 skel_to_frags —— 只保留允许的骨架
        self.skel_to_frags = {
            skel: frag_list
            for skel, frag_list in self.full_index['skel_to_frags'].items()
            if skel in allowed_skel_keys
        }
    
        # Step 5: 收集所有需要保留的 full_smiles
        allowed_smiles = set()
        for frag_list in self.skel_to_frags.values():
            allowed_smiles.update(frag_list)
    
        # Step 6: 过滤 fragments —— 只保留属于高频频骨架的 SMILES
        self.fragments = {
            smiles: info
            for smiles, info in self.full_index['fragments'].items()
            if smiles in allowed_smiles
        }
    
        print(f"Filtered index loaded:")
        print(f"  • {len(self.skel_freq):,} / {len(skel_freq_list)} skeletons retained")
        print(f"  • {len(self.fragments):,} unique fragment SMILES retained")
        # assert 0

    def get_candidates_by_skel(self, skel_key):
        """根据骨架key获取所有候选构象（molblock列表）"""
        
        if skel_key not in self.skel_to_frags:
            return []
        # print(self.skel_to_frags.keys())
        candidates = []
        for smiles in self.skel_to_frags[skel_key]:

            info = self.fragments.get(smiles, {})
            # print('FOUND SMILES:',smiles,info.get('molblocks', []))
            for molblock in info.get('molblocks', []):
                candidates.append({
                    'smiles': smiles,
                    'molblock': molblock,
                    'freq': self.skel_freq.get(skel_key, 0)
                })
        # print(f"FOUND {len(candidates)} CANDIDATES")
        return candidates

    def get_atom_mapping(self, input_frag, lib_mol):
        """建立骨架原子对应关系"""
        def make_skel(m):
            m = Chem.Mol(m)
            for a in m.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetIsotope(0)
                a.SetIsAromatic(False)
            for b in m.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
            return m

        skel_input = make_skel(input_frag)
        skel_lib = make_skel(lib_mol)

        matches = skel_input.GetSubstructMatches(skel_lib, uniquify=True)
        if not matches:
            return None

        # 取第一个匹配（因为是同构的，通常只有一个）
        match = matches[0]
        mapping = {}  # lib_idx -> input_idx
        for lib_i, input_i in enumerate(match):
            mapping[lib_i] = input_i
        return mapping

    def calculate_rmsd_and_transform(self, target_conf, frag_from_brics, lib_mol, mapping):
        """计算最佳刚性叠合的 RMSD 和变换参数"""
        lib_conf = lib_mol.GetConformer()
        pts_lib = []
        pts_target = []
        weights = []

        for lib_idx, frag_idx in mapping.items():
            lib_atom = lib_mol.GetAtomWithIdx(lib_idx)
            frag_atom = frag_from_brics.GetAtomWithIdx(frag_idx)

            if not frag_atom.HasProp("orig_idx"):
                continue

            global_idx = frag_atom.GetIntProp("orig_idx")

            l_pos = lib_conf.GetAtomPosition(lib_idx)
            t_pos = target_conf.GetAtomPosition(global_idx)

            pts_lib.append([l_pos.x, l_pos.y, l_pos.z])
            pts_target.append([t_pos.x, t_pos.y, t_pos.z])

            # 重原子权重1，虚拟原子权重0
            w = 1.0 if lib_atom.GetAtomicNum() > 0 else 0.0
            weights.append(w)

        if not pts_lib or sum(weights) < 1e-6:
            return None, None, float('inf')

        pts_lib = np.array(pts_lib)
        pts_target = np.array(pts_target)
        weights = np.array(weights)

        # 加权质心
        center_lib = np.average(pts_lib, axis=0, weights=weights)
        center_target = np.average(pts_target, axis=0, weights=weights)

        P = pts_lib - center_lib
        Q = pts_target - center_target

        H = P.T @ np.diag(weights) @ Q
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t_vec = center_target - (center_lib @ R.T)

        # 计算加权 RMSD
        transformed = (pts_lib @ R.T) + t_vec
        diff = transformed - pts_target
        weighted_sq = np.sum((np.linalg.norm(diff, axis=1)**2) * weights)
        rmsd = np.sqrt(weighted_sq / np.sum(weights))

        return R, t_vec, rmsd

    def find_best_candidate(self, input_frag, skel_key, target_conf, frag_with_orig_idx):
        """在给定骨架下寻找最佳构象匹配"""
        candidates = self.get_candidates_by_skel(skel_key)
        if not candidates:
            return None
        # print("FOUND CANDIDATE")
        input_fp = generate_fingerprint(input_frag)
        if input_fp is None:
            return None
        # print("FOUND FP")
        best = None
        best_score = -1

        for cand in candidates:
            lib_mol = Chem.MolFromMolBlock(cand['molblock'])
            if not lib_mol:
                continue

            # 指纹相似度
            lib_fp = generate_fingerprint(lib_mol)
            if lib_fp is None:
                continue
            chem_sim = DataStructs.TanimotoSimilarity(input_fp, lib_fp)

            # 几何匹配
            mapping = self.get_atom_mapping(input_frag, lib_mol)
            if not mapping:
                continue

            R, t, rmsd = self.calculate_rmsd_and_transform(
                target_conf, frag_with_orig_idx, lib_mol, mapping
            )

            if rmsd == float('inf'):
                continue

            # 综合评分（几何优先）
            geo_score = 1.0 / (1.0 + rmsd * 2)  # RMSD<0.5 → ~0.91
            score = 0.8 * geo_score * 0.2 * chem_sim # + 1e-6 * cand['freq'] # - 0.1 * chem_sim
            # print({
            #         # 'molblock': cand['molblock'],
            #         'rmsd': rmsd,
            #         'chem_sim': chem_sim,
            #         'score': score,
            #         'freq': cand['freq']
            #     })
            if score > best_score:
                best_score = score
                best = {
                    'mol': lib_mol,
                    'molblock': cand['molblock'],
                    'rmsd': rmsd,
                    'chem_sim': chem_sim,
                    'score': score,
                    'mapping': mapping,
                    'R': R,
                    't': t,
                    'freq': cand['freq']
                }

        return best

    def apply_replacement(self, target_mol, best, frag_from_brics):
        """应用最佳匹配的坐标变换"""
        # print('BEST:',best)
        if not best or not best['R'] is not None:
            return False

        conf = target_mol.GetConformer()
        lib_conf = best['mol'].GetConformer()
        mapping = best['mapping']
        R, t = best['R'], best['t']

        for lib_idx, frag_idx in mapping.items():
            frag_atom = frag_from_brics.GetAtomWithIdx(frag_idx)
            if not frag_atom.HasProp("orig_idx"):
                continue

            global_idx = frag_atom.GetIntProp("orig_idx")

            old_pos = np.array([
                lib_conf.GetAtomPosition(lib_idx).x,
                lib_conf.GetAtomPosition(lib_idx).y,
                lib_conf.GetAtomPosition(lib_idx).z
            ])

            new_pos_array = old_pos @ R.T + t
            new_pos = Point3D(new_pos_array[0], new_pos_array[1], new_pos_array[2])

            conf.SetAtomPosition(global_idx, new_pos)

        return True

    def process_molecule(self, mol):
        if mol is None:
            return None

        # 给原子打上原始索引标记
        mol_tagged = Chem.Mol(mol)
        for i, atom in enumerate(mol_tagged.GetAtoms()):
            atom.SetIntProp("orig_idx", i)

        try:
            frags_mols = BRICS.BreakBRICSBonds(mol_tagged)
            # frags_mols = custom_break_brics_bonds(mol_tagged)
            # print("BREAK!")
            fragments_indices = Chem.GetMolFrags(frags_mols, asMols=False)
        except:
            return mol

        output_mol = Chem.Mol(mol)
        replaced_count = 0

        for frag_indices in fragments_indices:
            curr_frag = extract_submol(frags_mols, frag_indices)
            if curr_frag is None:
                continue

            skel_key = get_skeleton_key(curr_frag)
            if not skel_key:
                continue

            target_conf = output_mol.GetConformer()
            best = self.find_best_candidate(
                curr_frag, skel_key, target_conf, curr_frag
            )
            # print(best)
            if best and self.apply_replacement(output_mol, best, curr_frag):
                replaced_count += 1
                # 可选打印：print(f"✓ Replaced fragment (skel: {skel_key[:40]}..., RMSD: {best['rmsd']:.3f})")

        # 最后轻微优化几何
        try:
            Chem.SanitizeMol(output_mol)
            if AllChem.MMFFHasAllMoleculeParams(output_mol):
                AllChem.MMFFOptimizeMolecule(output_mol, maxIters=15)
            else:
                AllChem.UFFOptimizeMolecule(output_mol, maxIters=15)
        except:
            pass

        # print(f"Refinement complete: {replaced_count} fragments replaced.")
        if replaced_count == 0:
            return None
        return output_mol


# ==========================================
# 主程序示例
# ==========================================

if __name__ == "__main__":
    INDEX_PATH = "/root/autodl-fs/DiffEDMol-unitrans/fragment_library/index_pdb.pkl.gz"
    INPUT_SDF = "/root/autodl-fs/DiffEDMol-unitrans/9c6k_D_TPP.sdf"
    OUTPUT_SDF = "rebuilt_refined_newindex.sdf"

    if not os.path.exists(INDEX_PATH):
        print(f"Index file not found: {INDEX_PATH}")
        exit()

    # refiner = MoleculeRefiner(INDEX_PATH,10000)
    refiner = MoleculeRefiner(INDEX_PATH)
    suppl = Chem.SDMolSupplier(INPUT_SDF)
    writer = Chem.SDWriter(OUTPUT_SDF)

    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        print(f"\nProcessing molecule {i+1}...")
        refined = refiner.process_molecule(mol)
        if refined:
            writer.write(refined)
            break

    writer.close()
    print(f"\nAll done. Refined molecules saved to: {OUTPUT_SDF}")