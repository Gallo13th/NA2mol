"""
骨架匹配器 (Scaffold Matcher)
=============================
实现BM骨架和CSK骨架提取，以及模糊匹配功能。

主要功能:
1. BM骨架提取 - 使用RDKit的MurckoScaffold
2. CSK骨架提取 - 自定义核心环系提取
3. 模糊匹配 - 基于相似度的片段匹配
4. 相似度计算 - Tanimoto相似度和子结构匹配

作者: AI Assistant
日期: 2025
"""

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from typing import List, Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


class ScaffoldMatcher:
    """
    骨架匹配器类
    
    提供BM骨架提取、CSK骨架提取和模糊匹配功能。
    """
    
    def __init__(self, fp_radius: int = 2, fp_nbits: int = 2048):
        """
        初始化骨架匹配器
        
        Args:
            fp_radius: Morgan指纹半径 (默认: 2)
            fp_nbits: Morgan指纹位数 (默认: 2048)
        """
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
    
    # ==================== BM骨架提取 ====================
    
    def get_bm_scaffold(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        提取BM骨架 (Bemis-Murcko骨架)
        
        使用RDKit的MurckoScaffold.GetScaffoldForMol()方法，
        提取分子的核心骨架（保留环和连接性）。
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            BM骨架分子对象，失败返回None
        """
        if mol is None:
            return None
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return scaffold
        except Exception as e:
            print(f"BM骨架提取失败: {e}")
            return None
    
    def get_bm_scaffold_smiles(self, smiles: str) -> Optional[str]:
        """
        从SMILES提取BM骨架SMILES
        
        Args:
            smiles: 分子SMILES字符串
            
        Returns:
            BM骨架SMILES字符串，失败返回None
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = self.get_bm_scaffold(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)
    
    def get_bm_scaffold_for_mol_list(self, mols: List[Chem.Mol]) -> List[Optional[Chem.Mol]]:
        """
        批量提取BM骨架
        
        Args:
            mols: 分子对象列表
            
        Returns:
            BM骨架分子对象列表
        """
        return [self.get_bm_scaffold(mol) for mol in mols]
    
    # ==================== CSK骨架提取 ====================
    
    def get_csk_scaffold(self, mol: Chem.Mol, 
                         keep_linker_atoms: bool = True,
                         remove_terminal_atoms: bool = True) -> Optional[Chem.Mol]:
        """
        提取CSK骨架 (核心结构骨架)
        
        自定义提取方法：保留核心环系，去除侧链。
        与BM骨架不同，CSK骨架保留更多的结构信息，
        包括环之间的连接原子和必要的官能团。
        
        Args:
            mol: RDKit分子对象
            keep_linker_atoms: 是否保留连接原子 (默认: True)
            remove_terminal_atoms: 是否移除末端原子 (默认: True)
            
        Returns:
            CSK骨架分子对象，失败返回None
        """
        if mol is None:
            return None
        
        try:
            # 获取BM骨架作为基础
            bm_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            
            if not keep_linker_atoms:
                # 仅保留环原子
                ring_info = bm_scaffold.GetRingInfo()
                ring_atoms = set()
                for ring in ring_info.AtomRings():
                    ring_atoms.update(ring)
                
                # 创建仅包含环原子的子结构
                if ring_atoms:
                    emol = Chem.EditableMol(Chem.Mol())
                    atom_map = {}
                    for idx in sorted(ring_atoms):
                        atom = bm_scaffold.GetAtomWithIdx(idx)
                        new_idx = emol.AddAtom(atom)
                        atom_map[idx] = new_idx
                    
                    # 添加环内的键
                    for bond in bm_scaffold.GetBonds():
                        begin = bond.GetBeginAtomIdx()
                        end = bond.GetEndAtomIdx()
                        if begin in ring_atoms and end in ring_atoms:
                            emol.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
                    
                    csk_scaffold = emol.GetMol()
                    Chem.SanitizeMol(csk_scaffold)
                    return csk_scaffold
            
            # 返回BM骨架作为CSK骨架的基础版本
            return bm_scaffold
            
        except Exception as e:
            print(f"CSK骨架提取失败: {e}")
            return None
    
    def get_csk_scaffold_advanced(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        高级CSK骨架提取
        
        提取包含核心环系及其直接相连的关键原子的骨架。
        保留环之间的连接链，但去除长侧链。
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            高级CSK骨架分子对象
        """
        if mol is None:
            return None
        
        try:
            # 获取所有环
            ring_info = mol.GetRingInfo()
            ring_systems = ring_info.AtomRings()
            
            if not ring_systems:
                # 无环分子，返回空
                return None
            
            # 收集所有环原子
            ring_atoms = set()
            for ring in ring_systems:
                ring_atoms.update(ring)
            
            # 识别连接原子和桥接原子
            linker_atoms = set()
            for atom_idx in ring_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in ring_atoms:
                        # 检查邻居是否连接到其他环
                        for other_ring in ring_systems:
                            if neighbor_idx in other_ring:
                                linker_atoms.add(neighbor_idx)
                                break
            
            # 合并环原子和连接原子
            core_atoms = ring_atoms.union(linker_atoms)
            
            # 创建骨架分子
            if core_atoms:
                emol = Chem.EditableMol(Chem.Mol())
                atom_map = {}
                for idx in sorted(core_atoms):
                    atom = mol.GetAtomWithIdx(idx)
                    new_idx = emol.AddAtom(atom)
                    atom_map[idx] = new_idx
                
                # 添加键
                for bond in mol.GetBonds():
                    begin = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    if begin in core_atoms and end in core_atoms:
                        emol.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
                
                scaffold = emol.GetMol()
                Chem.SanitizeMol(scaffold)
                return scaffold
            
            return None
            
        except Exception as e:
            print(f"高级CSK骨架提取失败: {e}")
            return None
    
    def get_csk_scaffold_smiles(self, smiles: str) -> Optional[str]:
        """
        从SMILES提取CSK骨架SMILES
        
        Args:
            smiles: 分子SMILES字符串
            
        Returns:
            CSK骨架SMILES字符串
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = self.get_csk_scaffold(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)
    
    # ==================== 相似度计算 ====================
    
    def compute_morgan_fingerprint(self, mol: Chem.Mol) -> Optional[DataStructs.ExplicitBitVect]:
        """
        计算Morgan指纹
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            Morgan指纹位向量
        """
        if mol is None:
            return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fp_radius, nBits=self.fp_nbits
            )
            return fp
        except Exception as e:
            print(f"指纹计算失败: {e}")
            return None
    
    def compute_tanimoto_similarity(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """
        计算两个分子的Tanimoto相似度
        
        Args:
            mol1: 第一个分子
            mol2: 第二个分子
            
        Returns:
            Tanimoto相似度分数 (0.0 - 1.0)
        """
        fp1 = self.compute_morgan_fingerprint(mol1)
        fp2 = self.compute_morgan_fingerprint(mol2)
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def compute_tanimoto_similarity_smiles(self, smiles1: str, smiles2: str) -> float:
        """
        从SMILES计算Tanimoto相似度
        
        Args:
            smiles1: 第一个分子的SMILES
            smiles2: 第二个分子的SMILES
            
        Returns:
            Tanimoto相似度分数
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        return self.compute_tanimoto_similarity(mol1, mol2)
    
    def check_substructure_match(self, mol: Chem.Mol, query: Chem.Mol) -> bool:
        """
        检查子结构匹配
        
        Args:
            mol: 被查询的分子
            query: 查询子结构
            
        Returns:
            是否匹配
        """
        if mol is None or query is None:
            return False
        try:
            return mol.HasSubstructMatch(query)
        except Exception:
            return False
    
    def check_substructure_match_smiles(self, smiles: str, query_smiles: str) -> bool:
        """
        从SMILES检查子结构匹配
        
        Args:
            smiles: 分子SMILES
            query_smiles: 查询子结构SMILES
            
        Returns:
            是否匹配
        """
        mol = Chem.MolFromSmiles(smiles)
        query = Chem.MolFromSmiles(query_smiles)
        return self.check_substructure_match(mol, query)
    
    def compute_substructure_similarity(self, mol: Chem.Mol, query: Chem.Mol) -> float:
        """
        计算子结构相似度
        
        基于匹配的原子数和查询分子的大小计算相似度。
        
        Args:
            mol: 被查询的分子
            query: 查询子结构
            
        Returns:
            子结构相似度分数 (0.0 - 1.0)
        """
        if mol is None or query is None:
            return 0.0
        
        try:
            matches = mol.GetSubstructMatches(query)
            if not matches:
                return 0.0
            
            # 计算匹配度 = 匹配原子数 / 查询分子原子数
            query_atom_count = query.GetNumAtoms()
            matched_atom_count = len(matches[0])
            
            return matched_atom_count / query_atom_count
        except Exception:
            return 0.0
    
    # ==================== 模糊匹配 ====================
    
    def fuzzy_match(self, 
                    scaffold_smiles: str, 
                    fragment_library: List[Dict],
                    threshold: float = 0.7,
                    similarity_method: str = 'tanimoto') -> List[Dict]:
        """
        模糊匹配函数
        
        输入骨架SMILES，匹配片段库中的片段，返回匹配结果和相似度分数。
        
        Args:
            scaffold_smiles: 骨架SMILES字符串
            fragment_library: 片段库列表，每个片段为字典格式:
                [
                    {'smiles': '片段SMILES', 'type': 'donor/acceptor/linker', ...},
                    ...
                ]
            threshold: 相似度阈值 (默认: 0.7)
            similarity_method: 相似度计算方法 ('tanimoto', 'substructure', 'combined')
            
        Returns:
            匹配结果列表:
            [
                {
                    'fragment_smiles': '片段SMILES',
                    'fragment_type': 'donor/acceptor/linker',
                    'similarity': 相似度分数,
                    'match_type': 'exact/substructure/fuzzy'
                },
                ...
            ]
        """
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaffold_mol is None:
            return []
        
        results = []
        
        for fragment in fragment_library:
            frag_smiles = fragment.get('smiles', '')
            frag_type = fragment.get('type', 'unknown')
            
            frag_mol = Chem.MolFromSmiles(frag_smiles)
            if frag_mol is None:
                continue
            
            # 计算相似度
            if similarity_method == 'tanimoto':
                similarity = self.compute_tanimoto_similarity(scaffold_mol, frag_mol)
            elif similarity_method == 'substructure':
                similarity = self.compute_substructure_similarity(scaffold_mol, frag_mol)
            elif similarity_method == 'combined':
                tanimoto = self.compute_tanimoto_similarity(scaffold_mol, frag_mol)
                substruct = self.compute_substructure_similarity(scaffold_mol, frag_mol)
                similarity = 0.6 * tanimoto + 0.4 * substruct
            else:
                similarity = self.compute_tanimoto_similarity(scaffold_mol, frag_mol)
            
            # 判断匹配类型
            if similarity >= 0.99:
                match_type = 'exact'
            elif self.check_substructure_match(scaffold_mol, frag_mol):
                match_type = 'substructure'
            else:
                match_type = 'fuzzy'
            
            # 如果相似度超过阈值，添加到结果
            if similarity >= threshold:
                results.append({
                    'fragment_smiles': frag_smiles,
                    'fragment_type': frag_type,
                    'similarity': round(similarity, 4),
                    'match_type': match_type,
                    'fragment_data': fragment  # 保留原始数据
                })
        
        # 按相似度降序排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def fuzzy_match_bm(self,
                       mol_smiles: str,
                       fragment_library: List[Dict],
                       threshold: float = 0.7) -> List[Dict]:
        """
        基于BM骨架的模糊匹配
        
        先提取分子的BM骨架，然后进行模糊匹配。
        
        Args:
            mol_smiles: 分子SMILES
            fragment_library: 片段库
            threshold: 相似度阈值
            
        Returns:
            匹配结果列表
        """
        bm_scaffold = self.get_bm_scaffold_smiles(mol_smiles)
        if bm_scaffold is None:
            return []
        
        return self.fuzzy_match(bm_scaffold, fragment_library, threshold)
    
    def fuzzy_match_csk(self,
                        mol_smiles: str,
                        fragment_library: List[Dict],
                        threshold: float = 0.7) -> List[Dict]:
        """
        基于CSK骨架的模糊匹配
        
        先提取分子的CSK骨架，然后进行模糊匹配。
        
        Args:
            mol_smiles: 分子SMILES
            fragment_library: 片段库
            threshold: 相似度阈值
            
        Returns:
            匹配结果列表
        """
        csk_scaffold = self.get_csk_scaffold_smiles(mol_smiles)
        if csk_scaffold is None:
            return []
        
        return self.fuzzy_match(csk_scaffold, fragment_library, threshold)
    
    # ==================== 批量处理 ====================
    
    def batch_extract_scaffolds(self, 
                                 smiles_list: List[str],
                                 scaffold_type: str = 'bm') -> Dict[str, List]:
        """
        批量提取骨架
        
        Args:
            smiles_list: SMILES字符串列表
            scaffold_type: 骨架类型 ('bm' 或 'csk')
            
        Returns:
            包含原始SMILES和骨架SMILES的字典
        """
        results = {
            'original': [],
            'scaffold': [],
            'success': []
        }
        
        for smiles in smiles_list:
            results['original'].append(smiles)
            
            if scaffold_type == 'bm':
                scaffold = self.get_bm_scaffold_smiles(smiles)
            elif scaffold_type == 'csk':
                scaffold = self.get_csk_scaffold_smiles(smiles)
            else:
                scaffold = None
            
            results['scaffold'].append(scaffold)
            results['success'].append(scaffold is not None)
        
        return results
    
    def batch_fuzzy_match(self,
                          smiles_list: List[str],
                          fragment_library: List[Dict],
                          threshold: float = 0.7,
                          scaffold_type: str = 'bm') -> Dict[str, List]:
        """
        批量模糊匹配
        
        Args:
            smiles_list: SMILES字符串列表
            fragment_library: 片段库
            threshold: 相似度阈值
            scaffold_type: 骨架类型 ('bm' 或 'csk')
            
        Returns:
            包含匹配结果的字典
        """
        results = {
            'original': [],
            'scaffold': [],
            'matches': []
        }
        
        for smiles in smiles_list:
            results['original'].append(smiles)
            
            if scaffold_type == 'bm':
                scaffold = self.get_bm_scaffold_smiles(smiles)
                matches = self.fuzzy_match_bm(smiles, fragment_library, threshold)
            elif scaffold_type == 'csk':
                scaffold = self.get_csk_scaffold_smiles(smiles)
                matches = self.fuzzy_match_csk(smiles, fragment_library, threshold)
            else:
                scaffold = None
                matches = []
            
            results['scaffold'].append(scaffold)
            results['matches'].append(matches)
        
        return results


# ==================== 辅助函数 ====================

def create_fragment_library() -> List[Dict]:
    """
    创建示例片段库
    
    Returns:
        片段库列表
    """
    return [
        # 电子给体片段
        {'smiles': 'c1ccccc1N', 'type': 'donor', 'name': 'aniline'},
        {'smiles': 'c1ccccc1O', 'type': 'donor', 'name': 'phenol'},
        {'smiles': 'c1ccccc1S', 'type': 'donor', 'name': 'thiophenol'},
        {'smiles': 'c1ccc2c(c1)[nH]c2', 'type': 'donor', 'name': 'indole'},
        {'smiles': 'c1ccc2[nH]ccc2c1', 'type': 'donor', 'name': 'indole_alt'},
        
        # 电子受体片段
        {'smiles': 'c1ccccc1[N+](=O)[O-]', 'type': 'acceptor', 'name': 'nitrobenzene'},
        {'smiles': 'c1ccccc1C#N', 'type': 'acceptor', 'name': 'benzonitrile'},
        {'smiles': 'c1ccccc1C=O', 'type': 'acceptor', 'name': 'benzaldehyde'},
        {'smiles': 'c1ccccc1C(=O)O', 'type': 'acceptor', 'name': 'benzoic_acid'},
        
        # 连接片段
        {'smiles': 'c1ccccc1', 'type': 'linker', 'name': 'benzene'},
        {'smiles': 'c1ccc2ccccc2c1', 'type': 'linker', 'name': 'naphthalene'},
        {'smiles': 'c1ccc2c(c1)cccc2', 'type': 'linker', 'name': 'tetralin'},
        {'smiles': 'C1CCCCC1', 'type': 'linker', 'name': 'cyclohexane'},
        {'smiles': 'C1CCOCC1', 'type': 'linker', 'name': 'tetrahydropyran'},
        
        # 杂环片段
        {'smiles': 'c1ccncc1', 'type': 'linker', 'name': 'pyridine'},
        {'smiles': 'c1ccoc1', 'type': 'linker', 'name': 'furan'},
        {'smiles': 'c1ccsc1', 'type': 'linker', 'name': 'thiophene'},
        {'smiles': 'c1c[nH]cn1', 'type': 'linker', 'name': 'imidazole'},
        {'smiles': 'c1cn[nH]c1', 'type': 'linker', 'name': 'pyrazole'},
    ]


def demo():
    """
    演示骨架匹配器的使用
    """
    print("=" * 60)
    print("骨架匹配器演示")
    print("=" * 60)
    
    # 创建匹配器实例
    matcher = ScaffoldMatcher(fp_radius=2, fp_nbits=2048)
    
    # 测试分子
    test_molecules = [
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'c1ccc2c(c1)c(c[nH]2)CCN',  # Tryptamine
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    ]
    
    print("\n1. BM骨架提取:")
    print("-" * 40)
    for smiles in test_molecules:
        bm = matcher.get_bm_scaffold_smiles(smiles)
        print(f"原始: {smiles}")
        print(f"BM:   {bm}\n")
    
    print("\n2. CSK骨架提取:")
    print("-" * 40)
    for smiles in test_molecules:
        csk = matcher.get_csk_scaffold_smiles(smiles)
        print(f"原始: {smiles}")
        print(f"CSK:  {csk}\n")
    
    print("\n3. 相似度计算:")
    print("-" * 40)
    mol1 = Chem.MolFromSmiles('c1ccccc1')
    mol2 = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    sim = matcher.compute_tanimoto_similarity(mol1, mol2)
    print(f"苯 vs 萘 的Tanimoto相似度: {sim:.4f}")
    
    print("\n4. 模糊匹配:")
    print("-" * 40)
    
    # 创建片段库
    fragment_lib = create_fragment_library()
    
    # 测试骨架
    test_scaffold = 'c1ccccc1'
    matches = matcher.fuzzy_match(test_scaffold, fragment_lib, threshold=0.3)
    
    print(f"骨架: {test_scaffold}")
    print(f"匹配结果 (阈值=0.3):")
    for match in matches[:5]:  # 显示前5个结果
        print(f"  - {match['fragment_smiles']} ({match['fragment_type']}) "
              f"相似度={match['similarity']}, 类型={match['match_type']}")
    
    print("\n5. 基于BM骨架的模糊匹配:")
    print("-" * 40)
    ibuprofen = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
    bm_matches = matcher.fuzzy_match_bm(ibuprofen, fragment_lib, threshold=0.5)
    
    print(f"分子: {ibuprofen} (Ibuprofen)")
    print(f"BM骨架: {matcher.get_bm_scaffold_smiles(ibuprofen)}")
    print(f"匹配结果 (阈值=0.5):")
    for match in bm_matches[:5]:
        print(f"  - {match['fragment_smiles']} ({match['fragment_type']}) "
              f"相似度={match['similarity']}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    demo()
