#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
荧光分子功能片段库工具包 (Fluorescent Molecule Fragment Library Utils)

该工具包用于：
1. 构建荧光分子功能片段库（电子供体、电子受体、连接子）
2. 识别分子中的功能片段
3. 通过BM骨架和CSK骨架模糊匹配功能片段
4. 将分子切分为BM骨架并输出标注片段的SDF文件

作者: AI Assistant
版本: 1.0.0
"""

import os
import pickle
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS, rdRascalMCES
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
import numpy as np


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class FragmentMatch:
    """片段匹配结果数据类"""
    smiles: str
    fragment_type: str  # 'donor', 'acceptor', 'linker'
    match_atoms: List[int] = field(default_factory=list)
    match_score: float = 1.0
    match_type: str = 'exact'  # 'exact', 'substructure', 'fuzzy'
    
    def to_dict(self) -> Dict:
        return {
            'smiles': self.smiles,
            'fragment_type': self.fragment_type,
            'match_atoms': self.match_atoms,
            'match_score': self.match_score,
            'match_type': self.match_type
        }


@dataclass
class ScaffoldInfo:
    """骨架信息数据类"""
    scaffold_smiles: str
    scaffold_type: str  # 'BM', 'CSK'
    atoms: List[int] = field(default_factory=list)
    original_atoms: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'scaffold_smiles': self.scaffold_smiles,
            'scaffold_type': self.scaffold_type,
            'atoms': self.atoms,
            'original_atoms': self.original_atoms
        }


@dataclass
class MoleculeAnnotation:
    """分子标注信息数据类"""
    donors: List[FragmentMatch] = field(default_factory=list)
    acceptors: List[FragmentMatch] = field(default_factory=list)
    linkers: List[FragmentMatch] = field(default_factory=list)
    scaffolds: List[ScaffoldInfo] = field(default_factory=list)
    
    @property
    def total_fragments(self) -> int:
        return len(self.donors) + len(self.acceptors) + len(self.linkers)
    
    def to_dict(self) -> Dict:
        return {
            'donors': [d.to_dict() for d in self.donors],
            'acceptors': [a.to_dict() for a in self.acceptors],
            'linkers': [l.to_dict() for l in self.linkers],
            'scaffolds': [s.to_dict() for s in self.scaffolds],
            'total_fragments': self.total_fragments
        }


# =============================================================================
# 片段库类
# =============================================================================

class FragmentLibrary:
    """荧光分子功能片段库"""
    
    # 预定义的供体SMARTS模式
    DONOR_PATTERNS = [
        # 苯环
        ('c1ccccc1', 'benzene'),
        # 含N杂环
        ('c1ccncc1', 'pyridine'),
        ('c1c[nH]cn1', 'imidazole'),
        ('c1ccc2[nH]ccc2c1', 'indole'),
        ('c1ccc2c(c1)[nH]c1ccccc12', 'carbazole'),
        # 含O杂环
        ('c1ccoc1', 'furan'),
        # 含S杂环
        ('c1ccsc1', 'thiophene'),
        # 氨基
        ('[NX3;H2,H1;!$(NC=O)]', 'amine'),
        ('c1ccccc1N', 'aniline'),
        # 羟基
        ('[OX2H]', 'hydroxyl'),
        # 烷氧基
        ('[OX2C]', 'alkoxy'),
        # 三苯胺
        ('c1ccccc1N(c2ccccc2)c2ccccc2', 'triphenylamine'),
    ]
    
    # 预定义的受体SMARTS模式
    ACCEPTOR_PATTERNS = [
        # 硝基
        ('[N+](=O)[O-]', 'nitro'),
        # 氰基
        ('C#N', 'cyano'),
        # 羰基
        ('C=O', 'carbonyl'),
        ('C(=O)O', 'carboxyl'),
        ('C(=O)N', 'amide'),
        # 砜
        ('S(=O)(=O)', 'sulfone'),
        # 卤素
        ('[F,Cl,Br,I]', 'halogen'),
        # 三氟甲基
        ('C(F)(F)F', 'trifluoromethyl'),
        # BODIPY核心
        ('B1N2C=CC=C2C=C1', 'BODIPY_core'),
    ]
    
    # 预定义的连接子SMARTS模式
    LINKER_PATTERNS = [
        # 乙烯基
        ('C=C', 'vinyl'),
        # 乙炔基
        ('C#C', 'ethynyl'),
        # 苯基桥连
        ('c1ccccc1-c2ccccc2', 'biphenyl'),
        # 共轭双键
        ('C=C-C=C', 'conjugated_diene'),
        # 苯乙烯基
        ('c1ccccc1C=C', 'styryl'),
    ]
    
    def __init__(self, library_path: Optional[str] = None):
        self.donors = []
        self.acceptors = []
        self.linkers = []
        self.stats = {}
        
        if library_path and os.path.exists(library_path):
            self.load_library(library_path)
        else:
            self._init_default_patterns()
    
    def _init_default_patterns(self):
        """初始化默认模式"""
        self.donors = [{'smarts': p[0], 'name': p[1]} for p in self.DONOR_PATTERNS]
        self.acceptors = [{'smarts': p[0], 'name': p[1]} for p in self.ACCEPTOR_PATTERNS]
        self.linkers = [{'smarts': p[0], 'name': p[1]} for p in self.LINKER_PATTERNS]
    
    def load_library(self, path: str):
        """加载片段库"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.donors = data.get('donors', [])
        self.acceptors = data.get('acceptors', [])
        self.linkers = data.get('linkers', [])
        self.stats = data.get('stats', {})
    
    def save_library(self, path: str):
        """保存片段库"""
        data = {
            'donors': self.donors,
            'acceptors': self.acceptors,
            'linkers': self.linkers,
            'stats': self.stats
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_donors(self) -> List[Dict]:
        """获取供体列表"""
        return self.donors
    
    def get_acceptors(self) -> List[Dict]:
        """获取受体列表"""
        return self.acceptors
    
    def get_linkers(self) -> List[Dict]:
        """获取连接子列表"""
        return self.linkers
    
    def add_donor(self, smarts: str, name: str = ''):
        """添加供体"""
        self.donors.append({'smarts': smarts, 'name': name})
    
    def add_acceptor(self, smarts: str, name: str = ''):
        """添加受体"""
        self.acceptors.append({'smarts': smarts, 'name': name})
    
    def add_linker(self, smarts: str, name: str = ''):
        """添加连接子"""
        self.linkers.append({'smarts': smarts, 'name': name})


# =============================================================================
# 片段识别器类
# =============================================================================

class FragmentIdentifier:
    """功能片段识别器"""
    
    def __init__(self, fragment_library: FragmentLibrary):
        self.library = fragment_library
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译SMARTS模式"""
        self.donor_mols = []
        self.acceptor_mols = []
        self.linker_mols = []
        
        for d in self.library.get_donors():
            mol = Chem.MolFromSmarts(d['smarts'])
            if mol:
                self.donor_mols.append((mol, d))
        
        for a in self.library.get_acceptors():
            mol = Chem.MolFromSmarts(a['smarts'])
            if mol:
                self.acceptor_mols.append((mol, a))
        
        for l in self.library.get_linkers():
            mol = Chem.MolFromSmarts(l['smarts'])
            if mol:
                self.linker_mols.append((mol, l))
    
    def identify(self, smiles_or_mol: Union[str, Chem.Mol]) -> MoleculeAnnotation:
        """
        识别分子中的功能片段
        
        Args:
            smiles_or_mol: SMILES字符串或RDKit Mol对象
            
        Returns:
            MoleculeAnnotation对象
        """
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol
        
        if mol is None:
            return MoleculeAnnotation()
        
        annotation = MoleculeAnnotation()
        
        # 识别供体
        for pattern_mol, pattern_info in self.donor_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.donors.append(FragmentMatch(
                    smiles=pattern_info['smarts'],
                    fragment_type='donor',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='substructure'
                ))
        
        # 识别受体
        for pattern_mol, pattern_info in self.acceptor_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.acceptors.append(FragmentMatch(
                    smiles=pattern_info['smarts'],
                    fragment_type='acceptor',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='substructure'
                ))
        
        # 识别连接子
        for pattern_mol, pattern_info in self.linker_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.linkers.append(FragmentMatch(
                    smiles=pattern_info['smarts'],
                    fragment_type='linker',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='substructure'
                ))
        
        return annotation
    
    def has_donor(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含供体"""
        annotation = self.identify(smiles_or_mol)
        return len(annotation.donors) > 0
    
    def has_acceptor(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含受体"""
        annotation = self.identify(smiles_or_mol)
        return len(annotation.acceptors) > 0
    
    def has_linker(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含连接子"""
        annotation = self.identify(smiles_or_mol)
        return len(annotation.linkers) > 0


# =============================================================================
# 骨架匹配器类
# =============================================================================

class ScaffoldMatcher:
    """骨架匹配器"""
    
    def __init__(self, fp_radius: int = 2, fp_nbits: int = 2048):
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
    
    def get_bm_scaffold(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        提取BM骨架（Bemis-Murcko骨架）
        
        Args:
            mol: RDKit Mol对象
            
        Returns:
            BM骨架Mol对象
        """
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return scaffold
        except:
            return None
    
    def get_bm_scaffold_smiles(self, smiles: str) -> Optional[str]:
        """从SMILES提取BM骨架SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = self.get_bm_scaffold(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)
    
    def get_csk_scaffold(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        提取CSK骨架（核心结构骨架）
        
        保留核心环系，去除侧链
        """
        try:
            # 获取BM骨架作为基础
            bm_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if bm_scaffold is None:
                return None
            
            # 进一步简化：只保留环系
            ring_info = bm_scaffold.GetRingInfo()
            if ring_info.NumRings() == 0:
                return bm_scaffold
            
            # 保留环上的原子
            ring_atoms = set()
            for ring in ring_info.AtomRings():
                ring_atoms.update(ring)
            
            # 创建子结构
            emol = Chem.EditableMol(Chem.Mol())
            atom_map = {}
            for idx in sorted(ring_atoms):
                atom = bm_scaffold.GetAtomWithIdx(idx)
                new_idx = emol.AddAtom(atom)
                atom_map[idx] = new_idx
            
            # 添加环之间的键
            for bond in bm_scaffold.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in atom_map and end in atom_map:
                    emol.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
            
            return emol.GetMol()
        except:
            return None
    
    def get_csk_scaffold_smiles(self, smiles: str) -> Optional[str]:
        """从SMILES提取CSK骨架SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = self.get_csk_scaffold(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)
    
    def compute_tanimoto_similarity(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """计算Tanimoto相似度"""
        try:
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol1, self.fp_radius, nBits=self.fp_nbits
            )
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol2, self.fp_radius, nBits=self.fp_nbits
            )
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0
    
    def compute_tanimoto_similarity_smiles(self, smiles1: str, smiles2: str) -> float:
        """从SMILES计算Tanimoto相似度"""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        return self.compute_tanimoto_similarity(mol1, mol2)
    
    def fuzzy_match(
        self,
        scaffold_smiles: str,
        fragment_library: FragmentLibrary,
        threshold: float = 0.5,
        similarity_method: str = 'tanimoto'
    ) -> List[Dict]:
        """
        模糊匹配功能片段
        
        Args:
            scaffold_smiles: 骨架SMILES
            fragment_library: 片段库
            threshold: 相似度阈值
            similarity_method: 相似度计算方法
            
        Returns:
            匹配结果列表
        """
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaffold_mol is None:
            return []
        
        matches = []
        
        # 匹配供体
        for donor in fragment_library.get_donors():
            donor_mol = Chem.MolFromSmarts(donor['smarts'])
            if donor_mol is None:
                continue
            
            similarity = self.compute_tanimoto_similarity(scaffold_mol, donor_mol)
            
            if similarity >= threshold:
                matches.append({
                    'fragment_smiles': donor['smarts'],
                    'fragment_type': 'donor',
                    'similarity': similarity,
                    'match_type': 'fuzzy'
                })
        
        # 匹配受体
        for acceptor in fragment_library.get_acceptors():
            acceptor_mol = Chem.MolFromSmarts(acceptor['smarts'])
            if acceptor_mol is None:
                continue
            
            similarity = self.compute_tanimoto_similarity(scaffold_mol, acceptor_mol)
            
            if similarity >= threshold:
                matches.append({
                    'fragment_smiles': acceptor['smarts'],
                    'fragment_type': 'acceptor',
                    'similarity': similarity,
                    'match_type': 'fuzzy'
                })
        
        # 匹配连接子
        for linker in fragment_library.get_linkers():
            linker_mol = Chem.MolFromSmarts(linker['smarts'])
            if linker_mol is None:
                continue
            
            similarity = self.compute_tanimoto_similarity(scaffold_mol, linker_mol)
            
            if similarity >= threshold:
                matches.append({
                    'fragment_smiles': linker['smarts'],
                    'fragment_type': 'linker',
                    'similarity': similarity,
                    'match_type': 'fuzzy'
                })
        
        # 按相似度排序
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches


# =============================================================================
# 分子处理器类
# =============================================================================

class MoleculeProcessor:
    """分子处理器"""
    
    def __init__(
        self,
        fragment_library: FragmentLibrary,
        scaffold_matcher: Optional[ScaffoldMatcher] = None
    ):
        self.library = fragment_library
        self.identifier = FragmentIdentifier(fragment_library)
        self.scaffold_matcher = scaffold_matcher or ScaffoldMatcher()
    
    def split_to_scaffolds(
        self,
        mol: Chem.Mol,
        scaffold_type: str = 'BM'
    ) -> List[ScaffoldInfo]:
        """
        将分子切分为骨架
        
        Args:
            mol: RDKit Mol对象
            scaffold_type: 骨架类型 ('BM' 或 'CSK')
            
        Returns:
            骨架信息列表
        """
        scaffolds = []
        
        if scaffold_type == 'BM':
            scaffold_mol = self.scaffold_matcher.get_bm_scaffold(mol)
        elif scaffold_type == 'CSK':
            scaffold_mol = self.scaffold_matcher.get_csk_scaffold(mol)
        else:
            return scaffolds
        
        if scaffold_mol is not None:
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
            # 获取骨架原子索引
            scaffold_atoms = list(range(scaffold_mol.GetNumAtoms()))
            
            scaffolds.append(ScaffoldInfo(
                scaffold_smiles=scaffold_smiles,
                scaffold_type=scaffold_type,
                atoms=scaffold_atoms,
                original_atoms=scaffold_atoms
            ))
        
        return scaffolds
    
    def process_molecule(
        self,
        input_smiles: str,
        output_sdf_path: Optional[str] = None
    ) -> Tuple[Chem.Mol, MoleculeAnnotation]:
        """
        处理单个分子
        
        Args:
            input_smiles: 输入SMILES
            output_sdf_path: 可选的输出SDF路径
            
        Returns:
            (分子对象, 标注信息)
        """
        # 解析分子
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            raise ValueError(f"无法解析SMILES: {input_smiles}")
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 识别功能片段
        annotation = self.identifier.identify(mol)
        
        # 提取骨架
        annotation.scaffolds = self.split_to_scaffolds(mol, 'BM')
        
        # 模糊匹配
        for scaffold in annotation.scaffolds:
            matches = self.scaffold_matcher.fuzzy_match(
                scaffold.scaffold_smiles,
                self.library,
                threshold=0.5
            )
            # 可以在这里存储匹配结果
        
        # 保存SDF
        if output_sdf_path:
            self.save_annotated_sdf(mol, annotation, output_sdf_path)
        
        return mol, annotation
    
    def save_annotated_sdf(
        self,
        mol: Chem.Mol,
        annotation: MoleculeAnnotation,
        output_path: str
    ):
        """
        保存标注的SDF文件
        
        Args:
            mol: RDKit Mol对象
            annotation: 标注信息
            output_path: 输出路径
        """
        # 创建分子副本
        mol_copy = Chem.Mol(mol)
        
        # 添加属性
        mol_copy.SetProp('donor_count', str(len(annotation.donors)))
        mol_copy.SetProp('acceptor_count', str(len(annotation.acceptors)))
        mol_copy.SetProp('linker_count', str(len(annotation.linkers)))
        mol_copy.SetProp('total_fragments', str(annotation.total_fragments))
        
        # 添加片段SMILES
        donor_smiles = [d.smiles for d in annotation.donors]
        acceptor_smiles = [a.smiles for a in annotation.acceptors]
        linker_smiles = [l.smiles for l in annotation.linkers]
        
        mol_copy.SetProp('donor_smiles', '|'.join(donor_smiles) if donor_smiles else '')
        mol_copy.SetProp('acceptor_smiles', '|'.join(acceptor_smiles) if acceptor_smiles else '')
        mol_copy.SetProp('linker_smiles', '|'.join(linker_smiles) if linker_smiles else '')
        
        # 添加骨架信息
        if annotation.scaffolds:
            scaffold = annotation.scaffolds[0]
            mol_copy.SetProp('bm_scaffold', scaffold.scaffold_smiles)
        
        # 写入SDF
        writer = Chem.SDWriter(output_path)
        writer.write(mol_copy)
        writer.close()
    
    def batch_process(
        self,
        input_csv: str,
        output_dir: str,
        smiles_column: str = 'smiles',
        id_column: str = 'id'
    ) -> List[Dict]:
        """
        批量处理分子
        
        Args:
            input_csv: 输入CSV文件路径
            output_dir: 输出目录
            smiles_column: SMILES列名
            id_column: ID列名
            
        Returns:
            处理结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取CSV
        df = pd.read_csv(input_csv)
        
        results = []
        for idx, row in df.iterrows():
            smiles = row.get(smiles_column, '')
            mol_id = row.get(id_column, f'mol_{idx:04d}')
            
            if not smiles:
                continue
            
            try:
                output_path = os.path.join(output_dir, f'{mol_id}.sdf')
                mol, annotation = self.process_molecule(smiles, output_path)
                
                results.append({
                    'id': mol_id,
                    'smiles': smiles,
                    'donor_count': len(annotation.donors),
                    'acceptor_count': len(annotation.acceptors),
                    'linker_count': len(annotation.linkers),
                    'total_fragments': annotation.total_fragments,
                    'output_path': output_path,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'id': mol_id,
                    'smiles': smiles,
                    'success': False,
                    'error': str(e)
                })
        
        return results


# =============================================================================
# 便捷函数
# =============================================================================

def create_default_library() -> FragmentLibrary:
    """创建默认片段库"""
    return FragmentLibrary()


def load_library(path: str) -> FragmentLibrary:
    """加载片段库"""
    return FragmentLibrary(path)


def identify_fragments(smiles: str, library: FragmentLibrary) -> MoleculeAnnotation:
    """识别分子中的功能片段"""
    identifier = FragmentIdentifier(library)
    return identifier.identify(smiles)


def process_molecule(
    smiles: str,
    library: FragmentLibrary,
    output_sdf_path: Optional[str] = None
) -> Tuple[Chem.Mol, MoleculeAnnotation]:
    """处理单个分子"""
    processor = MoleculeProcessor(library)
    return processor.process_molecule(smiles, output_sdf_path)


# =============================================================================
# 主函数（命令行接口）
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='荧光分子功能片段库工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 识别命令
    identify_parser = subparsers.add_parser('identify', help='识别分子中的功能片段')
    identify_parser.add_argument('smiles', help='分子SMILES')
    identify_parser.add_argument('--library', '-l', help='片段库路径')
    
    # 处理命令
    process_parser = subparsers.add_parser('process', help='处理分子')
    process_parser.add_argument('smiles', help='分子SMILES')
    process_parser.add_argument('--output', '-o', required=True, help='输出SDF路径')
    process_parser.add_argument('--library', '-l', help='片段库路径')
    
    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理')
    batch_parser.add_argument('input_csv', help='输入CSV文件')
    batch_parser.add_argument('output_dir', help='输出目录')
    batch_parser.add_argument('--smiles-col', default='smiles', help='SMILES列名')
    batch_parser.add_argument('--id-col', default='id', help='ID列名')
    batch_parser.add_argument('--library', '-l', help='片段库路径')
    
    args = parser.parse_args()
    
    if args.command == 'identify':
        library = FragmentLibrary(args.library) if args.library else create_default_library()
        annotation = identify_fragments(args.smiles, library)
        
        print(f"分子: {args.smiles}")
        print(f"供体数量: {len(annotation.donors)}")
        print(f"受体数量: {len(annotation.acceptors)}")
        print(f"连接子数量: {len(annotation.linkers)}")
        print(f"总片段数: {annotation.total_fragments}")
        
        if annotation.donors:
            print("\n供体片段:")
            for d in annotation.donors:
                print(f"  - {d.smiles}")
        
        if annotation.acceptors:
            print("\n受体片段:")
            for a in annotation.acceptors:
                print(f"  - {a.smiles}")
        
        if annotation.linkers:
            print("\n连接子片段:")
            for l in annotation.linkers:
                print(f"  - {l.smiles}")
    
    elif args.command == 'process':
        library = FragmentLibrary(args.library) if args.library else create_default_library()
        mol, annotation = process_molecule(args.smiles, library, args.output)
        
        print(f"已处理分子: {args.smiles}")
        print(f"输出文件: {args.output}")
        print(f"识别到 {annotation.total_fragments} 个功能片段")
    
    elif args.command == 'batch':
        library = FragmentLibrary(args.library) if args.library else create_default_library()
        processor = MoleculeProcessor(library)
        results = processor.batch_process(
            args.input_csv,
            args.output_dir,
            args.smiles_col,
            args.id_col
        )
        
        success_count = sum(1 for r in results if r['success'])
        print(f"批量处理完成: {success_count}/{len(results)} 成功")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
