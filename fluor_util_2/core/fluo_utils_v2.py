#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
荧光分子功能片段库工具包 V2.0
基于"环系整体标记"策略的高级版本

核心改进：
1. 如果供体或受体在芳香环系上，则将这个环系标记为供体或受体
2. 如果供体或受体是作为芳香环系的取代基，则将芳香环加上这个取代基标记为供体或受体

作者: AI Assistant
版本: 2.0.0
"""

import os
import pickle
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
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
    smarts: str
    fragment_type: str  # 'donor', 'acceptor', 'linker'
    match_atoms: List[int] = field(default_factory=list)
    match_score: float = 1.0
    match_type: str = 'exact'  # 'exact', 'substructure', 'fuzzy', 'ring_system'
    ring_system_name: str = ''  # 环系名称（如果是环系整体标记）
    
    def to_dict(self) -> Dict:
        return {
            'smiles': self.smiles,
            'smarts': self.smarts,
            'fragment_type': self.fragment_type,
            'match_atoms': self.match_atoms,
            'match_score': self.match_score,
            'match_type': self.match_type,
            'ring_system_name': self.ring_system_name
        }


@dataclass
class RingSystemPattern:
    """环系模式数据类"""
    smarts: str
    name: str
    fragment_type: str
    description: str
    examples: List[str] = field(default_factory=list)


@dataclass
class MoleculeAnnotation:
    """分子标注信息数据类"""
    donors: List[FragmentMatch] = field(default_factory=list)
    acceptors: List[FragmentMatch] = field(default_factory=list)
    linkers: List[FragmentMatch] = field(default_factory=list)
    ring_system_donors: List[FragmentMatch] = field(default_factory=list)  # 环系供体
    ring_system_acceptors: List[FragmentMatch] = field(default_factory=list)  # 环系受体
    
    @property
    def total_fragments(self) -> int:
        return (len(self.donors) + len(self.acceptors) + len(self.linkers) +
                len(self.ring_system_donors) + len(self.ring_system_acceptors))
    
    @property
    def total_donors(self) -> int:
        return len(self.donors) + len(self.ring_system_donors)
    
    @property
    def total_acceptors(self) -> int:
        return len(self.acceptors) + len(self.ring_system_acceptors)
    
    def to_dict(self) -> Dict:
        return {
            'donors': [d.to_dict() for d in self.donors],
            'acceptors': [a.to_dict() for a in self.acceptors],
            'linkers': [l.to_dict() for l in self.linkers],
            'ring_system_donors': [d.to_dict() for d in self.ring_system_donors],
            'ring_system_acceptors': [a.to_dict() for a in self.ring_system_acceptors],
            'total_fragments': self.total_fragments,
            'total_donors': self.total_donors,
            'total_acceptors': self.total_acceptors
        }


# =============================================================================
# 高级片段库类
# =============================================================================

class AdvancedFragmentLibrary:
    """高级片段库 - 基于环系整体标记策略"""
    
    def __init__(self, library_path: Optional[str] = None):
        self.donors = []
        self.acceptors = []
        self.linkers = []
        self.donor_ring_systems: List[RingSystemPattern] = []
        self.acceptor_ring_systems: List[RingSystemPattern] = []
        self.stats = {}
        
        if library_path and os.path.exists(library_path):
            self.load_library(library_path)
        else:
            self._init_default_patterns()
            self._init_ring_system_patterns()
    
    def _init_default_patterns(self):
        """初始化默认片段模式"""
        # 基础供体模式
        self.donors = [
            'c1ccccc1', 'c1ccncc1', 'c1ccsc1', 'c1ccoc1',
            'c1ccc2[nH]ccc2c1', 'c1ccc2c(c1)[nH]c1ccccc12',
            'c1ccc(N(c2ccccc2)c2ccccc2)cc1',
        ]
        
        # 基础受体模式
        self.acceptors = [
            'C#N', 'C=O', '[N+](=O)[O-]', 'S(=O)(=O)O',
            'P(=O)(O)O', '[N+](C)(C)C',
        ]
        
        # 连接子模式
        self.linkers = [
            'C=C', 'C#C', 'c1ccccc1-c2ccccc2',
        ]
    
    def _init_ring_system_patterns(self):
        """初始化环系整体标记模式"""
        
        # ==================== 供体环系模式 ====================
        self.donor_ring_systems = [
            RingSystemPattern(
                smarts='c1ccccc1N(c2ccccc2)c3ccccc3',
                name='triphenylamine_core',
                fragment_type='donor',
                description='三苯胺核心，整个结构作为供体单元',
                examples=['TPA', '经典空穴传输材料']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)[nH]c1ccccc12',
                name='carbazole_fused',
                fragment_type='donor',
                description='咔唑融合环系，整个结构作为供体单元',
                examples=['CBZ', '经典蓝色发光材料']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)c1ccccc1n2-c3ccccc3',
                name='carbazole_substituted',
                fragment_type='donor',
                description='N-取代咔唑，包括取代基整体作为供体',
                examples=['N-苯基咔唑']
            ),
            RingSystemPattern(
                smarts='c1ccc2[nH]ccc2c1',
                name='indole_core',
                fragment_type='donor',
                description='吲哚环系作为供体单元',
                examples=['Indole']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)Oc1ccccc1N2',
                name='phenoxazine_fused',
                fragment_type='donor',
                description='吩噁嗪融合环系作为供体单元',
                examples=['Phenoxazine']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)Sc1ccccc1N2',
                name='phenothiazine_fused',
                fragment_type='donor',
                description='吩噻嗪融合环系作为供体单元',
                examples=['Phenothiazine']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)oc1ccccc12',
                name='dibenzofuran_fused',
                fragment_type='donor',
                description='二苯并呋喃融合环系作为供体单元',
                examples=['DBF']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)sc1ccccc12',
                name='dibenzothiophene_fused',
                fragment_type='donor',
                description='二苯并噻吩融合环系作为供体单元',
                examples=['DBT']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)Cc1ccccc1-2',
                name='fluorene_fused',
                fragment_type='donor',
                description='芴融合环系作为供体单元',
                examples=['Fluorene']
            ),
            RingSystemPattern(
                smarts='C12(c3ccccc31)c1ccccc1-c1ccccc12',
                name='spirobifluorene_core',
                fragment_type='donor',
                description='螺芴核心作为供体单元',
                examples=['Spiro-OMeTAD']
            ),
        ]
        
        # ==================== 受体环系模式 ====================
        self.acceptor_ring_systems = [
            RingSystemPattern(
                smarts='[B-]1(F)(F)N2C=CC=C2C=C1',
                name='BODIPY_core_basic',
                fragment_type='acceptor',
                description='BODIPY基础核心，整个结构作为受体单元',
                examples=['BODIPY']
            ),
            RingSystemPattern(
                smarts='[B-]1(F)(F)n2cccc2C=c2ccccc21',
                name='BODIPY_core_extended',
                fragment_type='acceptor',
                description='BODIPY扩展核心（含苯环融合）',
                examples=['扩展BODIPY']
            ),
            RingSystemPattern(
                smarts='[B-]1(F)(F)n2c(C)cc(C)c2C',
                name='BODIPY_variant',
                fragment_type='acceptor',
                description='BODIPY变体核心',
                examples=['取代BODIPY']
            ),
            RingSystemPattern(
                smarts='c1nc2ccc1ccc2',
                name='porphyrin_core_simple',
                fragment_type='acceptor',
                description='卟啉简化核心，大环作为受体单元',
                examples=['Porphyrin']
            ),
            RingSystemPattern(
                smarts='c1c2nc1C=c1ccc([nH]1)=C=c1ccc([nH]1)=C2',
                name='porphyrin_core_full',
                fragment_type='acceptor',
                description='卟啉完整大环核心',
                examples=['完整卟啉']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(c1)N=c1ccc(=N2)c2ccccc2N1',
                name='phthalocyanine_core',
                fragment_type='acceptor',
                description='酞菁大环核心作为受体单元',
                examples=['Phthalocyanine']
            ),
            RingSystemPattern(
                smarts='c1ccc2ncccc2c1',
                name='quinoline_fused',
                fragment_type='acceptor',
                description='喹啉融合环系作为受体单元',
                examples=['Quinoline']
            ),
            RingSystemPattern(
                smarts='c1ccc2cnccc2c1',
                name='isoquinoline_fused',
                fragment_type='acceptor',
                description='异喹啉融合环系作为受体单元',
                examples=['Isoquinoline']
            ),
            RingSystemPattern(
                smarts='c1ccc2ncncc2c1',
                name='quinoxaline_fused',
                fragment_type='acceptor',
                description='喹喔啉融合环系作为受体单元',
                examples=['Quinoxaline']
            ),
            RingSystemPattern(
                smarts='c1ccc2c(nnc2)c1',
                name='phthalazine_fused',
                fragment_type='acceptor',
                description='酞嗪融合环系作为受体单元',
                examples=['Phthalazine']
            ),
            RingSystemPattern(
                smarts='c1cscn1',
                name='thiazole_ring',
                fragment_type='acceptor',
                description='噻唑环作为受体单元',
                examples=['Thiazole']
            ),
            RingSystemPattern(
                smarts='c1nsnc1',
                name='thiadiazole_ring',
                fragment_type='acceptor',
                description='噻二唑环作为受体单元',
                examples=['Thiadiazole']
            ),
            RingSystemPattern(
                smarts='c1ccc2nsnc2c1',
                name='benzothiadiazole_fused',
                fragment_type='acceptor',
                description='苯并噻二唑融合环系作为受体单元',
                examples=['BTZ']
            ),
            RingSystemPattern(
                smarts='c1ncc2nsnc2c1',
                name='pyridothiadiazole_fused',
                fragment_type='acceptor',
                description='吡啶并噻二唑融合环系作为受体单元',
                examples=['PTZ']
            ),
            RingSystemPattern(
                smarts='O=C1c2cccc3cccc(c23)C(=O)N1',
                name='naphthalimide_fused',
                fragment_type='acceptor',
                description='萘酰亚胺融合环系作为受体单元',
                examples=['Naphthalimide']
            ),
            RingSystemPattern(
                smarts='O=C1c2ccc3cccc4ccc(c2c34)C(=O)N1',
                name='peryleneimide_fused',
                fragment_type='acceptor',
                description='苝酰亚胺融合环系作为受体单元',
                examples=['Peryleneimide']
            ),
            RingSystemPattern(
                smarts='O=C1Oc2ccccc2C(=C2c3ccccc3Oc3ccccc32)c2ccccc21',
                name='rhodamine_core',
                fragment_type='acceptor',
                description='罗丹明核心作为受体单元',
                examples=['Rhodamine']
            ),
            RingSystemPattern(
                smarts='O=C1Oc2ccccc2C(=C2c3ccccc3Oc3cc(O)ccc32)c2ccccc21',
                name='fluorescein_core',
                fragment_type='acceptor',
                description='荧光素核心作为受体单元',
                examples=['Fluorescein']
            ),
            RingSystemPattern(
                smarts='O=c1ccc2ccccc2o1',
                name='coumarin_fused',
                fragment_type='acceptor',
                description='香豆素融合环系作为受体单元',
                examples=['Coumarin']
            ),
            RingSystemPattern(
                smarts='C(=C1C=CC=CC1)=C1C=CC=CC1',
                name='cyanine_core',
                fragment_type='acceptor',
                description='花菁核心（聚甲炔链）作为受体单元',
                examples=['Cyanine']
            ),
        ]
    
    def load_library(self, path: str):
        """加载片段库"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.donors = data.get('donors', [])
        self.acceptors = data.get('acceptors', [])
        self.linkers = data.get('linkers', [])
        
        # 加载环系模式
        if 'donor_ring_systems' in data:
            self.donor_ring_systems = [
                RingSystemPattern(**p) for p in data['donor_ring_systems']
            ]
        if 'acceptor_ring_systems' in data:
            self.acceptor_ring_systems = [
                RingSystemPattern(**p) for p in data['acceptor_ring_systems']
            ]
        
        self.stats = data.get('stats', {})
    
    def save_library(self, path: str):
        """保存片段库"""
        data = {
            'donors': self.donors,
            'acceptors': self.acceptors,
            'linkers': self.linkers,
            'donor_ring_systems': [
                {'smarts': p.smarts, 'name': p.name, 'fragment_type': p.fragment_type,
                 'description': p.description, 'examples': p.examples}
                for p in self.donor_ring_systems
            ],
            'acceptor_ring_systems': [
                {'smarts': p.smarts, 'name': p.name, 'fragment_type': p.fragment_type,
                 'description': p.description, 'examples': p.examples}
                for p in self.acceptor_ring_systems
            ],
            'stats': {
                'donor_count': len(self.donors),
                'acceptor_count': len(self.acceptors),
                'linker_count': len(self.linkers),
                'donor_ring_systems_count': len(self.donor_ring_systems),
                'acceptor_ring_systems_count': len(self.acceptor_ring_systems),
                'total_patterns': (len(self.donors) + len(self.acceptors) + len(self.linkers) +
                                  len(self.donor_ring_systems) + len(self.acceptor_ring_systems))
            }
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_all_donor_smarts(self) -> List[str]:
        """获取所有供体SMARTS（包括环系）"""
        return self.donors + [p.smarts for p in self.donor_ring_systems]
    
    def get_all_acceptor_smarts(self) -> List[str]:
        """获取所有受体SMARTS（包括环系）"""
        return self.acceptors + [p.smarts for p in self.acceptor_ring_systems]
    
    def print_summary(self):
        """打印摘要信息"""
        print("=" * 70)
        print("高级片段库 - 基于'环系整体标记'策略")
        print("=" * 70)
        print(f"\n📊 统计信息:")
        print(f"  基础供体: {len(self.donors)} 个")
        print(f"  基础受体: {len(self.acceptors)} 个")
        print(f"  连接子: {len(self.linkers)} 个")
        print(f"  供体环系: {len(self.donor_ring_systems)} 个")
        print(f"  受体环系: {len(self.acceptor_ring_systems)} 个")
        print(f"  总计: {len(self.donors) + len(self.acceptors) + len(self.linkers) + len(self.donor_ring_systems) + len(self.acceptor_ring_systems)} 个模式")


# =============================================================================
# 高级片段识别器类
# =============================================================================

class AdvancedFragmentIdentifier:
    """高级片段识别器 - 支持环系整体标记"""
    
    def __init__(self, fragment_library: AdvancedFragmentLibrary):
        self.library = fragment_library
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译SMARTS模式"""
        # 编译基础模式
        self.donor_mols = []
        self.acceptor_mols = []
        self.linker_mols = []
        
        for smarts in self.library.donors:
            mol = Chem.MolFromSmarts(smarts)
            if mol:
                self.donor_mols.append((mol, smarts, 'basic'))
        
        for smarts in self.library.acceptors:
            mol = Chem.MolFromSmarts(smarts)
            if mol:
                self.acceptor_mols.append((mol, smarts, 'basic'))
        
        for smarts in self.library.linkers:
            mol = Chem.MolFromSmarts(smarts)
            if mol:
                self.linker_mols.append((mol, smarts, 'basic'))
        
        # 编译环系模式
        self.donor_ring_mols = []
        self.acceptor_ring_mols = []
        
        for pattern in self.library.donor_ring_systems:
            mol = Chem.MolFromSmarts(pattern.smarts)
            if mol:
                self.donor_ring_mols.append((mol, pattern))
        
        for pattern in self.library.acceptor_ring_systems:
            mol = Chem.MolFromSmarts(pattern.smarts)
            if mol:
                self.acceptor_ring_mols.append((mol, pattern))
    
    def identify(self, smiles_or_mol: Union[str, Chem.Mol]) -> MoleculeAnnotation:
        """
        识别分子中的功能片段（包括环系整体标记）
        
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
        
        # 1. 识别基础供体
        for pattern_mol, smarts, ptype in self.donor_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.donors.append(FragmentMatch(
                    smiles=Chem.MolToSmiles(pattern_mol),
                    smarts=smarts,
                    fragment_type='donor',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='basic'
                ))
        
        # 2. 识别基础受体
        for pattern_mol, smarts, ptype in self.acceptor_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.acceptors.append(FragmentMatch(
                    smiles=Chem.MolToSmiles(pattern_mol),
                    smarts=smarts,
                    fragment_type='acceptor',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='basic'
                ))
        
        # 3. 识别连接子
        for pattern_mol, smarts, ptype in self.linker_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.linkers.append(FragmentMatch(
                    smiles=Chem.MolToSmiles(pattern_mol),
                    smarts=smarts,
                    fragment_type='linker',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='basic'
                ))
        
        # 4. 识别供体环系（环系整体标记）
        for pattern_mol, pattern in self.donor_ring_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.ring_system_donors.append(FragmentMatch(
                    smiles=Chem.MolToSmiles(pattern_mol),
                    smarts=pattern.smarts,
                    fragment_type='donor',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='ring_system',
                    ring_system_name=pattern.name
                ))
        
        # 5. 识别受体环系（环系整体标记）
        for pattern_mol, pattern in self.acceptor_ring_mols:
            matches = mol.GetSubstructMatches(pattern_mol)
            for match in matches:
                annotation.ring_system_acceptors.append(FragmentMatch(
                    smiles=Chem.MolToSmiles(pattern_mol),
                    smarts=pattern.smarts,
                    fragment_type='acceptor',
                    match_atoms=list(match),
                    match_score=1.0,
                    match_type='ring_system',
                    ring_system_name=pattern.name
                ))
        
        return annotation
    
    def has_donor(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含供体"""
        annotation = self.identify(smiles_or_mol)
        return annotation.total_donors > 0
    
    def has_acceptor(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含受体"""
        annotation = self.identify(smiles_or_mol)
        return annotation.total_acceptors > 0
    
    def has_ring_system_donor(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含环系供体"""
        annotation = self.identify(smiles_or_mol)
        return len(annotation.ring_system_donors) > 0
    
    def has_ring_system_acceptor(self, smiles_or_mol: Union[str, Chem.Mol]) -> bool:
        """检查分子是否包含环系受体"""
        annotation = self.identify(smiles_or_mol)
        return len(annotation.ring_system_acceptors) > 0


# =============================================================================
# 便捷函数
# =============================================================================

def create_advanced_library() -> AdvancedFragmentLibrary:
    """创建默认高级片段库"""
    return AdvancedFragmentLibrary()


def identify_fragments(smiles: str, library: AdvancedFragmentLibrary) -> MoleculeAnnotation:
    """识别分子中的功能片段"""
    identifier = AdvancedFragmentIdentifier(library)
    return identifier.identify(smiles)


# =============================================================================
# 主函数（命令行接口）
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='荧光分子功能片段库工具 V2.0')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 识别命令
    identify_parser = subparsers.add_parser('identify', help='识别分子中的功能片段')
    identify_parser.add_argument('smiles', help='分子SMILES')
    identify_parser.add_argument('--library', '-l', help='片段库路径')
    
    # 库信息命令
    info_parser = subparsers.add_parser('info', help='显示片段库信息')
    info_parser.add_argument('--library', '-l', help='片段库路径')
    
    args = parser.parse_args()
    
    if args.command == 'identify':
        library = AdvancedFragmentLibrary(args.library) if args.library else create_advanced_library()
        identifier = AdvancedFragmentIdentifier(library)
        annotation = identifier.identify(args.smiles)
        
        print(f"分子: {args.smiles}")
        print(f"\n基础供体: {len(annotation.donors)}")
        print(f"环系供体: {len(annotation.ring_system_donors)}")
        print(f"总供体: {annotation.total_donors}")
        print(f"\n基础受体: {len(annotation.acceptors)}")
        print(f"环系受体: {len(annotation.ring_system_acceptors)}")
        print(f"总受体: {annotation.total_acceptors}")
        print(f"\n连接子: {len(annotation.linkers)}")
        print(f"总片段数: {annotation.total_fragments}")
        
        if annotation.ring_system_donors:
            print("\n环系供体:")
            for d in annotation.ring_system_donors:
                print(f"  - {d.ring_system_name}: {d.smarts}")
        
        if annotation.ring_system_acceptors:
            print("\n环系受体:")
            for a in annotation.ring_system_acceptors:
                print(f"  - {a.ring_system_name}: {a.smarts}")
    
    elif args.command == 'info':
        library = AdvancedFragmentLibrary(args.library) if args.library else create_advanced_library()
        library.print_summary()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
