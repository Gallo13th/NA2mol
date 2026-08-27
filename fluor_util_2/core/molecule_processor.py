"""
分子切分与SDF输出功能模块

功能：
1. 将分子切分为BM（Bemis-Murcko）骨架
2. 识别骨架上的功能片段位置
3. 输出标注了功能片段的SDF文件

作者: AI Assistant
日期: 2024
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
import os
import json
import csv
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ScaffoldInfo:
    """骨架信息数据类"""
    scaffold_smiles: str
    scaffold_type: str  # 'BM' for Bemis-Murcko, 'CSK' for Core Scaffold
    atoms: List[int]  # 骨架中的原子索引
    original_atoms: List[int]  # 原始分子中的原子索引映射
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class FragmentAnnotation:
    """片段标注信息数据类"""
    donor_atoms: List[List[int]]
    acceptor_atoms: List[List[int]]
    linker_atoms: List[List[int]]
    donor_smiles: List[str]
    acceptor_smiles: List[str]
    linker_smiles: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


def split_to_bm_scaffolds(mol: Chem.Mol) -> List[Dict]:
    """
    将分子切分为BM（Bemis-Murcko）骨架
    
    参数:
        mol: RDKit Mol对象
        
    返回:
        骨架信息列表，每个元素包含:
        - scaffold_smiles: 骨架SMILES
        - scaffold_type: 骨架类型 ('BM' 或 'CSK')
        - atoms: 骨架中的原子索引列表
        - original_atoms: 原始分子中对应的原子索引列表
    """
    if mol is None:
        return []
    
    # 确保分子没有氢原子（用于正确识别骨架）
    mol = Chem.RemoveHs(mol)
    
    scaffolds = []
    
    # 1. 获取Bemis-Murcko骨架（包含所有环和连接它们的链）
    try:
        bm_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if bm_scaffold is not None and bm_scaffold.GetNumAtoms() > 0:
            # 获取BM骨架的原子映射
            bm_match = mol.GetSubstructMatch(bm_scaffold)
            if bm_match:
                scaffolds.append({
                    'scaffold_smiles': Chem.MolToSmiles(bm_scaffold),
                    'scaffold_type': 'BM',
                    'atoms': list(range(bm_scaffold.GetNumAtoms())),
                    'original_atoms': list(bm_match)
                })
    except Exception as e:
        print(f"获取BM骨架时出错: {e}")
    
    # 2. 获取核心骨架（仅保留环，移除连接链）
    try:
        csk_scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
        if csk_scaffold is not None and csk_scaffold.GetNumAtoms() > 0:
            csk_match = mol.GetSubstructMatch(csk_scaffold)
            if csk_match:
                scaffolds.append({
                    'scaffold_smiles': Chem.MolToSmiles(csk_scaffold),
                    'scaffold_type': 'CSK',
                    'atoms': list(range(csk_scaffold.GetNumAtoms())),
                    'original_atoms': list(csk_match)
                })
    except Exception as e:
        print(f"获取CSK骨架时出错: {e}")
    
    # 3. 如果以上方法都失败，使用BRICS分解获取骨架片段
    if len(scaffolds) == 0:
        try:
            frags = BRICS.BRICSDecompose(mol)
            for i, frag_smiles in enumerate(frags):
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol:
                    frag_match = mol.GetSubstructMatch(frag_mol)
                    if frag_match:
                        scaffolds.append({
                            'scaffold_smiles': frag_smiles,
                            'scaffold_type': 'BRICS',
                            'atoms': list(range(frag_mol.GetNumAtoms())),
                            'original_atoms': list(frag_match)
                        })
        except Exception as e:
            print(f"BRICS分解时出错: {e}")
    
    return scaffolds


def save_annotated_sdf(mol: Chem.Mol, 
                       fragment_annotations: Dict, 
                       output_path: str,
                       scaffold_info: List[Dict] = None) -> bool:
    """
    将标注了功能片段的分子输出为SDF格式
    
    参数:
        mol: RDKit Mol对象
        fragment_annotations: 片段标注信息，格式:
            {
                'donor_atoms': [[原子索引列表], ...],
                'acceptor_atoms': [[原子索引列表], ...],
                'linker_atoms': [[原子索引列表], ...],
                'donor_smiles': ['片段SMILES', ...],
                'acceptor_smiles': ['片段SMILES', ...],
                'linker_smiles': ['片段SMILES', ...]
            }
        output_path: 输出SDF文件路径
        scaffold_info: 可选的骨架信息
        
    返回:
        bool: 是否成功保存
    """
    if mol is None:
        print("错误: 输入分子为None")
        return False
    
    try:
        # 创建工作副本，不修改原始分子
        mol_copy = Chem.RWMol(mol)
        
        # 添加3D坐标（如果没有）
        if mol_copy.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol_copy)
        
        # 创建SDF写入器
        writer = Chem.SDWriter(output_path)
        
        # 添加属性字段
        mol_copy.SetProp("_Name", "AnnotatedMolecule")
        mol_copy.SetProp("SMILES", Chem.MolToSmiles(mol))
        
        # 添加片段标注信息
        if fragment_annotations:
            # Donor信息
            if 'donor_atoms' in fragment_annotations:
                donor_atoms_json = json.dumps(fragment_annotations['donor_atoms'])
                mol_copy.SetProp("Donor_Atoms", donor_atoms_json)
            if 'donor_smiles' in fragment_annotations:
                donor_smiles_json = json.dumps(fragment_annotations['donor_smiles'])
                mol_copy.SetProp("Donor_SMILES", donor_smiles_json)
            
            # Acceptor信息
            if 'acceptor_atoms' in fragment_annotations:
                acceptor_atoms_json = json.dumps(fragment_annotations['acceptor_atoms'])
                mol_copy.SetProp("Acceptor_Atoms", acceptor_atoms_json)
            if 'acceptor_smiles' in fragment_annotations:
                acceptor_smiles_json = json.dumps(fragment_annotations['acceptor_smiles'])
                mol_copy.SetProp("Acceptor_SMILES", acceptor_smiles_json)
            
            # Linker信息
            if 'linker_atoms' in fragment_annotations:
                linker_atoms_json = json.dumps(fragment_annotations['linker_atoms'])
                mol_copy.SetProp("Linker_Atoms", linker_atoms_json)
            if 'linker_smiles' in fragment_annotations:
                linker_smiles_json = json.dumps(fragment_annotations['linker_smiles'])
                mol_copy.SetProp("Linker_SMILES", linker_smiles_json)
        
        # 添加骨架信息
        if scaffold_info:
            scaffold_json = json.dumps(scaffold_info)
            mol_copy.SetProp("Scaffold_Info", scaffold_json)
        
        # 写入SDF文件
        writer.write(mol_copy)
        writer.close()
        
        print(f"SDF文件已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"保存SDF文件时出错: {e}")
        return False


def match_fragments_on_scaffold(mol: Chem.Mol, 
                                 scaffold_atoms: List[int],
                                 fragment_library: Dict) -> Dict:
    """
    在骨架上匹配功能片段
    
    参数:
        mol: 原始分子
        scaffold_atoms: 骨架在原始分子中的原子索引
        fragment_library: 片段库，格式:
            {
                'donors': [{'smiles': '...', 'name': '...'}, ...],
                'acceptors': [{'smiles': '...', 'name': '...'}, ...],
                'linkers': [{'smiles': '...', 'name': '...'}, ...]
            }
            
    返回:
        片段标注信息字典
    """
    if mol is None or not scaffold_atoms:
        return {
            'donor_atoms': [], 'acceptor_atoms': [], 'linker_atoms': [],
            'donor_smiles': [], 'acceptor_smiles': [], 'linker_smiles': []
        }
    
    # 创建骨架原子集合用于快速查找
    scaffold_set = set(scaffold_atoms)
    
    annotations = {
        'donor_atoms': [],
        'acceptor_atoms': [],
        'linker_atoms': [],
        'donor_smiles': [],
        'acceptor_smiles': [],
        'linker_smiles': []
    }
    
    # 匹配donor片段
    if 'donors' in fragment_library:
        for donor in fragment_library['donors']:
            donor_mol = Chem.MolFromSmiles(donor['smiles'])
            if donor_mol:
                matches = mol.GetSubstructMatches(donor_mol)
                for match in matches:
                    # 检查是否至少有一个原子在骨架上
                    if any(idx in scaffold_set for idx in match):
                        annotations['donor_atoms'].append(list(match))
                        annotations['donor_smiles'].append(donor['smiles'])
    
    # 匹配acceptor片段
    if 'acceptors' in fragment_library:
        for acceptor in fragment_library['acceptors']:
            acceptor_mol = Chem.MolFromSmiles(acceptor['smiles'])
            if acceptor_mol:
                matches = mol.GetSubstructMatches(acceptor_mol)
                for match in matches:
                    if any(idx in scaffold_set for idx in match):
                        annotations['acceptor_atoms'].append(list(match))
                        annotations['acceptor_smiles'].append(acceptor['smiles'])
    
    # 匹配linker片段
    if 'linkers' in fragment_library:
        for linker in fragment_library['linkers']:
            linker_mol = Chem.MolFromSmiles(linker['smiles'])
            if linker_mol:
                matches = mol.GetSubstructMatches(linker_mol)
                for match in matches:
                    if any(idx in scaffold_set for idx in match):
                        annotations['linker_atoms'].append(list(match))
                        annotations['linker_smiles'].append(linker['smiles'])
    
    return annotations


def process_molecule(input_smiles: str, 
                     output_sdf_path: str, 
                     fragment_library: Dict,
                     scaffold_matcher=None) -> Dict:
    """
    主流程函数：处理单个分子
    
    流程:
    1. 从SMILES解析分子
    2. 切分BM骨架
    3. 匹配功能片段
    4. 输出标注的SDF
    
    参数:
        input_smiles: 输入分子的SMILES字符串
        output_sdf_path: 输出SDF文件路径
        fragment_library: 片段库字典
        scaffold_matcher: 可选的骨架匹配器对象
        
    返回:
        处理结果字典，包含:
        - success: 是否成功
        - scaffolds: 骨架信息列表
        - annotations: 片段标注信息
        - output_path: 输出文件路径
        - error: 错误信息（如果有）
    """
    result = {
        'success': False,
        'scaffolds': [],
        'annotations': {},
        'output_path': output_sdf_path,
        'error': None
    }
    
    try:
        # 1. 解析SMILES
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            result['error'] = f"无法解析SMILES: {input_smiles}"
            return result
        
        print(f"处理分子: {input_smiles}")
        
        # 2. 切分BM骨架
        scaffolds = split_to_bm_scaffolds(mol)
        result['scaffolds'] = scaffolds
        print(f"  找到 {len(scaffolds)} 个骨架")
        
        # 3. 匹配功能片段
        all_annotations = {
            'donor_atoms': [],
            'acceptor_atoms': [],
            'linker_atoms': [],
            'donor_smiles': [],
            'acceptor_smiles': [],
            'linker_smiles': []
        }
        
        for scaffold in scaffolds:
            # 使用骨架匹配器（如果提供）
            if scaffold_matcher and hasattr(scaffold_matcher, 'match'):
                annotations = scaffold_matcher.match(mol, scaffold['original_atoms'], fragment_library)
            else:
                # 使用默认匹配方法
                annotations = match_fragments_on_scaffold(mol, scaffold['original_atoms'], fragment_library)
            
            # 合并标注信息
            for key in all_annotations:
                if key in annotations:
                    all_annotations[key].extend(annotations[key])
        
        # 去重
        for key in ['donor_atoms', 'acceptor_atoms', 'linker_atoms']:
            # 将列表转换为元组以便去重
            unique_items = []
            seen = set()
            for item in all_annotations[key]:
                item_tuple = tuple(sorted(item))
                if item_tuple not in seen:
                    seen.add(item_tuple)
                    unique_items.append(item)
            all_annotations[key] = unique_items
        
        # 对应的SMILES也去重
        for key in ['donor_smiles', 'acceptor_smiles', 'linker_smiles']:
            all_annotations[key] = list(dict.fromkeys(all_annotations[key]))
        
        result['annotations'] = all_annotations
        print(f"  Donor片段: {len(all_annotations['donor_atoms'])}")
        print(f"  Acceptor片段: {len(all_annotations['acceptor_atoms'])}")
        print(f"  Linker片段: {len(all_annotations['linker_atoms'])}")
        
        # 4. 输出SDF
        success = save_annotated_sdf(mol, all_annotations, output_sdf_path, scaffolds)
        result['success'] = success
        
        if success:
            print(f"  输出文件: {output_sdf_path}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"处理分子时出错: {e}")
    
    return result


def batch_process(input_csv: str, 
                  output_dir: str, 
                  fragment_library: Dict,
                  scaffold_matcher=None,
                  smiles_column: str = 'SMILES',
                  id_column: str = None) -> List[Dict]:
    """
    批量处理分子
    
    参数:
        input_csv: 输入CSV文件路径，需要包含SMILES列
        output_dir: 输出目录路径
        fragment_library: 片段库字典
        scaffold_matcher: 可选的骨架匹配器对象
        smiles_column: SMILES列名，默认为'SMILES'
        id_column: 分子ID列名，如果为None则使用行号
        
    返回:
        处理结果列表，每个元素是一个分子的处理结果
    """
    results = []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return results
    
    print(f"开始批量处理 {len(rows)} 个分子...")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 处理每个分子
    for i, row in enumerate(rows):
        smiles = row.get(smiles_column)
        if not smiles:
            print(f"跳过第 {i+1} 行: 缺少SMILES")
            continue
        
        # 生成分子ID
        if id_column and id_column in row:
            mol_id = row[id_column]
        else:
            mol_id = f"mol_{i+1:04d}"
        
        # 生成输出文件路径
        output_path = os.path.join(output_dir, f"{mol_id}.sdf")
        
        print(f"\n[{i+1}/{len(rows)}] 处理分子: {mol_id}")
        
        # 处理分子
        result = process_molecule(smiles, output_path, fragment_library, scaffold_matcher)
        result['mol_id'] = mol_id
        result['input_smiles'] = smiles
        results.append(result)
    
    # 输出统计信息
    print("\n" + "=" * 50)
    print("批量处理完成!")
    print(f"总分子数: {len(rows)}")
    print(f"成功处理: {sum(1 for r in results if r['success'])}")
    print(f"失败: {sum(1 for r in results if not r['success'])}")
    
    return results


# =============================================================================
# 辅助函数
# =============================================================================

def create_default_fragment_library() -> Dict:
    """
    创建默认的片段库（示例）
    """
    return {
        'donors': [
            {'smiles': 'c1ccccc1N', 'name': 'aniline'},
            {'smiles': 'c1ccccc1O', 'name': 'phenol'},
            {'smiles': 'N', 'name': 'amine'},
            {'smiles': 'c1ccc2c(c1)ccc3c2ccc4c3cccc4', 'name': 'triphenylene'},
            {'smiles': 'c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccccc5', 'name': 'pentacene'},
        ],
        'acceptors': [
            {'smiles': 'c1ccccc1C#N', 'name': 'benzonitrile'},
            {'smiles': 'c1ccccc1F', 'name': 'fluorobenzene'},
            {'smiles': 'c1ccccc1C(=O)', 'name': 'benzaldehyde'},
            {'smiles': 'c1ccc2nc3ccccc3cc2c1', 'name': 'quinoline'},
        ],
        'linkers': [
            {'smiles': 'C=C', 'name': 'vinyl'},
            {'smiles': 'C#C', 'name': 'ethynyl'},
            {'smiles': 'c1ccccc1', 'name': 'phenyl'},
            {'smiles': 'C', 'name': 'methyl'},
        ]
    }


def read_fragment_library(library_path: str) -> Dict:
    """
    从JSON文件读取片段库
    
    参数:
        library_path: 片段库JSON文件路径
        
    返回:
        片段库字典
    """
    try:
        with open(library_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取片段库时出错: {e}")
        return create_default_fragment_library()


def save_fragment_library(library: Dict, output_path: str) -> bool:
    """
    保存片段库到JSON文件
    
    参数:
        library: 片段库字典
        output_path: 输出文件路径
        
    返回:
        bool: 是否成功保存
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(library, f, indent=2, ensure_ascii=False)
        print(f"片段库已保存: {output_path}")
        return True
    except Exception as e:
        print(f"保存片段库时出错: {e}")
        return False


class ScaffoldMatcher:
    """
    骨架片段匹配器类
    用于自定义骨架上的片段匹配逻辑
    """
    
    def __init__(self, match_tolerance: float = 0.1):
        """
        初始化匹配器
        
        参数:
            match_tolerance: 匹配容差
        """
        self.match_tolerance = match_tolerance
    
    def match(self, mol: Chem.Mol, scaffold_atoms: List[int], 
              fragment_library: Dict) -> Dict:
        """
        在骨架上匹配功能片段
        
        参数:
            mol: 原始分子
            scaffold_atoms: 骨架原子索引
            fragment_library: 片段库
            
        返回:
            片段标注信息
        """
        return match_fragments_on_scaffold(mol, scaffold_atoms, fragment_library)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("分子切分与SDF输出功能模块")
    print("=" * 60)
    
    # 创建示例片段库
    fragment_lib = create_default_fragment_library()
    print("\n示例片段库:")
    print(f"  Donors: {len(fragment_lib['donors'])} 个")
    print(f"  Acceptors: {len(fragment_lib['acceptors'])} 个")
    print(f"  Linkers: {len(fragment_lib['linkers'])} 个")
    
    # 示例分子
    example_smiles = "c1ccc2c(c1)ccc3c2ccc4c3cccc4"  # triphenylene
    print(f"\n示例分子 SMILES: {example_smiles}")
    
    # 处理分子
    output_path = "example_output.sdf"
    result = process_molecule(example_smiles, output_path, fragment_lib)
    
    if result['success']:
        print("\n处理成功!")
        print(f"骨架数量: {len(result['scaffolds'])}")
        print(f"Donor片段: {len(result['annotations']['donor_atoms'])}")
        print(f"Acceptor片段: {len(result['annotations']['acceptor_atoms'])}")
        print(f"Linker片段: {len(result['annotations']['linker_atoms'])}")
    else:
        print(f"\n处理失败: {result['error']}")
