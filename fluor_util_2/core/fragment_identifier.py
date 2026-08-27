#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子功能片段识别器 (Molecular Fragment Identifier)

功能：识别分子中的电子供体、电子受体和连接子片段
作者：AI Assistant
版本：1.0.0
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os


def load_fragment_library(library_path='/Users/letiangao/Documents/GitHub/DiffEDMol/fluor_util_2/fluo_fragment_library.pkl'):
    """
    加载片段库
    
    Args:
        library_path: 片段库文件路径
        
    Returns:
        dict: 片段库字典，包含donors、acceptors、linkers三个类别
        
    Example:
        >>> library = load_fragment_library()
        >>> print(f"供体数量: {len(library['donors'])}")
    """
    with open(library_path, 'rb') as f:
        return pickle.load(f)


def preprocess_fragments(fragment_library):
    """
    预处理片段库，将SMILES转换为RDKit Mol对象
    
    Args:
        fragment_library: 片段库字典
        
    Returns:
        dict: 预处理后的片段库，每个片段包含mol对象
        
    Note:
        预处理可以提高后续识别效率，避免重复解析SMILES
    """
    processed = {
        'donors': [],
        'acceptors': [],
        'linkers': []
    }
    
    for category in ['donors', 'acceptors', 'linkers']:
        for fragment in fragment_library.get(category, []):
            smiles = fragment['smiles']
            frag_mol = Chem.MolFromSmiles(smiles)
            if frag_mol is not None:
                processed[category].append({
                    'name': fragment.get('name', 'unknown'),
                    'smiles': smiles,
                    'mol': frag_mol,
                    'type': fragment.get('type', category[:-1]),
                    'num_atoms': frag_mol.GetNumAtoms()
                })
    
    return processed


def calculate_match_score(mol, frag_mol, match_atoms):
    """
    计算匹配分数
    
    Args:
        mol: 目标分子 (RDKit Mol对象)
        frag_mol: 片段分子 (RDKit Mol对象)
        match_atoms: 匹配的原子索引列表
        
    Returns:
        float: 匹配分数 (0-1)，1.0表示完全匹配
        
    Note:
        匹配分数基于匹配原子数与片段总原子数的比值
    """
    if not match_atoms:
        return 0.0
    
    frag_atom_count = frag_mol.GetNumAtoms()
    match_atom_count = len(match_atoms)
    
    if frag_atom_count == 0:
        return 0.0
    
    base_score = match_atom_count / frag_atom_count
    
    if match_atom_count == frag_atom_count:
        base_score = 1.0
    
    return round(base_score, 3)


def identify_fragments(smiles_or_mol, fragment_library, match_threshold=0.5):
    """
    识别分子中的功能片段
    
    识别分子中包含的电子供体、电子受体和连接子片段，返回详细的匹配信息。
    
    Args:
        smiles_or_mol: SMILES字符串或RDKit Mol对象
        fragment_library: 片段库字典（原始或预处理后的）
        match_threshold: 匹配阈值 (0-1)，低于此值的匹配将被忽略
        
    Returns:
        dict: 识别结果字典，包含以下键：
            - donors: 供体片段列表
            - acceptors: 受体片段列表  
            - linkers: 连接子片段列表
            - summary: 摘要统计信息
            
        每个片段包含以下信息：
            - name: 片段名称
            - smiles: 片段SMILES
            - type: 片段类型
            - match_atoms: 匹配的原子索引列表
            - match_score: 匹配分数 (0-1)
            - num_atoms: 匹配的原子数
            
    Example:
        >>> library = load_fragment_library()
        >>> results = identify_fragments('CN(C)c1ccccc1', library)
        >>> print(f"发现 {results['summary']['total_donors']} 个供体")
        
    Raises:
        ValueError: 如果SMILES无法解析
    """
    # 解析输入分子
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"无法解析SMILES: {smiles_or_mol}")
    else:
        mol = smiles_or_mol
    
    # 预处理片段库（如果还没有预处理）
    if 'donors' in fragment_library and len(fragment_library['donors']) > 0:
        if 'mol' not in fragment_library['donors'][0]:
            fragment_library = preprocess_fragments(fragment_library)
    
    # 初始化结果
    results = {
        'donors': [],
        'acceptors': [],
        'linkers': [],
        'summary': {}
    }
    
    # 记录已匹配的原子，避免重复计数
    matched_atoms = set()
    
    # 识别各类片段
    for category in ['donors', 'acceptors', 'linkers']:
        for fragment in fragment_library.get(category, []):
            frag_mol = fragment['mol']
            
            # 使用RDKit进行子结构匹配
            matches = mol.GetSubstructMatches(frag_mol, uniquify=True)
            
            for match in matches:
                match_atoms = list(match)
                
                # 计算匹配分数
                score = calculate_match_score(mol, frag_mol, match_atoms)
                
                # 只保留超过阈值的匹配
                if score >= match_threshold:
                    # 检查是否有大量重叠
                    overlap = len(set(match_atoms) & matched_atoms)
                    overlap_ratio = overlap / len(match_atoms) if match_atoms else 0
                    
                    # 如果重叠超过80%，跳过
                    if overlap_ratio > 0.8:
                        continue
                    
                    # 添加到结果
                    results[category].append({
                        'name': fragment['name'],
                        'smiles': fragment['smiles'],
                        'type': fragment['type'],
                        'match_atoms': match_atoms,
                        'match_score': score,
                        'num_atoms': len(match_atoms)
                    })
                    
                    # 记录已匹配的原子
                    matched_atoms.update(match_atoms)
    
    # 按匹配分数排序
    for category in ['donors', 'acceptors', 'linkers']:
        results[category].sort(key=lambda x: x['match_score'], reverse=True)
    
    # 生成摘要
    results['summary'] = {
        'total_donors': len(results['donors']),
        'total_acceptors': len(results['acceptors']),
        'total_linkers': len(results['linkers']),
        'total_fragments': len(results['donors']) + len(results['acceptors']) + len(results['linkers']),
        'molecule_atoms': mol.GetNumAtoms(),
        'matched_atoms': len(matched_atoms),
        'coverage': round(len(matched_atoms) / mol.GetNumAtoms(), 3) if mol.GetNumAtoms() > 0 else 0
    }
    
    return results


def batch_identify(input_file, output_file, fragment_library, 
                   match_threshold=0.5, smiles_column='smiles', id_column='id'):
    """
    批量识别分子中的功能片段
    
    从CSV或SDF文件中批量读取分子，识别其中的功能片段，
    并将结果保存为标注的SDF文件。
    
    Args:
        input_file: 输入文件路径 (CSV或SDF格式)
        output_file: 输出SDF文件路径
        fragment_library: 片段库字典
        match_threshold: 匹配阈值 (0-1)
        smiles_column: CSV文件中SMILES列名（默认'smiles'）
        id_column: CSV文件中ID列名（默认'id'）
        
    Returns:
        dict: 处理统计信息
        {
            'total': 总分子数,
            'success': 成功处理的分子数,
            'failed': 失败的分子数,
            'total_fragments': 识别的总片段数,
            'output_sdf': 输出SDF文件路径,
            'output_csv': 输出CSV摘要文件路径
        }
        
    Example:
        >>> library = load_fragment_library()
        >>> stats = batch_identify('molecules.csv', 'output.sdf', library)
        >>> print(f"处理完成: {stats['success']}/{stats['total']}")
        
    Raises:
        ValueError: 如果文件格式不支持
    """
    import pandas as pd
    
    # 预处理片段库
    if 'donors' in fragment_library and len(fragment_library['donors']) > 0:
        if 'mol' not in fragment_library['donors'][0]:
            fragment_library = preprocess_fragments(fragment_library)
    
    results_list = []
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'total_fragments': 0
    }
    
    # 读取输入文件
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        # 读取CSV文件
        df = pd.read_csv(input_file)
        molecules = []
        for idx, row in df.iterrows():
            smiles = row.get(smiles_column, '')
            mol_id = row.get(id_column, f"mol_{idx}")
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol.SetProp('_Name', str(mol_id))
            molecules.append((mol_id, smiles, mol))
            
    elif file_ext == '.sdf':
        # 读取SDF文件
        supplier = Chem.SDMolSupplier(input_file)
        molecules = []
        for i, mol in enumerate(supplier):
            if mol is not None:
                mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{i}"
                smiles = Chem.MolToSmiles(mol)
                molecules.append((mol_id, smiles, mol))
            else:
                molecules.append((f"mol_{i}", '', None))
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}。请使用CSV或SDF格式。")
    
    # 处理每个分子
    writer = Chem.SDWriter(output_file)
    
    for mol_id, smiles, mol in molecules:
        stats['total'] += 1
        
        if mol is None:
            stats['failed'] += 1
            continue
        
        try:
            # 识别片段
            fragment_results = identify_fragments(mol, fragment_library, match_threshold)
            
            # 添加属性到分子
            mol.SetProp('Fragment_Donors', str(fragment_results['summary']['total_donors']))
            mol.SetProp('Fragment_Acceptors', str(fragment_results['summary']['total_acceptors']))
            mol.SetProp('Fragment_Linders', str(fragment_results['summary']['total_linkers']))
            mol.SetProp('Fragment_Total', str(fragment_results['summary']['total_fragments']))
            mol.SetProp('Fragment_Coverage', str(fragment_results['summary']['coverage']))
            
            # 添加详细片段信息
            donor_info = ';'.join([f"{d['name']}({d['match_score']})" for d in fragment_results['donors']])
            acceptor_info = ';'.join([f"{a['name']}({a['match_score']})" for a in fragment_results['acceptors']])
            linker_info = ';'.join([f"{l['name']}({l['match_score']})" for l in fragment_results['linkers']])
            
            mol.SetProp('Donor_Details', donor_info if donor_info else 'None')
            mol.SetProp('Acceptor_Details', acceptor_info if acceptor_info else 'None')
            mol.SetProp('Linker_Details', linker_info if linker_info else 'None')
            
            # 写入SDF
            writer.write(mol)
            
            # 记录结果
            results_list.append({
                'id': mol_id,
                'smiles': smiles,
                'donors': fragment_results['summary']['total_donors'],
                'acceptors': fragment_results['summary']['total_acceptors'],
                'linkers': fragment_results['summary']['total_linkers'],
                'total_fragments': fragment_results['summary']['total_fragments'],
                'coverage': fragment_results['summary']['coverage'],
                'success': True
            })
            
            stats['success'] += 1
            stats['total_fragments'] += fragment_results['summary']['total_fragments']
            
        except Exception as e:
            stats['failed'] += 1
            results_list.append({
                'id': mol_id,
                'smiles': smiles,
                'error': str(e),
                'success': False
            })
    
    writer.close()
    
    # 保存CSV摘要
    csv_output = output_file.replace('.sdf', '_summary.csv')
    summary_df = pd.DataFrame(results_list)
    summary_df.to_csv(csv_output, index=False)
    
    stats['output_sdf'] = output_file
    stats['output_csv'] = csv_output
    
    return stats


def get_fragment_statistics(results):
    """
    获取片段识别结果的统计信息
    
    Args:
        results: identify_fragments返回的结果字典
        
    Returns:
        dict: 格式化后的统计信息
    """
    summary = results['summary']
    
    stats = {
        '分子总原子数': summary['molecule_atoms'],
        '匹配原子数': summary['matched_atoms'],
        '覆盖率': f"{summary['coverage']*100:.1f}%",
        '供体片段数': summary['total_donors'],
        '受体片段数': summary['total_acceptors'],
        '连接子片段数': summary['total_linkers'],
        '总片段数': summary['total_fragments'],
        '供体列表': [d['name'] for d in results['donors']],
        '受体列表': [a['name'] for a in results['acceptors']],
        '连接子列表': [l['name'] for l in results['linkers']]
    }
    
    return stats


def visualize_fragments(smiles_or_mol, fragment_library, output_path=None):
    """
    可视化分子中的功能片段（需要rdkit.Chem.Draw）
    
    Args:
        smiles_or_mol: SMILES字符串或RDKit Mol对象
        fragment_library: 片段库字典
        output_path: 输出图片路径（可选）
        
    Returns:
        PIL.Image: 分子图片（如果可用）
    """
    try:
        from rdkit.Chem import Draw
        
        # 解析分子
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol
            
        if mol is None:
            return None
        
        # 识别片段
        results = identify_fragments(mol, fragment_library)
        
        # 收集所有匹配的原子
        highlight_atoms = []
        atom_colors = {}
        
        # 供体用绿色
        for donor in results['donors']:
            for atom_idx in donor['match_atoms']:
                if atom_idx not in [a for d in results['donors'] for a in d['match_atoms']]:
                    highlight_atoms.append(atom_idx)
                atom_colors[atom_idx] = (0.0, 1.0, 0.0)  # 绿色
        
        # 受体用红色
        for acceptor in results['acceptors']:
            for atom_idx in acceptor['match_atoms']:
                atom_colors[atom_idx] = (1.0, 0.0, 0.0)  # 红色
        
        # 连接子用蓝色
        for linker in results['linkers']:
            for atom_idx in linker['match_atoms']:
                atom_colors[atom_idx] = (0.0, 0.0, 1.0)  # 蓝色
        
        # 生成图片
        img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, 
                              highlightAtomColors=atom_colors,
                              size=(400, 400))
        
        if output_path:
            img.save(output_path)
            
        return img
        
    except ImportError:
        print("警告: rdkit.Chem.Draw 不可用，无法生成图片")
        return None


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    import sys
    
    # 简单的命令行接口
    if len(sys.argv) < 2:
        print("用法:")
        print("  python fragment_identifier.py <SMILES>")
        print("  python fragment_identifier.py --batch <input.csv> <output.sdf>")
        sys.exit(1)
    
    # 加载片段库
    library = load_fragment_library()
    
    if sys.argv[1] == '--batch':
        # 批量模式
        if len(sys.argv) < 4:
            print("批量模式需要输入和输出文件路径")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        
        print(f"批量处理: {input_file} -> {output_file}")
        stats = batch_identify(input_file, output_file, library)
        
        print("\n处理统计:")
        print(f"  总分子数: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        print(f"  总片段数: {stats['total_fragments']}")
        print(f"  输出SDF: {stats['output_sdf']}")
        print(f"  输出CSV: {stats['output_csv']}")
    else:
        # 单分子模式
        smiles = sys.argv[1]
        print(f"识别分子: {smiles}")
        print("="*60)
        
        try:
            results = identify_fragments(smiles, library)
            
            print("\n识别结果:")
            print(f"供体片段 ({results['summary']['total_donors']}个):")
            for d in results['donors']:
                print(f"  - {d['name']}: {d['smiles']}")
                print(f"    匹配原子: {d['match_atoms']}, 分数: {d['match_score']}")
            
            print(f"\n受体片段 ({results['summary']['total_acceptors']}个):")
            for a in results['acceptors']:
                print(f"  - {a['name']}: {a['smiles']}")
                print(f"    匹配原子: {a['match_atoms']}, 分数: {a['match_score']}")
            
            print(f"\n连接子片段 ({results['summary']['total_linkers']}个):")
            for l in results['linkers']:
                print(f"  - {l['name']}: {l['smiles']}")
                print(f"    匹配原子: {l['match_atoms']}, 分数: {l['match_score']}")
            
            print(f"\n摘要:")
            print(f"  总片段数: {results['summary']['total_fragments']}")
            print(f"  分子原子数: {results['summary']['molecule_atoms']}")
            print(f"  匹配原子数: {results['summary']['matched_atoms']}")
            print(f"  覆盖率: {results['summary']['coverage']*100:.1f}%")
            
        except Exception as e:
            print(f"错误: {e}")
            sys.exit(1)
