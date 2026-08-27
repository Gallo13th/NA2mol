#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块 - 分子结构可视化与荧光片段高亮显示

本模块提供荧光分子的可视化功能，支持：
1. 单个分子可视化与高亮
2. 多个分子的网格可视化
3. 基于片段识别结果的高亮显示
4. 从SDF文件读取标注信息进行可视化

作者: AI Assistant
版本: 2.0.0
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from typing import List, Dict, Optional, Tuple, Union
import io
from PIL import Image, ImageDraw, ImageFont


class MoleculeVisualizer:
    """分子可视化器 - 提供2D分子结构绘制与片段高亮功能"""

    def __init__(self, width: int = 600, height: int = 400):
        """
        初始化可视化器
        
        Args:
            width: 默认图像宽度
            height: 默认图像高度
        """
        self.default_width = width
        self.default_height = height
        # 颜色映射 (R, G, B) 0-1范围
        self.colors = {
            "donor": (0.2, 0.8, 0.2),      # 绿色
            "acceptor": (0.9, 0.2, 0.2),    # 红色
            "linker": (0.2, 0.4, 0.9),      # 蓝色
            "neutral": (0.8, 0.8, 0.8),     # 灰色
            "highlight": (1.0, 0.8, 0.0),   # 黄色
        }

    def mol_to_image(self, mol, width=None, height=None, 
                     highlight_atoms=None, highlight_bonds=None,
                     highlight_atom_colors=None, highlight_bond_colors=None):
        """
        将分子转换为图像
        
        Args:
            mol: RDKit Mol对象
            width: 图像宽度
            height: 图像高度
            highlight_atoms: 要高亮的原子索引列表
            highlight_bonds: 要高亮的键索引列表
            highlight_atom_colors: 原子颜色字典 {原子索引: (R, G, B)}
            highlight_bond_colors: 键颜色字典 {键索引: (R, G, B)}
            
        Returns:
            PIL.Image: 分子图像
        """
        if mol is None:
            w, h = width or 300, height or 300
            return Image.new("RGB", (w, h), "white")
        
        width = width or self.default_width
        height = height or self.default_height
        
        mol = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol)
        
        # 构建原子颜色字典
        atom_colors = {}
        if highlight_atom_colors:
            atom_colors = highlight_atom_colors
        elif highlight_atoms:
            for idx in highlight_atoms:
                atom_colors[idx] = self.colors["highlight"]
        
        # 构建键颜色字典
        bond_colors = {}
        if highlight_bond_colors:
            bond_colors = highlight_bond_colors
        elif highlight_bonds:
            for idx in highlight_bonds:
                bond_colors[idx] = self.colors["highlight"]
        
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = 2
        opts.atomLabelFontSize = 14
        
        if atom_colors or bond_colors:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(atom_colors.keys()) if atom_colors else [],
                highlightBonds=list(bond_colors.keys()) if bond_colors else [],
                highlightAtomColors=atom_colors,
                highlightBondColors=bond_colors
            )
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_data))

    def visualize_single(self, mol, title="", highlight_atoms=None, 
                        highlight_atom_colors=None, save_path=None, show_legend=True):
        """
        可视化单个分子
        
        Args:
            mol: RDKit Mol对象或SMILES字符串
            title: 图像标题
            highlight_atoms: 要高亮的原子索引列表
            highlight_atom_colors: 原子颜色字典 {原子索引: (R, G, B)}
            save_path: 保存路径（可选）
            show_legend: 是否显示图例
            
        Returns:
            PIL.Image或保存路径
        """
        # 解析SMILES
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            img = Image.new("RGB", (self.default_width, self.default_height), "white")
            if save_path:
                img.save(save_path, "PNG")
                return save_path
            return img
        
        img = self.mol_to_image(mol, highlight_atoms=highlight_atoms,
                               highlight_atom_colors=highlight_atom_colors)
        
        if title or show_legend:
            new_height = img.height + 60
            new_img = Image.new("RGB", (img.width, new_height), "white")
            new_img.paste(img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            
            # 安全添加文本（处理编码问题）
            if title:
                safe_title = title.encode('ascii', 'ignore').decode('ascii') if title else ""
                draw.text((10, img.height + 10), safe_title, fill="black")
            
            # 显示图例
            if show_legend and highlight_atom_colors:
                legend_items = []
                colors_used = set(highlight_atom_colors.values())
                
                if self.colors["donor"] in colors_used:
                    legend_items.append(("Donor", self.colors["donor"]))
                if self.colors["acceptor"] in colors_used:
                    legend_items.append(("Acceptor", self.colors["acceptor"]))
                if self.colors["linker"] in colors_used:
                    legend_items.append(("Linker", self.colors["linker"]))
                
                legend_x = img.width - 120
                legend_y = img.height + 10
                box_size = 15
                
                for label, color in legend_items:
                    r, g, b = color
                    rgb_color = (int(r*255), int(g*255), int(b*255))
                    draw.rectangle([legend_x, legend_y, legend_x + box_size, legend_y + box_size],
                                   fill=rgb_color, outline="black")
                    draw.text((legend_x + box_size + 5, legend_y), label, fill="black")
                    legend_y += 20
            
            img = new_img
        
        if save_path:
            img.save(save_path, "PNG")
            return save_path
        return img

    def visualize_grid(self, mols, labels=None, mols_per_row=4, sub_img_size=(300, 250),
                      highlight_lists=None, highlight_color_lists=None, 
                      title="", save_path=None):
        """
        网格可视化多个分子
        
        Args:
            mols: 分子列表（RDKit Mol对象或SMILES字符串）
            labels: 分子标签列表
            mols_per_row: 每行显示的分子数
            sub_img_size: 每个分子图像的大小 (宽, 高)
            highlight_lists: 每个分子的高亮原子列表的列表
            highlight_color_lists: 每个分子的原子颜色字典列表
            title: 图像标题
            save_path: 保存路径（可选）
            
        Returns:
            PIL.Image或保存路径
        """
        if not mols:
            return Image.new("RGB", (300, 300), "white")
        
        # 解析SMILES
        parsed_mols = []
        for mol in mols:
            if isinstance(mol, str):
                parsed_mol = Chem.MolFromSmiles(mol)
                parsed_mols.append(parsed_mol)
            else:
                parsed_mols.append(mol)
        mols = parsed_mols
        
        n_mols = len(mols)
        n_rows = (n_mols + mols_per_row - 1) // mols_per_row
        
        margin = 10
        header_height = 50 if title else 0
        footer_height = 40
        
        total_width = mols_per_row * sub_img_size[0] + (mols_per_row + 1) * margin
        total_height = n_rows * sub_img_size[1] + (n_rows + 1) * margin + header_height + footer_height
        
        grid_img = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(grid_img)
        
        # 安全添加标题
        if title:
            safe_title = title.encode('ascii', 'ignore').decode('ascii') if title else ""
            draw.text((10, 15), safe_title, fill="black")
        
        for idx, mol in enumerate(mols):
            if mol is None:
                continue
            
            row = idx // mols_per_row
            col = idx % mols_per_row
            
            x = margin + col * (sub_img_size[0] + margin)
            y = header_height + margin + row * (sub_img_size[1] + margin)
            
            # 获取高亮信息
            highlight_atoms = None
            highlight_atom_colors = None
            if highlight_lists and idx < len(highlight_lists):
                highlight_atoms = highlight_lists[idx]
            if highlight_color_lists and idx < len(highlight_color_lists):
                highlight_atom_colors = highlight_color_lists[idx]
            
            mol_img = self.mol_to_image(mol, width=sub_img_size[0], 
                                       height=sub_img_size[1]-30, 
                                       highlight_atoms=highlight_atoms,
                                       highlight_atom_colors=highlight_atom_colors)
            
            cell = Image.new("RGB", sub_img_size, "white")
            cell.paste(mol_img, (0, 0))
            
            # 安全添加标签
            if labels and idx < len(labels):
                cell_draw = ImageDraw.Draw(cell)
                label = labels[idx]
                safe_label = label.encode('ascii', 'ignore').decode('ascii') if label else ""
                if len(safe_label) > 35:
                    safe_label = safe_label[:32] + "..."
                cell_draw.text((5, sub_img_size[1] - 25), safe_label, fill="black")
            
            grid_img.paste(cell, (x, y))
        
        if save_path:
            grid_img.save(save_path, "PNG")
            return save_path
        return grid_img

    def visualize_fragment_result(self, mol, fragment_result, title="Fluorescence Analysis", 
                                  save_path=None):
        """
        根据片段识别结果可视化分子
        
        Args:
            mol: RDKit Mol对象或SMILES字符串
            fragment_result: 片段识别结果字典，格式为：
                {
                    'donors': [{'match_atoms': [0, 1, 2], ...}, ...],
                    'acceptors': [{'match_atoms': [3, 4], ...}, ...],
                    'linkers': [{'match_atoms': [5], ...}, ...]
                }
            title: 图像标题
            save_path: 保存路径（可选）
            
        Returns:
            PIL.Image或保存路径
        """
        # 解析SMILES
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            img = Image.new("RGB", (self.default_width, self.default_height), "white")
            if save_path:
                img.save(save_path, "PNG")
                return save_path
            return img
        
        # 构建高亮原子颜色字典
        highlight_atoms = {}
        
        # 高亮供体（绿色）
        if 'donors' in fragment_result:
            for donor in fragment_result['donors']:
                if 'match_atoms' in donor:
                    for atom_idx in donor['match_atoms']:
                        highlight_atoms[atom_idx] = self.colors["donor"]
        
        # 高亮受体（红色）
        if 'acceptors' in fragment_result:
            for acceptor in fragment_result['acceptors']:
                if 'match_atoms' in acceptor:
                    for atom_idx in acceptor['match_atoms']:
                        highlight_atoms[atom_idx] = self.colors["acceptor"]
        
        # 高亮连接子（蓝色）
        if 'linkers' in fragment_result:
            for linker in fragment_result['linkers']:
                if 'match_atoms' in linker:
                    for atom_idx in linker['match_atoms']:
                        highlight_atoms[atom_idx] = self.colors["linker"]
        
        return self.visualize_single(mol, title=title, 
                                    highlight_atom_colors=highlight_atoms if highlight_atoms else None,
                                    save_path=save_path, show_legend=True)

    def visualize_annotation(self, mol, annotation, title="Fluorescence Analysis", 
                            save_path=None):
        """
        根据MoleculeAnnotation对象可视化分子
        
        Args:
            mol: RDKit Mol对象或SMILES字符串
            annotation: MoleculeAnnotation对象（来自fluo_utils.py）
            title: 图像标题
            save_path: 保存路径（可选）
            
        Returns:
            PIL.Image或保存路径
        """
        # 解析SMILES
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            img = Image.new("RGB", (self.default_width, self.default_height), "white")
            if save_path:
                img.save(save_path, "PNG")
                return save_path
            return img
        
        # 构建高亮原子颜色字典
        highlight_atoms = {}
        
        # 高亮供体（绿色）
        if hasattr(annotation, 'donors'):
            for donor in annotation.donors:
                if hasattr(donor, 'match_atoms'):
                    for atom_idx in donor.match_atoms:
                        highlight_atoms[atom_idx] = self.colors["donor"]
        
        # 高亮受体（红色）
        if hasattr(annotation, 'acceptors'):
            for acceptor in annotation.acceptors:
                if hasattr(acceptor, 'match_atoms'):
                    for atom_idx in acceptor.match_atoms:
                        highlight_atoms[atom_idx] = self.colors["acceptor"]
        
        # 高亮连接子（蓝色）
        if hasattr(annotation, 'linkers'):
            for linker in annotation.linkers:
                if hasattr(linker, 'match_atoms'):
                    for atom_idx in linker.match_atoms:
                        highlight_atoms[atom_idx] = self.colors["linker"]
        
        return self.visualize_single(mol, title=title, 
                                    highlight_atom_colors=highlight_atoms if highlight_atoms else None,
                                    save_path=save_path, show_legend=True)

    def visualize_with_smarts(self, mol, donor_smarts=None, acceptor_smarts=None, 
                              linker_smarts=None, title="", save_path=None):
        """
        基于预定义SMARTS模式高亮片段
        
        Args:
            mol: RDKit Mol对象或SMILES字符串
            donor_smarts: 供体SMARTS列表
            acceptor_smarts: 受体SMARTS列表
            linker_smarts: 连接子SMARTS列表
            title: 图像标题
            save_path: 保存路径（可选）
            
        Returns:
            PIL.Image或保存路径
        """
        # 解析SMILES
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            img = Image.new("RGB", (self.default_width, self.default_height), "white")
            if save_path:
                img.save(save_path, "PNG")
                return save_path
            return img
        
        highlight_atoms = {}
        
        # 高亮供体
        if donor_smarts:
            for smarts in donor_smarts:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        for idx in match:
                            highlight_atoms[idx] = self.colors["donor"]
        
        # 高亮受体
        if acceptor_smarts:
            for smarts in acceptor_smarts:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        for idx in match:
                            highlight_atoms[idx] = self.colors["acceptor"]
        
        # 高亮连接子
        if linker_smarts:
            for smarts in linker_smarts:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        for idx in match:
                            highlight_atoms[idx] = self.colors["linker"]
        
        return self.visualize_single(mol, title=title, 
                                    highlight_atom_colors=highlight_atoms if highlight_atoms else None,
                                    save_path=save_path, show_legend=True)


# =============================================================================
# 便捷函数
# =============================================================================

def visualize_molecule(smiles_or_mol, save_path=None, width=600, height=400):
    """
    简单分子可视化函数
    
    Args:
        smiles_or_mol: SMILES字符串或RDKit Mol对象
        save_path: 保存路径（可选）
        width: 图像宽度
        height: 图像高度
        
    Returns:
        PIL.Image或保存路径
    """
    visualizer = MoleculeVisualizer(width=width, height=height)
    return visualizer.visualize_single(smiles_or_mol, save_path=save_path)


def visualize_fragments(smiles_or_mol, fragment_result, save_path=None):
    """
    根据片段识别结果可视化分子
    
    Args:
        smiles_or_mol: SMILES字符串或RDKit Mol对象
        fragment_result: 片段识别结果字典
        save_path: 保存路径（可选）
        
    Returns:
        PIL.Image或保存路径
    """
    visualizer = MoleculeVisualizer()
    return visualizer.visualize_fragment_result(smiles_or_mol, fragment_result, save_path=save_path)


def visualize_grid(mols, labels=None, mols_per_row=4, save_path=None):
    """
    网格可视化多个分子
    
    Args:
        mols: 分子列表（SMILES或Mol对象）
        labels: 标签列表
        mols_per_row: 每行分子数
        save_path: 保存路径（可选）
        
    Returns:
        PIL.Image或保存路径
    """
    visualizer = MoleculeVisualizer()
    return visualizer.visualize_grid(mols, labels=labels, mols_per_row=mols_per_row, save_path=save_path)


def visualize_sdf_annotated(sdf_path, output_dir=None, max_molecules=10):
    """
    可视化SDF文件中带有片段标注的分子
    
    从SDF文件读取分子及其片段标注信息，生成可视化图像
    
    Args:
        sdf_path: SDF文件路径
        output_dir: 输出目录（可选，默认为当前目录）
        max_molecules: 最多可视化的分子数
        
    Returns:
        List[str]: 生成的图像文件路径列表
    """
    import os
    
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    supplier = Chem.SDMolSupplier(sdf_path)
    visualizer = MoleculeVisualizer()
    output_paths = []
    
    for i, mol in enumerate(supplier):
        if i >= max_molecules:
            break
        
        if mol is None:
            continue
        
        # 获取分子名称
        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{i+1}"
        
        # 尝试读取片段标注信息
        fragment_result = {'donors': [], 'acceptors': [], 'linkers': []}
        
        # 从SDF属性读取片段信息（与molecule_processor.py格式兼容）
        if mol.HasProp('Donor_Atoms'):
            import json
            try:
                donor_atoms_list = json.loads(mol.GetProp('Donor_Atoms'))
                for atoms in donor_atoms_list:
                    fragment_result['donors'].append({'match_atoms': atoms})
            except:
                pass
        
        if mol.HasProp('Acceptor_Atoms'):
            import json
            try:
                acceptor_atoms_list = json.loads(mol.GetProp('Acceptor_Atoms'))
                for atoms in acceptor_atoms_list:
                    fragment_result['acceptors'].append({'match_atoms': atoms})
            except:
                pass
        
        if mol.HasProp('Linker_Atoms'):
            import json
            try:
                linker_atoms_list = json.loads(mol.GetProp('Linker_Atoms'))
                for atoms in linker_atoms_list:
                    fragment_result['linkers'].append({'match_atoms': atoms})
            except:
                pass
        
        # 也尝试读取fluo_utils.py格式的属性
        if mol.HasProp('donor_count') and int(mol.GetProp('donor_count')) > 0:
            # 尝试解析donor_smiles属性
            if mol.HasProp('donor_smiles'):
                # 这里我们只知道数量，不知道具体原子位置
                pass
        
        # 生成输出路径
        safe_name = "".join(c if c.isalnum() else "_" for c in mol_name)
        output_path = os.path.join(output_dir, f"{safe_name}_annotated.png")
        
        # 可视化
        if fragment_result['donors'] or fragment_result['acceptors'] or fragment_result['linkers']:
            visualizer.visualize_fragment_result(mol, fragment_result, 
                                                title=mol_name, save_path=output_path)
        else:
            visualizer.visualize_single(mol, title=mol_name, save_path=output_path)
        
        output_paths.append(output_path)
        print(f"已生成: {output_path}")
    
    return output_paths


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("荧光分子可视化模块")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\n用法:")
        print("  python visualization.py <SMILES> [output.png]")
        print("  python visualization.py --sdf <input.sdf> [output_dir]")
        print("\n示例:")
        print("  python visualization.py 'c1ccccc1' molecule.png")
        print("  python visualization.py --sdf annotated_molecules.sdf ./vis_output")
        sys.exit(1)
    
    if sys.argv[1] == '--sdf':
        # SDF可视化模式
        if len(sys.argv) < 3:
            print("错误: 请提供SDF文件路径")
            sys.exit(1)
        
        sdf_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "./visualization_output"
        
        print(f"\n处理SDF文件: {sdf_path}")
        print(f"输出目录: {output_dir}\n")
        
        output_paths = visualize_sdf_annotated(sdf_path, output_dir)
        
        print(f"\n共生成 {len(output_paths)} 个可视化图像")
    else:
        # 单分子可视化模式
        smiles = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"\n可视化分子: {smiles}")
        
        visualizer = MoleculeVisualizer()
        img = visualizer.visualize_single(smiles, title=smiles, save_path=output_path)
        
        if output_path:
            print(f"已保存: {output_path}")
        else:
            # 尝试显示图像
            try:
                img.show()
            except:
                print("无法显示图像，请提供输出路径保存")
