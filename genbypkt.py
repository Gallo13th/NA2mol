import os
import sys
import glob
import pickle
import random
import itertools
from typing import List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from accelerate import Accelerator
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit.Geometry import Point3D
import tqdm

from model.bfn4edmol import Config, BFN4SBDDScoreModel
from dataloader.featurizer import ProteinLigandDataLoader, GeomSeedMolOnlyPocket
from util.mol_refine import Minimize, prepare_seed_fragments, get_nci_info_batch
from util.mol_utils import sampletraj_to_sdf, sampletraj_to_sdf_byseed
from util.sascorer import calculateScore


# fix openbabel error

def load_compiled_model(model, state_dict):
    """
    加载模型参数，对于shape不一致的参数进行截断或填充
    返回不匹配的参数列表和加载成功比例
    
    Args:
        model: PyTorch模型
        state_dict: 要加载的状态字典（可能包含"dynamics."前缀）
        
    Returns:
        tuple: (mismatched_params, load_ratio)
            - mismatched_params: 需要学习的参数名称列表
            - load_ratio: 成功加载的参数比例(0-1)
    """
    model_state_dict = model.state_dict()
    new_state_dict = {}
    mismatched_params = []
    total_params = len(model_state_dict)
    loaded_params = 0
    
    # 预处理state_dict，去除"dynamics."前缀
    processed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")  # 去除"dynamics."前缀
        processed_state_dict[new_key] = value
    # print(processed_state_dict.keys())
    # 首先处理预训练权重中的参数
    for key, value in processed_state_dict.items():
        if key in model_state_dict:
            model_param = model_state_dict[key]
            
            # 如果shape相同，直接加载
            if value.shape == model_param.shape:
                new_state_dict[key] = value
                loaded_params += 1
            else:
                # 如果shape不同，进行截断或填充
                new_param = model_param.clone()  # 克隆原始参数
                print(key)
                # 计算每个维度的最小尺寸
                min_shape = [min(s1, s2) for s1, s2 in zip(value.shape, model_param.shape)]
                
                # 创建切片索引
                slices = tuple(slice(0, min_dim) for min_dim in min_shape)
                
                # 将state_dict中的值复制到新参数中（自动截断）
                new_param[slices] = value[slices]
                
                new_state_dict[key] = new_param
                # loaded_params += 1  # 部分加载也算作成功
                mismatched_params.append(key)
        else:
            # 如果key不在模型中，记录为不匹配参数
            mismatched_params.append(key)
    
    # 检查模型中是否有预训练权重中没有的参数
    for key in model_state_dict:
        if key not in processed_state_dict:
            mismatched_params.append(key)
    
    # 计算加载比例
    load_ratio = loaded_params / total_params if total_params > 0 else 0.0
    
    # 加载处理后的状态字典
    model.load_state_dict(new_state_dict, strict=False)
    print(mismatched_params)
    # 冻结所有匹配的参数
    for name, param in model.named_parameters():
        if name not in mismatched_params:
            param.requires_grad_(False)
    print(f"successed ratio: {load_ratio}")
    return new_state_dict

def read_sdf_ligand_coords(sdf_file):
    """读取 SDF 文件中的配体，并返回所有原子的 3D 坐标"""
    supplier = Chem.SDMolSupplier(sdf_file)
    print(sdf_file)
    mol = next(supplier)
    if mol is None:
        # try load from pdb
        mol = Chem.MolFromPDBFile(sdf_file.replace('.sdf','.pdb'), removeHs=True)
        if mol is None:
            raise ValueError("无法读取 SDF 文件中的配体，请检查文件格式和 3D 结构是否存在。")
    if not mol.GetNumConformers():
        raise ValueError("配体没有 3D 构象，请确保 SDF 文件包含坐标。")
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords)

# 常见水分子残基名称（PDB 标准及各类力场）
WATER_RESNAMES = {'HOH', 'WAT', 'H2O', 'DOD', 'TIP', 'TIP3', 'TIP4', 'SOL'}

# 标准氨基酸（20种 + 常见修饰）
AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'SEC', 'PYL', 'MSE', 'CAS', 'CSD', 'CSO', 'CSS', 'CSW', 'CSX', 'CYM',
    'CYX', 'HYP', 'KCX', 'LLP', 'MLY', 'NLE', 'OCS', 'PCA', 'PTR', 'SEP',
    'TPO', 'TPQ', 'TYS', 'UNK',
}

# 标准核酸
NUC_RESNAMES = {
    'A', 'G', 'C', 'T', 'U', 'DA', 'DG', 'DC', 'DT', 'DU',
    'RA', 'RG', 'RC', 'RU', 'DI', 'ADE', 'GUA', 'CYT', 'THY', 'URA', 'URI',
}

# 常见离子（明确保留）
ION_RESNAMES = {
    'NA', 'CL', 'CA', 'MG', 'ZN', 'K', 'FE', 'CU', 'MN', 'CO',
    'NI', 'CD', 'HG', 'PB', 'AG', 'AU', 'AL', 'BA', 'BE', 'CS',
    'LI', 'RB', 'SR', 'V', 'YB', 'CR', 'MO', 'W', 'SE', 'BR',
    'F', 'I', 'XE', 'KR', 'AR', 'HE', 'NE',
}

# 白名单：只保留氨基酸、核酸和离子
ALLOWED_RESNAMES = AA_RESNAMES | NUC_RESNAMES | ION_RESNAMES

def extract_pocket(residues, ligand_coords, distance_cutoff=20.0):
    pocket_residues = []
    ligand_center = np.mean(ligand_coords, axis=0)  # shape: (3,)

    for residue in residues:
        resname = residue.resname.strip().upper()
        # 跳过水分子
        if resname in WATER_RESNAMES:
            continue
        # 跳过配体及其他非口袋杂原子（只保留氨基酸、核酸、离子）
        if resname not in ALLOWED_RESNAMES:
            continue
        
        atom_coords = np.array([atom.coord for atom in residue])  # shape: (n_atoms, 3)
        if len(atom_coords) == 0:
            continue
        residue_center = np.mean(atom_coords, axis=0)  # shape: (3,)
        dist = np.linalg.norm(ligand_center - residue_center)  # scalar
        if dist < distance_cutoff:
            pocket_residues.append(residue)
    return pocket_residues

def save_pocket(pocket_residues, structure, output_pdb, new_chain_id='A'):
    """保存筛选出的口袋残基到新 PDB 文件，并将它们重命名为指定链"""
    class PocketSelect(Select):
        def __init__(self, residues):
            self.residue_set = set(residues)  # 使用 set 提高查找效率

        def accept_residue(self, residue):
            return residue in self.residue_set

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, PocketSelect(pocket_residues))
    print(f"口袋已保存至: {output_pdb}")

# ================= 新增：种子质量评分辅助函数 =================
def calc_seed_quality(frag_mol, affinity, nci_count):
    """
    计算片段的综合质量得分 (Score)。
    Score = w1*LBE + w2*QED + w3*NCI_Efficiency - w4*Complexity
    """
    try:
        num_heavy = frag_mol.GetNumHeavyAtoms()
        if num_heavy == 0: return -100, {}, False
        
        # 1. LBE (Ligand Binding Efficiency) 估算
        lbe = -affinity / num_heavy  # parent_affinity 通常为负，取反变正

        # 2. QED (Quantitative Estimate of Drug-likeness)
        frag_noH = Chem.RemoveHs(frag_mol)
        sascore = calculateScore(frag_noH)
        carbon_cnt = sum(1 for atom in frag_mol.GetAtoms() if atom.GetSymbol() == 'C')

        qed_val = QED.qed(frag_mol)
        
        # 3. NCI Efficiency (相互作用密度)
        nci_eff = nci_count

        # 综合评分公式 (权重可调)
        score = lbe + nci_eff * 0.1 + qed_val - sascore * 0.1
        
        metrics = {
            "LBE": lbe, "QED": qed_val, "NCI": nci_count, "SA Score": sascore, "Score": score
        }
        
        # 硬性门槛 (Hard Filters)
        is_pass = True
        if lbe < 0.2: is_pass = False          # 必须具备基本原子效率
        if qed_val < 0.3: is_pass = False          # 必须具备基本药物相似性
        if nci_count < 1: is_pass = False      # 必须至少有一个非共价相互作用
        if sascore > 3.0: is_pass = False   # 必须具备基本合成可行性
        if num_heavy < 6: is_pass = False
        if carbon_cnt < num_heavy // 2: is_pass = False
        return score, metrics, is_pass

    except Exception as e:
        return -100, {}, False

def merge_sdfs(sdf_list, output_filename):
    """将所有成功的分子合并到一个 SDF 文件"""
    writer = Chem.SDWriter(output_filename)
    count = 0
    for sdf in sdf_list:
        suppl = Chem.SDMolSupplier(sdf)
        for mol in suppl:
            if mol:
                writer.write(mol)
                count += 1
    writer.close()
    print(f"✅ 已将 {count} 个分子合并保存至: {output_filename}")

def deduplicate_fragments(fragment_paths, rmsd_threshold=2.0):
    """
    对片段列表去重：
      - 分组依据：Murcko 骨架 SMILES + 重原子元素类型序列（增强区分）
      - 组内计算 pairwise 原位 RMSD（不 align），保留 affinity 最高（值最小，即最负）者
    返回: 去重后的片段路径列表
    """
    scaffold_groups = defaultdict(list)
    frag_data = []

    for path in fragment_paths:
        suppl = Chem.SDMolSupplier(path)
        mol = next(suppl, None)
        if not mol:
            continue
        try:
            aff = float(mol.GetProp('minimizedAffinity')) if mol.HasProp('minimizedAffinity') else 0.0
            
            # 获取 Murcko 骨架
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            
            # 增强：加入重原子的元素类型序列，防止骨架相同但核心原子不同（如罕见情况）
            heavy_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
            heavy_atoms.sort()  # 排序以忽略顺序，只关注组成
            element_signature = ''.join(heavy_atoms)
            
            # 分组键：骨架 + 元素组成（双重保障）
            group_key = (scaffold_smiles, element_signature)
            
            scaffold_groups[group_key].append((path, mol, aff))
            frag_data.append((path, mol, aff, group_key))
        except Exception as e:
            # 可选：记录错误
            continue

    kept_paths = []
    for group_key, group in scaffold_groups.items():
        if len(group) == 1:
            kept_paths.append(group[0][0])
            continue

        # 选择代表：遍历所有成员，两两比较原位 RMSD
        # 策略：保留 affinity 最优（最小）且与其他已选分子 RMSD >= threshold 的分子
        # 但为简化，仍采用"第一个为参考，后续与之比较"的策略（可改进为聚类）
        best_path, best_aff = group[0][0], group[0][2]
        representative = group[0][1]

        for i in range(1, len(group)):
            path_i, mol_i, aff_i = group[i]
            
            # 检查原子数和元素序列是否一致（确保 CalcRMS 有效）
            if mol_i.GetNumAtoms() != representative.GetNumAtoms():
                # 原子数不同，不可能是重复（即使骨架相同）
                continue

            # 检查元素序列是否一致（顺序敏感！）
            rep_symbols = [a.GetSymbol() for a in representative.GetAtoms()]
            mol_i_symbols = [a.GetSymbol() for a in mol_i.GetAtoms()]
            if rep_symbols != mol_i_symbols:
                continue  # 原子类型顺序不同，跳过（或视为不同）

            # 原位 RMSD（不 align）
            try:
                rmsd = AllChem.CalcRMS(representative, mol_i)
                if rmsd < rmsd_threshold:
                    # 视为重复，保留 affinity 更低（更负）者
                    if aff_i < best_aff:
                        best_aff = aff_i
                        best_path = path_i
                # else: 不重复，但我们仍只保留一个代表（当前策略）
                # 注意：此策略可能漏掉多个不相似但同骨架的分子
                # 如需保留所有非重复项，需改为聚类
            except Exception:
                continue

        kept_paths.append(best_path)

    return kept_paths

def min_atom_distance(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """高效计算两分子间最小原子距离（带提前终止优化）"""
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    min_dist = float('inf')
    
    for i in range(mol1.GetNumAtoms()):
        pos_i = conf1.GetAtomPosition(i)
        for j in range(mol2.GetNumAtoms()):
            pos_j = conf2.GetAtomPosition(j)
            dx = pos_i.x - pos_j.x
            dy = pos_i.y - pos_j.y
            dz = pos_i.z - pos_j.z
            d = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            if d < min_dist:
                min_dist = d
                # 提前终止：一旦发现冲突距离，无需继续计算
                if min_dist <= 4.0:
                    return min_dist
    return min_dist

def merge_fragments_via_cliques(
    fragment_paths: List[str],
    output_dir: str,
    max_seeds: int = 50,
    max_attempts: int = 500,  # 保留接口兼容性，实际不再使用
    min_inter_frag_dist: float = 4.0,
    reward_per_frag: float = 0.01
) -> List[str]:
    """
    基于团（Clique）搜索的片段合并策略 - 系统化替代随机拼接
    
    核心逻辑：
      1. 选取Top-200高质量片段（按_SeedScore降序）
      2. 构建距离图：片段间最小原子距离 > min_inter_frag_dist 时建立边
      3. 搜索所有2-4节点的完全子图（clique）：
         - 2节点团 = 有效边（两两可合并）
         - 3节点团 = 三角形（三者两两可合并）
         - 4节点团 = 完全子图K4（四者两两可合并）
      4. 优先级规则：
         (a) 节点数多的团优先（4 > 3 > 2）
         (b) 同节点数时，按团内最高_SeedScore降序
         (c) 隐含规则：大团自动包含小团的所有节点关系，优先级天然更高
    
    参数:
        fragment_paths: 高质量片段SDF路径列表（已含_SeedScore）
        output_dir: 新种子保存目录
        max_seeds: 最多生成的新种子数量
        max_attempts: 保留接口兼容性（实际由团数量决定）
        min_inter_frag_dist: 片段间最小允许距离（Å）
        reward_per_frag: 每增加一个片段的奖励分
    
    返回:
        成功生成的新种子文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # === 步骤1: 加载并排序片段（取Top-200）===
    frag_data = []
    for path in fragment_paths:
        try:
            suppl = Chem.SDMolSupplier(path)
            if not suppl or len(suppl) == 0:
                continue
            mol = suppl[0]
            if mol is None:
                continue
            score = float(mol.GetProp('_SeedScore'))
            frag_data.append((score, mol, path))
        except Exception as e:
            print(f"⚠️ 跳过无效片段 {path}: {e}")
            continue
    
    if len(frag_data) < 2:
        print("❌ 片段数量不足2个，无法合并")
        return []
    
    # 按_SeedScore降序排序，取Top-200
    frag_data.sort(key=lambda x: x[0], reverse=True)
    frag_data = frag_data[:200]  # 限制规模
    
    scores = [d[0] for d in frag_data]
    mols = [d[1] for d in frag_data]
    paths = [d[2] for d in frag_data]
    n_frags = len(mols)
    
    print(f"✓ 选取Top-{n_frags}高质量片段（最高_SeedScore: {scores[0]:.3f}）")
    
    # === 步骤2: 构建距离图（邻接矩阵）===
    print("正在计算片段间距离矩阵...")
    adj_matrix = np.zeros((n_frags, n_frags), dtype=bool)
    
    # 仅计算上三角（对称矩阵）
    for i in range(n_frags):
        for j in range(i + 1, n_frags):
            dist = min_atom_distance(mols[i], mols[j])
            if dist > min_inter_frag_dist:
                adj_matrix[i, j] = adj_matrix[j, i] = True
    
    print("="*10 + f"EXISTS EDGES : {adj_matrix.sum()}")
    
    # 构建邻接表（用于高效团搜索）
    neighbors = [np.where(adj_matrix[i])[0].tolist() for i in range(n_frags)]
    
    # === 步骤3: 搜索2-4节点团（Clique）===
    cliques: List[Tuple[int, ...]] = []

    # 2-节点团（边）
    for i in range(n_frags):
        for j in neighbors[i]:
            if j > i:
                cliques.append((i, j))

    # 3-节点团（三角形）
    triangles = []  # 单独保存3-节点团，用于后续扩展
    for i in range(n_frags):
        for j in neighbors[i]:
            if j <= i:
                continue
            common = [k for k in neighbors[i] if k in neighbors[j] and k > j]
            for k in common:
                tri = (i, j, k)
                cliques.append(tri)
                triangles.append(tri)

    # 4-节点团（K4完全子图）— 仅从3-节点团扩展
    for (i, j, k) in triangles:  # ✅ 安全：triangles 中全是3元组
        # 找共同邻居 l > k
        common = [l for l in neighbors[i] 
                  if l in neighbors[j] and l in neighbors[k] and l > k]
        for l in common:
            cliques.append((i, j, k, l))
    
    print(f"✓ 发现有效团数量: {len(cliques)} (2节点:{sum(1 for c in cliques if len(c)==2)}, "
          f"3节点:{sum(1 for c in cliques if len(c)==3)}, 4节点:{sum(1 for c in cliques if len(c)==4)})")
    
    if not cliques:
        print("⚠️ 未找到满足距离约束的片段组合")
        return []
    
    # === 步骤4: 按优先级排序团 ===
    def clique_priority(clique: Tuple[int, ...]) -> Tuple[int, float]:
        """优先级规则: (1) 节点数降序 (2) 团内最高_SeedScore降序"""
        max_score = max(scores[i] for i in clique)
        return (-len(clique), -max_score)  # 负值实现降序
    
    cliques.sort(key=clique_priority)
    
    # === 步骤5: 生成新种子 ===
    new_seed_paths = []
    used_combinations: Set[Tuple[int, ...]] = set()  # 防止重复组合
    
    print(f"{'='*60}\n开始生成新种子 (目标: {max_seeds}个)...\n{'='*60}")
    
    for clique in cliques:
        if len(new_seed_paths) >= max_seeds:
            break
            
        # 严格去重：相同片段组合（忽略顺序）只生成一次
        sorted_clique = tuple(sorted(clique))
        if sorted_clique in used_combinations:
            continue
        used_combinations.add(sorted_clique)
        
        # 提取团内片段
        selected_mols = [mols[i] for i in clique]
        selected_scores = [scores[i] for i in clique]
        selected_paths = [paths[i] for i in clique]
        
        # 合并片段（依次CombineMols）
        combined_mol = selected_mols[0]
        for mol in selected_mols[1:]:
            combined_mol = Chem.CombineMols(combined_mol, mol)
        
        # 计算新种子分数: max(片段分数) + 奖励*片段数
        new_score = max(selected_scores) + reward_per_frag * len(clique)
        
        # 设置分子属性
        combined_mol.SetProp("_SeedScore", f"{new_score:.4f}")
        combined_mol.SetProp("_SourceFrags", ";".join(os.path.basename(p) for p in selected_paths))
        combined_mol.SetProp("_CliqueSize", str(len(clique)))  # 记录团大小
        
        # 保存结果
        seed_name = f"merged_seed_{len(new_seed_paths):04d}_clique{len(clique)}.sdf"
        seed_path = os.path.join(output_dir, seed_name)
        
        try:
            writer = Chem.SDWriter(seed_path)
            writer.write(combined_mol)
            writer.close()
            new_seed_paths.append(seed_path)
            print(f"✓ 生成种子 #{len(new_seed_paths):02d} | 团大小: {len(clique)} | "
                  f"Score: {new_score:.4f} | 来源: {', '.join(os.path.basename(p)[:15] for p in selected_paths)}")
        except Exception as e:
            print(f"⚠️ 保存失败 (团{clique}): {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ 成功生成 {len(new_seed_paths)}/{max_seeds} 个新种子")
    print(f"✓ 最高分种子: {max([float(Chem.SDMolSupplier(p)[0].GetProp('_SeedScore')) for p in new_seed_paths]):.4f}")
    print(f"{'='*60}")
    
    return new_seed_paths

# ── SPEC 02: lazy cached fluorophore-guidance singleton ──────────────────────
_FLUOR_GUIDANCE = None
_FLUOR_GUIDANCE_LIB = 'fragment_library/targeted_fluorophore_focused.pkl'
_FLUOR_GUIDANCE_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'casestudy', '8hzk', 'fluor_guidance.py')

# When True, every generation iteration is seeded from the original scaffold file and the
# fragment-library branch is never taken. The seed atoms are placed first and marked fixed
# in GeomSeedMolOnlyPocket, so the scaffold stays pinned for the whole run. Used for
# scaffold-constrained sampling; False reproduces the released behaviour.
PIN_SCAFFOLD = False


def generate(task, output_prefix, seed_used=False, total_target=10000,
             batch_size=100, iter_cap=150):

   
    protein_file = task[1]
    mol_file = task[0]
    seed_file = task[0]
    pocket_file = task[1].replace('.pdb','_pocket.pdb')
   
    # 预处理口袋
   
    pdbid = os.path.basename(protein_file)[:4]

    cutoff = 14
    basename = os.path.basename(mol_file)[:-4]
   
    output_base_path = f'output/{output_prefix}/{basename}'
    if os.path.exists(f'{output_base_path}/{basename}.sdf'):
        print(f"✅ 已存在合并后的分子文件: {output_base_path}/{basename}.sdf")
        return  # 跳过生成流程
   
    # 1. 读取配体坐标
    print("正在读取配体坐标...")
    ligand_coords = read_sdf_ligand_coords(mol_file)
   
    # 2. 读取受体结构
    print("正在读取受体结构...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", protein_file)
   
    # 获取所有残基
    residues = [residue for residue in structure.get_residues()]
   
    # 3. 提取口袋
    print(f"正在提取距离配体 < {cutoff} Å 的残基...")
    pocket_residues = extract_pocket(residues, ligand_coords, distance_cutoff=cutoff)
    print(f"共保留 {len(pocket_residues)} 个残基。")
   
    # 4. 保存结果
    save_pocket(pocket_residues, structure, pocket_file)
   
   
    # ================= 初始化变量 =================
    basename = os.path.basename(mol_file)[:-4]
    output_base_path = f'output/{output_prefix}/{basename}'  # 修改输出路径以免覆盖旧数据
   
    # 结果容器
    good_affinity_mols = []       # 存储 affinity < 0 的最终分子路径
    seed_library = []             # 存储所有制作好的种子文件路径
    processed_parent_files = set() # 记录已经切分过的父分子，防止重复切分
   
    iteration = 0
   
    # 确保口袋已被处理 (rd_pkt 在循环外加载一次即可)
    if not os.path.exists(pocket_file.replace('_pocket.pdb','_pocket_addh.pdb')) or True:
        os.system(f"obabel {pocket_file} -O {pocket_file.replace('_pocket.pdb','_pocket_addh.pdb')} -h")
    rd_pkt = Chem.MolFromPDBFile(pocket_file.replace('_pocket.pdb','_pocket_addh.pdb'), removeHs=False)
   
    print(f"🚀 开始迭代生成流程，目标：获得 {total_target} 个 Affinity < 0 的分子")
   
    # ================= 进入主循环 =================
    while len(good_affinity_mols) < total_target:
        iteration += 1
        if iteration > iter_cap:
            print(f"\n=== REACH MAX ITERATION NUMS ! STOP ITERATION | 当前达标分子数: {len(good_affinity_mols)}/{total_target} | 种子库大小: {len(seed_library)} ===")
            break # Fix: Add break to actually stop
       
        print(f"\n=== Iteration {iteration} | 当前达标分子数: {len(good_affinity_mols)}/{total_target} | 种子库大小: {len(seed_library)} ===")
       
        # 1. 准备 DataLoader (Batch Size = 100)
        current_batch_size = batch_size
       
        if len(seed_library) == 0 or not seed_used or iteration <= 15 or PIN_SCAFFOLD:
            # --- 模式 A: 初始无种子生成 (Cold Start) ---
            print(">> 进行初始无种子生成...")
            dataset = GeomSeedMolOnlyPocket(pocket=pocket_file, seed=seed_file, mode='general', extra_range=(-8,8))
            # 复制 dataset 以凑够 batch_size
            batch_dataset = ConcatDataset([dataset] * current_batch_size)
            use_seed_mode = False
        else:
            sorted_seeds = seed_library.copy()
           
            try:
                seed_scores = []
                for seed_path in sorted_seeds:
                    score = 0.0
                    if os.path.exists(seed_path):
                        suppl = Chem.SDMolSupplier(seed_path)
                        mol = next(suppl, None)
                        if mol and mol.HasProp('_SeedScore'):
                            score = float(mol.GetProp('_SeedScore'))
                    seed_scores.append(score)
               
                # 按分数降序排序
                sorted_pairs = sorted(zip(sorted_seeds, seed_scores), key=lambda x: x[1], reverse=True)
                sorted_seeds = [pair[0] for pair in sorted_pairs]
            except:
                # 如果读取分数失败，保持原顺序
                pass
           
            # 选择前 current_batch_size 个最好的种子
            selected_seeds = sorted_seeds[:min(len(sorted_seeds), current_batch_size)]
           
            # 如果种子不足 100，循环填充直到 100 (保证 batch 填满)
            if len(selected_seeds) < current_batch_size:
                selected_seeds = (selected_seeds * (current_batch_size // len(selected_seeds) + 1))[:current_batch_size]
           
            print(f">> 使用 {len(selected_seeds)} 个种子进行生成...")
            sgl_ds = []
            for s_file in selected_seeds:
                ds = GeomSeedMolOnlyPocket(
                    pocket=pocket_file,
                    seed=s_file,
                    mode='general',
                    extra_range=(-8,8),
                    remain_seed=True,
                )
                sgl_ds.append(ds)
            batch_dataset = ConcatDataset(sgl_ds)
            use_seed_mode = True
   
        dataloader = ProteinLigandDataLoader(batch_dataset, batch_size=current_batch_size, shuffle=False)
        dataloader = accelerator.prepare(dataloader)
       
        # 定义当前迭代的目录
        iter_dir = os.path.join(output_base_path, f'iter_{iteration}')
        wkd_raw = os.path.join(iter_dir, 'raw_output')
        wkd_smn = os.path.join(iter_dir, 'smina_output')
        wkd_seed_iter = os.path.join(iter_dir, 'new_seeds')
        wkd_frag_smn = os.path.join(iter_dir, 'frag_smina_output')  # 新目录：片段的smina输出
       
        os.makedirs(wkd_raw, exist_ok=True)
        os.makedirs(wkd_smn, exist_ok=True)
        os.makedirs(wkd_seed_iter, exist_ok=True)
        os.makedirs(wkd_frag_smn, exist_ok=True)
   
        # 2. 模型采样 (Generation)
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=f"Iter {iteration} Generating"):
                batch = batch.to(accelerator.device)
               
                if not use_seed_mode:
                    # 无种子采样
                    unwrapped_model = accelerator.unwrap_model(model)
                    theta_traj, sample_traj, y_traj, final_e, final_ei = unwrapped_model.sample(
                        pocket_pos=batch.pocket_pos / 2,
                        pocket_v=batch.pocket_atom_feature.float(),
                        batch_pocket=batch.pocket_element_batch,
                        ed_pos=torch.empty((0, 3), device=accelerator.device),
                        ed_v=torch.empty((0, 4), device=accelerator.device),
                        batch_ed=batch.ed_ed_code_batch,
                        batch_ligand=batch.mol_element_batch,
                        n_nodes=batch_size,
                        sample_steps=100,
                        pocket_res=None,
                        pocket_ele=None,
                        return_traj=False
                    )
                    new_traj = (sample_traj[-1][0] * 2 + batch.mol_geom_center[:3], sample_traj[-1][1])
                    sampletraj_to_sdf(
                        {'trajs': new_traj, 'batch_idx': batch.mol_element_batch.cpu()},
                        wkd_raw,
                    )
                else:
                    # 有种子采样
                    unwrapped_model = accelerator.unwrap_model(model)
                    theta_traj, sample_traj, y_traj, final_e, final_ei = unwrapped_model.sample_by_seed(
                        pocket_pos=batch.pocket_pos / 2,
                        pocket_v=batch.pocket_atom_feature.float(),
                        batch_pocket=batch.pocket_element_batch,
                        ed_pos=torch.empty((0,3), device=accelerator.device),
                        ed_v=torch.empty((0,4), device=accelerator.device),
                        batch_ed=batch.ed_ed_code_batch,
                        batch_ligand=batch.mol_element_batch,
                        n_nodes=batch_size,
                        fixed_atom_mask=batch.mol_fixed_atom_mask,
                        fixed_atom_pos=batch.mol_fixed_atom_pos / 2,
                        fixed_atom_types=batch.mol_fixed_atom_types,
                        sample_steps=100,
                        pocket_res=None,
                        pocket_ele=None,
                        return_traj=False
                    )
               
                    # 保存 Raw SDF
                    current_batch_seeds = [next(Chem.SDMolSupplier(p)) for p in batch.mol_seed_path]
                    new_traj = (sample_traj[-1][0] * 2 + batch.mol_geom_center[:3], sample_traj[-1][1])
                    sampletraj_to_sdf_byseed(
                        {'trajs': new_traj, 'batch_idx': batch.mol_element_batch.cpu()},
                        wkd_raw, current_batch_seeds, batch.mol_fixed_atom_mask, dist_tol=0.4,
                    )
    
        # 3. 能量最小化 (Minimize)
        raw_sdfs = glob.glob(os.path.join(wkd_raw, "*.sdf"))
        if not raw_sdfs:
            print("⚠️ 本轮未生成有效 SDF，跳过...")
            continue

        for input_sdf in raw_sdfs:
            Minimize(input_sdf, pocket_file, wkd_smn)
   
        minimized_sdfs = glob.glob(os.path.join(wkd_smn, "*.sdf"))  
        fragments_to_screen = []
       
        print(">> 正在预处理分子并切分片段...")
        for min_sdf in minimized_sdfs:
            try:
                suppl = Chem.SDMolSupplier(min_sdf)
            except:
                continue
            for mol in suppl:
                if mol is None: continue
                try:
                    # A. 检查 Affinity
                    aff = 0.0
                    if mol.HasProp('minimizedAffinity'):
                        aff = float(mol.GetProp('minimizedAffinity'))
                        if aff < 0:
                            mol.SetProp('Iteration', str(iteration))
                            # write back updated mol with Iteration prop
                            writer = Chem.SDWriter(min_sdf)
                            writer.write(mol)
                            writer.close()
                            good_affinity_mols.append(min_sdf) # 记录达标分子路径

                    # B. 检查是否需要切分
                    if seed_used:
                        if min_sdf in processed_parent_files:
                            continue

                        processed_parent_files.add(min_sdf) # 标记为已处理

                        mol_noH = Chem.RemoveHs(mol)
                        seed_frags, _ = prepare_seed_fragments(mol_noH)

                        if seed_frags:
                            for frag in seed_frags:
                                # 记录片段、父文件路径
                                fragments_to_screen.append((frag, min_sdf))
                           
                except Exception as e:
                    print(f"Error processing {min_sdf}: {e}")
   
        # C. 对所有片段进行smina评分并保存
        if not fragments_to_screen or not seed_used:
            continue
           
        print(f">> 正在对 {len(fragments_to_screen)} 个片段进行 smina 评分并保存...")
       
        frag_minimized_paths = []  # 存储每个片段的minimized sdf路径
        frag_affinities = []       # 存储每个片段的affinity
        successed_idx = []
        for idx, (frag_mol, parent_path) in enumerate(fragments_to_screen):
            # 创建临时片段sdf
            frag_temp_sdf = os.path.join(wkd_frag_smn, f"frag_{idx:04d}.sdf")
            sdf_written = False
            try:
                writer = Chem.SDWriter(frag_temp_sdf)
                frag_mol = Chem.RemoveHs(frag_mol)
                writer.write(frag_mol)
                writer.close()
                sdf_written = True
            except:
                pass  # 不 continue，继续追加占位符
           
            if sdf_written:
                try:
                    Minimize(frag_temp_sdf, pocket_file, wkd_frag_smn)
                    minimized_frag_sdf = os.path.join(wkd_frag_smn, f"frag_{idx:04d}_lig_score.sdf")
                    aff = 0.0
                    try:
                        suppl = Chem.SDMolSupplier(minimized_frag_sdf)
                        min_frag_mol = next(suppl, None)
                        if min_frag_mol and min_frag_mol.HasProp('minimizedAffinity'):
                            aff = float(min_frag_mol.GetProp('minimizedAffinity'))
                    except:
                        pass
                    frag_minimized_paths.append(minimized_frag_sdf)
                    frag_affinities.append(aff)
                    successed_idx.append(idx)
                except:
                    # Minimize 或后续出错，也追加占位符
                    pass
            else:
                # SDF 写入失败，追加占位符
                pass
   
        # D. 批量 NCI 筛选与综合打分（现在用片段自己的affinity）
        print(f">> 正在对 {len(fragments_to_screen)} 个片段进行 NCI 计算与综合评分...")
        fragments_to_screen = [fragments_to_screen[i] for i in successed_idx]
        frag_mols_only = [t[0] for t in fragments_to_screen]  # 原frag_mol用于NCI
       
        nci_dfs = get_nci_info_batch(rd_pkt, frag_mols_only)
       
        # 存储通过筛选的片段：key=parent_path, value=list of (score, frag_mol, minimized_path)
        parent_to_valid_frags = {}
        valid_frag_count = 0
        print("Check Lenght of nci_dfs:", len(nci_dfs), " vs fragments_to_screen:", len(fragments_to_screen), " vs frag_affinities:", len(frag_affinities))
        for i, df in enumerate(nci_dfs):
            # 1. 计算 NCI 总数
            ligands = df.index.get_level_values('ligand').unique()
            full_index = pd.MultiIndex.from_product(
                [ligands, ['HBAcceptor', 'HBDonor', 'PiStacking']],
                names=['ligand', 'interaction']
            )
            nci_df = df.reindex(full_index, fill_value=False)
            nci_counts = nci_df.sum().sum()
           
            # 2. 获取元数据
            frag_mol = fragments_to_screen[i][0]
            parent_path = fragments_to_screen[i][1]
            frag_aff = frag_affinities[i]  # 用片段自己的affinity
   
            # 3. 综合评分与筛选
            # 只有存在非共价相互作用时才进入评分
            if nci_counts > 0:
                score, metrics, is_pass = calc_seed_quality(frag_mol, frag_aff, nci_counts)
                if is_pass:
                    if parent_path not in parent_to_valid_frags:
                        parent_to_valid_frags[parent_path] = []
                    # 存储 (score, frag_mol, minimized_path)，方便后续合并使用minimized版本
                    parent_to_valid_frags[parent_path].append((score, frag_mol, frag_minimized_paths[i]))
                    valid_frag_count += 1
   
        print(f">> 筛选完成，保留了 {valid_frag_count}/{len(fragments_to_screen)} 个高质量类药片段。")
   
        # E. 合并片段 & 保存种子
        all_valid_fragments = []  # 每项: (score, frag_mol_from_minimized_sdf, minimized_sdf_path)
        for parent_path, scored_frags in parent_to_valid_frags.items():
            for score, orig_frag_mol, minimized_path in scored_frags:
                if minimized_path and os.path.exists(minimized_path):
                    suppl = Chem.SDMolSupplier(minimized_path)
                    mol_min = next(suppl, None)
                    if mol_min:
                        mol_min.SetProp("_SeedScore", str(score))
                        writer = Chem.SDWriter(minimized_path)
                        # mol_min_noH = Chem.RemoveHs(mol_min)
                        writer.write(mol_min)
                        writer.close()
                        all_valid_fragments.append((score, mol_min, minimized_path))
                        seed_library.append(minimized_path)
                        

        print(f">> 全局高质量片段池大小: {len(all_valid_fragments)}")


        # 如果片段不足，跳过合并
        if len(all_valid_fragments) == 0:
            continue


        # 按分数降序排序，优先使用高分片段
        all_valid_fragments.sort(key=lambda x: x[0], reverse=True)


        # 随机打乱以引入多样性（可选，但避免总是选前几个）
        random.shuffle(all_valid_fragments)

        seed_library = deduplicate_fragments(seed_library)
        new_seeds_count = 0
        used_indices = set()
        max_seeds_this_iter = min(50, len(all_valid_fragments))  # 控制新增种子数量
        
        # 尝试构建新种子：最多3片段，两两距离 > 4Å
        
        merged_seed = merge_fragments_via_cliques(
            fragment_paths=[frag[2] for frag in all_valid_fragments],
            output_dir=wkd_seed_iter,
            max_seeds=max_seeds_this_iter,
            min_inter_frag_dist=4.0,
            reward_per_frag=0.02
        )
        
        seed_library.extend(merged_seed)
        new_seeds_count = len(merged_seed)

        print(f"✅ Iteration {iteration} 结束: 新增 {new_seeds_count} 个优质种子（基于全局随机匹配），累计种子 {len(seed_library)} 个。")
       
        # 5. 维护种子库大小，仅保留分值靠前的100个种子片段
        if len(seed_library) > 100:
            print(f">> 维护种子库：当前有 {len(seed_library)} 个种子，需要缩减到100个...")
           
            # 读取所有种子的分数
            seed_scores = []
            for seed_path in seed_library:
                score = 0.0
                try:
                    if os.path.exists(seed_path):
                        suppl = Chem.SDMolSupplier(seed_path)
                        mol = next(suppl, None)
                        if mol and mol.HasProp('_SeedScore'):
                            score = float(mol.GetProp('_SeedScore'))
                except:
                    pass
                seed_scores.append(score)
           
            # 按分数降序排序
            sorted_pairs = sorted(zip(seed_library, seed_scores), key=lambda x: x[1], reverse=True)
           
            # 保留前100个
            seed_library = [pair[0] for pair in sorted_pairs[:100]]
           
            print(f">> 种子库已缩减为 {len(seed_library)} 个高分种子。")


    if good_affinity_mols:
        final_sdf_path = os.path.join(output_base_path, f'{basename}.sdf')
        final_seed_path = os.path.join(output_base_path, f'{basename}_finalseed.sdf')
        merge_sdfs(good_affinity_mols, final_sdf_path)
        merge_sdfs(seed_library, final_seed_path)
    else:
        print("未生成任何 Affinity < 0 的分子。")
    print(f"\n🎉 任务完成！共获得 {len(good_affinity_mols)} 个 Affinity < 0 的分子。")
    print(f"结果保存在: {output_base_path}")
    os.system(f"rm -rf {output_base_path}/iter_*/")  # 清理中间迭代文件夹
    os.system(f"rm -rf {output_base_path}/iter_*/")  # 清理中间迭代文件夹

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pocket-conditioned ligand generation (cold start or seed inpainting).')
    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint (.pth)")
    parser.add_argument("--pocket", type=str, required=True,
                        help="full receptor PDB; a <stem>_pocket.pdb is derived from it")
    parser.add_argument("--ligand", type=str, required=True,
                        help="reference/dummy ligand SDF defining the pocket center")
    parser.add_argument("--output_prefix", type=str, default="gen")
    parser.add_argument("--seed_used", action="store_true",
                        help="seed inpainting mode (anchor the ligand SDF as fixed atoms)")
    parser.add_argument("--total_target", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)

    cfg = Config(config_file='config.yaml')
    model = BFN4SBDDScoreModel(net_config=cfg.dynamics.net_config,
                               protein_atom_feature_dim=11,
                               ligand_atom_feature_dim=13, time_emb_dim=1,
                               ed_feature_dim=4)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    model.load_state_dict(
        load_compiled_model(model, torch.load(args.ckpt, map_location='cpu')),
        strict=False)
    model = accelerator.prepare(model)
    model.eval()

    generate((args.ligand, args.pocket), output_prefix=args.output_prefix,
             seed_used=args.seed_used, total_target=args.total_target,
             batch_size=args.batch_size)
