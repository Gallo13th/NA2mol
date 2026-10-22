import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataloader.dataloader_pointcloud import MolPocketPointCloud
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import GetPeriodicTable
import pandas as pd
from util.ligand_code_util import parse_pdb_file, parse_sdf_file
from dataloader.atom_num import sample_atom_num, get_space_size
def build_vdw_radius_dict():
    """
    使用 RDKit 构建一个元素符号到范德华半径的字典
    通过创建单个原子（Carbon-like）并设置其原子序数来获取 GetVdwRad()
    """
    vdw_dict = {}
    pt = GetPeriodicTable()
    for atomic_num in range(1, 119):  # 1 到 118 号元素
        try:
            vdw_radius = pt.GetRvdw(atomic_num)
            vdw_dict[atomic_num] = vdw_radius
        except Exception as e:
            print(f"Could not get VDW radius for atomic number {atomic_num}: {e}")
    return vdw_dict

# 生成查找表（只需运行一次，可以保存下来）
VDW_RADIUS_DICT = build_vdw_radius_dict()


'''
Reference: MolCRAFT

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

'''
pocket: {
    'smiles': str,
    'element': np.ndarray,  # shape: (num_atoms,)
    'pos': np.ndarray,  # shape: (num_atoms, 3)
    'bond_index': np.ndarray,  # shape: (2, num_bonds)
    'bond_type': np.ndarray,  # shape: (num_bonds,)
    'center_of_mass': np.ndarray,  # shape: (3,)
    'atom_feature': np.ndarray,  # shape: (num_atoms, num_features)
    'hybridization': list[str],  # length: num_atoms
}
ed: {
    'ed_code': list[int],  # length: num_ed
    'ed_pos': np.ndarray,  # shape: (num_ed, 3)
}

mol:{
    'smiles': str,
    'element': np.ndarray,  # shape: (num_atoms,)
    'pos': np.ndarray,  # shape: (num_atoms, 3)
    'bond_index': np.ndarray,  # shape: (2, num_bonds)
    'bond_type': np.ndarray,  # shape: (num_bonds,)
    'center_of_mass': np.ndarray,  # shape: (3,)
    'atom_feature': np.ndarray,  # shape: (num_atoms, num_features)
    'hybridization': list[str],  # length: num_atoms
}
sdf_path: str,  # path to the SDF file
pdb_id: str,  # PDB ID of the molecule
'''

HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2' ,'UNSPECIFIED']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}

POCKET_COMMON_ATOMIC_NUMBER = torch.LongTensor([1, 6, 7, 8, 16, 34])

VOLUMNE_TO_NATOM = pd.read_csv("dataloader/volume_to_natom_prob_table.csv")

AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

AA_NAME_NUMBER = {
    k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
}

RES_NAME_NUMBER = {
    'PR' + v: i for i, (_, v) in enumerate(AA_NAME_SYM.items())
}

RES_NAME_NUMBER.update({
    'NTA': 20, 'NTC': 21, 'NTG': 22, 'NTU': 23, 'NTT': 24, 'UNK': 25
})

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

def estimate_pointcloud_volume_fast(coords, alpha_scale=2.0, verbose=False):
    """
    ⚡️ 为多进程优化的体积估算（仅凸包，避免 pyvista 序列化问题）
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    if len(coords) < 4:
        return None

    # 快速估算平均最近邻距离
    try:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', n_jobs=1).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        avg_dist = np.mean(distances[:, 1])
    except:
        avg_dist = 1.0

    # 仅使用凸包（最快、稳定、可并行）
    try:
        hull = ConvexHull(coords, qhull_options='QJ')
        return hull.volume
    except:
        return None

def rotate_random_3d():
    """
    生成一个随机的3D旋转矩阵 (SO(3))
    """
    # 使用随机正交矩阵（通过QR分解）
    rand_mat = np.random.randn(3, 3)
    q, r = np.linalg.qr(rand_mat)
    # 保证行列式为1（右手系）
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float32)

def get_index(atom_num, is_aromatic):
    return MAP_ATOM_TYPE_AROMATIC_TO_INDEX.get((atom_num,is_aromatic),0)

def index_to_atom_type_aromatic(index):
    return MAP_INDEX_TO_ATOM_TYPE_AROMATIC[index]

def pocket_to_tensor(pocket, mode='general'):
    pocket_pos = torch.tensor(pocket['pos'], dtype=torch.float32)
    pocket_element = torch.tensor(pocket['element'], dtype=torch.int64)
    # pocket_bond_index = torch.tensor(pocket['bond_index'], dtype=torch.int64)
    # pocket_bond_type = torch.tensor(pocket['bond_type'], dtype=torch.int64)
    # pocket_center_of_mass = torch.tensor(pocket['center_of_mass'], dtype=torch.float32)
    pocket_atom_feature = torch.tensor(pocket['atom_feature'], dtype=torch.float32)
    if mode == 'general':
        pocket_hybridization = torch.tensor([HYBRIDIZATION_TYPE_ID.get(h,6) for h in pocket['hybridization']], dtype=torch.int64)
        pocket_atom_feature[:,-1] = pocket_hybridization # don't insert backbone feat
        vdw_radius = [VDW_RADIUS_DICT.get(int(elem), 2.0) for elem in pocket_element]  # 默认 2.0 Å
        partialcharges = torch.nan_to_num(torch.exp(torch.tensor(pocket['partialcharges'], dtype=torch.float16)))
        pocket_atom_feature = torch.cat([pocket_atom_feature, partialcharges[:,None], torch.tensor(vdw_radius)[:,None]],dim=-1)
        if 'residue_name' in pocket:
            residue = torch.tensor([RES_NAME_NUMBER.get(res,0) for res in pocket['residue_name']], dtype=torch.int64)
        elif 'residue_type' in pocket:
            residue = torch.tensor([RES_NAME_NUMBER.get(res,0) for res in pocket['residue_type']], dtype=torch.int64)
        else:
            residue = torch.zeros(len(pocket_element), dtype=torch.int64)
        # pocket_atom_feature = torch.cat([pocket_atom_feature, torch.zeros(len(vdw_radius),1), torch.tensor(vdw_radius)[:,None]],dim=-1)
        
        return {
            'pos': pocket_pos,
            'element': pocket_element,
            'vdw_radius': torch.tensor(vdw_radius),
            'atom_feature': pocket_atom_feature,
            'hybridization': pocket_hybridization,
            'residue': residue
        }
    elif mode == 'molcraft':
        element = pocket_element.view(-1, 1) == POCKET_COMMON_ATOMIC_NUMBER.view(1, -1)
        residue_type = [AA_NAME_NUMBER.get(res,0) for res in pocket['residue_type']]
        aa_feat = F.one_hot(torch.tensor(residue_type,dtype=torch.long),num_classes=20)
        is_backbone = pocket_atom_feature[:,-1]
        pocket_atom_feature = torch.cat([element,aa_feat,is_backbone[:,None]],dim=-1)
        return {
            'pos': pocket_pos,
            'element': pocket_element,
            'residue': torch.tensor(residue_type),
            'atom_feature': pocket_atom_feature
        }
    # pocket_atom_feature = torch.where(torch.all(pocket_atom_feature[:,None] == pocket_atom_feat_set[None,:], dim=-1))[1]
    else:
        raise NotImplementedError(f"No {mode} Mode")
    
def mol_to_tensor(mol, mode='general'):
    mol_pos = torch.tensor(mol['pos'], dtype=torch.float32)
    mol_element = torch.tensor(mol['element'], dtype=torch.int64)
    mol_atom_feature = torch.tensor(mol['atom_feature'], dtype=torch.float32)
    mol_smiles = mol['smiles']
    if mode == 'general':
        mol_hybridization = torch.tensor([HYBRIDIZATION_TYPE_ID[h] for h in mol['hybridization']], dtype=torch.int64)


        return {
            'smiles': mol_smiles,
            'pos': mol_pos,
            'element': mol_element,
            'atom_feature': mol_atom_feature,
            'hybridization': mol_hybridization
        }
    elif mode == 'molcraft':
        mol_feat = torch.tensor(
            [get_index(int(element), bool(item[2])) for element, item in zip(list(mol_element), list(mol_atom_feature))]
        )
        # nonh_mask = (mol_element != 1)
        try:
            bond_type = torch.tensor(mol['bond_type'], dtype=torch.int64)
            bond_index = torch.tensor(mol['bond_index'], dtype=torch.int64)
        except Exception:
            bond_type = torch.zeros(0, dtype=torch.int64)
            bond_index = torch.zeros(2, 0, dtype=torch.int64)
        return {
            'smiles': mol_smiles,
            'pos': mol_pos,
            'element': mol_element,
            'atom_feature': mol_feat,
            'bond_type': bond_type,
            'bond_index': bond_index
        }
    else:
        raise NotImplementedError("No {mode} Mode")
    
from torch_scatter import scatter_max
import numpy as np

def ed_to_tensor(ed, min_distance=None):
    """
    将 ed 转为 tensor，并使用 torch-scatter 基于体素网格进行降采样。
    
    参数:
        ed: dict, 包含 'ed_code' 和 'ed_pos'
        min_distance: float, 体素大小（即最小距离阈值）
    """
    ed_code = torch.tensor(ed['ed_code'], dtype=torch.int64)      # (N,)
    ed_pos = torch.tensor(ed['ed_pos'], dtype=torch.float32)      # (N, 3)

    if min_distance is not None and len(ed_pos) > 0:
        # 1. 将点坐标量化到体素网格
        voxel_coords = (ed_pos / min_distance).round().to(torch.int64)  # (N, 3)
        
        # 2. 将体素坐标展平为唯一键（hash）
        flat_coords = voxel_coords[:, 0] * 1000000 + voxel_coords[:, 1] * 1000 + voxel_coords[:, 2]
        
        # 3. 使用 scatter_max 获取每个体素中任意一个点的索引（或保留第一个）
        idx_per_voxel, unq_inv = flat_coords.unique(return_inverse=True)
        perm = torch.arange(len(flat_coords), dtype=torch.long, device=flat_coords.device)
        idx_of_voxels = scatter_max(perm, unq_inv)[1]  # 每个体素中保留最大 perm 的点（即最后出现）
        
        # 4. 只保留每个体素中的一个点
        mask = torch.zeros(len(flat_coords), dtype=torch.bool)
        mask[idx_of_voxels] = True
        
        ed_code = ed_code[mask]
        ed_pos = ed_pos[mask]

    return {
        'ed_code': ed_code,
        'ed_pos': ed_pos
    }

# Cold-start heavy-atom count range. Defaults to the benchmark setting (20, 28).
# Override with NA2SMOL_ATOM_RANGE="26,29" for a tighter, series-matched distribution.
def _atom_range_from_env(default=(20, 28)):
    import os
    raw = os.environ.get('NA2SMOL_ATOM_RANGE', '')
    if not raw:
        return default
    try:
        lo, hi = (int(x) for x in raw.split(','))
    except Exception:
        return default
    return (lo, hi) if 0 < lo <= hi else default


COLD_START_ATOM_RANGE = _atom_range_from_env()


class PocketEDMol(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def from_protein_ligand_dicts(pocket_dict=None,ed_dict=None,mol_dict=None, **kwargs):
        instance = PocketEDMol(**kwargs)

        if pocket_dict is not None:
            for key, item in pocket_dict.items():
                instance['pocket_' + key] = item

        if ed_dict is not None:
            for key, item in ed_dict.items():
                instance['ed_' + key] = item

        if mol_dict is not None:
            for key, item in mol_dict.items():
                instance['mol_' + key] = item

        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'pocket_bond_index':
            return self['pocket_element'].size(0)
        elif key == 'mol_bond_index':
            return self['mol_element'].size(0)
        else:
            return super().__inc__(key, value)


class GeomMolOnlyPocketPretrain(Dataset):
    def __init__(self, raw_dataset, mode='general', pocket_radius=None):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        self.mode = mode
        self.pocket_radius = pocket_radius  # 新增：定义口袋裁剪半径

    def __getitem__(self, idx):

        item = self.raw_dataset[idx]

        mol = item[1]
        pocket = item[0]

        # 正常处理数据
        pocket_tensor = pocket_to_tensor(pocket, mode=self.mode)
        mol_tensor = mol_to_tensor(mol, mode="molcraft")

        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        pocket_geom_center = pocket_tensor['pos'].mean(dim=0)

        pocket_tensor['geom_center'] = pocket_geom_center
        mol_tensor['geom_center'] = mol_geom_center
        
        # 将坐标中心化到口袋几何中心
        pocket_tensor['pos'] -= mol_geom_center
        mol_tensor['pos'] -= mol_geom_center
        R = rotate_random_3d()

        mol_tensor['pos'] = mol_tensor['pos'] @ R.T  # (N,3) @ (3,3) -> (N,3)
        pocket_tensor['pos'] = pocket_tensor['pos'] @ R.T
        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        
        # 如果设置了口袋半径，则进行裁剪
        if self.pocket_radius is not None:
            distances = torch.norm(pocket_tensor['pos'] - mol_geom_center.unsqueeze(0), dim=1)
            mask = distances <= self.pocket_radius

            # 只保留满足条件的原子
            pocket_tensor['pos'] = pocket_tensor['pos'][mask]
            pocket_tensor['element'] = pocket_tensor['element'][mask]
            pocket_tensor['atom_feature'] = pocket_tensor['atom_feature'][mask]
            if 'hybridization' in pocket_tensor:
                pocket_tensor['hybridization'] = pocket_tensor['hybridization'][mask]
            if 'residue' in pocket_tensor:
                pocket_tensor['residue'] = pocket_tensor['residue'][mask]
        ed_tensor = {}
        ed_tensor['ed_code'] = torch.empty((0,4),device=mol_tensor['pos'].device)
        ed_tensor['ed_pos'] = torch.empty((0,3),device=mol_tensor['pos'].device)
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)

class GeomMolPocketPointCloud(Dataset):
    def __init__(self, raw_dataset, mode='general', pocket_radius=None):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        self.mode = mode
        self.pocket_radius = pocket_radius  # 新增：定义口袋裁剪半径

    def __getitem__(self, idx):

        item = self.raw_dataset[idx]

        mol = item['mol']
        pocket = item['pocket']
        residx = random.randint(0, len(item['ed']))
        print(residx)
        # residx = 0
        ed = item['ed']#[residx]
        # print(ed)
        # 正常处理数据
        pocket_tensor = pocket_to_tensor(pocket, mode=self.mode)
        ed_tensor = ed_to_tensor(ed)
        mol_tensor = mol_to_tensor(mol, mode="molcraft")

        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        pocket_geom_center = pocket_tensor['pos'].mean(dim=0)

        pocket_tensor['geom_center'] = pocket_geom_center
        mol_tensor['geom_center'] = mol_geom_center
        # 将坐标中心化到口袋几何中心
        pocket_tensor['pos'] -= mol_geom_center
        mol_tensor['pos'] -= mol_geom_center
        R = rotate_random_3d()
        if ed_tensor['ed_pos'].shape[0] != 0:
            ed_tensor['ed_pos'] =  ed_tensor['ed_pos']  - mol_geom_center
            ed_tensor['ed_pos'] = ed_tensor['ed_pos'] # @ R.T
        # 应用旋转
        mol_tensor['pos'] = mol_tensor['pos'] # @ R.T  # (N,3) @ (3,3) -> (N,3)
        pocket_tensor['pos'] = pocket_tensor['pos'] # @ R.T
        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        # 如果设置了口袋半径，则进行裁剪
        if self.pocket_radius is not None:
            distances = torch.norm(pocket_tensor['pos'] - mol_geom_center.unsqueeze(0), dim=1)
            mask = distances <= self.pocket_radius

            # 只保留满足条件的原子
            pocket_tensor['pos'] = pocket_tensor['pos'][mask]
            pocket_tensor['element'] = pocket_tensor['element'][mask]
            pocket_tensor['atom_feature'] = pocket_tensor['atom_feature'][mask]
            if 'hybridization' in pocket_tensor:
                pocket_tensor['hybridization'] = pocket_tensor['hybridization'][mask]
            if 'residue' in pocket_tensor:
                pocket_tensor['residue'] = pocket_tensor['residue'][mask]
        # pocket_tensor = {'pos':torch.empty((0,3)),'element':torch.empty((0,)),'atom_feature':torch.empty((0,11))}
        volume = estimate_pointcloud_volume_fast(ed_tensor['ed_pos'].cpu().numpy())

        # print(volume)
        n_atoms_params = VOLUMNE_TO_NATOM[(VOLUMNE_TO_NATOM['volume_low'] < volume) & (VOLUMNE_TO_NATOM['volume_high'] > volume)].iloc[0]
        # print(n_atoms_params)
        mu, sigma = n_atoms_params['n_atoms_mu'], n_atoms_params['n_atoms_sigma']
        # 假设 mu, sigma 是 float，从 prob_table 中查表得到
        mu = torch.tensor(mu, dtype=torch.float32)
        sigma = torch.tensor(sigma, dtype=torch.float32)
        
        # 方法2：如果你想限制在 [μ-2σ, μ+2σ]（覆盖95%），但仍允许超出后 clamp
        n_atoms = torch.normal(mu, sigma).clamp(min=mu - 1.5*sigma, max=mu + 1.5*sigma).clamp(min=1.0).long().item()

        mol_tensor = {
            'pos': torch.randn(n_atoms,3),
            'element': torch.zeros(n_atoms),
            'atom_feature': torch.zeros(n_atoms),
            'geom_center': mol_tensor['geom_center']
        }
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)

class GeomMolOnlyPocket(Dataset):
    def __init__(self, raw_dataset, mode='general', pocket_radius=None, is_train=False, is_rna=False):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        self.mode = mode
        self.is_train = is_train
        self.pocket_radius = pocket_radius  # 新增：定义口袋裁剪半径
        self.is_rna = is_rna

    def __getitem__(self, idx):

        item = self.raw_dataset[idx]
        if isinstance(item,dict):
            mol = item['mol']
            pocket = item['pocket']
        else:
            pocket, mol = item
        
        ed = item['ed']
        if isinstance(ed, (list, tuple)):
            if len(ed) > 0 and isinstance(ed[0], dict):
                # expdb 格式：列表中每个元素是 {'ed_code', 'ed_pos'}
                ed = ed[0]
            else:
                code, pos = ed
                if isinstance(code[0], str):
                    code = [{'N': 290, 'O': 291, 'F': 292, 'C': 293}[c] for c in code]
                ed = {'ed_code':code, 'ed_pos':pos}
        if 'file' in item:
            pdb_id = item['file'][:4].lower()
        elif 'pdb_id' in item:
            pdb_id = item['pdb_id'].lower()
        else:
            pdb_id = f'unk_{idx}'

        # 正常处理数据
        pocket_tensor = pocket_to_tensor(pocket, mode=self.mode)
        ed_tensor = ed_to_tensor(ed)
        mol_tensor = mol_to_tensor(mol, mode="molcraft")

        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        pocket_geom_center = pocket_tensor['pos'].mean(dim=0)

        pocket_tensor['geom_center'] = pocket_geom_center
        mol_tensor['geom_center'] = mol_geom_center
        # 将坐标中心化到口袋几何中心
        pocket_tensor['pos'] -= mol_geom_center
        mol_tensor['pos'] -= mol_geom_center
        ed_tensor['ed_pos'] -= mol_geom_center

        R = rotate_random_3d()
        # 应用随机旋转（数据增强）
        mol_tensor['pos'] = mol_tensor['pos'] @ R.T
        pocket_tensor['pos'] = pocket_tensor['pos'] @ R.T
        ed_tensor['ed_pos'] = ed_tensor['ed_pos'] @ R.T

        # RNA 专用更强数据增强（训练时）
        if self.is_rna and self.is_train:
            # 轻微坐标抖动（模拟晶体学不确定性）
            noise_scale = 0.15
            mol_tensor['pos'] += torch.randn_like(mol_tensor['pos']) * noise_scale
            pocket_tensor['pos'] += torch.randn_like(pocket_tensor['pos']) * noise_scale
            ed_tensor['ed_pos'] += torch.randn_like(ed_tensor['ed_pos']) * noise_scale
            # 随机平移
            shift = torch.randn(3) * 0.3
            mol_tensor['pos'] += shift
            pocket_tensor['pos'] += shift
            ed_tensor['ed_pos'] += shift

        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        # 如果设置了口袋半径，则进行裁剪
        if self.pocket_radius is not None:
            distances = torch.norm(pocket_tensor['pos'] - mol_geom_center.unsqueeze(0), dim=1)
            mask = distances <= self.pocket_radius

            # 只保留满足条件的原子
            pocket_tensor['pos'] = pocket_tensor['pos'][mask]
            pocket_tensor['element'] = pocket_tensor['element'][mask]
            pocket_tensor['atom_feature'] = pocket_tensor['atom_feature'][mask]
            if 'hybridization' in pocket_tensor:
                pocket_tensor['hybridization'] = pocket_tensor['hybridization'][mask]
            if 'residue' in pocket_tensor:
                pocket_tensor['residue'] = pocket_tensor['residue'][mask]

        pocket_tensor['pdb_id'] = pdb_id
        pocket_tensor['is_rna'] = self.is_rna
        pocket_tensor['file'] = item.get('file', pdb_id)
        
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)


class GeomSeedMolOnlyPocket(Dataset):
    def __init__(self, pocket, seed, mode='general', pocket_radius=None, extra_range = (0,6), remain_seed = False):
        self.pocket_path = pocket
        self.seed_path = seed
        self.mode = mode
        self.pocket_radius = pocket_radius
        self.extra_range = extra_range
        self.remain_seed = remain_seed
        # ✅ 预处理：只执行一次
        self._preprocess()

    def _preprocess(self):
        # Parse files once
        mol = parse_sdf_file(path=self.seed_path)
        pocket = parse_pdb_file(path=self.pocket_path)

        # Convert to tensor
        mol_tensor = mol_to_tensor(mol, mode="molcraft")
        pocket_tensor = pocket_to_tensor(pocket, mode=self.mode)

        # Center on seed molecule's geometric center
        mol_center = mol_tensor['pos'].mean(dim=0)
        mol_tensor['pos'] = mol_tensor['pos'] - mol_center
        pocket_tensor['pos'] = pocket_tensor['pos'] - mol_center

        # Crop pocket if radius is specified
        if self.pocket_radius is not None:
            distances = torch.norm(pocket_tensor['pos'], dim=1)
            mask = distances <= self.pocket_radius
            pocket_tensor = self._apply_mask(pocket_tensor, mask)
        self.pocket_size = get_space_size(pocket_tensor['pos'].detach().cpu().numpy()) 
        # ✅ Cache the preprocessed tensors
        self._fixed_mol_tensor = {
            'pos': mol_tensor['pos'].clone(),
            'element': mol_tensor['element'].clone(),
            'atom_feature': mol_tensor['atom_feature'].clone(),
            'bond_index': mol_tensor['bond_index'].clone(),
            'bond_type': mol_tensor['bond_type'].clone(),
            'geom_center': mol_center,
        }
        self._pocket_tensor = pocket_tensor  # already cropped & centered

    def _apply_mask(self, d, mask):
        return {k: v[mask] if torch.is_tensor(v) and v.shape[0] == mask.shape[0] else v for k, v in d.items()}

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset contains only one sample.")

        # Get cached fixed part
        fixed_pos = self._fixed_mol_tensor['pos']
        fixed_elem = self._fixed_mol_tensor['element']
        fixed_feat = self._fixed_mol_tensor['atom_feature']
        origin_n = fixed_pos.shape[0]

        # Sample number of extra atoms
        # extra_n = random.randint(self.extra_range[0], self.extra_range[1])
        if self.remain_seed:
            total_n = max(origin_n + 6, sample_atom_num(self.pocket_size))
        else:
            # SPEC06: limit cold-start atom count to known_mols heavy-atom range (22-27) + margin.
            # The benchmark used (20, 28); set NA2SMOL_ATOM_RANGE="lo,hi" to narrow it, e.g.
            # to match a series of known actives without editing this file.
            lo, hi = COLD_START_ATOM_RANGE
            total_n = random.randint(lo, hi)
        # Initialize full tensors
        pos = torch.zeros(total_n, 3)
        element = torch.zeros(total_n, dtype=torch.long)
        atom_feature = torch.zeros(total_n, dtype=torch.float)
        fixed_mask = torch.zeros(total_n, dtype=torch.bool)
        # Place fixed atoms at the beginning
        if origin_n < total_n:
            pos[:origin_n] = fixed_pos
            element[:origin_n] = fixed_elem
            atom_feature[:origin_n] = fixed_feat
            # Fixed mask: first `origin_n` are fixed
            fixed_mask[:origin_n] = True
        else:
            pos = fixed_pos[:total_n]
            element = fixed_elem[:total_n]
            atom_feature = fixed_feat[:total_n]
            fixed_pos = fixed_pos[:total_n]
            fixed_feat = fixed_feat[:total_n]
            fixed_mask[:] = True

        # Construct final mol dict
        mol_out = {
            'pos': pos,
            'element': element,
            'atom_feature': atom_feature,
            'bond_index': self._fixed_mol_tensor['bond_index'],
            'bond_type':self._fixed_mol_tensor['bond_type'],
            'fixed_atom_mask': fixed_mask,
            'fixed_atom_pos': fixed_pos.clone(),
            'fixed_atom_types': fixed_feat.clone(),
            'geom_center': self._fixed_mol_tensor['geom_center'],  # already centered at origin
        }

        # Empty ed dict
        device = pos.device
        ed_tensor = {
            'ed_code': torch.empty((0, 4), device=device),
            'ed_pos': torch.empty((0, 3), device=device),
        }

        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=self._pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_out,
        )
        # Carry the seed path on the Data so that after torch_geometric collate
        # ``batch.mol_seed_path`` is the per-sample list of seed SDF paths
        # (string fields are collected into a Python list of length batch_size).
        data.mol_seed_path = self.seed_path
        return data

    def __len__(self):
        return 1

class GeomMolOnlyPocketDefinedNatom(Dataset):
    def __init__(self, raw_dataset, mode='general', natom=1, pocket_radius=None):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        self.mode = mode
        self.pocket_radius = pocket_radius  # 新增：定义口袋裁剪半径
        self.natom = natom

    def __getitem__(self, idx):

        item = self.raw_dataset[idx]

        mol = item['mol']
        pocket = item['pocket']
        ed = item['ed']
        if isinstance(ed,list):
            ed = ed[0]

        # 正常处理数据
        pocket_tensor = pocket_to_tensor(pocket, mode=self.mode)
        ed_tensor = ed_to_tensor(ed)
        mol_tensor = mol_to_tensor(mol, mode="molcraft")

        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        pocket_geom_center = pocket_tensor['pos'].mean(dim=0)

        pocket_tensor['geom_center'] = pocket_geom_center
        mol_tensor['geom_center'] = mol_geom_center
        # 将坐标中心化到口袋几何中心
        pocket_tensor['pos'] -= mol_geom_center
        mol_tensor['pos'] -= mol_geom_center
        R = rotate_random_3d()
        if ed_tensor['ed_pos'].shape[0] != 0:
            ed_tensor['ed_pos'] =  ed_tensor['ed_pos']  - mol_geom_center
            ed_tensor['ed_pos'] = ed_tensor['ed_pos'] # @ R.T
        # 应用旋转
        mol_tensor['pos'] = mol_tensor['pos'] # @ R.T  # (N,3) @ (3,3) -> (N,3)
        pocket_tensor['pos'] = pocket_tensor['pos'] # @ R.T
        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        # 如果设置了口袋半径，则进行裁剪
        if self.pocket_radius is not None:
            distances = torch.norm(pocket_tensor['pos'] - mol_geom_center.unsqueeze(0), dim=1)
            mask = distances <= self.pocket_radius

            # 只保留满足条件的原子
            pocket_tensor['pos'] = pocket_tensor['pos'][mask]
            pocket_tensor['element'] = pocket_tensor['element'][mask]
            pocket_tensor['atom_feature'] = pocket_tensor['atom_feature'][mask]
            if 'hybridization' in pocket_tensor:
                pocket_tensor['hybridization'] = pocket_tensor['hybridization'][mask]
            if 'residue' in pocket_tensor:
                pocket_tensor['residue'] = pocket_tensor['residue'][mask]
        # pocket_tensor = {'pos':torch.empty((0,3)),'element':torch.empty((0,)),'atom_feature':torch.empty((0,11))}
        ed_tensor['ed_code'] = torch.empty((0,4),device=ed_tensor['ed_code'].device)
        ed_tensor['ed_pos'] = torch.empty((0,3),device=ed_tensor['ed_pos'].device)
        mol_tensor.update({
            'pos': torch.randn(self.natom,3),
            'element': torch.zeros(self.natom),
            'atom_feature': torch.zeros(self.natom),
        })
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)

class ProteinLigandDataLoader(DataLoader):
    """
    Example data format:
    PocketEDMol(
        sdf_path='C:/github_official/EDMol/data/mol_sdf\10gs_ligand.sdf',
        pdb_id='10gs', 
        pocket_pos=[265, 3], 
        pocket_element=[265], 
        pocket_bond_index=[2, 518], 
        pocket_bond_type=[518], 
        pocket_center_of_mass=[3], 
        pocket_atom_feature=[265, 8], 
        pocket_hybridization=[265], 
        ed_ed_code=[499], 
        ed_ed_pos=[499, 3], 
        mol_pos=[33, 3], 
        mol_element=[33], 
        mol_bond_index=[2, 68], 
        mol_bond_type=[68], 
        mol_center_of_mass=[3], 
        mol_atom_feature=[33, 8], 
        mol_hybridization=[33]
        ) 
    """
    FOLLOW_BATCH = (
        'pocket_element',
        'mol_element', 
        'mol_bond_type',
        'ed_ed_code',
        )

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)

class GeomMolPointCloudPretrain(Dataset):
    def __init__(self, raw_dataset, pocket_radius=None):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        
    def __getitem__(self, idx):

        item = self.raw_dataset[idx]
        mol = item[2]
        pocket_tensor = {'pos':torch.empty((0,3)),'element':torch.empty((0,)),'atom_feature':torch.empty((0,11))}
        residx = random.randint(0,2)
        residx = 1
        # print(len(item[0]))
        ed = {'ed_pos':item[0][residx],'ed_code':item[1][residx]}

        
        # 正常处理数据
        ed_tensor = ed_to_tensor(ed)
        mol_tensor = mol_to_tensor(mol, mode="molcraft")
        # pocket_tensor = pocket_to_tensor(pocket)
        
        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        
        # print(mol_tensor['pos'].mean(dim=0),ed_tensor['ed_pos'].mean(dim=0))
        # 将坐标中心化到口袋几何中心
        mol_tensor['pos'] -= mol_geom_center
        ed_tensor['ed_pos'] -= mol_geom_center

        # 生成随机旋转矩阵（对分子和ed使用相同的旋转）
        # R = rotate_random_3d()
    
        # 应用旋转
        mol_tensor['pos'] = mol_tensor['pos'] # @ R.T  # (N,3) @ (3,3) -> (N,3)
        ed_tensor['ed_pos'] = ed_tensor['ed_pos'] # @ R.T
        
        # print(mol_tensor['pos'].mean(dim=0),ed_tensor['ed_pos'].mean(dim=0))
        
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)


class GeomMolPointCloudExp(Dataset):
    def __init__(self, raw_dataset, pocket_radius=None):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        
    def __getitem__(self, idx):

        item = self.raw_dataset[idx]
        mol = item['mol']
        pocket = item['pocket']
        ed = item['ed']
        pocket_tensor = {'pos':torch.empty((0,3)),'element':torch.empty((0,)),'atom_feature':torch.empty((0,11))}
        # residx = random.randint(0,2)
        # residx = 1
        # # print(len(item[0]))
        # ed = {'ed_pos':item[0][residx],'ed_code':item[1][residx]}

        
        # 正常处理数据
        ed_tensor = ed_to_tensor(ed)
        mol_tensor = mol_to_tensor(mol, mode="molcraft")
        # pocket_tensor = pocket_to_tensor(pocket)
        
        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        
        # print(mol_tensor['pos'].mean(dim=0),ed_tensor['ed_pos'].mean(dim=0))
        # 将坐标中心化到口袋几何中心
        mol_tensor['pos'] -= mol_geom_center
        ed_tensor['ed_pos'] -= mol_geom_center

        # 生成随机旋转矩阵（对分子和ed使用相同的旋转）
        # R = rotate_random_3d()
    
        # 应用旋转
        mol_tensor['pos'] = mol_tensor['pos'] # @ R.T  # (N,3) @ (3,3) -> (N,3)
        ed_tensor['ed_pos'] = ed_tensor['ed_pos'] # @ R.T
        
        # print(mol_tensor['pos'].mean(dim=0),ed_tensor['ed_pos'].mean(dim=0))
        
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)

class TestPointCloudPretrain(Dataset):
    def __init__(self, raw_dataset, pocket_radius=None):
        self.raw_dataset = raw_dataset
        self.skipped_indices = set()  # 记录被跳过的索引
        
    def __getitem__(self, idx):

        item = self.raw_dataset[idx]
        # mol = item[2]
        pocket_tensor = {'pos':torch.empty((0,3)),'element':torch.empty((0,)),'atom_feature':torch.empty((0,11))}
        mol = item['mol']
        ed = item['ed']
        # print(ed)
        # 正常处理数据
        
        ed_tensor = ed_to_tensor(ed)
        # print(ed)
        # print(ed_tensor['ed_pos'].cpu().numpy())
        mol_tensor = mol_to_tensor(mol,mode='molcraft')

        mol_geom_center = mol_tensor['pos'].mean(dim=0)
        mol_tensor['geom_center'] = mol_geom_center
        mol_tensor['pos'] -= mol_geom_center
        ed_tensor['ed_pos'] -= mol_geom_center
        volume = estimate_pointcloud_volume_fast(ed_tensor['ed_pos'].cpu().numpy())

        # print(volume)
        try:
            n_atoms_params = VOLUMNE_TO_NATOM[(VOLUMNE_TO_NATOM['volume_low'] < volume) & (VOLUMNE_TO_NATOM['volume_high'] > volume)].iloc[0]
            # print(n_atoms_params)
            mu, sigma = n_atoms_params['n_atoms_mu'], n_atoms_params['n_atoms_sigma']
            # 假设 mu, sigma 是 float，从 prob_table 中查表得到
            mu = torch.tensor(mu, dtype=torch.float32)
            sigma = torch.tensor(sigma, dtype=torch.float32)
            
            # 方法2：如果你想限制在 [μ-2σ, μ+2σ]（覆盖95%），但仍允许超出后 clamp
            n_atoms = torch.normal(mu, sigma).clamp(min=mu - 1.5*sigma, max=mu + 1.5*sigma).clamp(min=1.0).long().item()
        except:
            n_atoms = mol_tensor['pos'].shape[0]
        mol_tensor = {
            'pos': torch.randn(n_atoms,3),
            'element': torch.zeros(n_atoms),
            'atom_feature': torch.zeros(n_atoms),
            'geom_center': mol_geom_center
        }
        
        data = PocketEDMol.from_protein_ligand_dicts(
            pocket_dict=pocket_tensor,
            ed_dict=ed_tensor,
            mol_dict=mol_tensor,
        )
        return data

    def __len__(self):
        return len(self.raw_dataset) - len(self.skipped_indices)

class PretrainLigandDataLoader(DataLoader):
    FOLLOW_BATCH = (
        'pocket_element',
        'mol_element', 
        'mol_bond_type',
        'ed_ed_code',
        )

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


if __name__ == "__main__":
    # Example usage
    raw_dataset = MolPocketPointCloud(lmdb_path="/root/autodl-tmp/merged/")
    # print(raw_dataset[0])
    dataset = GeomMolPointCloudPretrain(raw_dataset)
    print(dataset[0])
    dataloader = PretrainLigandDataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        print(batch.ed_ed_code)
        # print(batch.ed_ed_pos)
        break

