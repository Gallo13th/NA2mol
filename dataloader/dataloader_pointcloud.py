
import os
import gc
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch.utils.data as data
import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# 关闭 RDKit 警告
RDLogger.DisableLog('rdApp.warning')
from util.pocket_code_all import PocketCode, FasterPocketCode
import random

from dataloader.data_utils import ed_pdb_read
# from util.point_cloud_util import find_centroids, augment_point_cloud_torch, compute_resolution_ratio
from util.ligand_code_util import FastLigandCodeUtil, parse_sdf_file, parse_pdb_file
# from joblib import Parallel, delayed 
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import os
import lmdb
import pickle
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from torch.utils import data
import io
import threading
import uuid
import warnings
warnings.filterwarnings("ignore", message=".*RingInfo not initialized.*")

class PocketPointCloud(data.Dataset):
    def __init__(self, root=os.getcwd() + '/data/', precompute=True, lmdb_path=os.getcwd() + '/data/lmdbcache_new'):
        self.root = root
        self.lmdb_path = lmdb_path
        self.pocketUtil = FasterPocketCode()
        self.fragUtil = FastLigandCodeUtil()
        self.symbol_point_cloud = {'N': 290, 'O': 291, 'F': 292, 'C': 293}
        # self.resolutions = [1.5, 2.7, 3.5, 5.0, 8.0]
        self.return_all = False
        # Ensure LMDB directory exists
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        
        # Load pocket files
        pocket_files = os.listdir(root + 'pocket_pdb/')
        error_file = os.path.join(os.getcwd(), 'error')
        self.error_set = set()
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                self.error_set = {line.strip().split('/')[-1][:-4] for line in f}

        self.items = []
        for pocket_file in pocket_files:
            if 'pdb' in pocket_file:
                pdb_id = pocket_file[:-4]
                if pdb_id in self.error_set:
                    continue
                pocket_path = os.path.join(root, 'pocket_pdb', pocket_file)
                sdf_path = os.path.join(root, 'mol_sdf', f'{pdb_id}_ligand.sdf')
                # point_cloud_paths = [
                #     os.path.join(root, 'ed', f'{pdb_id}_res{r:.1f}.pdb') 
                #     for r in self.resolutions
                # ]
                point_cloud_path = os.path.join(root, 'ed_3_5A', f'{pdb_id}.pdb')
                self.items.append({
                    'pdb_id': pdb_id,
                    'pocket_path': pocket_path,
                    'sdf_path': sdf_path,
                    'point_cloud_path': point_cloud_path
                })

        print(f'======= Loading {len(self.items)} samples =======')

        # Precompute data and store in LMDB
        if precompute:
            print("Precomputing data...")
            self._precompute_data()

    def _precompute_data(self):
        """Precompute data with batch LMDB writes"""
        num_workers = min(cpu_count()*3, 24)
        batch_size = 100  # Write in batches to reduce LMDB overhead
        results = []
        
        # Initialize LMDB environment
        map_size = 5 * 1024 * 1024 * 1024  # 5 GB
        env = lmdb.open(self.lmdb_path, map_size=map_size, readonly=False, create=True)
        lock = threading.Lock()  # For safe LMDB writes

        def write_batch(batch_data):
            with lock:
                with env.begin(write=True) as txn:
                    for idx, data in batch_data:
                        if data is not None:
                            # Serialize efficiently
                            buffer = io.BytesIO()
                            torch.save(data, buffer)
                            txn.put(str(idx).encode(), buffer.getvalue())

        with Pool(num_workers) as pool:
            batch = []
            for idx, result in enumerate(tqdm(
                pool.imap(self._preprocess_wrapper, range(len(self.items))),
                total=len(self.items)
            )):
                if result is not None:
                    batch.append((idx, result))
                if len(batch) >= batch_size or idx == len(self.items) - 1:
                    write_batch(batch)
                    batch = []
                results.append(result)

        # Write any remaining items
        if batch:
            write_batch(batch)
        
        env.close()
        successful = sum(1 for r in results if r is not None)
        print(f"Precomputed {successful} items (failed on {len(self.items) - successful}).")

    def _preprocess_wrapper(self, idx):
        """Wrapper function for multiprocessing"""
        try:
            return self.preprocess_data(idx)
        except Exception as e:
            print(f"Error precomputing index {idx}: {e}")
            return None

    def preprocess_data(self, idx):
        item = self.items[idx]
        pocket_path = item['pocket_path']
        sdf_path = item['sdf_path']
        point_cloud_path = item['point_cloud_path']

        # Pocket processing
        pocket_symbol, mask, pocket_pos, _ = self.pocketUtil.pocketCodeNCI(pocket_path)
        atom_pos, atom_type = self.fragUtil.encode(sdf_path)
        atom_pos_center = np.mean(atom_pos, axis=0)
        atom_pos = atom_pos - atom_pos_center
                            
        pc_poses = []
        pc_codes = []
        pocket_pos = (pocket_pos - 120) / 120

        # Pre-allocate arrays for efficiency
        max_points = 499
        point_cloud_file = point_cloud_path
            
        # Point cloud processing
        pc_code, pc_pos = ed_pdb_read(point_cloud_file)

        pc_code = [self.symbol_point_cloud.get(c, 0) for c in pc_code]
        pc_pos = self.convert_pos_pointcloud(np.array(pc_pos), center=np.array([0, 0, 0]))

        # Convert to tensors
        pc_pos = torch.from_numpy(pc_pos).long()
        pc_code = torch.tensor(pc_code, dtype=torch.long)

        atom_pos = torch.tensor(atom_pos, dtype=torch.float)
        atom_type = torch.tensor(atom_type, dtype=torch.long)            
        
        # Normalize positions
        pc_pos = (pc_pos - 120) / 120

        # Pad to fixed size
        if pc_pos.shape[0] < max_points:
            pad_size = max_points - pc_pos.shape[0]
            pc_pos = torch.nn.functional.pad(pc_pos, (0, 0, 0, pad_size))
            pc_code = torch.nn.functional.pad(pc_code, (0, pad_size))
        
        pc_poses.append(pc_pos)
        pc_codes.append(pc_code)
        
        pocket_symbol = torch.tensor(pocket_symbol, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        pocket_pos = torch.tensor(pocket_pos, dtype=torch.float)

        # Validate shapes
        if pocket_symbol.shape[0] != 500 or pc_codes[0].shape[0] != max_points:
            raise ValueError("Invalid shape")

        return {
            'pointcloud': [pc_poses, pc_codes],
            'pocket': [pocket_symbol, mask, pocket_pos],
            'mol': [atom_pos, atom_type],
        }

    def __getitem__(self, index):
        index = index % len(self.items)
        # Check LMDB cache
        with lmdb.open(self.lmdb_path, map_size=5*1024*1024*1024, readonly=True) as env:
            with env.begin() as txn:
                cached = txn.get(str(index).encode())
                if cached is not None:
                    # Deserialize
                    try:
                        buffer = io.BytesIO(cached)
                        data = torch.load(buffer)
                        if self.return_all:
                            data['sdf_path'] = self.items[index]['sdf_path']
                            if data['pointcloud'][0][0].shape[0] != 499:
                                return self.__getitem__((index + 1) % len(self.items))
                            return data
                        else:
                            resolution_idx = 0
                            if data['pointcloud'][0][0].shape[0] != 499 or torch.all(data['pointcloud'][0][0] == -1):
                                return self.__getitem__((index + 1) % len(self.items)) 
                            data['pointcloud'][0] = data['pointcloud'][0][resolution_idx][:499]
                            data['pointcloud'][1] = data['pointcloud'][1][resolution_idx][:499]  # Fixed bug: was using [0]
                            mol_len = data['mol'][0].shape[0]
                            data['mol'][1] = torch.nn.functional.one_hot(data['mol'][1], num_classes=12).float()
                            data['mol'] = self.sample_with_preserved_original(data['mol'][0] /12, data['mol'][1], 100 - mol_len, noise_std=0.05, k=5, add_noise_to_original=True)
                            return data
                    except:
                        return self.__getitem__((index + 1) % len(self.items))
        
        # Fallback: preprocess on-the-fly
        try:
            data = self.preprocess_data(index)
            if self.return_all:
                data['sdf_path'] = self.items[index]['sdf_path']
                if data['pointcloud'][0][0].shape[0] != 999:
                    return self.__getitem__((index + 1) % len(self.items))
                return data
            else:
                resolution_idx = 0
                if data['pointcloud'][0][0].shape[0] != 999:
                    return self.__getitem__((index + 1) % len(self.items))
                data['pointcloud'][0] = data['pointcloud'][0][resolution_idx][:999]
                data['pointcloud'][1] = data['pointcloud'][1][resolution_idx][:999]
                return data
        except Exception as e:
            print(e)
            self.error_set.add(self.items[index]['pdb_id'])
            return self.__getitem__((index + 1) % len(self.items))

    def __len__(self):
        return len(self.items)

    def convert_pos_pointcloud(self, pc_pos, center):
        grid_size = (self.fragUtil.grid_size - 1) / 2
        if center is None:
            center = np.array([0, 0, 0])
        pc_pos = (pc_pos - center) / self.fragUtil.resolution + np.array([grid_size, grid_size, grid_size])
        pc_pos = np.rint(pc_pos).astype(np.int32)
        
        # Vectorized boundary check
        invalid = (pc_pos < 0) | (pc_pos >= self.fragUtil.grid_size)
        pc_pos[invalid.any(axis=1)] = [-1, -1, -1]
        return pc_pos
    
    def get_lmdb_stats(self):
        with lmdb.open(self.lmdb_path, map_size=5*1024*1024*1024, readonly=True) as env:
            stats = env.stat()
            page_size = stats['psize']
            pages_used = stats['entries']
            total_size = page_size * pages_used
            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024
            return {
                'entries': stats['entries'],
                'page_size': page_size,
                'total_size_bytes': total_size,
                'total_size_mb': size_mb,
                'total_size_gb': size_gb
            }
    
    def save_errors(self):
        """Save errors to file at the end"""
        if self.error_set:
            with open(os.path.join(os.getcwd(), 'error'), 'w') as f:
                for pdb_id in self.error_set:
                    print(f"{self.root}/pocket_pdb/{pdb_id}.pdb", file=f)
    
    @staticmethod
    def sample_with_preserved_original(R, C, M_add, noise_std=0.05, k=5, add_noise_to_original=True):
        """
        采样新点云，保留所有原始点，并额外增加 M_add 个采样点
        适用于训练时监督原始点、利用新增点做正则化

        输入：
            R: [N, 3]              - 原始坐标
            C: [N, n_type]         - 原始类型（one-hot 或概率）
            M_add: int             - 新增采样的点数
            noise_std: float       - 加噪强度（建议小值，如 0.05 Å）
            k: int                 - KNN 平滑用的邻居数
            add_noise_to_original: bool - 是否对原始点加轻微噪声（增强鲁棒性）

        输出：
            R_out: [N + M_add, 3]
            C_out: [N + M_add, n_type]  # 所有都是概率分布
        """
        device = R.device
        N = R.size(0)

        # ======================
        # Step 1: 保留原始点（可加轻微噪声）
        # ======================
        R_orig = R
        if add_noise_to_original:
            R_orig = R_orig + torch.randn_like(R_orig) * noise_std

        C_orig = C.clone()
        # 将原始标签轻微“软化”，避免 one-hot 导致 loss 数值问题
        C_orig = C_orig + 0.01  # 引入小噪声
        C_orig = torch.nn.functional.normalize(C_orig, p=1, dim=1)  # 重新归一化为概率分布

        # ======================
        # Step 2: 采样 M_add 个新点（从原始点中采样 + 加噪）
        # ======================
        idx = torch.randint(0, N, (M_add,), device=device)
        R_new = R[idx] + torch.randn(M_add, 3, device=device) * noise_std

        # 使用 KNN 平滑获取新点的类型概率
        dist = torch.cdist(R_new.unsqueeze(0), R.unsqueeze(0))  # [1, M_add, N]
        _, indices = torch.topk(dist, k=k, dim=2, largest=False)  # 最近 k 个原始点

        C_knn = C[indices]  # [1, M_add, k, n_type]
        C_new = C_knn.mean(dim=2).squeeze(0)  # [M_add, n_type]

        # 轻微扰动 + 归一化
        C_new = C_new + torch.randn_like(C_new) * 0.02
        C_new = torch.clamp(C_new, 0.01, 0.99)
        C_new = torch.nn.functional.normalize(C_new, p=1, dim=1)

        # ======================
        # Step 3: 拼接结果
        # ======================
        R_out = torch.cat([R_orig, R_new], dim=0)  # [N + M_add, 3]
        C_out = torch.cat([C_orig, C_new], dim=0)  # [N + M_add, n_type]

        return R_out, C_out

# class MolPocketPointCloud(data.Dataset):
#     def __init__(self, root=os.getcwd() + '/data/', precompute=True, lmdb_path=os.getcwd() + '/data/lmdbcache_edmol'):
#         self.root = root
#         self.lmdb_path = lmdb_path
#         self.symbol_point_cloud = {'N': 290, 'O': 291, 'F': 292, 'C': 293}
#         self.return_all = False
        
#         os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        
#         # Load pocket files
#         pocket_files = os.listdir(root + 'pocket_pdb/')
#         error_file = os.path.join(os.getcwd(), 'error')
#         self.error_set = set()
#         if os.path.exists(error_file):
#             with open(error_file, 'r') as f:
#                 self.error_set = {line.strip().split('/')[-1][:-4] for line in f}

#         self.items = []
#         for pocket_file in pocket_files:
#             if 'pdb' in pocket_file:
#                 pdb_id = pocket_file[:-4]
#                 if pdb_id in self.error_set:
#                     continue
#                 pocket_path = os.path.join(root, 'pocket_pdb', pocket_file)
#                 sdf_path = os.path.join(root, 'mol_sdf', f'{pdb_id}_ligand.sdf')
#                 point_cloud_path = os.path.join(root, 'ed_3_5A', f'{pdb_id}.pdb')
#                 self.items.append({
#                     'pdb_id': pdb_id,
#                     'pocket_path': pocket_path,
#                     'sdf_path': sdf_path,
#                     'point_cloud_path': point_cloud_path
#                 })

#         print(f'======= Loading {len(self.items)} samples =======')

#         # Precompute data and store in LMDB
#         if precompute:
#             print("Precomputing data...")
#             self._precompute_data()

#     def _precompute_data(self):
#         """Precompute data with batch LMDB writes"""
#         num_workers = min(cpu_count()*3, 24)
#         batch_size = num_workers * 4
#         results = []
        
#         # Initialize LMDB environment
#         map_size = 5 * 1024 * 1024 * 1024  # 5 GB
#         env = lmdb.open(self.lmdb_path, map_size=map_size, readonly=False, create=True)
#         lock = threading.Lock()  # For safe LMDB writes

#         def write_batch(batch_data):
#             with lock:
#                 with env.begin(write=True) as txn:
#                     for idx, data in batch_data:
#                         if data is not None:
#                             # Serialize efficiently
#                             buffer = io.BytesIO()
#                             torch.save(data, buffer)
#                             txn.put(str(idx).encode(), buffer.getvalue())

#         with Pool(num_workers) as pool:
#             batch = []
#             for idx, result in enumerate(tqdm(
#                 pool.imap(self._preprocess_wrapper, range(len(self.items))),
#                 total=len(self.items)
#             )):
#                 if result is not None:
#                     batch.append((idx, result))
#                 if len(batch) >= batch_size or idx == len(self.items) - 1:
#                     write_batch(batch)
#                     batch = []
#                 results.append(result)

#         # Write any remaining items
#         if batch:
#             write_batch(batch)
        
#         env.close()
#         successful = sum(1 for r in results if r is not None)
#         print(f"Precomputed {successful} items (failed on {len(self.items) - successful}).")

#     def _preprocess_wrapper(self, idx):
#         """Wrapper function for multiprocessing"""
#         try:
#             return self.preprocess_data(idx)
#         except Exception as e:
#             print(f"Error precomputing index {idx}: {e}")
#             return None

#     def preprocess_data(self, idx):
#         item = self.items[idx]
#         pocket_path = item['pocket_path']
#         sdf_path = item['sdf_path']
#         point_cloud_path = item['point_cloud_path']

#         # Pocket processing
#         pocket_dict = parse_pdb_file(pocket_path)
#         ligand_dict = parse_sdf_file(sdf_path)

#         point_cloud_file = point_cloud_path
            
#         # Point cloud processing
#         ed_code, ed_pos = ed_pdb_read(point_cloud_file)
#         ed_code = [self.symbol_point_cloud.get(c, 0) for c in ed_code]

#         ed_dict = {
#             'ed_code': ed_code,
#             'ed_pos': ed_pos
#         }
#         return {
#             'pocket': pocket_dict,
#             'mol': ligand_dict,
#             'ed': ed_dict,
#             'sdf_path': item['sdf_path'],
#             'pdb_id': item['pdb_id']
#         }

#     def __getitem__(self, index):
#         index = index % len(self.items)
#         # Check LMDB cache
#         with lmdb.open(self.lmdb_path, map_size=5*1024*1024*1024, readonly=True) as env:
#             with env.begin() as txn:
#                 cached = txn.get(str(index).encode())
#                 if cached is not None:
#                     # Deserialize
#                     try:
#                         buffer = io.BytesIO(cached)
#                         data = torch.load(buffer)
#                         return data
#                     except:
#                         return self.__getitem__((index + 1) % len(self.items))
        
#         # Fallback: preprocess on-the-fly
#         try:
#             data = self.preprocess_data(index)
#             if self.return_all:
#                 data['sdf_path'] = self.items[index]['sdf_path']
#                 if data['pointcloud'][0][0].shape[0] != 499:
#                     return self.__getitem__((index + 1) % len(self.items))
#                 return data
#             else:
#                 resolution_idx = 0
#                 if data['pointcloud'][0][0].shape[0] != 499:
#                     return self.__getitem__((index + 1) % len(self.items))
#                 data['pointcloud'][0] = data['pointcloud'][0][resolution_idx][:499]
#                 data['pointcloud'][1] = data['pointcloud'][1][resolution_idx][:499]
#                 return data
#         except Exception as e:
#             print(e)
#             self.error_set.add(self.items[index]['pdb_id'])
#             return self.__getitem__((index + 1) % len(self.items))

#     def __len__(self):
#         return len(self.items)
    
#     def get_lmdb_stats(self):
#         with lmdb.open(self.lmdb_path, map_size=5*1024*1024*1024, readonly=True) as env:
#             stats = env.stat()
#             page_size = stats['psize']
#             pages_used = stats['entries']
#             total_size = page_size * pages_used
#             size_mb = total_size / (1024 * 1024)
#             size_gb = size_mb / 1024
#             return {
#                 'entries': stats['entries'],
#                 'page_size': page_size,
#                 'total_size_bytes': total_size,
#                 'total_size_mb': size_mb,
#                 'total_size_gb': size_gb
#             }
    
#     def save_errors(self):
#         """Save errors to file at the end"""
#         if self.error_set:
#             with open(os.path.join(os.getcwd(), 'error'), 'w') as f:
#                 for pdb_id in self.error_set:
#                     print(f"{self.root}/pocket_pdb/{pdb_id}.pdb", file=f)

class MolPocketPointCloud(data.Dataset):
    def __init__(self, root=os.getcwd() + '/data/', lmdb_path=os.getcwd() + '/data/rnadata', num_workers=8):
        self.root = root
        self.lmdb_path = lmdb_path
        self.num_workers = num_workers
        self.symbol_point_cloud = {'N': 290, 'O': 291, 'F': 292, 'C': 293}

        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

        # 检查是否已有缓存
        if not self._lmdb_exists():
            self._preprocess_and_cache()
        else:
            print("LMDB cache already exists. Skipping preprocessing.")

        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        
        self.length = self.txn.stat()['entries']

        print(f'======= Loaded {self.length} samples from LMDB =======')
        
    def _lmdb_exists(self):
        """检查 LMDB 是否已经存在"""
        lock_file = self.lmdb_path + '/lock.mdb'
        return os.path.exists(lock_file)

    def _preprocess_and_cache(self):
        """流式多进程预处理并写入 LMDB，避免内存爆炸"""

        folder = "molecular-share-data/others/zhouwenbiao/gen_pockets_pretrain_108w_with_genpocket/"
        chunk_files = sorted(os.listdir(folder))  # 确保顺序一致（可选）

        # 打开 LMDB 环境（预留足够空间）
        map_size = 120 * 1024 * 1024 * 1024  # 80 GB，根据实际调整
        env = lmdb.open(self.lmdb_path, map_size=map_size)

        total_written = 0

        for chunk_idx, chunkfile in enumerate(tqdm(chunk_files)):
            # 1. 读取单个 chunk
            test_pkl = os.path.join(folder, chunkfile)
            with open(test_pkl, 'rb') as f:
                chunk = pickle.load(f)

            # 2. 构造当前 chunk 的 items 列表（仅当前 chunk，非全部）
            items = []
            for data in chunk:
                items.append({
                    'pdb_id': f'pretrain_{total_written + len(items)}',  # 临时 ID，最终以写入顺序为准
                    'pocket_pdb': data['pockets'][0],
                    'mol_conf_sdf': data['mol_conf_sdf'],
                    'ed_block': data['mol_ed_pdb'][0],
                })

            # 3. 多进程处理当前 chunk
            with mp.Pool(self.num_workers) as pool:
                results = list(pool.imap(self._process_single, items))

            # 4. 写入 LMDB（仅有效结果）
            with env.begin(write=True) as txn:
                for pdb_id, processed_data in results:
                    if pdb_id is not None and processed_data is not None:
                        key = str(total_written).encode()
                        value = pickle.dumps(processed_data)
                        txn.put(key, value)
                        total_written += 1

            # 可选：显式删除大对象以释放内存
            del chunk, items, results
            gc.collect()

        env.close()
        print(f"Finished. Total {total_written} entries written to LMDB.")

    @staticmethod
    def _process_single(item):
        try:
            """单个样本的预处理逻辑"""
            pdb_id = item['pdb_id']
            pocket_block = item['pocket_pdb']
            sdf_block = item['mol_conf_sdf']
            ed_block = item['ed_block']

            # Pocket processing
            pocket_dict = parse_pdb_file(block=pocket_block)
            ligand_dict = parse_sdf_file(block=sdf_block)
            uid = str(uuid.uuid4())
            with open(f'temp_ed_{uid}.pdb','w') as f:
                f.write(ed_block)
            ed_dicts = ed_pdb_read(f'temp_ed_{uid}.pdb')
            os.remove(f'temp_ed_{uid}.pdb')
            processed_data = {
                'pdb_id': pdb_id,
                'pocket': pocket_dict,
                'mol': ligand_dict,
                'ed': ed_dicts,
            }
        except Exception as e:
            print(f"Error processing PDB ID {pdb_id}: {e}")
            pdb_id = None
            processed_data = None
        return pdb_id, processed_data

    def __getitem__(self, index):
        """从 LMDB 中读取数据"""
        return pickle.loads(self.txn.get(str(index).encode()))

    def __len__(self):
        return self.length

    def get_lmdb_stats(self):
        with lmdb.open(self.lmdb_path, map_size=120*1024*1024*1024, readonly=True) as env:
            stats = env.stat()
            page_size = stats['psize']
            pages_used = stats['entries']
            total_size = page_size * pages_used
            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024
            return {
                'entries': stats['entries'],
                'page_size': page_size,
                'total_size_bytes': total_size,
                'total_size_mb': size_mb,
                'total_size_gb': size_gb
            }


if __name__ == '__main__':
    from tqdm import tqdm
    import os
    from torch.utils.data import DataLoader

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    m = MolPocketPointCloud(lmdb_path=os.getcwd() + '/data/pretrain_pocket', num_workers=32)
    print(m[0])
    print(m.get_lmdb_stats())
