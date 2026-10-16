#!/usr/bin/env python3
"""
过滤 ExpDB 中与 DUDE 测试集口袋过于相似的结构，避免信息泄露。

策略：
1. USRC (Ultrafast Shape Recognition) 预过滤 —— SE3 不变、O(N) 提取、O(1) 比较，
   540 万对在几秒内完成粗筛。
2. ICP-style 精筛 —— 对 USRC 候选对做迭代最近邻 + Kabsch 对齐，计算 RMSD。
3. 卡阈值：RMSD < threshold（默认 2.0 Å）认为结构过于相似，丢弃。

输出：expdb_dude_filtered_indices.pkl，供训练脚本使用。
"""

import os
from pathlib import Path
import sys
import pickle
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import lmdb

# DUDE 解析
from rdkit import Chem
from Bio.PDB import PDBParser
ROOT = Path(__file__).resolve().parents[1]  # repository root


def parse_dude_ligand_center(sdf_path):
    """读取 DUDE 配体 SDF，返回几何中心 [3]。"""
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
    mol = next(iter(suppl))
    if mol is None:
        return None
    conf = mol.GetConformer()
    coords = np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])
    return coords.mean(axis=0)


def extract_dude_pocket(protein_pdb, ligand_center, cutoff=10.0):
    """
    从 DUDE 蛋白 PDB 中提取配体中心 cutoff Å 内的口袋原子坐标。
    只保留重原子（C, N, O, S, P）以减少噪音。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('p', protein_pdb)
    heavy_elements = {'C', 'N', 'O', 'S', 'P'}
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element in heavy_elements:
                        coord = np.array(atom.get_coord())
                        if np.linalg.norm(coord - ligand_center) <= cutoff:
                            atoms.append(coord)
    return np.array(atoms) if atoms else np.empty((0, 3))


def load_all_dude_pockets(dude_dir, cutoff=10.0):
    """
    加载所有 DUDE 口袋。
    返回: list of dict [{'name': str, 'pos': ndarray(N,3)}, ...]
    """
    protein_dir = os.path.join(dude_dir, 'DUDE_proteins')
    ligand_dir = os.path.join(dude_dir, 'cocrystal_ligands')

    pockets = []
    ligand_files = sorted([
        f for f in os.listdir(ligand_dir)
        if f.startswith('dude_nh_') and f.endswith('.sdf')
    ])

    for ligand_file in tqdm(ligand_files, desc='Extracting DUDE pockets'):
        # dude_nh_1b9v.sdf -> 1b9v
        pdb_id = ligand_file.replace('dude_nh_', '').replace('.sdf', '')
        protein_file = f"{pdb_id}_protein_wo_ligand.pdb"
        protein_path = os.path.join(protein_dir, protein_file)
        ligand_path = os.path.join(ligand_dir, ligand_file)

        if not os.path.exists(protein_path):
            continue

        center = parse_dude_ligand_center(ligand_path)
        if center is None:
            continue

        pocket_pos = extract_dude_pocket(protein_path, center, cutoff=cutoff)
        if len(pocket_pos) == 0:
            continue

        pockets.append({
            'name': pdb_id,
            'pos': pocket_pos,
            'n_atoms': len(pocket_pos),
        })

    print(f"Loaded {len(pockets)} DUDE pockets")
    return pockets


def load_all_expdb_pockets(expdb_lmdb_path):
    """
    加载所有 ExpDB 口袋坐标。
    返回: list of dict [{'idx': int, 'pos': ndarray(N,3)}, ...]
    """
    env = lmdb.open(expdb_lmdb_path, readonly=True, max_dbs=0)
    pockets = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc='Loading ExpDB pockets', total=53698):
            data = pickle.loads(value)
            pocket = data['pocket']
            pos = np.array(pocket['pos'], dtype=np.float32)
            # 同样只保留重原子（通过元素号过滤）
            elements = np.array(pocket['element'], dtype=np.int32)
            # 原子序数 > 1 排除 H
            heavy_mask = elements > 1
            pos = pos[heavy_mask]
            pockets.append({
                'idx': int(key.decode()),
                'pos': pos,
                'n_atoms': len(pos),
            })
    print(f"Loaded {len(pockets)} ExpDB pockets")
    return pockets


def usr_features(pos):
    """
    计算 USRC (Ultrafast Shape Recognition) 特征 —— 向量化加速版。
    12 维向量，SE3 不变：
    - mu, sd, skew, kurt of distances-to-centroid  (4)
    - mu, sd, skew, kurt of pairwise distances     (4)
    - mu, sd, skew, kurt of triangle areas         (4)
    """
    if len(pos) < 3:
        return np.full(12, np.nan)

    rng = np.random.default_rng(42)
    n = len(pos)

    # 1. distances to centroid
    centroid = pos.mean(axis=0)
    d1 = np.linalg.norm(pos - centroid, axis=1)
    f1 = [d1.mean(), d1.std(ddof=1), skew(d1), kurtosis(d1)]

    # 2. pairwise distances (采样加速)
    if n > 500:
        sampled = pos[rng.choice(n, size=500, replace=False)]
    else:
        sampled = pos
    diff = sampled[:, None, :] - sampled[None, :, :]
    d2 = np.linalg.norm(diff, axis=2)
    triu_idx = np.triu_indices_from(d2, k=1)
    d2 = d2[triu_idx]
    f2 = [d2.mean(), d2.std(ddof=1), skew(d2), kurtosis(d2)]

    # 3. triangle areas —— 向量化采样
    if n > 300:
        sampled = pos[rng.choice(n, size=300, replace=False)]
    else:
        sampled = pos
    n_s = len(sampled)
    n_samples = min(2000, n_s * (n_s - 1) * (n_s - 2) // 6)

    # 向量化采样三元组 [n_samples, 3]
    triples = np.stack([rng.choice(n_s, size=3, replace=False) for _ in range(n_samples)])
    tri = sampled[triples]  # [n_samples, 3, 3]

    # 向量化计算边长
    a = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    b = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    c = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)

    # Heron's formula (向量化)
    s = (a + b + c) / 2.0
    areas = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    f3 = [areas.mean(), areas.std(ddof=1), skew(areas), kurtosis(areas)]

    feats = np.array(f1 + f2 + f3, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def kabsch_align(P, Q):
    """
    Kabsch 算法：将 P 对齐到 Q。
    P, Q: [N, 3] 或 [M, 3]（若大小不同，取较小者）
    返回: aligned_P, RMSD
    """
    # 若大小不同，随机采样到相同大小
    if len(P) != len(Q):
        min_len = min(len(P), len(Q))
        rng = np.random.default_rng(42)
        if len(P) > min_len:
            P = P[rng.choice(len(P), size=min_len, replace=False)]
        if len(Q) > min_len:
            Q = Q[rng.choice(len(Q), size=min_len, replace=False)]

    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (R @ P_c.T).T + Q.mean(axis=0)
    rmsd = np.sqrt(np.mean(np.sum((aligned - Q) ** 2, axis=1)))
    return aligned, rmsd


def icp_rmsd(P, Q, max_iter=10, tol=1e-4):
    """
    ICP (Iterative Closest Point) + Kabsch。
    先 Kabsch 全局对齐，再迭代最近邻 refine。
    返回最终 RMSD。
    """
    # 初始全局对齐
    aligned, rmsd = kabsch_align(P, Q)
    Q_centered = Q - Q.mean(axis=0)

    prev_rmsd = rmsd + 1.0
    for _ in range(max_iter):
        tree = cKDTree(Q_centered)
        # aligned 也需要中心化
        aligned_c = aligned - aligned.mean(axis=0)
        distances, indices = tree.query(aligned_c, k=1)
        Q_matched = Q_centered[indices]

        # Kabsch 对齐
        H = aligned_c.T @ Q_matched
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        aligned_c = (R @ aligned_c.T).T
        aligned = aligned_c + Q.mean(axis=0)

        rmsd = np.sqrt(np.mean(distances ** 2))
        if abs(prev_rmsd - rmsd) < tol:
            break
        prev_rmsd = rmsd

    return rmsd


def usr_distance(f1, f2):
    """USRC 特征欧氏距离。"""
    return np.linalg.norm(f1 - f2)


# 全局变量，用于多进程 ICP worker
_DUDE_POCKETS = None
_EXPDB_POCKETS = None


def _usr_features_worker(pocket_dict):
    return usr_features(pocket_dict['pos'])


def find_similar_pairs(dude_pockets, expdb_pockets,
                       usr_threshold=3.0, rmsd_threshold=2.0,
                       n_workers=None):
    """
    主过滤逻辑：USRC 预过滤 + 口袋大小过滤 + ICP 精筛。
    返回需要丢弃的 ExpDB 索引列表。
    """
    n_workers = n_workers or max(1, cpu_count() - 2)
    print(f"\n=== Step 1: Computing USRC features (using {n_workers} workers) ===")

    with Pool(n_workers) as pool:
        dude_feats = list(tqdm(
            pool.imap(_usr_features_worker, dude_pockets, chunksize=1),
            total=len(dude_pockets), desc='DUDE USRC'
        ))
        expdb_feats = list(tqdm(
            pool.imap(_usr_features_worker, expdb_pockets, chunksize=max(1, len(expdb_pockets)//n_workers//4)),
            total=len(expdb_pockets), desc='ExpDB USRC'
        ))

    dude_feats = np.stack(dude_feats)
    expdb_feats = np.stack(expdb_feats)

    print("\n=== Step 2: Pre-filtering (size + USRC) ===")
    # 口袋大小预过滤：原子数差异超过 50% 直接跳过
    dude_sizes = np.array([p['n_atoms'] for p in dude_pockets], dtype=np.float32)
    expdb_sizes = np.array([p['n_atoms'] for p in expdb_pockets], dtype=np.float32)
    size_mat = np.minimum(dude_sizes[:, None], expdb_sizes[None, :]) / np.maximum(dude_sizes[:, None], expdb_sizes[None, :])
    size_mask = size_mat >= 0.5

    # USRC 距离矩阵 [n_dude, n_expdb]
    dist_mat = np.linalg.norm(dude_feats[:, None, :] - expdb_feats[None, :, :], axis=2)
    usr_mask = dist_mat < usr_threshold

    candidate_mask = size_mask & usr_mask
    candidate_pairs = np.argwhere(candidate_mask)
    candidate_pairs = [(int(i), int(j)) for i, j in candidate_pairs]

    total_pairs = len(dude_pockets) * len(expdb_pockets)
    print(f"Pre-filter: {len(candidate_pairs)} / {total_pairs} pairs retained (size_ratio>=0.5 & USRC<{usr_threshold})")

    if len(candidate_pairs) == 0:
        return []

    print("\n=== Step 3: ICP refinement ===")
    similar_expdb_idx = set()
    chunk = max(1, len(candidate_pairs) // n_workers // 10)
    with Pool(n_workers, initializer=_init_icp_worker, initargs=(dude_pockets, expdb_pockets)) as pool:
        for j, rmsd in tqdm(
            pool.imap_unordered(_icp_worker_init, candidate_pairs, chunksize=chunk),
            total=len(candidate_pairs),
            desc='ICP refinement'
        ):
            if rmsd < rmsd_threshold:
                similar_expdb_idx.add(expdb_pockets[j]['idx'])

    print(f"ICP filter: {len(similar_expdb_idx)} ExpDB structures will be dropped (RMSD < {rmsd_threshold} Å)")
    return sorted(list(similar_expdb_idx))


def _init_icp_worker(dude_pockets, expdb_pockets):
    """多进程 initializer：在每个子进程中设置全局 pockets。"""
    global _DUDE_POCKETS, _EXPDB_POCKETS
    _DUDE_POCKETS = dude_pockets
    _EXPDB_POCKETS = expdb_pockets


def _icp_worker_init(pair):
    """模块级 worker，供多进程调用（依赖 initializer 设置的全局变量）。"""
    i, j = pair
    rmsd = icp_rmsd(_DUDE_POCKETS[i]['pos'], _EXPDB_POCKETS[j]['pos'])
    return j, rmsd


def main():
    parser = argparse.ArgumentParser(description='Filter ExpDB by DUDE similarity')
    parser.add_argument('--dude_dir', type=str,
                        default=str(ROOT / 'data/DUDE_complex'),
                        help='DUDE complex directory')
    parser.add_argument('--expdb_lmdb', type=str,
                        default=str(ROOT / 'data/expdb'),
                        help='ExpDB LMDB path')
    parser.add_argument('--output', type=str,
                        default='./clean_exped_dataset/expdb_dude_filtered_indices.pkl',
                        help='Output pickle with dropped indices')
    parser.add_argument('--usr_threshold', type=float, default=3.0,
                        help='USRC distance threshold for pre-filtering')
    parser.add_argument('--rmsd_threshold', type=float, default=2.0,
                        help='ICP RMSD threshold (Å)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Load pockets
    dude_pockets = load_all_dude_pockets(args.dude_dir)
    expdb_pockets = load_all_expdb_pockets(args.expdb_lmdb)

    # 2. Find similar pairs
    dropped_indices = find_similar_pairs(
        dude_pockets, expdb_pockets,
        usr_threshold=args.usr_threshold,
        rmsd_threshold=args.rmsd_threshold,
        n_workers=args.n_workers
    )

    # 3. Save
    with open(args.output, 'wb') as f:
        pickle.dump(dropped_indices, f)

    print(f"\nSaved dropped indices ({len(dropped_indices)} items) to {args.output}")


if __name__ == '__main__':
    main()
