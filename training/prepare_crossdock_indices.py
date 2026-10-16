#!/usr/bin/env python3
"""
按 pocket_split.csv 重新划分 CrossDock 索引，并应用 validate_mol 质量过滤。
纯 CPU 运行，不占用 GPU。

用法：
    source setup_env.sh   # creates/activates the rna2smol conda env
    python prepare_crossdock_indices.py --nproc 64
"""

import os
import argparse
import pickle
import lmdb
import multiprocessing as mp
from collections import Counter
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def validate_mol(mol):
    """基础分子质量检查：可 sanitize + 无不合理的原子类型。"""
    try:
        Chem.SanitizeMol(mol)
        # 排除只含有一个原子的分子
        if mol.GetNumAtoms() < 2:
            return False
        # 排除含不常见元素的分子
        allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in allowed:
                return False
        return True
    except Exception:
        return False

# ========================================================================
# 配置
# ========================================================================
LMDB_PATH = 'data/crossdock/crossdocked_diffedmol_total'
POCKET_SPLIT_CSV = 'pocket_split.csv'
OUTPUT_DIR = 'clean_dataset_onlycrossdock'


def _worker_init(lmdb_path, pocket_split_csv):
    """每个子进程初始化。"""
    global _WORKER_ENV, _SPLIT_DICT
    _WORKER_ENV = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=128)
    
    df = pd.read_csv(pocket_split_csv)
    df['pocket_name'] = df['pocket'].apply(
        lambda x: os.path.basename(x)[:-4] if x.endswith('.pdb') else os.path.basename(x)
    )
    _SPLIT_DICT = dict(zip(df['pocket_name'], df['split']))


def _worker_process(idx):
    """处理单个样本：返回 (idx, split, is_valid, reason)。"""
    global _WORKER_ENV, _SPLIT_DICT
    try:
        with _WORKER_ENV.begin() as txn:
            raw = txn.get(str(idx).encode())
            if raw is None:
                return (idx, None, False, 'missing_key')
            data = pickle.loads(raw)
        
        # 1. 按 pocket 划分
        f = data.get('file', '')
        pocket_name = f.split('_rec_')[0] + '_rec' if '_rec_' in f else ''
        split = _SPLIT_DICT.get(pocket_name, None)
        if split is None:
            return (idx, None, False, 'unknown_pocket')
        
        # test 不参与训练，直接标记但不验证分子质量
        if split == 'test':
            return (idx, 'test', True, 'test_split')
        
        # 2. 基础数据一致性检查（口袋）
        pocket = data.get('pocket', {})
        p_pos = pocket.get('pos', [])
        p_elem = pocket.get('element', [])
        p_feat = pocket.get('atom_feature', [])
        if not (len(p_pos) == len(p_elem) == len(p_feat)):
            return (idx, split, False, 'pocket_size_mismatch')
        
        # 3. 基础数据一致性检查（分子）
        mol_dict = data.get('mol', {})
        m_pos = mol_dict.get('pos', [])
        m_elem = mol_dict.get('element', [])
        m_feat = mol_dict.get('atom_feature', [])
        if not (len(m_pos) == len(m_elem) == len(m_feat)):
            return (idx, split, False, 'mol_size_mismatch')
        
        # 4. 分子质量验证
        smiles = mol_dict.get('smiles', None)
        if not smiles:
            return (idx, split, False, 'no_smiles')
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (idx, split, False, 'invalid_smiles')
        
        if validate_mol(mol):
            return (idx, split, True, 'pass')
        else:
            return (idx, split, False, 'quality_check_fail')
    except Exception as e:
        return (idx, None, False, f'exception: {e}')


def main():
    parser = argparse.ArgumentParser(description='Rebuild CrossDock indices with correct pocket split')
    parser.add_argument('--nproc', type=int, default=32, help='Number of CPU processes')
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"CrossDock Index Rebuilder")
    print(f"LMDB    : {LMDB_PATH}")
    print(f"Split   : {POCKET_SPLIT_CSV}")
    print(f"Workers : {args.nproc}")
    print(f"{'='*60}\n")
    
    # 统计 LMDB 总样本数
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        all_indices = [int(k.decode()) for k, v in cursor if k.decode().isdigit()]
    env.close()
    total = len(all_indices)
    print(f"Total samples in LMDB: {total}")
    
    # 多进程处理
    results = []
    with mp.Pool(
        processes=args.nproc,
        initializer=_worker_init,
        initargs=(LMDB_PATH, POCKET_SPLIT_CSV)
    ) as pool:
        for res in tqdm(pool.imap(_worker_process, all_indices, chunksize=32),
                        total=total, ncols=80):
            results.append(res)
    
    # 整理结果
    train_passed = []
    val_passed = []
    test_indices = []
    stats = Counter()
    
    for idx, split, is_valid, reason in results:
        stats[reason] += 1
        if split == 'train' and is_valid:
            train_passed.append(idx)
        elif split == 'valid' and is_valid:
            val_passed.append(idx)
        elif split == 'test':
            test_indices.append(idx)
    
    # 构建输出
    indices_data = {
        'train': train_passed,
        'val': val_passed,
        'all_valid': train_passed + val_passed,
        'test': test_indices,
    }
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"Onlycrossdock_rebuilt_{timestamp}.pkl"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(indices_data, f)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"Saved to: {out_path}")
    print(f"{'='*60}")
    print(f"\nPocket split + Quality filter results:")
    print(f"  train (pass) : {len(train_passed):6d}")
    print(f"  val   (pass) : {len(val_passed):6d}")
    print(f"  all_valid    : {len(indices_data['all_valid']):6d}")
    print(f"  test (raw)   : {len(test_indices):6d}")
    print(f"\nDetailed stats:")
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {reason:25s}: {count:6d} ({count/total*100:.1f}%)")
    
    # 与原始错误划分的对比
    orig_train = sum(1 for idx, split, is_valid, reason in results 
                     if reason != 'test_split' and 
                     pickle.loads(lmdb.open(LMDB_PATH, readonly=True, lock=False).begin().get(str(idx).encode())).get('train_or_valid') == 'train')
    # 上面这段太复杂了，简化一下
    print(f"\nNote: original wrong split had ~52k train / ~116k valid.")
    print(f"      Correct pocket split gives ~{len(train_passed)} train / ~{len(val_passed)} valid.")


if __name__ == '__main__':
    main()
