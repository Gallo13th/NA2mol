#!/usr/bin/env python3
"""诊断 DataLoader batch 中 tensor shape 不一致的问题（模拟训练配置）。"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from dataloader.dataloader_pointcloud import MolPocketPointCloud
from dataloader.featurizer import GeomMolOnlyPocket, ProteinLigandDataLoader

# 加载三个数据集
print("Loading datasets...")

# CrossDock
raw_cd = MolPocketPointCloud(lmdb_path="./data/crossdocked_diffedmol_total")
cd_dataset = GeomMolOnlyPocket(raw_cd, mode='general', pocket_radius=10, is_train=True, is_rna=False)
with open("./clean_dataset_onlycrossdock/Onlycrossdock_rebuilt_20260503_181740.pkl", 'rb') as f:
    cd_indices = pickle.load(f)
cd_train = Subset(cd_dataset, list(cd_indices['train']))
cd_val = Subset(cd_dataset, cd_indices['val'])
print(f"CrossDock train={len(cd_train)}, val={len(cd_val)}")

# ExpDB
raw_exp = MolPocketPointCloud(lmdb_path="./data/lmdbcache_expdb")
exp_dataset = GeomMolOnlyPocket(raw_exp, mode='general', pocket_radius=8, is_train=True, is_rna=False)
with open("./clean_exped_dataset/expdb_indices_filtered_20260503_180911.pkl", 'rb') as f:
    exp_indices = pickle.load(f)
exp_train = Subset(exp_dataset, list(exp_indices['train']))
exp_val = Subset(exp_dataset, exp_indices['val'])
print(f"ExpDB train={len(exp_train)}, val={len(exp_val)}")

# RNA
raw_rna = MolPocketPointCloud(lmdb_path="./rnaligandwithedid")
rna_dataset = GeomMolOnlyPocket(raw_rna, mode='general', pocket_radius=8, is_train=True, is_rna=True)
with open("./clean_dataset_onlyrna/OnlyRNA_filtered_20260503_180819.pkl", 'rb') as f:
    rna_indices = pickle.load(f)
rna_train = Subset(rna_dataset, list(rna_indices['train']))
rna_val = Subset(rna_dataset, rna_indices['val'])
print(f"RNA train={len(rna_train)}, val={len(rna_val)}")

# 模拟训练时的混合配置
protein_set = ConcatDataset([exp_train, cd_train])
repeated_rna = ConcatDataset([rna_train] * 5)
combined = ConcatDataset([repeated_rna, protein_set])
print(f"Protein total={len(protein_set)}, Repeated RNA={len(repeated_rna)}, Combined={len(combined)}")

class BalancedMixedSampler(torch.utils.data.Sampler):
    def __init__(self, len_a, len_b, samples_per_epoch, ratio_a=0.3, seed=42):
        self.len_a = len_a
        self.len_b = len_b
        self.samples_per_epoch = samples_per_epoch
        self.ratio_a = ratio_a
        self.seed = seed
    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        n_a = int(self.samples_per_epoch * self.ratio_a)
        n_b = self.samples_per_epoch - n_a
        idx_a = rng.choice(self.len_a, size=n_a, replace=True)
        idx_b = rng.choice(self.len_b, size=n_b, replace=True) + self.len_a
        all_idx = np.concatenate([idx_a, idx_b])
        rng.shuffle(all_idx)
        yield from all_idx.tolist()
    def __len__(self):
        return self.samples_per_epoch

sampler = BalancedMixedSampler(
    len_a=len(repeated_rna), len_b=len(protein_set),
    samples_per_epoch=500, ratio_a=0.3, seed=42
)

loader = ProteinLigandDataLoader(
    combined, batch_size=8, sampler=sampler, num_workers=0, pin_memory=True
)

print("\nIterating mixed batches...")
for i, batch in enumerate(loader):
    mol_pos_size = batch.mol_pos.size(0)
    mol_element_size = batch.mol_element.size(0)
    mol_atom_feature_size = batch.mol_atom_feature.size(0)
    mol_element_batch_size = batch.mol_element_batch.size(0)
    
    sizes = {
        'mol_pos': mol_pos_size,
        'mol_element': mol_element_size,
        'mol_atom_feature': mol_atom_feature_size,
        'mol_element_batch': mol_element_batch_size,
    }
    
    consistent = len(set(sizes.values())) == 1
    status = "OK" if consistent else "MISMATCH!"
    print(f"Batch {i}: {status} | {sizes}")
    
    if not consistent:
        print(f"  mol_element_batch max: {batch.mol_element_batch.max().item()}")
        # 找出哪个样本有问题
        unique_batches = batch.mol_element_batch.unique().tolist()
        for bidx in unique_batches:
            mask = batch.mol_element_batch == bidx
            n_pos = batch.mol_pos[mask].size(0)
            n_elem = batch.mol_element[mask].size(0)
            n_feat = batch.mol_atom_feature[mask].size(0)
            if n_pos != n_elem or n_elem != n_feat:
                print(f"  Sample batch_idx={bidx}: pos={n_pos}, element={n_elem}, feat={n_feat}")
        break
else:
    print("All 10 batches OK")

print("\nDone.")
