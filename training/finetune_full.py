import os
import sys
import logging
import argparse
import random
import pickle
import glob
import math
from datetime import datetime
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.distributed import all_gather_object
from torch.utils.data import ConcatDataset, Subset, DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Lipinski, rdMolDescriptors

# 自定义模块导入
from model.bfn4edmol import Config, BFN4SBDDScoreModel
from dataloader.dataloader_pointcloud import MolPocketPointCloud
from dataloader.featurizer import GeomMolOnlyPocket, ProteinLigandDataLoader
from util.optims import SingleDeviceMuonWithAuxAdam, Muon
ROOT = Path(__file__).resolve().parents[1]  # repository root

# 抑制 RDKit 日志
RDLogger.DisableLog('rdApp.*')

# ============================================================================
# 辅助函数与配置常量
# ============================================================================

def setup_logger(name, log_file, level=logging.INFO, rank=0):
    """配置日志记录器，仅 rank 0 输出到控制台，所有 rank 都写文件。"""
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 文件处理器（所有 rank）
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        if rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def remove_compiled_prefix(state_dict):
    """移除 state_dict 中可能由 torch.compile 添加的 '_orig_mod.' 前缀。"""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_sd[k[len('_orig_mod.'):]] = v
        else:
            new_sd[k] = v
    return new_sd


def calculate_properties(mol):
    """计算分子属性，用于过滤。"""
    if mol is None:
        return None
    properties = {}
    try:
        # properties['QED'] = QED.qed(mol)
        # properties['LogP'] = Crippen.MolLogP(mol)
        properties['rotB'] = Lipinski.NumRotatableBonds(mol)
        # properties['MolWt'] = rdMolDescriptors.CalcExactMolWt(mol)
    except Exception:
        return None
    return properties


def is_sample_valid(item):
    """检查一个样本是否有效（用于过滤）。"""
    try:
        pdb_id = item.pocket_pdb_id
        if pdb_id.lower() in TEST_PDBIDS:          # TEST_PDBIDS 在下方定义
            return False

        # 检查各特征长度一致
        sizes = {
            'pocket_pos': item.pocket_pos.size(0),
            'pocket_element': item.pocket_element.size(0),
            'pocket_atom_feature': item.pocket_atom_feature.size(0),
        }
        all_sizes = list(sizes.values())
        if not all(size == all_sizes[0] for size in all_sizes):
            return False

        if torch.any(item.mol_atom_feature == 0):
            return False

        smiles = item.mol_smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        prop = calculate_properties(mol)
        if prop and prop['rotB'] > 10:
            return False

        return True
    except Exception:
        return False


# 数据集相关常量
DUDE_PDBIDS = [os.path.basename(c)[-8:-4].lower()
               for c in glob.glob('DUDE_complex/cocrystal_ligands/dude_nh_*.sdf')]
ROBIN_PDBIDS = ['3fu2','8g4w','8g7e','8g8z','2qwy','9c9k','5btp','6od9','6wzs']

SMRTBENCH_DF = pd.read_csv('pocket_grouping_strict_no_smrt.csv', index_col=False)
CROSSDOCK_DF = pd.read_csv('pocket_split.csv', index_col=False)
CROSSDOCK_DF['pocket_name'] = CROSSDOCK_DF['pocket'].apply(lambda x: os.path.basename(x[:-4]))
print(CROSSDOCK_DF)
CROSSDOCK_SPLIT_DICT = {x['pocket_name']: x['split'] for _, x in CROSSDOCK_DF.iterrows()}

SMRTBENCH_PDBIDS = SMRTBENCH_DF['pdb_id'][:489].tolist()
SMRTBENCH_PDBIDS = [pdbid.lower() for pdbid in SMRTBENCH_PDBIDS]

TEST_PDBIDS = DUDE_PDBIDS + ROBIN_PDBIDS + SMRTBENCH_PDBIDS


def load_clean_dataset(lmdb_path, indices_path, pocket_radius=10, is_rna=False):
    """从预先生成的索引文件加载清洗后的数据集。"""
    print(f"Loading dataset from indices file: {indices_path}")
    with open(indices_path, 'rb') as f:
        indices_data = pickle.load(f)

    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    dataset = GeomMolOnlyPocket(raw_dataset, mode='general',
                                pocket_radius=pocket_radius, is_train=True,
                                is_rna=is_rna)

    train_set = Subset(dataset, list(indices_data['train']))
    val_set = Subset(dataset, indices_data['val'])

    print(f"✅ Clean dataset loaded. Train: {len(train_set)}, Val: {len(val_set)}, is_rna={is_rna}")
    return train_set, val_set


def load_or_create_expdb_dataset(lmdb_path, output_dir, val_size=5000, seed=42):
    """
    加载 expdb 数据集。若索引文件不存在，则动态划分 train/val 并保存。
    expdb 是真实晶体结构，质量高于 CrossDock 对接数据。
    """
    os.makedirs(output_dir, exist_ok=True)
    indices_path = os.path.join(output_dir, 'expdb_indices.pkl')

    if os.path.exists(indices_path):
        print(f"Loading expdb indices from {indices_path}")
        with open(indices_path, 'rb') as f:
            indices_data = pickle.load(f)
    else:
        print(f"Creating expdb indices: val_size={val_size}")
        raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
        total = len(raw_dataset)
        indices = list(range(total))
        random.seed(seed)
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        indices_data = {'train': train_indices, 'val': val_indices}
        with open(indices_path, 'wb') as f:
            pickle.dump(indices_data, f)
        print(f"Saved expdb indices to {indices_path}")

    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    dataset = GeomMolOnlyPocket(raw_dataset, mode='general',
                                pocket_radius=8, is_train=True, is_rna=False)

    train_set = Subset(dataset, list(indices_data['train']))
    val_set = Subset(dataset, indices_data['val'])
    print(f"✅ ExpDB loaded. Train: {len(train_set)}, Val: {len(val_set)}")
    return train_set, val_set


def create_clean_crossdock_dataset(lmdb_path, output_dir, name, train_size=40000, seed=42):
    """（已弃用，但保留）创建 CrossDock 数据集并保存索引。"""
    # 原函数保留，逻辑未变
    # ... 完整实现（此处省略，与原始一致）
    pass


def create_clean_rna_dataset(lmdb_path, output_dir, name, seed=42):
    """（已弃用，但保留）创建 RNA 数据集并保存索引。"""
    # 原函数保留，逻辑未变
    # ... 完整实现（此处省略，与原始一致）
    pass


class BalancedMixedSampler(torch.utils.data.Sampler):
    """
    类别平衡采样器：从两个数据集中按指定比例采样。
    假设 dataset = ConcatDataset([dataset_a, dataset_b])，
    dataset_a 的索引为 0 ~ len_a-1，dataset_b 的索引为 len_a ~ len_a+len_b-1。
    """
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


def create_balanced_dataloader(rna_set, protein_set,
                               rna_repeat=5, samples_per_epoch=100000,
                               rna_ratio=0.3, batch_size=20,
                               num_workers=16, seed=42):
    """
    构建类别平衡的混合训练数据加载器。
    RNA 数据集重复 rna_repeat 次，与蛋白数据集（ExpDB + CrossDock）拼接后使用 BalancedMixedSampler。
    """
    repeated_rna = ConcatDataset([rna_set] * rna_repeat)
    combined_dataset = ConcatDataset([repeated_rna, protein_set])

    sampler = BalancedMixedSampler(
        len_a=len(repeated_rna),
        len_b=len(protein_set),
        samples_per_epoch=samples_per_epoch,
        ratio_a=rna_ratio,
        seed=seed
    )

    loader = ProteinLigandDataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    # 1. 初始化 Accelerator（必须在其他 torch 相关操作前）
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    device = accelerator.device
    if not hasattr(accelerator, 'gather_object'):
        def gather_object(obj):
            if accelerator.num_processes == 1:
                return [obj]
            output = [None for _ in range(accelerator.num_processes)]
            all_gather_object(output, obj)
            return output
        accelerator.gather_object = gather_object
    # 2. 解析命令行参数
    parser = argparse.ArgumentParser()
    # Meta
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    # Global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="info")
    # Data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)
    parser.add_argument("--pos_normalizer", type=float, default=2.0)
    # Training params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')
    # BFN params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=1.5)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf',
                        choices=['vanilla', 'end_back_pmf'])
    parser.add_argument("--time_emb_mode", type=str, default="simple",
                        choices=["simple", "sin", 'rbf', 'rbfnn'])
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero',
                        choices=['zero', 'randn'])
    # Eval params
    parser.add_argument('--ckpt_path', type=str, default='best')
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref',
                        choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score",
                        choices=['vina_score', 'vina_dock'])

    args = parser.parse_args()
    cfg = Config(**args.__dict__)

    # 3. 设置日志（每个进程独立文件，仅 rank0 输出到终端）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"./logs/20260421", exist_ok=True)
    log_file = f"./logs/20260421/ft_rna_constraint_{timestamp}_rank{accelerator.process_index}.log"
    logger = setup_logger(__name__, log_file, level=logging.INFO, rank=accelerator.process_index)
    logger.info(f"Starting experiment on rank {accelerator.process_index}, device: {device}")

    # 4. 数据路径配置
    LMDB_PATH_CROSSDOCKED = str(ROOT / "data/crossdock/crossdocked_diffedmol_total")
    INDICES_PATH_CROSSDOCKED = "./clean_dataset_onlycrossdock/Onlycrossdock.pkl"
    LMDB_PATH_RNA = str(ROOT / "data/rnaligand")
    INDICES_PATH_RNA = "./clean_dataset_onlyrna/OnlyRNA.pkl"
    LMDB_PATH_EXPDB = str(ROOT / "data/expdb")
    INDICES_PATH_EXPDB = "./clean_exped_dataset"

    # 5. 加载数据集
    # CrossDock（对接数据，作为蛋白多样性补充）
    train_crossdock, _ = load_clean_dataset(
        LMDB_PATH_CROSSDOCKED, INDICES_PATH_CROSSDOCKED, pocket_radius=8, is_rna=False
    )
    # ExpDB（真实晶体结构蛋白数据，质量更高）
    train_expdb, val_expdb = load_or_create_expdb_dataset(
        LMDB_PATH_EXPDB, INDICES_PATH_EXPDB, val_size=5000, seed=42
    )
    # 排除与 DUDE 测试集过于相似的 ExpDB 结构（ICP 过滤）
    filter_path = './clean_exped_dataset/expdb_dude_filtered_indices.pkl'
    if os.path.exists(filter_path):
        with open(filter_path, 'rb') as f:
            dropped_indices = set(pickle.load(f))
        # 获取原始训练索引并过滤
        raw_dataset = train_expdb.dataset
        original_indices = [train_expdb.indices[i] for i in range(len(train_expdb))]
        filtered_indices = [idx for idx in original_indices if idx not in dropped_indices]
        train_expdb = Subset(raw_dataset, filtered_indices)
        logger.info(f"Applied DUDE filter: dropped {len(original_indices) - len(filtered_indices)} ExpDB structures")
    # 合并蛋白训练集：ExpDB(真实结构) + CrossDock(对接pose)
    train_protein = ConcatDataset([train_expdb, train_crossdock])
    logger.info(f"Combined protein train set: {len(train_protein)} (ExpDB {len(train_expdb)} + CrossDock {len(train_crossdock)})")

    # RNA 数据集
    train_set, val_set = load_clean_dataset(
        LMDB_PATH_RNA, INDICES_PATH_RNA, pocket_radius=8, is_rna=True
    )

    # 6. 构建模型
    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11,
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    teacher_model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11,
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )

    # 冻结 v_inference 参数（仅学生模型）
    for name, p in model.named_parameters():
        if 'v_inference' in name:
            p.requires_grad = False

    # 7. 设置优化器（Muon 混合优化）
    muon_params = [
        p for name, p in model.named_parameters()
        if p.ndim >= 2 and "ed_emb" not in name and "emb" in name and 'v_inference' not in name
    ]
    adamw_params = [
        p for name, p in model.named_parameters()
        if not (p.ndim >= 2 and "emb" in name and 'v_inference' in name)
    ]
    optimizer = Muon(
        lr=1e-5,
        wd=0.01,
        muon_params=muon_params,
        adamw_params=adamw_params,
    )

    # 8. 学习率调度器（warmup + plateau）
    steps_per_epoch = 4900   # 原脚本中定义的步数，用于 warmup
    warmup_steps = steps_per_epoch * 1
    from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=warmup_steps)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8,
                                          patience=50, min_lr=1e-6)

    # 9. 加载预训练权重（若存在）
    ckpt_path_student = 'ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth'
    ckpt_path_teacher = 'ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth'

    if os.path.exists(ckpt_path_student):
        logger.info(f"Loading student model from {ckpt_path_student}")
        sd = torch.load(ckpt_path_student, map_location='cpu')
        model.load_state_dict(remove_compiled_prefix(sd), strict=False)

    if os.path.exists(ckpt_path_teacher):
        logger.info(f"Loading teacher model from {ckpt_path_teacher}")
        sd = torch.load(ckpt_path_teacher, map_location='cpu')
        teacher_model.load_state_dict(remove_compiled_prefix(sd), strict=False)

    # 10. 使用 Accelerator 准备模型、优化器、调度器
    model, optimizer, warmup_scheduler, plateau_scheduler = accelerator.prepare(
        model, optimizer, warmup_scheduler, plateau_scheduler
    )
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # 11. 初始化实验监控（仅主进程）
    pkt_model = None   # 未使用，保持原样
    if accelerator.is_main_process:
        import swanlab
        swanlab.init(
            project="RNA2mol",
            workspace="LetianGao",
            experiment_name="pkt_20260421_rna30to10_expdb_filter_contrastive",
            description="RNA finetuning with ExpDB+DUDE filter, contrastive queue, balanced sampler. Data: CrossDock(81k)+ExpDB(45k)+RNA(356). RNA ratio 30%->10%, weight 10x. Contrastive weight 0.1, queue 4096.",
        )

        # 打印模型中的可学习缩放因子（若存在）
        unwrapped = accelerator.unwrap_model(model)
        try:
            logger.info({
                'v_scale': torch.sigmoid(unwrapped.v_scale).item(),
                'res_scale': torch.sigmoid(unwrapped.res_scale).item(),
                'ele_scale': torch.sigmoid(unwrapped.ele_scale).item()
            })
        except AttributeError:
            logger.warning("Scale attributes not found in model.")

    # 12. 训练配置常量
    BATCHSIZE = 20
    VAL_FREQ = 1013   # 每约 1225 步验证一次
    best_val_loss = float('inf')
    global_step = 0

    # 13. 构建验证数据加载器（固定）
    # RNA 验证集（每 epoch 验证，最终目标）
    val_loader = ProteinLigandDataLoader(
        val_set,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    val_loader = accelerator.prepare(val_loader)

    # ExpDB 验证集（真实晶体结构蛋白，每 5 个 epoch 验证一次，监控蛋白能力退化）
    val_expdb_loader = ProteinLigandDataLoader(
        val_expdb,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    val_expdb_loader = accelerator.prepare(val_expdb_loader)

    # 14. 训练循环
    logger.info("Starting training loop...")
    for epoch in range(1000):
        model.train()

        # 每 epoch 动态构建训练数据加载器（类别平衡采样）
        # 前 5 个 epoch RNA 比例 30%，之后降至 10%
        current_rna_ratio = 0.3 if epoch < 5 else 0.1
        train_loader = create_balanced_dataloader(
            rna_set=train_set,
            protein_set=train_protein,
            rna_repeat=5,
            samples_per_epoch=100000,
            rna_ratio=current_rna_ratio,
            batch_size=BATCHSIZE,
            num_workers=16,
            seed=args.seed + epoch
        )
        train_loader = accelerator.prepare(train_loader)

        # 统计量
        train_loss = train_c_loss = train_d_loss = 0
        train_align_d_loss = train_align_c_loss = 0
        train_align_d_pkt_loss = train_align_c_pkt_loss = 0
        train_contrastive_loss = 0
        train_samples = 0

        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}',
                         disable=not accelerator.is_local_main_process)

        for batch_idx, batch in enumerate(pbar):
            # 在前一个 epoch 的第一个 step 中，如果 epoch < 1 则更新 warmup scheduler
            if epoch < 1:
                warmup_scheduler.step()

            # ---------- 验证步骤 ----------
            if global_step % VAL_FREQ == (VAL_FREQ - 1):
                model.eval()
                val_losses_list = []

                with torch.no_grad():
                    for val_batch in val_loader:
                        current_bs = val_batch.mol_element_batch.size(0)
                        t = torch.rand(current_bs, 1, device=device).index_select(0,
                                              val_batch.mol_element_batch)
                        unwrapped = accelerator.unwrap_model(model)
                        (c_loss, d_loss, e_loss,
                         discretized_loss, coord_align_loss, type_align_loss,
                         coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss) = \
                            unwrapped.constrain_loss_one_step(
                                rep_model=teacher_model,
                                t=t,
                                pocket_pos=val_batch.pocket_pos / 2,
                                pocket_v=val_batch.pocket_atom_feature.float(),
                                batch_pocket=val_batch.pocket_element_batch,
                                ed_pos=val_batch.ed_ed_pos / 2,
                                ed_v=F.one_hot(val_batch.ed_ed_code - 290, num_classes=4).float(),
                                batch_ed=val_batch.ed_ed_code_batch,
                                ligand_pos=val_batch.mol_pos / 2,
                                ligand_v=val_batch.mol_atom_feature,
                                batch_ligand=val_batch.mol_element_batch,
                                pkt_model=pkt_model,
                                pocket_res=val_batch.pocket_residue,
                                pocket_ele=val_batch.pocket_element,
                                onlygeneral=True
                            )

                        loss = torch.mean(c_loss + d_loss +
                                          0.1 * coord_align_loss + 0.1 * type_align_loss +
                                          0.1 * contrastive_loss)
                        bs = val_batch.mol_element_batch.size(0)
                        val_losses_list.append({
                            'loss': loss.item() * bs,
                            'c_loss': torch.mean(c_loss).item() * bs,
                            'd_loss': torch.mean(d_loss).item() * bs,
                            'align_d': torch.mean(coord_align_loss).item() * bs,
                            'align_c': torch.mean(type_align_loss).item() * bs,
                            'align_d_pkt': torch.mean(coord_align_pkt_loss).item() * bs,
                            'align_c_pkt': torch.mean(type_align_pkt_loss).item() * bs,
                            'contrast': torch.mean(contrastive_loss).item() * bs,
                            'n': bs
                        })

                # 聚合多卡结果
                all_val_losses = accelerator.gather_object(val_losses_list)

                if accelerator.is_main_process:
                    total_n = sum(item['n'] for sublist in all_val_losses for item in sublist)
                    avg_val_loss = sum(item['loss'] for sublist in all_val_losses
                                       for item in sublist) / total_n

                    if epoch >= 1:
                        plateau_scheduler.step(avg_val_loss)

                    metrics = {
                        'val_loss': avg_val_loss,
                        'val_c_loss': sum(item['c_loss'] for sublist in all_val_losses
                                          for item in sublist) / total_n,
                        'val_d_loss': sum(item['d_loss'] for sublist in all_val_losses
                                          for item in sublist) / total_n,
                        'align_d_loss': sum(item['align_d'] for sublist in all_val_losses
                                            for item in sublist) / total_n,
                        'align_c_loss': sum(item['align_c'] for sublist in all_val_losses
                                            for item in sublist) / total_n,
                        'contrastive_loss': sum(item['contrast'] for sublist in all_val_losses
                                                for item in sublist) / total_n,
                        'step': global_step,
                        'lr': optimizer.param_groups[0]['lr']
                    }
                    logger.info(f"[VAL] Step {global_step}: Loss={avg_val_loss:.4f}, LR={metrics['lr']:.2e}")
                    swanlab.log(metrics)

                    # 保存最佳模型
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_path = f"ckpt/20260421/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_{global_step}.pth"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                        logger.info(f"Saved best model to {save_path}")

                model.train()

            # ---------- ExpDB 蛋白验证步骤（每 5 倍 VAL_FREQ 验证一次）----------
            if global_step % (VAL_FREQ * 5) == (VAL_FREQ * 5 - 1):
                model.eval()
                expdb_val_losses_list = []

                with torch.no_grad():
                    # 为加速验证，每轮只随机采样 1000 个 ExpDB 样本
                    num_expdb_val = len(val_expdb)
                    subset_size = min(1000, num_expdb_val)
                    subset_indices = np.random.choice(num_expdb_val, size=subset_size, replace=False)
                    val_expdb_subset = Subset(val_expdb, subset_indices)
                    val_expdb_subset_loader = ProteinLigandDataLoader(
                        val_expdb_subset,
                        batch_size=BATCHSIZE,
                        shuffle=False,
                        num_workers=16,
                        pin_memory=True
                    )
                    val_expdb_subset_loader = accelerator.prepare(val_expdb_subset_loader)

                    for val_batch in val_expdb_subset_loader:
                        current_bs = val_batch.mol_element_batch.size(0)
                        t = torch.rand(current_bs, 1, device=device).index_select(0,
                                          val_batch.mol_element_batch)
                        unwrapped = accelerator.unwrap_model(model)
                        (c_loss, d_loss, e_loss,
                         discretized_loss, coord_align_loss, type_align_loss,
                         coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss) = \
                            unwrapped.constrain_loss_one_step(
                                rep_model=teacher_model,
                                t=t,
                                pocket_pos=val_batch.pocket_pos / 2,
                                pocket_v=val_batch.pocket_atom_feature.float(),
                                batch_pocket=val_batch.pocket_element_batch,
                                ed_pos=val_batch.ed_ed_pos / 2,
                                ed_v=F.one_hot(val_batch.ed_ed_code - 290, num_classes=4).float(),
                                batch_ed=val_batch.ed_ed_code_batch,
                                ligand_pos=val_batch.mol_pos / 2,
                                ligand_v=val_batch.mol_atom_feature,
                                batch_ligand=val_batch.mol_element_batch,
                                pkt_model=pkt_model,
                                pocket_res=val_batch.pocket_residue,
                                pocket_ele=val_batch.pocket_element,
                                onlygeneral=True
                            )

                        loss = torch.mean(c_loss + d_loss +
                                          0.1 * coord_align_loss + 0.1 * type_align_loss +
                                          0.1 * contrastive_loss)
                        bs = val_batch.mol_element_batch.size(0)
                        expdb_val_losses_list.append({
                            'loss': loss.item() * bs,
                            'c_loss': torch.mean(c_loss).item() * bs,
                            'd_loss': torch.mean(d_loss).item() * bs,
                            'align_d': torch.mean(coord_align_loss).item() * bs,
                            'align_c': torch.mean(type_align_loss).item() * bs,
                            'contrast': torch.mean(contrastive_loss).item() * bs,
                            'n': bs
                        })

                    # 清理临时 loader
                    accelerator.free_memory(val_expdb_subset_loader)

                all_expdb_val_losses = accelerator.gather_object(expdb_val_losses_list)

                if accelerator.is_main_process:
                    total_n = sum(item['n'] for sublist in all_expdb_val_losses for item in sublist)
                    avg_expdb_loss = sum(item['loss'] for sublist in all_expdb_val_losses
                                         for item in sublist) / total_n

                    expdb_metrics = {
                        'expdb_val_loss': avg_expdb_loss,
                        'expdb_val_c_loss': sum(item['c_loss'] for sublist in all_expdb_val_losses
                                                for item in sublist) / total_n,
                        'expdb_val_d_loss': sum(item['d_loss'] for sublist in all_expdb_val_losses
                                                for item in sublist) / total_n,
                        'expdb_align_d_loss': sum(item['align_d'] for sublist in all_expdb_val_losses
                                                  for item in sublist) / total_n,
                        'expdb_align_c_loss': sum(item['align_c'] for sublist in all_expdb_val_losses
                                                  for item in sublist) / total_n,
                        'expdb_contrastive_loss': sum(item['contrast'] for sublist in all_expdb_val_losses
                                                      for item in sublist) / total_n,
                        'step': global_step,
                    }
                    logger.info(f"[EXPDB VAL] Step {global_step}: Loss={avg_expdb_loss:.4f}")
                    swanlab.log(expdb_metrics)

                model.train()

            # ---------- 训练步骤 ----------
            current_bs = batch.mol_element_batch.size(0)
            t = torch.rand(current_bs, 1, device=device).index_select(0, batch.mol_element_batch)
            unwrapped = accelerator.unwrap_model(model)
            (c_loss, d_loss, e_loss,
             discretized_loss, coord_align_loss, type_align_loss,
             coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss) = \
                unwrapped.constrain_loss_one_step(
                    rep_model=teacher_model,
                    t=t,
                    pocket_pos=batch.pocket_pos / 2,
                    pocket_v=batch.pocket_atom_feature.float(),
                    batch_pocket=batch.pocket_element_batch,
                    ed_pos=batch.ed_ed_pos / 2,
                    ed_v=F.one_hot(batch.ed_ed_code - 290, num_classes=4).float(),
                    batch_ed=batch.ed_ed_code_batch,
                    ligand_pos=batch.mol_pos / 2,
                    ligand_v=batch.mol_atom_feature,
                    batch_ligand=batch.mol_element_batch,
                    pkt_model=pkt_model,
                    pocket_res=batch.pocket_residue,
                    pocket_ele=batch.pocket_element,
                    onlygeneral=True
                )

            # RNA 样本加权（如果数据包含 is_rna 标记）
            if hasattr(batch, 'pocket_is_rna'):
                rna_mask = batch.pocket_is_rna.bool().to(device)
                weights = torch.where(rna_mask, 10.0, 1.0)
            else:
                weights = torch.ones(c_loss.size(0), device=device)

            loss = torch.mean(
                (c_loss + d_loss) * weights +
                0.1 * coord_align_loss + 0.1 * type_align_loss +
                0.1 * coord_align_pkt_loss + 0.1 * type_align_pkt_loss +
                0.1 * contrastive_loss
            )

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Rank {accelerator.process_index}: NaN/Inf loss at step {global_step}")
                optimizer.zero_grad()
                global_step += 1
                continue

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=8.0)

            optimizer.step()
            optimizer.zero_grad()

            # 累积训练统计量
            bs = batch.mol_element_batch.size(0)
            train_loss += loss.item() * bs
            train_c_loss += torch.mean(c_loss).item() * bs
            train_d_loss += torch.mean(d_loss).item() * bs
            train_align_d_loss += torch.mean(coord_align_loss).item() * bs
            train_align_c_loss += torch.mean(type_align_loss).item() * bs
            train_align_d_pkt_loss += torch.mean(coord_align_pkt_loss).item() * bs
            train_align_c_pkt_loss += torch.mean(type_align_pkt_loss).item() * bs
            train_contrastive_loss += torch.mean(contrastive_loss).item() * bs
            train_samples += bs

            global_step += 1

            # 定期打印训练日志（每 50 步）
            if global_step % 50 == 0 and accelerator.is_main_process:
                logs = {
                    'train_loss': train_loss / train_samples,
                    'main_loss': (train_c_loss + train_d_loss) / train_samples,
                    'train_c_loss': train_c_loss / train_samples,
                    'train_d_loss': train_d_loss / train_samples,
                    'align_ed': (train_align_d_loss + train_align_c_loss) / train_samples,
                    'contrastive_loss': train_contrastive_loss / train_samples,
                    'lr': optimizer.param_groups[0]['lr'],
                    'step': global_step
                }
                logger.info(f"[TRAIN] Step {global_step}: Loss={logs['train_loss']:.4f}")
                swanlab.log(logs)

                # 重置统计量
                train_loss = train_c_loss = train_d_loss = 0
                train_align_d_loss = train_align_c_loss = 0
                train_align_d_pkt_loss = train_align_c_pkt_loss = 0
                train_contrastive_loss = 0
                train_samples = 0

        # 每个 epoch 结束后保存最新 checkpoint
        if accelerator.is_main_process:
            save_path = f"ckpt/20260421/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_latest.pth"
            torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
            logger.info(f"Epoch {epoch+1} finished. Saved latest checkpoint.")


if __name__ == '__main__':
    main()
