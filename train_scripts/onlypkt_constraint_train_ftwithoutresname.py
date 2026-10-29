import os
import sys
# === 必须加在 import torch 之前 ===
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
# os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 如果出错，强制报错而不是无限等待
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 替换旧的
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1" # 既然已经用了 Socket，保持禁用 P2P
# os.environ["NCCL_DEBUG"] = "INFO"       # 打印详细的通信日志，方便看哪里卡住了
from model.bfn4edmol import Config, BFN4SBDDScoreModel
from dataloader.dataloader_pointcloud import MolPocketPointCloud
from dataloader.featurizer import GeomMolOnlyPocket, ProteinLigandDataLoader
import torch
import torch.nn.functional as F
import pickle
import math
from util.optims import SingleDeviceMuonWithAuxAdam, Muon
from rdkit.Chem import QED, Crippen, Lipinski
import pandas as pd
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem, RDLogger
import glob
from accelerate import Accelerator
from torch.distributed import all_gather_object
import torch

if not hasattr(Accelerator, 'gather_object'):
    def gather_object(self, obj):
        if self.num_processes == 1:
            return [obj]
        output = [None for _ in range(self.num_processes)]
        all_gather_object(output, obj)
        return output
    Accelerator.gather_object = gather_object
    
DUDE_PDBIDS = [os.path.basename(c)[-8:-4].lower() for c in glob.glob('DUDE_complex/cocrystal_ligands/dude_nh_*.sdf')]
ROBIN_PDBIDS = ['3fu2','8g4w','8g7e','8g8z','2qwy','9c9k','5btp','6od9','6wzs']

TEST_PDBIDS = DUDE_PDBIDS + ROBIN_PDBIDS

def load_model(model, state_dict):
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
        new_key = key.replace("dynamics.", "")  # 去除"dynamics."前缀
        processed_state_dict[new_key] = value
    # print(processed_state_dict.keys())
    # 首先处理预训练权重中的参数
    for key, value in processed_state_dict.items():
        if key in model_state_dict:
            model_param = model_state_dict[key]
            
            # 如果shape相同，直接加载
            if value.shape == model_param.shape:
                new_state_dict[key] = value  * 0.1
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

def calculate_properties(mol):
    """
    计算分子的一些类药属性。
    Args:
        mol (rdkit.Chem.Mol): RDKit 分子对象。
    Returns:
        dict: 包含计算出的属性值的字典，失败时返回 None。
              属性包括 'QED', 'LogP', 'rotB'。
    """
    if mol is None:
        print("Warning: Input molecule is None in calculate_properties.")
        return None
        
    properties = {}
    try:
        properties['QED'] = QED.qed(mol)
    except Exception as e:
        print(f"Error calculating QED: {e}")
        properties['QED'] = None

    try:
        properties['LogP'] = Crippen.MolLogP(mol)
    except Exception as e:
        print(f"Error calculating LogP: {e}")
        properties['LogP'] = None

    try:
        properties['rotB'] = Lipinski.NumRotatableBonds(mol)
    except Exception as e:
        print(f"Error calculating rotB: {e}")
        properties['rotB'] = None
    try:
        mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
        properties['MolWt'] = mol_wt
    except Exception as e:
        print(f"Error calculating MolWt: {e}")
        properties['MolWt'] = None
    return properties


def is_sample_valid(item):
    """
    判断一个数据样本是否有效。
    检查所有 pocket 相关属性的节点数量是否一致。
    """
    
    try:
        # 提取 pocket 部分的所有张量
        pdb_id = item.pocket_pdb_id
        # print(pdb_id)
        pocket_pos = item.pocket_pos
        pocket_element = item.pocket_element
        pocket_atom_feature = item.pocket_atom_feature
        # pocket_hybridization = item.pocket_hybridization  # 如果也需要检查
        if pdb_id.lower() in TEST_PDBIDS:
            return False
        # 获取它们的节点数量
        sizes = {
            'pocket_pos': pocket_pos.size(0),
            'pocket_element': pocket_element.size(0),
            'pocket_atom_feature': pocket_atom_feature.size(0),
            # 'pocket_hybridization': pocket_hybridization.size(0)
        }

        # 检查所有大小是否相等
        all_sizes = list(sizes.values())
        is_valid = all(size == all_sizes[0] for size in all_sizes)

        if not is_valid:
            print(f"❌ Invalid sample: Size mismatch. Details: {sizes}")
            return False

        # 检查额外的过滤条件
        min_pocket_atoms = 12
        min_mol_atoms = 8

        mol_feat = item.mol_atom_feature
        if torch.any(mol_feat == 0):
            # print(f"[Wwarning] Found Unknown Atom Element in Ligand, Skip")
            return False
        
        smiles = item.mol_smiles
        mol = Chem.MolFromSmiles(smiles)
        
        if pocket_element.size(0) < min_pocket_atoms:
            # print(f"Skipping: pocket atoms ({pocket_element.size(0)}) < {min_pocket_atoms}")
            return False
        if item.mol_element.size(0) < min_mol_atoms:
            # print(f"Skipping: ligand atoms ({item.mol_element.size(0)}) < {min_mol_atoms}")
            return False
        prop = calculate_properties(mol)
        if prop['rotB'] > min(12, item.mol_element.size(0) // 1.5):
            # print(f"Skipping: ligand rotable bonds ({prop['rotB']}) > 10")
            return False

        return True

    except AttributeError as e:
        print(f"❌ Invalid sample: Missing attribute {e}")
        return False
    except Exception as e:
        print(f"❌ Invalid sample: Unexpected error {e}")
        return False


import multiprocessing as mp
from functools import partial

# 全局变量（用于子进程共享参数）
_GLOBAL_LMDB_PATH = None
_GLOBAL_DATASET = None

def init_worker(lmdb_path):
    """子进程初始化：加载数据集"""
    global _GLOBAL_LMDB_PATH, _GLOBAL_DATASET
    _GLOBAL_LMDB_PATH = lmdb_path
    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    _GLOBAL_DATASET = GeomMolOnlyPocket(raw_dataset, mode='general', pocket_radius=8.0, is_train=True)

def validate_single_index(idx):
    """单个样本验证函数（供子进程调用）"""
    global _GLOBAL_DATASET
    try:
        item = _GLOBAL_DATASET[idx]
        if is_sample_valid(item):
            return idx
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
    return None  # 无效样本返回 None

def create_clean_dataset(lmdb_path, output_dir, name, train_size=40000, seed=42, num_workers=20):
    """
    多进程版本：创建干净数据集子集。
    Args:
        num_workers (int): 并行进程数（建议设为 CPU 核心数）。
    """
    import random
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # 初始化主进程数据集（仅用于获取长度）
    print("Loading raw dataset...")
    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    dataset_len = len(raw_dataset)  # 只需长度，不加载全部数据
    print(f"Total samples in dataset: {dataset_len}")

    # 多进程验证
    print(f"Validating samples with {num_workers} workers...")
    valid_indices = []
    
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(lmdb_path,)) as pool:
        # 使用 imap 保持顺序 + tqdm 进度条
        for result in tqdm.tqdm(
            pool.imap(validate_single_index, range(dataset_len)),
            total=dataset_len
        ):
            if result is not None:
                valid_indices.append(result)

    print(f"Found {len(valid_indices)} valid samples.")
    
    # 后续逻辑不变（划分、保存等）
    train_size = min(train_size, len(valid_indices))
    train_indices = random.sample(valid_indices, train_size)
    train_index_set = set(train_indices)
    val_indices = [idx for idx in valid_indices if idx not in train_index_set]


    # 保存索引文件
    indices_data = {
        'all_valid': valid_indices,
        'train': train_indices,
        'val': val_indices
    }
    indices_path = os.path.join(output_dir, name)
    with open(indices_path, 'wb') as f:
        pickle.dump(indices_data, f)
    print(f"✅ Clean dataset indices saved to: {indices_path}")

    # 保存摘要
    summary_path = os.path.join(output_dir, 'pktftwithrna_dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Total raw samples: {len(dataset)}\n")
        f.write(f"Total valid samples: {len(valid_indices)}\n")
        f.write(f"Training set size: {len(train_set)}\n")
        f.write(f"Validation set size: {len(val_set)}\n")
        f.write(f"Train ratio: {len(train_set)/len(dataset):.3f}\n")
        f.write(f"Val ratio: {len(val_set)/len(dataset):.3f}\n")
        if seed is not None:
            f.write(f"Random seed: {seed}\n")
    print(f"📝 Dataset summary saved to: {summary_path}")

    # 创建子集
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"Final dataset sizes:")
    print(f"  Total valid samples: {len(valid_indices)}")
    print(f"  Training set: {len(train_set)}")
    print(f"  Validation set: {len(val_set)}")
    
    return train_set, val_set


def load_clean_dataset(lmdb_path, indices_path):
    """
    从保存的索引文件加载干净的数据集。
    
    Args:
        lmdb_path (str): 原始 LMDB 数据库路径。
        indices_path (str): 之前保存的索引文件路径。
    
    Returns:
        train_set, val_set: torch.utils.data.Subset 对象。
    """
    print(f"Loading dataset from indices file: {indices_path}")
    
    # 加载索引
    with open(indices_path, 'rb') as f:
        indices_data = pickle.load(f)
    
    # 重建数据集
    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    dataset = GeomMolOnlyPocket(raw_dataset, mode='general',pocket_radius=10, is_train=True)
    
    # 创建子集
    train_set = Subset(dataset, indices_data['train'])
    val_set = Subset(dataset, indices_data['val'])
    
    print(f"✅ Clean dataset loaded.")
    print(f"  Training set size: {len(train_set)}")
    print(f"  Validation set size: {len(val_set)}")
    
    return train_set, val_set


if __name__ == '__main__':
    import argparse
    import tqdm
    import logging
    import os
    from datetime import datetime
    from accelerate import Accelerator
    from torch.utils.data.dataset import Subset
    import torch
    import torch.nn.functional as F
    
    # === Setup Accelerator FIRST ===
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    device = accelerator.device
    print(f"Device: {device}, Process index: {accelerator.process_index}, Num processes: {accelerator.num_processes}")

    LMDB_PATH = "./data/pretrain_pocket"
    OUTPUT_DIR = "./data/pretrain_pocket/clean_dataset_onlypro"
    TARGET_TRAIN_SIZE = 1200000
    # train_set, val_set = create_clean_dataset(
    #     lmdb_path=LMDB_PATH,
    #     name='OnlyPro.pkl',
    #     output_dir=OUTPUT_DIR,
    #     train_size=TARGET_TRAIN_SIZE
    # )

    train_set, val_set = load_clean_dataset(
        lmdb_path=LMDB_PATH,
        indices_path="./data/pretrain_pocket/clean_dataset_onlypro/OnlyPro.pkl"
    )
    
    # === Logger Setup (per-process log file to avoid conflict) ===
    def setup_logger(name, log_file, level=logging.INFO):
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            logger.addHandler(handler)
            logger.addHandler(file_handler)
        logger.propagate = False
        return logger

    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")

    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")  # now unused
    parser.add_argument("--logging_level", type=str, default="info")

    # train data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)    
    parser.add_argument("--pos_normalizer", type=float, default=2.0)    

    # train params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=1.5)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf'])

    parser.add_argument("--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn'])
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])

    # eval params
    parser.add_argument('--ckpt_path', type=str, default='best')
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score", choices=['vina_score', 'vina_dock'])

    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)

    # Use unique log file per process to avoid write conflict
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/{_args.exp_name}_{timestamp}_rank{accelerator.process_index}.log"
    logger = setup_logger(__name__, log_file, level=getattr(logging, _args.logging_level.upper()))

    logger.info(f"Starting experiment: {_args.exp_name} on rank {accelerator.process_index}")
    logger.info(f"Arguments: {_args}")

    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11,
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    logger.info(f"Rank {accelerator.process_index}: MODEL BUILD, entering training loop.")
    # Parameter grouping for Muon optimizer
    # muon_params = [
    #     p for name, p in model.named_parameters()
    #     if p.ndim >= 2 and "ed_emb" not in name and "emb" in name and 'v_inference' not in name
    # ]
    # adamw_params = [
    #     p for name, p in model.named_parameters()
    #     if not (p.ndim >= 2 and "emb" in name and 'v_inference' in name)
    # ]

    trainable_params = [
        p for name, p in model.named_parameters()
        if 'v_inference' not in name and p.requires_grad
    ]
    
    
    # 使用 AdamW（更推荐，尤其带 weight decay 时）
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=4e-5,               # 微调建议更小 lr，如 1e-5 ~ 1e-4，根据任务调整
        weight_decay=0.01,     # 可调，embedding 层有时不加 wd，但统一加也常见
    )
    
    logger.info(f"Rank {accelerator.process_index}: optimizer prepared, entering training loop.")
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.6,
        patience=5,
        min_lr=1e-6,
        verbose=False
    )

    teacher_model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11,
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    def remove_compiled_prefix(state_dict):
        """Remove '_orig_mod.' prefix from state_dict keys (from torch.compile)."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_k = k[len('_orig_mod.'):]
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    # Compile BEFORE prepare (common practice, but test compatibility)
    # model = torch.compile(model)
    # teacher_model = torch.compile(teacher_model)
    # Load state dict BEFORE prepare (state dict is on CPU)
    model.load_state_dict(remove_compiled_prefix(torch.load('ckpt/onlypkt_ftonlyprotein_withresele_constraint_train_latest.pth', map_location='cpu')), strict=False)
    teacher_model.load_state_dict(remove_compiled_prefix(torch.load('ckpt/molcraft_edmol_onlyed_multires_epoch_9_step_1125000.pth', map_location='cpu')), strict=False)
    accelerator.wait_for_everyone()
    logger.info(f"Rank {accelerator.process_index}: CKPT LOADED, entering training loop.")
    
    model = accelerator.prepare(model)
    logger.info(f"Rank {accelerator.process_index}: OPTIMIZER PREPARED, entering training loop.")
    optimizer = accelerator.prepare(optimizer)
    teacher_model = teacher_model.to(device)
    # teacher_model = accelerator.prepare(teacher_model)  # AFTER loading state dict
    logger.info(f"Rank {accelerator.process_index}: MODEL prepared, entering training loop.")
    pkt_model = None

    BATCHSIZE = 12
    VAL_FREQ = 5000
    best_val_loss = float('inf')
    global_step = 0 + 25 * 2500
    
    
    train_loader = ProteinLigandDataLoader(
        train_set, batch_size=BATCHSIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = ProteinLigandDataLoader(
        val_set, batch_size=BATCHSIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    logger.info(f"Rank {accelerator.process_index}: DataLoaders prepared, entering training loop.")
    for epoch in range(25, 100):
        model.train()
        # Accumulators reset every logging interval, not per epoch
        train_loss = 0
        train_c_loss = 0
        train_e_loss = 0
        train_d_loss = 0
        train_align_d_loss = 0
        train_align_c_loss = 0
        train_align_d_pkt_loss = 0
        train_align_c_pkt_loss = 0
        train_contrastive_loss = 0
        train_samples = 0

        pbar = tqdm.tqdm(train_loader, desc=f'Training Epoch {epoch+1}', disable=not accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(pbar):

            # === VALIDATION ===
            if global_step % VAL_FREQ == (VAL_FREQ - 1):
                model.eval()
                val_losses = []  # collect per-batch losses for reduction

                with torch.no_grad():
                    for val_batch in tqdm.tqdm(val_loader, desc=f'Valid Epoch {epoch+1}', disable=not accelerator.is_local_main_process):
                        # All tensors already on correct device via DataLoader + prepare
                        unwrapped_model = accelerator.unwrap_model(model)
                        c_loss, d_loss, e_loss, \
                        discretized_loss, coord_align_loss, type_align_loss, \
                        coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss = unwrapped_model.constrain_loss_one_step(
                            rep_model=teacher_model,
                            t=torch.rand(BATCHSIZE, 1, device=device).index_select(0, val_batch.mol_element_batch),
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
                            pocket_res=None,
                            pocket_ele=None,
                            onlygeneral=True
                        )

                        loss = torch.mean(
                            c_loss + d_loss +
                            0.1 * coord_align_loss + 0.1 * type_align_loss +
                            0.05 * coord_align_pkt_loss + 0.05 * type_align_pkt_loss +
                            0.02 * contrastive_loss
                        )

                        # Gather batch size for weighting (important in DDP)
                        batch_size = val_batch.mol_element_batch.size(0)
                        val_losses.append({
                            'loss': loss * batch_size,
                            'c_loss': torch.mean(c_loss) * batch_size,
                            'd_loss': torch.mean(d_loss) * batch_size,
                            'e_loss': torch.mean(e_loss) * batch_size,
                            'coord_align_loss': torch.mean(coord_align_loss) * batch_size,
                            'type_align_loss': torch.mean(type_align_loss) * batch_size,
                            'coord_align_pkt_loss': torch.mean(coord_align_pkt_loss) * batch_size,
                            'type_align_pkt_loss': torch.mean(type_align_pkt_loss) * batch_size,
                            'contrastive_loss': torch.mean(contrastive_loss) * batch_size,
                            'n': batch_size
                        })

                # === REDUCE ACROSS PROCESSES ===
                # Gather all dicts from all processes
                all_val_losses = accelerator.gather_object(val_losses)  # list of lists

                if accelerator.is_main_process:
                    total_loss = sum(item['loss'].item() for sublist in all_val_losses for item in sublist)
                    total_c = sum(item['c_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_d = sum(item['d_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_coord_align = sum(item['coord_align_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_type_align = sum(item['type_align_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_coord_pkt = sum(item['coord_align_pkt_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_type_pkt = sum(item['type_align_pkt_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_contrastive = sum(item['contrastive_loss'].item() for sublist in all_val_losses for item in sublist)
                    total_n = sum(item['n'] for sublist in all_val_losses for item in sublist)

                    avg_val_loss = total_loss / total_n
                    plateau_scheduler.step(avg_val_loss)

                    logger.info("[VAL] step={}, val_loss={:.6f}, main_val_loss={:.6f}, "
                                "align_ed={:.6f}, align_pkt={:.6f}, contrastive={:.6f}, lr={:.2e}".format(
                        global_step,
                        avg_val_loss,
                        (total_c + total_d) / total_n,
                        (total_coord_align + total_type_align) / total_n,
                        (total_coord_pkt + total_type_pkt) / total_n,
                        total_contrastive / total_n,
                        optimizer.param_groups[0]['lr']
                    ))

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        ckpt_path = f"ckpt/onlypkt_ftonlyprotein_withresele_train_best_val_loss_step_{global_step}.pth"
                        # Save unwrapped model
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), ckpt_path)
                        logger.info(f"[SAVE] New best model at step {global_step}: val_loss={best_val_loss:.6f} -> {ckpt_path}")

                model.train()

            global_step += 1

            # === TRAINING STEP ===
            unwrapped_model = accelerator.unwrap_model(model)
            c_loss, d_loss, e_loss, \
            discretized_loss, coord_align_loss, type_align_loss, \
            coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss = unwrapped_model.constrain_loss_one_step(
                rep_model=teacher_model,
                t=torch.rand(BATCHSIZE, 1, device=device).index_select(0, batch.mol_element_batch),
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
                pocket_res=None,
                pocket_ele=None,
                onlygeneral=True
            )

            loss = torch.mean(
                c_loss + d_loss +
                0.1 * coord_align_loss + 0.1 * type_align_loss +
                0.05 * coord_align_pkt_loss + 0.05 * type_align_pkt_loss +
                0.02 * contrastive_loss
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logger.info(f"Rank {accelerator.process_index}: Detected NaN loss!")
                continue # 或者 raise error
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate for logging (local only, no need to sync until log)
            batch_size = batch.mol_element_batch.size(0)
            train_loss += loss.item() * batch_size
            train_c_loss += torch.mean(c_loss).item() * batch_size
            train_d_loss += torch.mean(d_loss).item() * batch_size
            train_align_d_loss += torch.mean(coord_align_loss).item() * batch_size
            train_align_c_loss += torch.mean(type_align_loss).item() * batch_size
            train_align_d_pkt_loss += torch.mean(coord_align_pkt_loss).item() * batch_size
            train_align_c_pkt_loss += torch.mean(type_align_pkt_loss).item() * batch_size
            train_contrastive_loss += torch.mean(contrastive_loss).item() * batch_size
            train_samples += batch_size

            pbar.set_postfix({
                'train_loss': train_loss / train_samples,
                'main_loss': (train_c_loss + train_d_loss) / train_samples,
                'align_ed': (train_align_d_loss + train_align_c_loss) / train_samples,
                'lr': optimizer.param_groups[0]['lr']
            })

            if global_step % 50 == 0:
                if accelerator.is_main_process:
                    logger.info("[TRAIN] step={}, train_loss={:.6f}, main_loss={:.6f}, "
                                "align_ed={:.6f}, align_pkt={:.6f}, contrastive={:.6f}, lr={:.2e}".format(
                        global_step,
                        train_loss / train_samples,
                        (train_c_loss + train_d_loss) / train_samples,
                        (train_align_d_loss + train_align_c_loss) / train_samples,
                        (train_align_d_pkt_loss + train_align_c_pkt_loss) / train_samples,
                        train_contrastive_loss / train_samples,
                        optimizer.param_groups[0]['lr']
                    ))
                # Reset accumulators
                train_loss = 0
                train_samples = 0
                train_c_loss = 0
                train_d_loss = 0
                train_align_d_loss = 0
                train_align_c_loss = 0
                train_align_d_pkt_loss = 0
                train_align_c_pkt_loss = 0
                train_contrastive_loss = 0

        # Save latest checkpoint (only main process)
        if accelerator.is_main_process:
            latest_ckpt = "ckpt/onlypkt_ftonlyprotein_withresele_constraint_train_latest.pth"
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), latest_ckpt)
            logger.info(f"[EPOCH END] Epoch {epoch+1} completed. Latest checkpoint saved: {latest_ckpt}")