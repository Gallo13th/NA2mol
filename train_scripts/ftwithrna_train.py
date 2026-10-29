import os
import sys

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
DUDE_PDBIDS = [os.path.basename(c)[-8:-4].lower() for c in glob.glob('DUDE_complex/cocrystal_ligands/dude_nh_*.sdf')]
ROBIN_PDBIDS = ['3fu2','8g4w','8g7e','8g8z','2qwy','9c9k','5btp','6od9','6wzs']

TEST_PDBIDS = DUDE_PDBIDS + ROBIN_PDBIDS
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
        if prop['rotB'] > 12:
            # print(f"Skipping: ligand rotable bonds ({prop['rotB']}) > 10")
            return False

        return True

    except AttributeError as e:
        print(f"❌ Invalid sample: Missing attribute {e}")
        return False
    except Exception as e:
        print(f"❌ Invalid sample: Unexpected error {e}")
        return False


def create_clean_dataset(lmdb_path, output_dir, name, train_size=40000, seed=42):
    """
    创建一个干净的数据集子集（随机采样），并保存索引文件。
    
    Args:
        lmdb_path (str): 原始 LMDB 数据库路径。
        output_dir (str): 输出目录，用于保存索引文件。
        train_size (int): 训练集目标大小。
        seed (int): 随机种子，用于可复现性（设为 None 表示不固定）。
    """
    import random
    if seed is not None:
        random.seed(seed)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据集
    print("Loading raw dataset...")
    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    # print(raw_dataset[0])
    # assert 0
    print("Processing dataset with GeomMolPocketPointCloud...")
    dataset = GeomMolOnlyPocket(raw_dataset, mode='general', pocket_radius=8.0, is_train=True)

    print(f"Total samples in dataset: {len(dataset)}")
    print("Validating samples...")

    valid_indices = []
    for idx in tqdm.tqdm(range(len(dataset))):
        try:
            item = dataset[idx]
            if is_sample_valid(item):
                valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"Found {len(valid_indices)} valid samples.")

    # 随机划分训练集和验证集（无放回采样）
    train_size = min(train_size, len(valid_indices))
    train_indices = random.sample(valid_indices, train_size)
    # 验证集 = 剩余的 valid_indices \ train_indices（保持顺序无关）
    val_indices = [idx for idx in valid_indices if idx not in set(train_indices)]

    # 创建子集
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"Final dataset sizes:")
    print(f"  Total valid samples: {len(valid_indices)}")
    print(f"  Training set: {len(train_set)}")
    print(f"  Validation set: {len(val_set)}")

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

from torch.utils.data import ConcatDataset, Subset, DataLoader
import random

# 假设 train_set 和 train_aug_set 已经加载好（如你代码所示）
# 注意：train_set 是 RNA 数据（490 个），train_aug_set 是 protein 数据（35000 个）

def create_mixed_dataloader(train_set, train_aug_set, aug_sample_size=490, batch_size=..., num_workers=16):
    # 随机采样 aug_sample_size 个样本从 train_aug_set
    aug_indices = random.sample(range(len(train_aug_set)), min(aug_sample_size, len(train_aug_set)))
    sampled_aug_set = Subset(train_aug_set, aug_indices)
    
    # 合并原始 train_set（RNA）和采样的 protein 数据
    combined_dataset = ConcatDataset([train_set, sampled_aug_set])
    
    loader = ProteinLigandDataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

if __name__ == '__main__': 
    import argparse
    import tqdm
    from accelerate import Accelerator
    from torch.utils.data.dataset import Subset
    import swanlab
    
    swanlab.init(
        # mode='disabled',
        project="pdfedmol",
        workspace="LetianGao",
        experiment_name="pkt_ftonlyrna_withresele_addconstraintloss",
        # id="ljauemsnb7bvzrr1yy7sn",
        # resume=True
        )
    
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"
    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--config_file", type=str, default="config.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

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
    parser.add_argument('--max_grad_norm', type=str, default='Q')  # '8.0' for

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=1.5)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf']) #vanilla or end_back

    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])

    # eval params
    parser.add_argument('--ckpt_path', type=str, default='best', help='path to the checkpoint')
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score", choices=['vina_score', 'vina_dock'])
    _args = parser.parse_args()
    
    cfg = Config(**_args.__dict__)
    
    # Protein Set for Augment
    LMDB_PATH = "/root/autodl-fs/DiffEDMol-unitrans/pocket_with_pdbid"
    OUTPUT_DIR = "./clean_dataset_onlypro"  # 保存索引文件的目录
    TARGET_TRAIN_SIZE = 35000       # 您期望的训练集大小

    train_aug_set, _ = load_clean_dataset(
        lmdb_path=LMDB_PATH,
        indices_path="./clean_dataset_onlypro/OnlyPro.pkl"
    )
    
    LMDB_PATH = "/root/autodl-fs/DiffEDMol-unitrans/rnaligandwithedid"
    OUTPUT_DIR = "./clean_dataset_onlyrna"  # 保存索引文件的目录
    TARGET_TRAIN_SIZE = 490       # 您期望的训练集大小
    
    train_set, val_set = load_clean_dataset(
        lmdb_path=LMDB_PATH,
        indices_path="./clean_dataset_onlyrna/OnlyRNA.pkl"
    )
    
    
    point_transformer_config = cfg.point_transformer
    
    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11, 
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    

    # 冻结所有包含 'v_inference' 的参数（不参与优化）
    frozen_params_names = {name for name, p in model.named_parameters() if 'v_inference' in name}
    for name, p in model.named_parameters():
        if 'v_inference' in name:
            p.requires_grad = False
    
    # 可训练参数：排除 frozen 的
    trainable_params = [
        p for name, p in model.named_parameters()
        if 'v_inference' not in name and p.requires_grad
    ]
    
    
    # 使用 AdamW（更推荐，尤其带 weight decay 时）
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=3e-4,               # 微调建议更小 lr，如 1e-5 ~ 1e-4，根据任务调整
        weight_decay=0.01,     # 可调，embedding 层有时不加 wd，但统一加也常见
    )

    from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

    steps_per_epoch = 70   
    warmup_steps = steps_per_epoch * 20   
    
    # 1. Warmup 调度器：从 lr * 0.01 或 0 线性升到 base_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # 2. 主调度器：ReduceLROnPlateau（从 warmup 结束后的 lr 开始）
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=20,
        min_lr=1e-7,
        verbose=True
    )
    
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    teacher_model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11, 
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    pkt_model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        protein_atom_feature_dim=11, 
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    model = torch.compile(model)
    teacher_model = torch.compile(teacher_model)
    pkt_model = torch.compile(pkt_model)
    model, optimizer, warmup_scheduler, plateau_scheduler = accelerator.prepare(model, optimizer, warmup_scheduler, plateau_scheduler)
    model.load_state_dict(torch.load('ckpt/onlypkt_ftonlyprotein_withresele_train_best_val_loss_step_222999.pth'), strict=False)
    # model.load_state_dict(torch.load("ckpt/onlypkt_gt_ftonlyrna_constraint_train_latest.pth"), strict=False)
    teacher_model.load_state_dict(torch.load('ckpt/molcraft_edmol_onlyed_multires_epoch_9_step_1125000.pth'), strict=False)
    teacher_model = accelerator.prepare(teacher_model)
    teacher_model.eval()
    pkt_model.load_state_dict(torch.load('ckpt/onlypkt_ftonlyprotein_withresele_train_best_val_loss_step_222999.pth'), strict=False)
    pkt_model = accelerator.prepare(pkt_model)
    pkt_model = None
    # pkt_model.eval()
    print({'v_scale': torch.sigmoid(model.v_scale).item(),
    'res_scale': torch.sigmoid(model.res_scale).item(),
    'ele_scale': torch.sigmoid(model.ele_scale).item()})
    
    BATCHSIZE = 14
    VAL_FREQ = 70*1  # 每1000个batch验证一次
    best_val_loss = float('inf')
    global_step = 0  # 全局步数计数器
    
    # 准备数据加载器
    # train_loader = ProteinLigandDataLoader(
    #     train_set, batch_size=BATCHSIZE, shuffle=True, 
    #     num_workers=16, pin_memory=True
    # )
    val_loader = ProteinLigandDataLoader(
        val_set, batch_size=BATCHSIZE, shuffle=False,
        num_workers=16, pin_memory=True
    )
    val_loader = accelerator.prepare(val_loader)
    
    for epoch in range(0, 1000):
        model.train()
        train_loader = create_mixed_dataloader(
            train_set=train_set,
            train_aug_set=train_aug_set,
            aug_sample_size=490,
            batch_size=BATCHSIZE,
            num_workers=16
        )
        train_loader = accelerator.prepare(train_loader)
        train_loss = 0
        train_c_loss = 0
        train_e_loss = 0
        train_d_loss = 0
        train_align_d_loss = 0    # ED → self coord
        train_align_c_loss = 0    # ED → self type
        # >>> ADD: new accumulators (minimal change)
        train_align_d_pkt_loss = 0  # PKT → self coord
        train_align_c_pkt_loss = 0  # PKT → self type
        train_contrastive_loss = 0
        train_samples = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            if epoch < 20:
                warmup_scheduler.step()
    
            # 验证步骤 (每VAL_FREQ个batch执行一次)
            if global_step % VAL_FREQ == (VAL_FREQ - 1):
                model.eval()
                val_loss = 0
                val_c_loss = 0
                val_e_loss = 0
                val_d_loss = 0
                val_align_d_loss = 0
                val_align_c_loss = 0
                # >>> ADD
                val_align_d_pkt_loss = 0
                val_align_c_pkt_loss = 0
                val_contrastive_loss = 0
                val_samples = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        c_loss, d_loss, e_loss, \
                        discretized_loss, coord_align_loss, type_align_loss, \
                        coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss = model.constrain_loss_one_step(
                            rep_model=teacher_model,
                            t=torch.rand(BATCHSIZE, 1).to('cuda').index_select(0, val_batch.mol_element_batch),
                            pocket_pos=val_batch.pocket_pos.to('cuda') / 2,
                            pocket_v=val_batch.pocket_atom_feature.float().to('cuda'),
                            batch_pocket=val_batch.pocket_element_batch.to('cuda'),
                            ed_pos=val_batch.ed_ed_pos.to('cuda') / 2,
                            ed_v=F.one_hot(val_batch.ed_ed_code - 290, num_classes=4).float().to('cuda'),
                            batch_ed=val_batch.ed_ed_code_batch.to('cuda'),
                            ligand_pos=val_batch.mol_pos.to('cuda') / 2,
                            ligand_v=val_batch.mol_atom_feature.to('cuda'),
                            batch_ligand=val_batch.mol_element_batch.to('cuda'),
                            pkt_model=pkt_model,
                            pocket_res=val_batch.pocket_residue.to('cuda'),
                            pocket_ele=val_batch.pocket_element.to('cuda')
                        )
                        
                        
                        loss = torch.mean(c_loss + d_loss + 0.1 * coord_align_loss + 0.1 * type_align_loss)
                        # loss = torch.mean(
                        #     c_loss + d_loss +
                        #     0.1 * coord_align_loss + 0.1 * type_align_loss +
                        #     0.05 * coord_align_pkt_loss + 0.05 * type_align_pkt_loss +
                        #     0.02 * contrastive_loss
                        # )
    
                        val_c_loss += torch.mean(c_loss).item() * len(val_batch.mol_element_batch)
                        val_e_loss += torch.mean(e_loss).item() * len(val_batch.mol_element_batch)
                        val_d_loss += torch.mean(d_loss).item() * len(val_batch.mol_element_batch)
                        val_align_d_loss += torch.mean(coord_align_loss).item() * len(val_batch.mol_element_batch)
                        val_align_c_loss += torch.mean(type_align_loss).item() * len(val_batch.mol_element_batch)
                        # >>> ADD
                        val_align_d_pkt_loss += torch.mean(coord_align_pkt_loss).item() * len(val_batch.mol_element_batch)
                        val_align_c_pkt_loss += torch.mean(type_align_pkt_loss).item() * len(val_batch.mol_element_batch)
                        val_contrastive_loss += torch.mean(contrastive_loss).item() * len(val_batch.mol_element_batch)
    
                        val_loss += loss.item() * len(val_batch.mol_element_batch)
                        val_samples += len(val_batch.mol_element_batch)
                
                avg_val_loss = val_loss / val_samples
                if epoch >= 20:
                    plateau_scheduler.step(avg_val_loss)
    
                # >>> ADD new metrics to log
                swanlab.log({
                    'val_loss': avg_val_loss,
                    'val_c_loss': val_c_loss / val_samples,
                    'val_d_loss': val_d_loss / val_samples,
                    'main_val_loss': (val_c_loss + val_d_loss) / val_samples,
                    'align_d_loss': val_align_d_loss / val_samples,          # ED→self coord
                    'align_c_loss': val_align_c_loss / val_samples,           # ED→self type
                    # >>> ADD
                    'align_d_pkt_loss': val_align_d_pkt_loss / val_samples,   # PKT→self coord
                    'align_c_pkt_loss': val_align_c_pkt_loss / val_samples,   # PKT→self type
                    'contrastive_loss': val_contrastive_loss / val_samples,
                    'step': global_step
                })
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), f"ckpt/onlypkt_ftonlyrna_withresele_train_best_val_loss_step_{global_step}.pth")
                    print(f"New best validation loss at step {global_step}: {best_val_loss:.4f}")
                
                model.train()
    
            global_step += 1
    
            # >>> Training step (same as val, but on `batch`)
            c_loss, d_loss, e_loss, \
            discretized_loss, coord_align_loss, type_align_loss, \
            coord_align_pkt_loss, type_align_pkt_loss, contrastive_loss = model.constrain_loss_one_step(
                rep_model=teacher_model,
                t=torch.rand(BATCHSIZE, 1).to('cuda').index_select(0, batch.mol_element_batch),
                pocket_pos=batch.pocket_pos.to('cuda') / 2,
                pocket_v=batch.pocket_atom_feature.float().to('cuda'),
                batch_pocket=batch.pocket_element_batch.to('cuda'),
                ed_pos=batch.ed_ed_pos.to('cuda') / 2,
                ed_v=F.one_hot(batch.ed_ed_code - 290, num_classes=4).float().to('cuda'),
                batch_ed=batch.ed_ed_code_batch.to('cuda'),
                ligand_pos=batch.mol_pos.to('cuda') / 2,
                ligand_v=batch.mol_atom_feature.to('cuda'),
                batch_ligand=batch.mol_element_batch.to('cuda'),
                pkt_model=pkt_model,
                pocket_res=batch.pocket_residue.to('cuda'),
                pocket_ele=batch.pocket_element.to('cuda')
            )
            
            # Keep original loss formula for stability
            # loss = torch.mean(c_loss + d_loss + 0.1 * coord_align_loss + 0.1 * type_align_loss)
            # ↑ To enable new terms:
            loss = torch.mean(
                c_loss + d_loss +
                0.1 * coord_align_loss + 0.1 * type_align_loss +
                0.05 * coord_align_pkt_loss + 0.05 * type_align_pkt_loss +
                0.02 * contrastive_loss
            )
    
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=8.0)
            optimizer.step()
            optimizer.zero_grad()
    
            train_loss += loss.item() * len(batch.mol_element_batch)
            train_c_loss += torch.mean(c_loss).item() * len(batch.mol_element_batch)
            train_e_loss += torch.mean(e_loss).item() * len(batch.mol_element_batch)
            train_d_loss += torch.mean(d_loss).item() * len(batch.mol_element_batch)
            train_align_d_loss += torch.mean(coord_align_loss).item() * len(batch.mol_element_batch)
            train_align_c_loss += torch.mean(type_align_loss).item() * len(batch.mol_element_batch)
            train_align_d_pkt_loss += torch.mean(coord_align_pkt_loss).item() * len(batch.mol_element_batch)
            train_align_c_pkt_loss += torch.mean(type_align_pkt_loss).item() * len(batch.mol_element_batch)
            train_contrastive_loss += torch.mean(contrastive_loss).item() * len(batch.mol_element_batch)
    
            train_samples += len(batch.mol_element_batch)
            
            # Update progress bar with existing + new items (可选加，避免太长)
            pbar.set_postfix({
                'train_loss': train_loss / train_samples,
                'main_loss': (train_c_loss + train_d_loss) / train_samples,
                'align_ed': (train_align_d_loss + train_align_c_loss) / train_samples,
                'align_pkt': (train_align_d_pkt_loss + train_align_c_pkt_loss) / train_samples,
                'contrast': train_contrastive_loss / train_samples,
                'lr': optimizer.param_groups[0]['lr']
            })
    
            if global_step % 50 == 0:
                swanlab.log({
                    'train_loss': train_loss / train_samples,
                    'train_c_loss': train_c_loss / train_samples,
                    'train_d_loss': train_d_loss / train_samples,
                    'main_loss': (train_c_loss + train_d_loss) / train_samples,
                    'align_d_loss': train_align_d_loss / train_samples,        # ED→self coord
                    'align_c_loss': train_align_c_loss / train_samples,         # ED→self type
                    'align_d_pkt_loss': train_align_d_pkt_loss / train_samples, # PKT→self coord
                    'align_c_pkt_loss': train_align_c_pkt_loss / train_samples, # PKT→self type
                    'contrastive_loss': train_contrastive_loss / train_samples,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'v_scale': torch.sigmoid(model.v_scale).item(),
                    'res_scale': torch.sigmoid(model.res_scale).item(),
                    'ele_scale': torch.sigmoid(model.ele_scale).item(),
                    'step': global_step
                })
                # Reset accumulators (as before)
                train_loss = 0
                train_samples = 0
                train_c_loss = 0
                train_d_loss = 0
                train_e_loss = 0
                train_align_d_loss = 0
                train_align_c_loss = 0
                train_align_d_pkt_loss = 0
                train_align_c_pkt_loss = 0
                train_contrastive_loss = 0
        
        torch.save(model.state_dict(), f"ckpt/onlypkt_ftonlyrna_withresele_constraint_train_latest.pth")