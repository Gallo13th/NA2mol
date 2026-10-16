import os
import sys

from model.bfn4edmol import Config, BFN4SBDDScoreModel
from dataloader.dataloader_pointcloud import MolPocketPointCloud
from dataloader.featurizer import GeomMolPocketPointCloud, ProteinLigandDataLoader, GeomMolPointCloudExp, PretrainLigandDataLoader
import torch
import torch.nn.functional as F
import pickle
import math
from util.optims import SingleDeviceMuonWithAuxAdam, Muon
import numpy as np

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
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]  # repository root
def is_sample_valid(item):
    """
    判断一个数据样本是否有效。
    检查所有 pocket 相关属性的节点数量是否一致。
    """
    try:
        # min_mol_atoms=10
        # if item.mol_element.size(0) < min_mol_atoms:
        #     # print(f"Skipping: ligand atoms ({item.mol_element.size(0)}) < {min_mol_atoms}")
        #     return False

        ed_pos = item.ed_ed_pos
        mol_pos = item.mol_pos
        tree = cKDTree(mol_pos)
        distances, indices = tree.query(ed_pos, k=1)
        if max(distances) >= 5.0 or min(distances) <= 1e-3:
            return False
        # print(item.mol_atom_feature)
        if (item.mol_atom_feature == 0).any():
            print(f"Found Unseen Element {item.mol_element.cpu().numpy()}")
            return False
        return True

    except AttributeError as e:
        print(f"❌ Invalid sample: Missing attribute {e}")
        return False
    except Exception as e:
        print(f"❌ Invalid sample: Unexpected error {e}")
        return False


def create_clean_dataset(lmdb_path, output_dir, train_size=2000000):
    """
    创建一个干净的数据集子集，并保存索引文件。
    
    Args:
        lmdb_path (str): 原始 LMDB 数据库路径。
        output_dir (str): 输出目录，用于保存索引文件。
        train_size (int): 训练集目标大小。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据集
    print("Loading raw dataset...")
    raw_dataset = MolPocketPointCloud(lmdb_path=lmdb_path)
    
    print("Processing dataset with GeomMolPocketPointCloud...")
    dataset = GeomMolPointCloudExp(raw_dataset)

    print(f"Total samples in dataset: {len(dataset)}")
    print("Validating samples...")

    # valid_indices = list(range(len(dataset)))
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

    # 划分训练集和验证集
    train_size = min(train_size, len(valid_indices))
    train_indices = valid_indices[:train_size]
    val_indices = valid_indices[train_size:]

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
    indices_path = os.path.join(output_dir, 'clean_dataset_indices.pkl')
    with open(indices_path, 'wb') as f:
        pickle.dump(indices_data, f)
    print(f"✅ Clean dataset indices saved to: {indices_path}")

    # 可选：保存一个简短的摘要
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Total raw samples: {len(dataset)}\n")
        f.write(f"Total valid samples: {len(valid_indices)}\n")
        f.write(f"Training set size: {len(train_set)}\n")
        f.write(f"Validation set size: {len(val_set)}\n")
        f.write(f"Train ratio: {len(train_set)/len(dataset):.3f}\n")
        f.write(f"Val ratio: {len(val_set)/len(dataset):.3f}\n")
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
    dataset = GeomMolPointCloudExp(raw_dataset)
    
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
    from accelerate import Accelerator
    from torch.utils.data.dataset import Subset
    import swanlab
    
    swanlab.init(
        # mode='disabled',
        project="pdfedmol",
        workspace="LetianGao",
        experiment_name="ExpED",
        # id="ot4mryt67lhkff986ffjg",
        # resume=True
        )
    
    # print(torch.load("ckpt/molcraft_epoch26-val_loss5.96-mol_stable0.95-complete0.97.ckpt")['state_dict'].keys())
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
    

    LMDB_PATH = str(ROOT / "data/expdb")  # 数据集路径
    OUTPUT_DIR = "./clean_exped_dataset"  # 保存索引文件的目录
    TARGET_TRAIN_SIZE = 35000       # 训练集大小

    # 第一次运行：创建干净数据集并保存索引
    train_set, val_set = create_clean_dataset(
        lmdb_path=LMDB_PATH,
        output_dir=OUTPUT_DIR,
        train_size=TARGET_TRAIN_SIZE
    )

    # 之后的运行：直接从索引文件加载
    # train_set, val_set = load_clean_dataset(
    #     lmdb_path=LMDB_PATH,
    #     indices_path="./clean_edpretrained_dataset/clean_dataset_indices.pkl"
    # )
    
    point_transformer_config = cfg.point_transformer
    
    model = BFN4SBDDScoreModel(
        net_config=cfg.dynamics.net_config,
        # point_transformer_config=point_transformer_config,
        protein_atom_feature_dim=11, 
        ligand_atom_feature_dim=13,
        time_emb_dim=1,
        ed_feature_dim=4
    )
    # model.load_state_dict(torch.load("ckpt/molcraft_edmol_all_param.pth"))
    # load_model(model, torch.load("ckpt/molcraft_epoch26-val_loss5.96-mol_stable0.95-complete0.97.ckpt")['state_dict'])
    
    # 优化器配置

    muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" in name and "lm_head" not in name
        ]
    adamw_params = [
        p
        for name, p in model.named_parameters()
        if not (
            p.ndim >= 2 and "emb" in name and "lm_head" in name
        )
    ]

    optimizer = Muon(
        lr=5e-4,
        wd=0.01,
        muon_params=muon_params,
        adamw_params=adamw_params,
    )
    state_dict = load_compiled_model(model, torch.load('ckpt/molcraft_edmol_onlyed_multires_best_val_loss_step_239999.pth'))
    model.load_state_dict(state_dict,strict=False)
    model.requires_grad_(True)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.6,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    
    model = torch.compile(model)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.load_state_dict(torch.load('ckpt/molcraft_edmol_onlyed_multires_epoch_9_step_1125000.pth'))
    BATCHSIZE = 14
    VAL_FREQ = 2500  # 每10000个batch验证一次
    best_val_loss = float('inf')
    global_step = 0  # 全局步数计数器
    
    # 准备数据加载器
    train_loader = PretrainLigandDataLoader(
        train_set, batch_size=BATCHSIZE, shuffle=True, 
        num_workers=16, pin_memory=True
    )
    val_loader = PretrainLigandDataLoader(
        val_set, batch_size=BATCHSIZE, shuffle=False,
        num_workers=16, pin_memory=True
    )
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        train_c_loss = 0
        train_d_loss = 0
        train_samples = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # print(batch)
            # continue
            # 训练步骤
            # print(batch.mol_atom_feature)

            # 验证步骤 (每VAL_FREQ个batch执行一次)
            if global_step % VAL_FREQ == (VAL_FREQ - 1):
                model.eval()
                val_loss = 0
                val_c_loss = 0
                val_d_loss = 0
                val_samples = 0
                
                with torch.no_grad():
                    num_val_samples = len(val_set)
                    subset_indices = np.random.choice(num_val_samples, size=num_val_samples, replace=False)
                    val_subset = Subset(val_set, subset_indices)
                    val_loader = PretrainLigandDataLoader(
                        val_subset, batch_size=BATCHSIZE, shuffle=False,
                        num_workers=16, pin_memory=True
                    )
                    val_loader = accelerator.prepare(val_loader)
                    for val_batch in val_loader:
                        c_loss, d_loss, e_loss, discretised_loss = model.loss_one_step(
                            t=torch.rand(BATCHSIZE,1).to('cuda').index_select(0,batch.mol_element_batch),
                            pocket_pos=batch.pocket_pos.to('cuda') / 2,
                            pocket_v=batch.pocket_atom_feature.float().to('cuda'),
                            batch_pocket=batch.pocket_element_batch.to('cuda'),
                            ed_pos=batch.ed_ed_pos.to('cuda') / 2,
                            ed_v=F.one_hot(batch.ed_ed_code,num_classes=4).to('cuda'),
                            batch_ed=batch.ed_ed_code_batch.to('cuda'),
                            ligand_pos=batch.mol_pos.to('cuda') / 2,
                            ligand_v=batch.mol_atom_feature.to('cuda'),
                            batch_ligand=batch.mol_element_batch.to('cuda')
                        )
                        
                        loss = torch.mean(c_loss + d_loss + discretised_loss)
                        val_c_loss += torch.mean(c_loss).item() * len(val_batch.mol_element_batch)
                        val_d_loss += torch.mean(d_loss).item() * len(val_batch.mol_element_batch)
                        val_loss += loss.item() * len(val_batch.mol_element_batch)
                        val_samples += len(val_batch.mol_element_batch)
                
                avg_val_loss = val_loss / val_samples
                scheduler.step(avg_val_loss)  # 根据验证损失调整学习率
                
                swanlab.log({
                    'val_loss': avg_val_loss,
                    'val_c_loss': val_c_loss / val_samples,
                    'val_d_loss': val_d_loss / val_samples,
                    'step': global_step
                })
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), f"ckpt/edbfn_exped_best_val_loss_step_{global_step}.pth")
                    print(f"New best validation loss at step {global_step}: {best_val_loss:.4f}")
                
                model.train()  # 验证后切换回训练模式


            global_step += 1
            c_loss, d_loss, e_loss, discretised_loss = model.loss_one_step(
                t=torch.rand(BATCHSIZE,1).to('cuda').index_select(0,batch.mol_element_batch),
                pocket_pos=batch.pocket_pos.to('cuda') / 2,
                pocket_v=batch.pocket_atom_feature.float().to('cuda'),
                batch_pocket=batch.pocket_element_batch.to('cuda'),
                ed_pos=batch.ed_ed_pos.to('cuda') / 2,
                ed_v=F.one_hot(batch.ed_ed_code,num_classes=4).to('cuda'),
                batch_ed=batch.ed_ed_code_batch.to('cuda'),
                ligand_pos=batch.mol_pos.to('cuda') / 2,
                ligand_v=batch.mol_atom_feature.to('cuda'),
                batch_ligand=batch.mol_element_batch.to('cuda')
            )
            
            loss = torch.mean(c_loss + d_loss + discretised_loss)
            accelerator.backward(loss)
            if accelerator.sync_gradients:  # 通常只在累积梯度的最后一步裁剪
                accelerator.clip_grad_norm_(model.parameters(), max_norm=8.0)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item() * len(batch.mol_element_batch)
            train_c_loss += torch.mean(c_loss).item() * len(batch.mol_element_batch)
            train_d_loss += torch.mean(d_loss).item() * len(batch.mol_element_batch)
            train_samples += len(batch.mol_element_batch)
            
            pbar.set_postfix({
                'train_loss': train_loss / train_samples,
                'train_c_loss': train_c_loss / train_samples,
                'train_d_loss': train_d_loss / train_samples,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # 每50个batch记录一次训练损失
            if global_step % 50 == 0:
                swanlab.log({
                    'train_loss': train_loss / train_samples,
                    'train_c_loss': train_c_loss / train_samples,
                    'train_d_loss': train_d_loss / train_samples,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': global_step
                })
                train_loss = 0
                train_samples = 0
                train_c_loss = 0
                train_d_loss = 0
            
        
        # 每个epoch结束后保存一次模型
        torch.save(model.state_dict(), f"ckpt/edbfn_exped_epoch_{epoch+1}_step_{global_step}.pth")
