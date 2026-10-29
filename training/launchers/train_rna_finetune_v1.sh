#!/bin/bash
# =============================================================================
# DiffEDMol RNA Finetuning 训练脚本 (Screen 后台运行)
# 日期: 2026-04-21
# =============================================================================
# 
# 【项目设置】
# - 模型: BFN (Bayesian Flow Network) + unio2net
# - 基础权重: ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth
# - 训练模式: Teacher-Student 蒸馏 + 对比学习 (InfoNCE with Memory Bank)
# - GPU: 4 x NVIDIA RTX 4090 (24GB) —— 从8张中选取4张使用
# - 混合精度: bf16 (可改为 fp16 若出现 NaN)
#
# 【数据集】
# - CrossDock:  52,961 train / 116,898 val  (蛋白对接 poses)
# - ExpDB:      ~45k train / 5,000 val    (真实晶体，已过滤 DUDE 相似结构)
# - RNA:           356 train / 382 val      (真实 RNA 结构)
# - 合计蛋白训练: ~98k (ExpDB + CrossDock)
#
# 【数据增强】
# - 随机旋转 (所有数据)
# - RNA 专用增强: noise_scale=0.15, random shift=[-0.3, 0.3]
#
# 【训练策略】
# - BalancedMixedSampler: RNA ratio 30% (ep 0-4) → 10% (ep 5+)
# - RNA loss weight: 5.0  (降低自 10.0，防止 4090 上梯度爆炸/NaN)
# - Contrastive weight: 0.1 (finetuning)
# - Memory Bank: FIFO queue size 4096, cross-GPU all_gather
# - Temperature: learnable (init log(5.0)), clamp_max=100
# - Grad Clip: max_norm=2.0 (收紧自 8.0，增强 4090 稳定性)
#
# 【验证策略】
# - RNA val:  每个 VAL_FREQ 步
# - ExpDB val: 每 5 倍 VAL_FREQ 步，采样 1,000
#
# 【输出路径】
# - Logs:   ./logs/<运行日期>/
# - CKPT:   ./ckpt/<运行日期>/
# - SwanLab: project RNA2mol, run 见云端
#
# 【Screen 用法】
#   启动:   screen -S rna_train -dm bash train.sh
#   查看:   screen -ls
#   接入:   screen -r rna_train
#   退出:   (screen 内) Ctrl+A, D
#   停止:   (screen 内) Ctrl+C, 然后 exit
# =============================================================================

set -euo pipefail

# 使用当前目录（无需 cd，应在项目根目录执行本脚本）

# 激活环境（使用本机实际环境路径）
source "$PROJ_DIR/setup_env.sh" 2>/dev/null || eval "$(conda shell.bash hook)" && conda activate rna2smol

# =============================================================================
# 【GPU 选择】本机有 8 张 4090，以下选择前 4 张 (0,1,2,3)。
# 如需使用其他卡，请修改此行，例如：
#   export CUDA_VISIBLE_DEVICES=4,5,6,7
# =============================================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 性能相关环境变量
export OMP_NUM_THREADS=16

echo "===== 启动 RNA Finetuning 训练 ====="
echo "时间: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "日志目录: ./logs/<date>/"
echo "模型目录: ./ckpt/<date>/"

# =============================================================================
# 正式训练命令
# 
# 关键参数说明（针对 4090 NaN 问题优化）：
#   --rna_loss_weight 5.0      降低 RNA 权重（原 10.0），防止梯度爆炸
#   --max_grad_norm 2.0         收紧梯度裁剪（原 8.0）
#   --mixed_precision bf16      可选 bf16/fp16/no。若仍出现 NaN，可改为 fp16
#   --resume_ckpt ...           若需从上一轮断点恢复，取消注释下面 resume 行
# =============================================================================

# 如需从上一轮 checkpoint 恢复，取消下行注释并注释掉上一行 ckpt_path
RESUME_ARG="--resume_ckpt ckpt/20260421/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_latest.pth"
RESUME_SWANLAB_ID="loitmd5czmd8feis77e3g"

# accelerate launch 参数说明：
#   --multi_gpu          启用多卡分布式
#   --num_processes 4    使用 4 个进程（对应 4 张 GPU）
#   --mixed_precision bf16  混合精度（可通过命令行覆盖为 fp16/no）
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    finetune_full.py \
    --epochs 50 \
    --batch_size 20 \
    --lr 1e-4 \
    --scheduler cosine \
    --v_loss_weight 1.0 \
    --rna_loss_weight 5.0 \
    --max_grad_norm 2.0 \
    --mixed_precision bf16 \
    --ckpt_path ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth \
    ${RESUME_ARG:-}

echo "===== 训练结束 ====="
echo "时间: $(date)"
