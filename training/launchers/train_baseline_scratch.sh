#!/bin/bash
# =============================================================================
# Baseline 训练脚本 —— 从头训练，不使用预训练、教师模型和对比学习
# =============================================================================
# 
# 【与 finetune_full.py 的区别】
# - 不加载预训练权重（ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth）
# - 不创建 teacher_model，不使用 distillation
# - 不使用对比学习（contrastive loss），只优化坐标+类型主损失
# - 调用 model.loss_one_step() 而非 constrain_loss_one_step()
#
# 【适用场景】
# - 作为 ablation baseline，验证预训练/蒸馏/对比学习的增益
# - 资源受限时简化训练流程
#
# 【GPU】
# - 默认使用 4 张 RTX 4090 (0,1,2,3)
# =============================================================================

set -euo pipefail

source "$PROJ_DIR/setup_env.sh" 2>/dev/null || eval "$(conda shell.bash hook)" && conda activate rna2smol

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16

echo "===== 启动 Baseline 训练（从头训练，无教师/无对比学习） ====="
echo "时间: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "日志目录: ./logs/<date>/"
echo "模型目录: ./ckpt/<date>/"

# 如需从断点恢复，取消下行注释
RESUME_ARG="--resume_ckpt ckpt/20260423/baseline_latest.pth"
# RESUME_SWANLAB_ID="loitmd5czmd8feis77e3g"

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    baseline_train.py \
    --epochs 1000 \
    --batch_size 5 \
    --lr 1e-4 \
    --scheduler cosine \
    --v_loss_weight 1.0 \
    --rna_loss_weight 5.0 \
    --max_grad_norm 2.0 \
    --mixed_precision bf16 \
    ${RESUME_ARG:-}

echo "===== Baseline 训练结束 ====="
echo "时间: $(date)"
