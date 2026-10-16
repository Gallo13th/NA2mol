#!/bin/bash
# =============================================================================
# Ablation 2 训练 —— 有预训练，有约束学习，无对比学习
# epochs=100, 动态获取空闲 GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJ_DIR"

# ── Conda 环境 ───────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate rna2smol
echo "Conda env: rna2smol ($(which python))"

# ── 动态获取空闲 GPU ─────────────────────────────────────────────────
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "GPU (user-specified): $CUDA_VISIBLE_DEVICES"
else
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
        sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
    if [ -z "$IDLE_GPU" ]; then
        echo "ERROR: No GPU found via nvidia-smi"
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$IDLE_GPU
    echo "GPU (auto): $CUDA_VISIBLE_DEVICES  (memory-ordered, lowest first)"
fi
export OMP_NUM_THREADS=16

echo "===== Ablation 2: pretrain + constraint, no contrastive ====="
echo "时间: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Epochs: 100"

python training/ablation_pretrain_constrain.py \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --scheduler cosine \
    --v_loss_weight 1.0 \
    --rna_loss_weight 5.0 \
    --max_grad_norm 2.0 \
    --mixed_precision bf16 \
    --ckpt_path ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth \
    --ckpt_path_teacher ckpt/molcraft_edmol_onlyed_cst_best_val_loss_step_369999.pth

echo "===== Ablation 2 训练结束 ====="
echo "时间: $(date)"
