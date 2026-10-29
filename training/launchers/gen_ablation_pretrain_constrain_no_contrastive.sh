#!/bin/bash
# =============================================================================
# 消融2 生成测试脚本 —— 有预训练有约束学习无对比学习
# 禁用 refiner
# =============================================================================

set -euo pipefail

cd "$PROJ_DIR"
source "$PROJ_DIR/setup_env.sh" 2>/dev/null || eval "$(conda shell.bash hook)" && conda activate rna2smol

export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=16

echo "===== 启动消融2生成测试（有预训练有约束学习无对比学习） ====="
echo "时间: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "模型: ckpt/ablation_best_models/ablation_pretrain_constrain_only_best.pth"
echo "输出: output/ablation_pretrain_constrain_only/"

python genbypkt.py \
    --ckpt ckpt/ablation_best_models/ablation_pretrain_constrain_only_best.pth \
    --output_prefix ablation_pretrain_constrain_only \
    --feat_mode generalfeat \
    --sample_steps 100

echo "===== 消融2生成测试结束 ====="
echo "时间: $(date)"
