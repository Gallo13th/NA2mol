#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}开始设置 rna2smol 环境${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 环境名称
ENV_NAME="rna2smol"

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: 未找到 conda 命令${NC}"
    echo "请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 初始化 conda（重要！）
eval "$(conda shell.bash hook)"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo -e "${YELLOW}警告: 环境 ${ENV_NAME} 已存在${NC}"
    if [ "${RECREATE_ENV:-n}" = "y" ]; then
        REPLY=y
    else
        read -p "是否删除并重新创建? (y/n) [或设置 RECREATE_ENV=y]: " -n 1 -r
        echo
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}删除现有环境...${NC}"
        conda env remove -n ${ENV_NAME} -y
    else
        echo -e "${YELLOW}退出安装${NC}"
        exit 0
    fi
fi

# 步骤1: 从版本化的 environment.yml 创建环境
# The environment spec is committed, not generated: environment.yml pins every
# dependency (torch, torch-scatter, torch-geometric, openbabel, rdkit). The
# conda environment is REQUIRED -- the system Python lacks torch_scatter and
# openbabel, so the model and molecule reconstruction cannot run outside it.
echo -e "\n${GREEN}[步骤 1/5] 从 environment.yml 创建 conda 环境...${NC}"

ENV_FILE="$SCRIPT_DIR/environment.yml"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}错误: 缺少 $ENV_FILE${NC}"
    exit 1
fi

echo "执行: conda env create -f environment.yml"
conda env create -f "$ENV_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 创建环境失败${NC}"
    exit 1
fi

rm -f temp_base_env.yml
echo -e "${GREEN}✓ 基础环境创建成功${NC}"

# 等待一下确保环境完全创建
sleep 2

# 步骤2: 激活环境
echo -e "\n${GREEN}[步骤 2/5] 激活环境...${NC}"

# 使用 conda activate 的多种方式
if ! conda activate ${ENV_NAME} 2>/dev/null; then
    echo "尝试使用 source activate..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${ENV_NAME}
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 激活环境失败${NC}"
    echo "尝试列出可用环境:"
    conda info --envs
    exit 1
fi

echo -e "${GREEN}✓ 环境激活成功: ${ENV_NAME}${NC}"
echo "当前 Python 路径: $(which python)"

# 步骤3: 优先安装 PyTorch 和相关包
echo -e "\n${GREEN}[步骤 3/5] 安装 PyTorch 核心包 (优先级最高)...${NC}"

echo "1/5 安装 torch==2.3.1 (CUDA 11.8)..."
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: torch 安装失败${NC}"
    exit 1
fi

echo "2/5 安装 torchvision==0.18.1..."
pip install torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

echo "3/5 安装 torch-geometric==2.6.1..."
pip install torch-geometric==2.6.1

echo "4/5 安装 torch-cluster==1.6.3..."
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_cluster-1.6.3%2Bpt23cu118-cp310-cp310-linux_x86_64.whl

echo "5/5 安装 torch-scatter==2.1.2..."
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_scatter-2.1.2%2Bpt23cu118-cp310-cp310-linux_x86_64.whl

echo -e "${GREEN}✓ PyTorch 相关包安装完成${NC}"

# 步骤4: 安装其他依赖
echo -e "\n${GREEN}[步骤 4/5] 安装其他 Python 依赖...${NC}"

# 创建其他依赖的 requirements 文件
cat > temp_other_requirements.txt << 'EOF'
numpy==1.26.4
scipy==1.15.3
pandas==2.2.2
rdkit==2024.9.6
biopython==1.85
tqdm==4.67.1
lmdb==1.6.2
hydra-core
omegaconf==2.1.1
matplotlib==3.10.0
huggingface-hub==0.30.2
transformers==4.51.3
accelerate==1.9.0
einops==0.8.1
pystow==0.7.0
networkx==3.4.2
scikit-learn==1.6.1
opencv-python==4.11.0.86
pillow==10.4.0
h5py==3.13.0
protobuf==4.25.7
pyyaml==6.0.2
absl-py==2.2.2
typing-extensions==4.12.2
prolif
EOF

echo "正在安装其他依赖包..."
pip install -r temp_other_requirements.txt
PIP_EXIT_CODE=$?
rm -f temp_other_requirements.txt

if [ $PIP_EXIT_CODE -ne 0 ]; then
    echo -e "${YELLOW}警告: 部分依赖安装失败，请检查上面的错误信息${NC}"
else
    echo -e "${GREEN}✓ 其他依赖安装完成${NC}"
fi

# 步骤5: 验证安装
echo -e "\n${GREEN}[步骤 5/5] 验证安装...${NC}"

cat > /tmp/verify_install.py << 'EOF'
#!/usr/bin/env python
import sys

def test_imports():
    errors = []
    success_count = 0
    
    # 测试 PyTorch
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
        success_count += 1
    except ImportError as e:
        errors.append(f"✗ torch: {e}")
    
    # 测试其他重要包
    packages = [
        ("torch_geometric", "torch_geometric"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("rdkit", "rdkit"),
        ("biopython", "Bio"),
        ("transformers", "transformers"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]
    
    for name, module in packages:
        try:
            __import__(module)
            print(f"✓ {name}")
            success_count += 1
        except ImportError as e:
            errors.append(f"✗ {name}: {e}")
    
    print(f"\n成功安装: {success_count}/{success_count + len(errors)} 个包")
    
    if errors:
        print("\n失败项:")
        for err in errors:
            print(f"  {err}")
        return False
    return True

if __name__ == "__main__":
    print("验证包安装情况:\n")
    success = test_imports()
    sys.exit(0 if success else 1)
EOF

python /tmp/verify_install.py
VERIFY_RESULT=$?
rm -f /tmp/verify_install.py

if [ $VERIFY_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ 环境安装成功！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\n激活环境: ${YELLOW}conda activate ${ENV_NAME}${NC}"
    echo -e "退出环境: ${YELLOW}conda deactivate${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}⚠ 部分验证失败，请检查输出${NC}"
    echo -e "${RED}========================================${NC}"
fi

# 显示环境信息
echo -e "\n${GREEN}当前环境 Python 包信息:${NC}"
conda list -n ${ENV_NAME} 2>/dev/null | grep -E "torch|python|pip|numpy|pandas" | head -20