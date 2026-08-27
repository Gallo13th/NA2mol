# 荧光分子功能片段库工具包 - 快速开始指南

## 安装

```bash
pip install rdkit pandas numpy
```

## 5分钟快速上手

### 1. 识别分子中的功能片段

```python
from fluo_utils import create_default_library, FragmentIdentifier

# 创建片段库
library = create_default_library()

# 创建识别器
identifier = FragmentIdentifier(library)

# 识别分子
annotation = identifier.identify('CN(C)c1ccccc1')  # N,N-二甲基苯胺

print(f"供体: {len(annotation.donors)}")
print(f"受体: {len(annotation.acceptors)}")
print(f"连接子: {len(annotation.linkers)}")
```

### 2. 提取骨架并模糊匹配

```python
from fluo_utils import ScaffoldMatcher

matcher = ScaffoldMatcher()

# 提取BM骨架
bm = matcher.get_bm_scaffold_smiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
print(f"BM骨架: {bm}")  # c1ccccc1

# 模糊匹配
matches = matcher.fuzzy_match('c1ccccc1', library, threshold=0.5)
for match in matches[:3]:
    print(f"{match['fragment_smiles']} - {match['fragment_type']} - {match['similarity']:.2f}")
```

### 3. 处理分子并输出SDF

```python
from fluo_utils import MoleculeProcessor

processor = MoleculeProcessor(library)

# 处理分子
mol, annotation = processor.process_molecule(
    'CN(C)c1ccccc1',
    output_sdf_path='output.sdf'
)

print(f"识别到 {annotation.total_fragments} 个片段")
print(f"BM骨架: {annotation.scaffolds[0].scaffold_smiles}")
```

### 4. 批量处理

```python
# 批量处理CSV文件
results = processor.batch_process(
    input_csv='molecules.csv',
    output_dir='output_dir/',
    smiles_column='smiles',
    id_column='id'
)

for result in results:
    print(f"{result['id']}: {result['total_fragments']} 片段")
```

## 命令行使用

```bash
# 识别分子
python fluo_utils.py identify "CN(C)c1ccccc1"

# 处理分子
python fluo_utils.py process "CN(C)c1ccccc1" -o output.sdf

# 批量处理
python fluo_utils.py batch input.csv output_dir/
```

## 运行示例

```bash
python example_usage.py
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `fluo_utils.py` | 主工具包，导入这个即可 |
| `fluo_fragment_library.pkl` | 预构建的片段库 |
| `README.md` | 完整文档 |
| `example_usage.py` | 使用示例 |

## 核心类

| 类 | 功能 |
|----|------|
| `FragmentLibrary` | 管理供体/受体/连接子片段 |
| `FragmentIdentifier` | 识别分子中的功能片段 |
| `ScaffoldMatcher` | 提取BM/CSK骨架，模糊匹配 |
| `MoleculeProcessor` | 处理分子，输出标注SDF |

## 功能片段定义

- **供体**: 苯环、杂环、氨基、羟基等（低价态原子）
- **受体**: 硝基、氰基、羰基、卤素等（高价态原子）
- **连接子**: 乙烯基、乙炔基、联苯等（碳链连接）

## 下一步

- 阅读 `README.md` 获取完整文档
- 查看 `example_usage.py` 了解更多示例
- 参考 `PROJECT_SUMMARY.md` 了解项目详情
