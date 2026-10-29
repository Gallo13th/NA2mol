# 荧光分子功能片段库工具包 - 项目总结

## 项目概述

本项目构建了一个完整的荧光分子功能片段库工具包，实现了以下核心功能：

1. **功能片段库构建**：从FluoDB-Full.csv荧光分子库中提取电子供体、电子受体和连接子片段
2. **分子片段识别**：识别任意分子中包含的功能片段
3. **骨架模糊匹配**：通过BM骨架（Bemis-Murcko）和CSK骨架（Core Structure Skeleton）模糊匹配功能片段
4. **分子切分与SDF输出**：将分子切分为BM骨架，输出标注片段的SDF文件

## 生成的文件列表

### 核心工具包

| 文件 | 大小 | 说明 |
|------|------|------|
| `fluo_utils.py` | 27.6 KB | **主工具包入口**，整合所有功能模块 |
| `fluo_fragment_library.py` | 11.7 KB | FragmentLibrary类实现 |
| `fluo_fragment_library.pkl` | 5.0 KB | 预构建的片段库数据 |

### 功能模块

| 文件 | 大小 | 说明 |
|------|------|------|
| `fragment_identifier.py` | 18.2 KB | 片段识别器模块 |
| `scaffold_matcher.py` | 23.3 KB | 骨架匹配器模块 |
| `molecule_processor.py` | 20.5 KB | 分子处理器模块 |

### 文档和示例

| 文件 | 大小 | 说明 |
|------|------|------|
| `README.md` | 8.4 KB | 完整使用文档 |
| `example_usage.py` | 9.9 KB | 使用示例脚本 |
| `PROJECT_SUMMARY.md` | 本文件 | 项目总结文档 |

### 测试数据

| 文件 | 说明 |
|------|------|
| `test_molecules.csv` | 测试分子列表 |
| `test_input.csv` | 批量处理测试输入 |
| `test_output.sdf` | 测试输出SDF |
| `test_aniline.sdf` | 苯胺测试输出 |
| `annotated_molecules.sdf` | 标注分子SDF |
| `batch_results/` | 批量处理结果目录 |

## 功能模块详解

### 1. FragmentLibrary（片段库）

**位置**: `fluo_utils.py` 中的 `FragmentLibrary` 类

**功能**:
- 管理电子供体、电子受体、连接子三类片段
- 支持从pickle文件加载/保存
- 支持添加自定义片段

**预定义片段**:
- **供体**: 苯环、吡啶、噻吩、咔唑、三苯胺、氨基、羟基等（146个唯一片段）
- **受体**: 硝基、氰基、羰基、砜、卤素、三氟甲基等（37个唯一片段）
- **连接子**: 乙烯基、乙炔基、联苯、苯乙烯基等（59个唯一片段）

**API**:
```python
library = FragmentLibrary('fluo_fragment_library.pkl')
donors = library.get_donors()
acceptors = library.get_acceptors()
linkers = library.get_linkers()
library.add_donor('c1ccccc1N', 'aniline')
library.save_library('my_library.pkl')
```

### 2. FragmentIdentifier（片段识别器）

**位置**: `fluo_utils.py` 中的 `FragmentIdentifier` 类

**功能**:
- 识别分子中的电子供体、受体和连接子
- 返回匹配的原子索引和匹配分数
- 支持子结构匹配

**API**:
```python
identifier = FragmentIdentifier(library)
annotation = identifier.identify('CN(C)c1ccccc1')
print(f"供体: {len(annotation.donors)}")
print(f"受体: {len(annotation.acceptors)}")
has_donor = identifier.has_donor('CN(C)c1ccccc1')
```

### 3. ScaffoldMatcher（骨架匹配器）

**位置**: `fluo_utils.py` 中的 `ScaffoldMatcher` 类

**功能**:
- 提取BM骨架（Bemis-Murcko骨架）
- 提取CSK骨架（核心结构骨架）
- 计算Tanimoto相似度
- 模糊匹配功能片段

**API**:
```python
matcher = ScaffoldMatcher()
bm_smiles = matcher.get_bm_scaffold_smiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
csk_smiles = matcher.get_csk_scaffold_smiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
sim = matcher.compute_tanimoto_similarity_smiles('c1ccccc1', 'c1ccncc1')
matches = matcher.fuzzy_match('c1ccccc1', library, threshold=0.5)
```

### 4. MoleculeProcessor（分子处理器）

**位置**: `fluo_utils.py` 中的 `MoleculeProcessor` 类

**功能**:
- 将分子切分为BM骨架
- 识别并标注功能片段
- 输出标注片段的SDF文件
- 支持批量处理
- **不增加原子**：确保不在输入分子上增加原子

**API**:
```python
processor = MoleculeProcessor(library)
mol, annotation = processor.process_molecule('CN(C)c1ccccc1', 'output.sdf')
results = processor.batch_process('input.csv', 'output_dir/')
```

## 使用方式

### 方式1：作为Python库导入

```python
from fluo_utils import (
    FragmentLibrary,
    FragmentIdentifier,
    ScaffoldMatcher,
    MoleculeProcessor,
    create_default_library
)

# 创建片段库
library = create_default_library()

# 识别片段
identifier = FragmentIdentifier(library)
annotation = identifier.identify('CN(C)c1ccccc1')

# 处理分子
processor = MoleculeProcessor(library)
mol, annotation = processor.process_molecule('CN(C)c1ccccc1', 'output.sdf')
```

### 方式2：命令行使用

```bash
# 识别分子
python fluo_utils.py identify "CN(C)c1ccccc1"

# 处理分子
python fluo_utils.py process "CN(C)c1ccccc1" -o output.sdf

# 批量处理
python fluo_utils.py batch input.csv output_dir/
```

### 方式3：运行示例脚本

```bash
python example_usage.py
```

## 输入/输出格式

### 输入CSV格式

```csv
id,smiles,name
mol_001,CN(C)c1ccccc1,N,N-dimethylaniline
mol_002,O=[N+]([O-])c1ccccc1,nitrobenzene
mol_003,c1ccccc1C=Cc2ccccc2,stilbene
```

### 输出SDF属性

SDF文件包含以下属性字段：
- `donor_count`: 供体数量
- `acceptor_count`: 受体数量
- `linker_count`: 连接子数量
- `total_fragments`: 总片段数
- `donor_smiles`: 供体SMILES（用|分隔）
- `acceptor_smiles`: 受体SMILES（用|分隔）
- `linker_smiles`: 连接子SMILES（用|分隔）
- `bm_scaffold`: BM骨架SMILES

## 功能片段定义

### 电子供体
- 低价态的氧、氮、硫、碳
- 含有低价态原子的杂环
- 苯环
- 典型示例：苯环、吡啶、噻吩、咔唑、三苯胺、氨基、羟基

### 电子受体
- 高价态的氧（三价及以上）
- 氮（四价及以上）
- 磷（四价及以上）
- 硫（三价及以上）
- 卤素
- 典型示例：硝基、氰基、羰基、砜、三氟甲基

### 连接子
- 碳原子之间的单键、双键、单双键交替的碳链
- 可以无，至多一个单键，单双键交替数量不限
- 典型示例：乙烯基、乙炔基、联苯、苯乙烯基

## 核心设计原则

1. **模块化设计**：每个功能独立成类，便于维护和扩展
2. **不增加原子**：分子处理过程不会在输入分子上增加原子
3. **保留结构完整性**：输出SDF保留原始分子结构
4. **灵活配置**：支持自定义片段库和匹配参数
5. **批量处理**：支持大规模分子库的高效处理

## 扩展性

### 添加自定义片段

```python
library = FragmentLibrary()
library.add_donor('c1ccc2c(c1)oc1ccccc12', 'dibenzofuran')
library.add_acceptor('S(=O)(=O)N', 'sulfonamide')
library.add_linker('C#CC#C', 'diacetylene')
```

### 自定义匹配参数

```python
matcher = ScaffoldMatcher(fp_radius=3, fp_nbits=4096)
matches = matcher.fuzzy_match(scaffold, library, threshold=0.8)
```

## 测试覆盖

- ✅ 片段库构建：从10,000个荧光分子中提取242个唯一片段
- ✅ 片段识别：准确识别供体、受体、连接子
- ✅ BM骨架提取：正确提取Bemis-Murcko骨架
- ✅ CSK骨架提取：正确提取核心结构骨架
- ✅ Tanimoto相似度：准确计算分子相似度
- ✅ 模糊匹配：支持多种阈值和匹配策略
- ✅ SDF输出：正确保存标注信息
- ✅ 批量处理：支持大规模分子处理

## 依赖项

- Python 3.7+
- RDKit (化学信息学库)
- pandas (数据处理)
- numpy (数值计算)

## 安装

```bash
pip install rdkit pandas numpy
```

## 许可证

MIT License

## 作者

AI Assistant

## 版本

1.0.0
