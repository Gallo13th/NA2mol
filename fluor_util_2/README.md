# 荧光分子功能片段库工具包 (Fluorescent Molecule Fragment Library Utils)

## 概述

该工具包用于荧光分子的功能片段分析、识别和再设计，支持以下核心功能：

1. **功能片段库构建**：从荧光分子库中提取电子供体、电子受体和连接子片段
2. **分子片段识别**：识别任意分子中包含的功能片段
3. **骨架模糊匹配**：通过BM骨架和CSK骨架模糊匹配功能片段
4. **分子切分与SDF输出**：将分子切分为BM骨架，输出标注片段的SDF文件

## 安装依赖

```bash
pip install rdkit pandas numpy
```

## 核心模块

### 1. 片段库 (FragmentLibrary)

```python
from fluo_utils import FragmentLibrary, create_default_library

# 创建默认片段库
library = create_default_library()

# 或从文件加载
library = FragmentLibrary('fluo_fragment_library.pkl')

# 获取各类片段
donors = library.get_donors()
acceptors = library.get_acceptors()
linkers = library.get_linkers()

# 添加自定义片段
library.add_donor('c1ccccc1N', 'aniline')
library.add_acceptor('C#N', 'cyano')
library.add_linker('C=C', 'vinyl')

# 保存片段库
library.save_library('my_library.pkl')
```

### 2. 片段识别器 (FragmentIdentifier)

```python
from fluo_utils import FragmentIdentifier, create_default_library

library = create_default_library()
identifier = FragmentIdentifier(library)

# 识别单分子
annotation = identifier.identify('CN(C)c1ccccc1')  # N,N-二甲基苯胺

print(f"供体数量: {len(annotation.donors)}")
print(f"受体数量: {len(annotation.acceptors)}")
print(f"连接子数量: {len(annotation.linkers)}")

# 检查特定片段类型
has_donor = identifier.has_donor('CN(C)c1ccccc1')
has_acceptor = identifier.has_acceptor('O=[N+]([O-])c1ccccc1')
```

### 3. 骨架匹配器 (ScaffoldMatcher)

```python
from fluo_utils import ScaffoldMatcher, create_default_library

matcher = ScaffoldMatcher()
library = create_default_library()

# 提取BM骨架
bm_smiles = matcher.get_bm_scaffold_smiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
print(f"BM骨架: {bm_smiles}")  # c1ccccc1

# 提取CSK骨架
csk_smiles = matcher.get_csk_scaffold_smiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
print(f"CSK骨架: {csk_smiles}")

# 计算相似度
sim = matcher.compute_tanimoto_similarity_smiles('c1ccccc1', 'c1ccncc1')
print(f"Tanimoto相似度: {sim}")

# 模糊匹配
matches = matcher.fuzzy_match('c1ccccc1', library, threshold=0.5)
for match in matches:
    print(f"{match['fragment_smiles']} - 类型: {match['fragment_type']} - 相似度: {match['similarity']:.2f}")
```

### 4. 分子处理器 (MoleculeProcessor)

```python
from fluo_utils import MoleculeProcessor, create_default_library

library = create_default_library()
processor = MoleculeProcessor(library)

# 处理单分子
mol, annotation = processor.process_molecule(
    'CN(C)c1ccccc1',
    output_sdf_path='output.sdf'
)

print(f"识别到 {annotation.total_fragments} 个功能片段")
print(f"BM骨架: {annotation.scaffolds[0].scaffold_smiles if annotation.scaffolds else 'None'}")

# 批量处理
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

### 识别分子

```bash
python fluo_utils.py identify "CN(C)c1ccccc1"
```

输出：
```
分子: CN(C)c1ccccc1
供体数量: 2
受体数量: 0
连接子数量: 0
总片段数: 2

供体片段:
  - c1ccccc1
  - [NX3;H2,H1;!$(NC=O)]
```

### 处理分子

```bash
python fluo_utils.py process "CN(C)c1ccccc1" -o output.sdf
```

### 批量处理

```bash
python fluo_utils.py batch input.csv output_dir/ --smiles-col smiles --id-col id
```

## 输入/输出格式

### 输入CSV格式

```csv
id,smiles,name
mol_001,CN(C)c1ccccc1,N,N-dimethylaniline
mol_002,O=[N+]([O-])c1ccccc1,nitrobenzene
mol_003,c1ccccc1C=Cc2ccccc2,stilbene
```

### 输出SDF格式

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

## 高级用法

### 自定义片段库

```python
from fluo_utils import FragmentLibrary

library = FragmentLibrary()

# 添加自定义供体
library.add_donor('c1ccc2c(c1)oc1ccccc12', 'dibenzofuran')
library.add_donor('c1ccc2c(c1)sc1ccccc12', 'dibenzothiophene')

# 添加自定义受体
library.add_acceptor('S(=O)(=O)N', 'sulfonamide')
library.add_acceptor('C(F)(F)(F)', 'trifluoromethyl')

# 添加自定义连接子
library.add_linker('C#CC#C', 'diacetylene')
library.add_linker('c1ccccc1C#Cc2ccccc2', 'tolane')
```

### 自定义骨架匹配

```python
from fluo_utils import ScaffoldMatcher

# 创建自定义匹配器
matcher = ScaffoldMatcher(fp_radius=3, fp_nbits=4096)

# 使用更高的相似度阈值
matches = matcher.fuzzy_match(
    scaffold_smiles='c1ccccc1',
    fragment_library=library,
    threshold=0.8
)
```

### 处理复杂分子

```python
from fluo_utils import MoleculeProcessor, create_default_library

library = create_default_library()
processor = MoleculeProcessor(library)

# 处理一个复杂的荧光分子
complex_smiles = 'CCN1C(=O)/C(=c2\\s/c(=C\\c3cc4c(c5ccccc35)N(c3ccc(C=C(c5ccccc5)c5ccccc5)cc3)C3CCCC43)c(=O)n2CC(=O)O)SC1=S'

mol, annotation = processor.process_molecule(
    complex_smiles,
    output_sdf_path='complex_mol.sdf'
)

print(f"总片段数: {annotation.total_fragments}")
print(f"供体: {len(annotation.donors)}")
print(f"受体: {len(annotation.acceptors)}")
print(f"连接子: {len(annotation.linkers)}")
```

## API参考

### FragmentLibrary

| 方法 | 说明 |
|------|------|
| `load_library(path)` | 加载片段库 |
| `save_library(path)` | 保存片段库 |
| `get_donors()` | 获取供体列表 |
| `get_acceptors()` | 获取受体列表 |
| `get_linkers()` | 获取连接子列表 |
| `add_donor(smarts, name)` | 添加供体 |
| `add_acceptor(smarts, name)` | 添加受体 |
| `add_linker(smarts, name)` | 添加连接子 |

### FragmentIdentifier

| 方法 | 说明 |
|------|------|
| `identify(smiles_or_mol)` | 识别分子中的功能片段 |
| `has_donor(smiles_or_mol)` | 检查是否包含供体 |
| `has_acceptor(smiles_or_mol)` | 检查是否包含受体 |
| `has_linker(smiles_or_mol)` | 检查是否包含连接子 |

### ScaffoldMatcher

| 方法 | 说明 |
|------|------|
| `get_bm_scaffold(mol)` | 提取BM骨架 |
| `get_bm_scaffold_smiles(smiles)` | 从SMILES提取BM骨架 |
| `get_csk_scaffold(mol)` | 提取CSK骨架 |
| `get_csk_scaffold_smiles(smiles)` | 从SMILES提取CSK骨架 |
| `compute_tanimoto_similarity(mol1, mol2)` | 计算Tanimoto相似度 |
| `fuzzy_match(scaffold_smiles, library, threshold)` | 模糊匹配功能片段 |

### MoleculeProcessor

| 方法 | 说明 |
|------|------|
| `split_to_scaffolds(mol, scaffold_type)` | 切分分子为骨架 |
| `process_molecule(smiles, output_sdf_path)` | 处理单分子 |
| `save_annotated_sdf(mol, annotation, output_path)` | 保存标注的SDF |
| `batch_process(input_csv, output_dir, ...)` | 批量处理 |

## 示例数据

工具包包含以下示例文件：

- `fluo_fragment_library.pkl` - 预构建的片段库
- `test_molecules.csv` - 测试分子列表
- `test_output.sdf` - 示例输出SDF

## 注意事项

1. **不增加原子**：分子处理过程不会在输入分子上增加原子
2. **保留结构完整性**：输出SDF保留原始分子结构
3. **片段标注**：SDF文件中的属性字段包含完整的片段标注信息
4. **批量处理**：支持大规模分子库的批量处理

## 许可证

MIT License
