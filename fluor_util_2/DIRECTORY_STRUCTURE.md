# FluorUtil 2 目录结构说明

## 目录概览

```
fluor_util_2/
├── core/                           # 核心模块
│   ├── __init__.py
│   ├── fluo_utils.py              # 基础版本 (v1.x)
│   ├── fluo_utils_v2.py           # 高级版本 - 环系整体标记策略 (v2.x)
│   ├── fragment_identifier.py     # 片段识别模块
│   ├── scaffold_matcher.py        # 骨架匹配模块
│   ├── molecule_processor.py      # 分子处理模块
│   ├── visualization.py           # 可视化模块
│   └── fluo_fragment_library.py   # 片段库管理模块
│
├── data/                          # 数据文件
│   ├── fluo_fragment_library.pkl              # 基础片段库
│   └── fluo_fragment_library_combined.pkl     # 组合片段库 (156 donors, 57 acceptors, 59 linkers)
│
├── examples/                      # 示例脚本
│   ├── example_usage.py           # 基础使用示例
│   └── visualization_example.py   # 可视化示例
│
├── tests/                         # 测试文件
│   ├── annotated_molecules.sdf
│   ├── test_*.sdf/csv
│   ├── batch_results/
│   ├── sdf_visualization/
│   └── visualization_output/
│
├── __init__.py                    # 包初始化
├── README.md                      # 项目说明
├── QUICKSTART.md                  # 快速开始指南
├── PROJECT_SUMMARY.md             # 项目总结
├── DIRECTORY_STRUCTURE.md         # 本文件
└── 最终交付清单.md
```

## 核心模块说明

### 1. fluo_utils.py (v1.x)
基础功能模块，包含：
- `FragmentLibrary`: 基础片段库类
- `FragmentIdentifier`: 片段识别器
- `ScaffoldMatcher`: 骨架匹配器
- `MoleculeProcessor`: 分子处理器
- `MoleculeAnnotation`: 分子标注数据类

### 2. fluo_utils_v2.py (v2.x) ⭐ 新
高级功能模块，基于"环系整体标记"策略：
- `AdvancedFragmentLibrary`: 高级片段库类
- `RingSystemPattern`: 环系模式数据类
- 支持咔唑、三苯胺、BODIPY等复杂环系的整体标记

### 3. fragment_identifier.py
片段识别模块：
- `identify_fragments()`: 识别分子中的功能片段
- `batch_identify()`: 批量识别
- `load_fragment_library()`: 加载片段库

### 4. scaffold_matcher.py
骨架匹配模块：
- `ScaffoldMatcher`: 骨架匹配器类
- BM骨架和CSK骨架提取
- 模糊匹配功能

### 5. molecule_processor.py
分子处理模块：
- `process_molecule()`: 处理单个分子
- `batch_process()`: 批量处理
- `save_annotated_sdf()`: 保存标注的SDF文件

### 6. visualization.py
可视化模块：
- `MoleculeVisualizer`: 分子可视化器类
- 支持片段高亮显示（绿色=供体，红色=受体，蓝色=连接子）
- 网格可视化功能

## 快速开始

### 基础用法 (v1.x)

```python
from fluor_util_2 import FragmentLibrary, FragmentIdentifier

# 创建片段库
library = FragmentLibrary()

# 识别分子片段
from fluor_util_2 import identify_fragments
result = identify_fragments('CN(C)c1ccccc1', library)
print(f"供体: {result['summary']['total_donors']}")
print(f"受体: {result['summary']['total_acceptors']}")
```

### 高级用法 (v2.x) - 环系整体标记

```python
from fluor_util_2 import AdvancedFragmentLibrary

# 创建高级片段库
lib_v2 = AdvancedFragmentLibrary()

# 使用环系整体标记功能
# ... (详见 fluo_utils_v2.py 文档)
```

### 可视化

```python
from fluor_util_2 import MoleculeVisualizer, identify_fragments

# 识别片段
result = identify_fragments('CN(C)c1ccccc1', library)

# 可视化
visualizer = MoleculeVisualizer()
img = visualizer.visualize_fragment_result('CN(C)c1ccccc1', result)
img.save('output.png')
```

## 数据文件

### fluo_fragment_library.pkl
基础片段库，包含预定义的SMARTS模式。

### fluo_fragment_library_combined.pkl ⭐ 新
扩展片段库，包含：
- 156 个供体片段
- 57 个受体片段
- 59 个连接子片段

使用方式：
```python
import pickle

with open('data/fluo_fragment_library_combined.pkl', 'rb') as f:
    library = pickle.load(f)
    
print(f"Donors: {len(library['donors'])}")
print(f"Acceptors: {len(library['acceptors'])}")
print(f"Linkers: {len(library['linkers'])}")
```

## 导入参考

### 基础类
```python
from fluor_util_2 import (
    FragmentLibrary,        # 基础片段库 (v1.x)
    AdvancedFragmentLibrary, # 高级片段库 (v2.x)
    FragmentIdentifier,
    ScaffoldMatcher,
    MoleculeProcessor,
    MoleculeVisualizer,
)
```

### 函数
```python
from fluor_util_2 import (
    identify_fragments,
    load_fragment_library,
    batch_identify,
    visualize_molecule,
    visualize_grid,
)
```

### 工具函数
```python
from fluor_util_2 import (
    get_module_info,   # 获取模块信息
    get_data_path,     # 获取数据文件路径
    quick_visualize,   # 快速可视化
)
```

## 版本说明

- **v1.x**: 基础功能，使用简单SMARTS匹配
- **v2.x**: 高级功能，使用"环系整体标记"策略，更适合复杂荧光分子

建议：对于简单分子使用 v1.x，对于包含咔唑、BODIPY等复杂结构的分子使用 v2.x。
