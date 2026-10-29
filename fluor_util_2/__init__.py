#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FluorUtil 2 - 荧光分子功能片段库工具包
======================================

用于荧光分子功能片段识别、骨架匹配和分子处理的高级工具包。

目录结构:
---------
fluor_util_2/
├── core/                   # 核心模块
│   ├── fluo_utils.py       # 原始版本 (v1.x)
│   ├── fluo_utils_v2.py    # 新版本 - 环系整体标记策略 (v2.x)
│   ├── fragment_identifier.py
│   ├── scaffold_matcher.py
│   ├── molecule_processor.py
│   └── visualization.py
├── data/                   # 数据文件
│   ├── fluo_fragment_library.pkl
│   └── fluo_fragment_library_combined.pkl
├── examples/               # 示例脚本
│   ├── example_usage.py
│   └── visualization_example.py
└── tests/                  # 测试文件

快速开始:
---------
>>> from fluor_util_2 import FragmentLibrary, FragmentIdentifier
>>> 
>>> # 加载片段库
>>> library = FragmentLibrary()
>>> 
>>> # 识别分子片段
>>> from fluor_util_2 import identify_fragments
>>> result = identify_fragments('CN(C)c1ccccc1', library)
>>> print(f"供体数量: {result['summary']['total_donors']}")

>>> # 可视化分子
>>> from fluor_util_2 import MoleculeVisualizer
>>> visualizer = MoleculeVisualizer()
>>> img = visualizer.visualize_fragment_result('CN(C)c1ccccc1', result)
>>> img.save('output.png')

作者: AI Assistant
版本: 2.0.0
"""

import os

# 从 core.fluo_utils 导入原始版本 (v1.x)
try:
    from .core.fluo_utils import (
        FragmentLibrary,
        FragmentIdentifier,
        ScaffoldMatcher,
        MoleculeProcessor,
        FragmentMatch,
        ScaffoldInfo,
        MoleculeAnnotation,
        create_default_library,
        load_library,
        identify_fragments as identify,
        process_molecule,
    )
    _has_fluo_utils = True
except ImportError as e:
    _has_fluo_utils = False
    # print(f"Debug: Could not import fluo_utils: {e}")

# 从 core.fluo_utils_v2 导入新版本 (v2.x) - 环系整体标记策略
try:
    from .core.fluo_utils_v2 import (
        AdvancedFragmentLibrary,
        FragmentMatch as FragmentMatchV2,
        RingSystemPattern,
        MoleculeAnnotation as MoleculeAnnotationV2,
    )
    _has_fluo_utils_v2 = True
except ImportError as e:
    _has_fluo_utils_v2 = False
    # print(f"Debug: Could not import fluo_utils_v2: {e}")

# 从 core.fragment_identifier 导入函数
try:
    from .core.fragment_identifier import (
        load_fragment_library,
        identify_fragments,
        batch_identify,
        get_fragment_statistics,
        preprocess_fragments,
        calculate_match_score,
    )
    _has_fragment_identifier = True
except ImportError as e:
    _has_fragment_identifier = False
    # print(f"Debug: Could not import fragment_identifier: {e}")

# 从 core.scaffold_matcher 导入类
try:
    from .core.scaffold_matcher import (
        ScaffoldMatcher as ScaffoldMatcherV2,
        create_fragment_library,
    )
    _has_scaffold_matcher = True
except ImportError as e:
    _has_scaffold_matcher = False
    # print(f"Debug: Could not import scaffold_matcher: {e}")

# 从 core.molecule_processor 导入函数
try:
    from .core.molecule_processor import (
        process_molecule as process_mol,
        batch_process,
        split_to_bm_scaffolds,
        save_annotated_sdf,
        match_fragments_on_scaffold,
        read_fragment_library,
        create_default_fragment_library,
    )
    _has_molecule_processor = True
except ImportError as e:
    _has_molecule_processor = False
    # print(f"Debug: Could not import molecule_processor: {e}")

# 从 core.visualization 导入可视化类
try:
    from .core.visualization import (
        MoleculeVisualizer,
        visualize_molecule,
        visualize_fragments,
        visualize_grid,
        visualize_sdf_annotated,
    )
    _has_visualization = True
except ImportError as e:
    _has_visualization = False
    # print(f"Debug: Could not import visualization: {e}")

# 从 core.fluo_fragment_library 导入
try:
    from .core.fluo_fragment_library import (
        FragmentLibrary as FragmentLibraryModule,
    )
    _has_fragment_library_module = True
except ImportError as e:
    _has_fragment_library_module = False
    # print(f"Debug: Could not import fluo_fragment_library: {e}")

__version__ = "2.0.0"
__author__ = "AI Assistant"

# 导出列表
__all__ = []

# V1.x 版本导出
if _has_fluo_utils:
    __all__.extend([
        "FragmentLibrary",
        "FragmentIdentifier", 
        "ScaffoldMatcher",
        "MoleculeProcessor",
        "FragmentMatch",
        "ScaffoldInfo",
        "MoleculeAnnotation",
        "create_default_library",
        "load_library",
        "identify",
        "process_molecule",
    ])

# V2.x 版本导出（环系整体标记策略）
if _has_fluo_utils_v2:
    __all__.extend([
        "AdvancedFragmentLibrary",
        "FragmentMatchV2",
        "RingSystemPattern",
        "MoleculeAnnotationV2",
    ])

if _has_fragment_identifier:
    __all__.extend([
        "load_fragment_library",
        "identify_fragments",
        "batch_identify",
        "get_fragment_statistics",
        "preprocess_fragments",
        "calculate_match_score",
    ])

if _has_scaffold_matcher:
    __all__.extend([
        "ScaffoldMatcherV2",
        "create_fragment_library",
    ])

if _has_molecule_processor:
    __all__.extend([
        "process_mol",
        "batch_process",
        "split_to_bm_scaffolds",
        "save_annotated_sdf",
        "match_fragments_on_scaffold",
        "read_fragment_library",
        "create_default_fragment_library",
    ])

if _has_visualization:
    __all__.extend([
        "MoleculeVisualizer",
        "visualize_molecule",
        "visualize_fragments",
        "visualize_grid",
        "visualize_sdf_annotated",
    ])

if _has_fragment_library_module:
    __all__.extend([
        "FragmentLibraryModule",
    ])


def get_module_info():
    """
    获取模块信息
    
    Returns:
        dict: 包含模块版本和可用组件的信息
    """
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'version': __version__,
        'author': __author__,
        'available_modules': {
            'fluo_utils (v1.x)': _has_fluo_utils,
            'fluo_utils_v2 (v2.x)': _has_fluo_utils_v2,
            'fragment_identifier': _has_fragment_identifier,
            'scaffold_matcher': _has_scaffold_matcher,
            'molecule_processor': _has_molecule_processor,
            'visualization': _has_visualization,
            'fluo_fragment_library': _has_fragment_library_module,
        },
        'data_files': {
            'fluo_fragment_library.pkl': os.path.exists(
                os.path.join(_current_dir, 'data', 'fluo_fragment_library.pkl')
            ),
            'fluo_fragment_library_combined.pkl': os.path.exists(
                os.path.join(_current_dir, 'data', 'fluo_fragment_library_combined.pkl')
            ),
        }
    }


def get_data_path(filename: str = None) -> str:
    """
    获取数据文件路径
    
    Args:
        filename: 数据文件名（可选）
        
    Returns:
        str: 数据目录路径或完整文件路径
        
    Example:
        >>> pkl_path = get_data_path('fluo_fragment_library.pkl')
        >>> print(pkl_path)
        '/path/to/fluor_util_2/data/fluo_fragment_library.pkl'
    """
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_current_dir, 'data')
    if filename:
        return os.path.join(data_dir, filename)
    return data_dir


def quick_visualize(smiles: str, output_path: str = None):
    """
    快速可视化分子
    
    Args:
        smiles: 分子SMILES字符串
        output_path: 输出文件路径（可选）
        
    Returns:
        PIL.Image 或 保存路径
        
    Example:
        >>> img = quick_visualize('c1ccccc1', 'benzene.png')
    """
    if not _has_visualization:
        raise ImportError("可视化模块不可用，请确保 core/visualization.py 存在")
    
    visualizer = MoleculeVisualizer()
    return visualizer.visualize_single(smiles, title=smiles, save_path=output_path)
