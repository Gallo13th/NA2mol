#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FluorUtil 2 - 核心模块
======================

核心功能模块，包含以下子模块:

1. fluo_utils.py          - 基础功能 (v1.x)
2. fluo_utils_v2.py       - 环系整体标记策略 (v2.x)
3. fragment_identifier.py - 片段识别
4. scaffold_matcher.py    - 骨架匹配
5. molecule_processor.py  - 分子处理
6. visualization.py       - 可视化
7. fluo_fragment_library.py - 片段库管理

使用建议:
---------
- 对于基础功能，使用 fluo_utils 中的类
- 对于高级环系标记功能，使用 fluo_utils_v2 中的类
- 所有模块都通过 fluor_util_2 包直接导入

示例:
-----
>>> from fluor_util_2 import FragmentLibrary, MoleculeVisualizer
>>> from fluor_util_2 import AdvancedFragmentLibrary  # v2.x
"""

# 尝试导入所有核心模块
from .fluo_utils import (
    FragmentLibrary,
    FragmentIdentifier,
    ScaffoldMatcher,
    MoleculeProcessor,
    FragmentMatch,
    ScaffoldInfo,
    MoleculeAnnotation,
    create_default_library,
    load_library,
    identify_fragments,
    process_molecule,
)

from .fluo_utils_v2 import (
    AdvancedFragmentLibrary,
    FragmentMatch as FragmentMatchV2,
    RingSystemPattern,
    MoleculeAnnotation as MoleculeAnnotationV2,
)

from .fragment_identifier import (
    load_fragment_library,
    identify_fragments as identify_fragments_v2,
    batch_identify,
    get_fragment_statistics,
    preprocess_fragments,
    calculate_match_score,
)

from .scaffold_matcher import (
    ScaffoldMatcher as ScaffoldMatcherV2,
    create_fragment_library,
)

from .molecule_processor import (
    process_molecule as process_molecule_v2,
    batch_process,
    split_to_bm_scaffolds,
    save_annotated_sdf,
    match_fragments_on_scaffold,
    read_fragment_library,
    create_default_fragment_library,
)

from .visualization import (
    MoleculeVisualizer,
    visualize_molecule,
    visualize_fragments,
    visualize_grid,
    visualize_sdf_annotated,
)

from .fluo_fragment_library import (
    FragmentLibrary as FragmentLibraryModule,
)

__all__ = [
    # V1.x
    "FragmentLibrary",
    "FragmentIdentifier",
    "ScaffoldMatcher",
    "MoleculeProcessor",
    "FragmentMatch",
    "ScaffoldInfo",
    "MoleculeAnnotation",
    "create_default_library",
    "load_library",
    "identify_fragments",
    "process_molecule",
    # V2.x
    "AdvancedFragmentLibrary",
    "FragmentMatchV2",
    "RingSystemPattern",
    "MoleculeAnnotationV2",
    # Fragment Identifier
    "load_fragment_library",
    "identify_fragments_v2",
    "batch_identify",
    "get_fragment_statistics",
    "preprocess_fragments",
    "calculate_match_score",
    # Scaffold Matcher
    "ScaffoldMatcherV2",
    "create_fragment_library",
    # Molecule Processor
    "process_molecule_v2",
    "batch_process",
    "split_to_bm_scaffolds",
    "save_annotated_sdf",
    "match_fragments_on_scaffold",
    "read_fragment_library",
    "create_default_fragment_library",
    # Visualization
    "MoleculeVisualizer",
    "visualize_molecule",
    "visualize_fragments",
    "visualize_grid",
    "visualize_sdf_annotated",
    # Fragment Library
    "FragmentLibraryModule",
]
