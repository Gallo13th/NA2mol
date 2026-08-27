# SPEC 02 — 基于概率的荧光构件评分与分阶段生成引导

> 依赖 SPEC 01 产出的 `fragment_library/targeted_fluorophore.pkl`。本 spec 改造 `genbypkt.py` 的生成后处理流程,**不改模型、不改采样逻辑、不改 checkpoint**。

## 1. 背景与核心思想

BFN 采样输出每个原子位置在 13 类原子类型上的概率分布 `p0_h_final` `[N,13]`(`bfn4edmol.py:2290`)。当前 `sampletraj_to_sdf` 只取 argmax、**丢弃概率矩阵**。

核心机制:**对生成骨架上匹配到某荧光构件 F 的那组原子,计算 sumlogp = Σ log P_i(type_i)**。sumlogp 越高,模型越确信生成了该构件。sumlogp 同时驱动**片段排序**与**替换选型**(匹配上的骨架位置替换为 sumlogp 最高的构件构象)。构件频率不参与(构件库仅作"待验证假设集")。

实测耗时(benchmark):全库 1853 构件扫描单分子 ~5ms,100 分子/轮 ~0.6s。**无需分桶优化,直接全库扫描。**

## 2. 硬性约束

1. **不改模型权重/网络/采样**;只改生成后处理(`sampletraj_to_sdf`、`generate` 主循环)。
2. **不改 checkpoint**。
3. 全部新增逻辑以"可开关"方式接入(命令行 flag),默认行为可回退到原流程。
4. 构件库只读 `fragment_library/targeted_fluorophore.pkl`,不再读 FluoDB/known_mols。
5. sumlogp 用于**片段排序**与**替换选型**,不用于分子级阈值判断(分子级用纯二值完整性)。

## 3. 改造后的生成流程

对照 `genbypkt.py:generate` 现有步骤(行号为现状):

| 步骤 | 现状 | 改造后 |
|---|---|---|
| ②采样 | `sample_traj[-1]` 含 `[坐标, 概率P]`,但 `sampletraj_to_sdf` 丢弃 P | **保留 P**,随分子传递 |
| 建骨架 | `sampletraj_to_sdf`→`reconstruct_from_generated`(碳模板建键) | 同,额外保留 P 与 orig_idx 映射 |
| — | (无) | **【新增】匹配**:全库扫描,sumlogp + 完整性标记 |
| — | (无) | **【新增】替换**:命中位置用构件3D构象替换(MoleculeRefiner) |
| ③minimize | smina `Minimize` | smina `Minimize`(对替换后结构) |
| ④筛选 | `aff<0` | `aff<0` **AND** D/A/π 三类齐全 |
| ⑤种子 | BRICS切片段→NCI→`calc_seed_quality` | BRICS切片段→**sumlogp排序**(替代 `_SeedScore`) |

关键:**匹配与替换插入在"建骨架后、smina前"**——此时坐标是生成的原始坐标(未被 smina 扰动),匹配稳定;替换后统一 smina relax。

## 4. 核心机制详述

### 4.1 概率矩阵保留与原子索引映射(前提)
- `sampletraj_to_sdf` / `sampletraj_to_sdf_byseed` 在重建分子时,**保留输入的概率矩阵 P `[N,13]`**。
- 重建过程中(OpenBabel AddHs/RemoveHs/DeleteHydrogens 会改原子顺序),用 `SetProp('orig_idx')` 建立"重建后原子 → 生成原子"的映射(参考 `mol_refine.py:851` 的 orig_idx 做法)。
- 输出的每个 SDF 分子,在内存中附带其 P 与 orig_idx(可通过返回值或临时 sidecar 文件传递,实现自选)。

### 4.2 全库匹配 + sumlogp
对每个重建分子:
1. 加载构件库全部构件(预编译 SMARTS,一次性 0.06s)。
2. 对每个构件 `pat`:`mol.GetSubstructMatches(pat)` 取所有命中。
3. 对每个命中原子集 `m`:`sumlogp = Σ_{i∈m} log( P[orig_idx[i]][type_i] + ε )`,其中 `type_i` 为骨架原子 i 还原后的真实类型在 13 类中的下标(查 `MAP_ATOM_TYPE_AROMATIC_TO_INDEX`)。
4. 记录:每类(D/A/π)的命中构件列表 + 各自 sumlogp。
5. **完整性**:`is_complete_dpa = (donor命中数>0) AND (acceptor命中数>0) AND (bridge命中数>0)`。

### 4.3 替换(sumlogp 选型 + 几何叠合)
- **替换选型由 sumlogp 决定**:对分子的每个骨架位置,在所有匹配命中的构件中,**选 sumlogp 最高的那个构件**作为替换源。
- 选定构件后,用其 3D molblock,通过 MoleculeRefiner 的刚性叠合(`mol_refine.py:667 get_atom_mapping`、`694 calculate_rmsd_and_transform`、`814 apply_replacement`)计算 SVD 变换矩阵,替换骨架对应原子坐标。
- 即:**sumlogp 决定"换成谁",几何叠合决定"怎么对齐"**。MoleculeRefiner 的 RMSD 仅用于验证叠合质量(可记录但不作为选型标准)。
- 若 sumlogp 最高的构件无 3D molblock(罕见),退到次高;均无则跳过该位置替换。
- 替换后整体交 smina。

> 说明:替换用已知荧光构象精修生成骨架。若某分子无任何匹配命中,跳过替换直接 smina(降级不报错)。

### 4.4 分子级筛选(二值完整性)
`good` 分子条件:`affinity < 0` **AND** `is_complete_dpa == True`。
不设 sumlogp 阈值。

### 4.5 种子排序(sumlogp)
切出的每个功能片段(BRICS),按其在母分子概率矩阵上的 `sumlogp` 排序(降序),替代原 `_SeedScore`(LBE+QED+NCI+SA)。种子库维护逻辑不变(Top-100 裁剪等)。

## 5. 接入方式(可开关)

`genbypkt.py` 新增命令行参数:
- `--use_fluor_guidance`:启用荧光引导(匹配+完整性筛选+sumlogp种子排序)。默认 False(回退原流程)。
- `--use_fluor_replace`:启用构件替换。默认 False。仅当 `--use_fluor_guidance` 时生效。

`generate()` 函数签名扩展对应参数,默认值保持原行为不变。

## 6. 新增/修改文件

| 文件 | 动作 | 说明 |
|---|---|---|
| `casestudy/8hzk/fluor_guidance.py` | 新增 | 核心逻辑:`FluorGuidance` 类(加载库、匹配、sumlogp、完整性、替换委托) |
| `util/mol_utils.py` | 修改 | `sampletraj_to_sdf` / `sampletraj_to_sdf_byseed` 保留 P + orig_idx(向后兼容:无 P 时走原逻辑) |
| `genbypkt.py` | 修改 | `generate()` 主循环接入匹配/替换/完整性/sumlogp种子(全可开关) |
| `casestudy/8hzk/verify_guidance.py` | 新增 | 验收脚本 |

## 7. 验收标准(`verify_guidance.py`)

1. **概率保留**:对一个模拟采样输出(坐标+P),走改造后 `sampletraj_to_sdf`,能取回与重建原子对应的 P 切片。
2. **sumlogp 正确性**:对 4 个 known_mol(赋予模拟 P),全库匹配后 D/A/π 三类均有命中且 `is_complete_dpa=True`。
3. **完整性筛选**:模拟一批分子(含完整与残缺 D-π-A),筛选结果正确(只留三类齐全者)。
4. **替换可用**:对一个 known_mol 触发替换流程,不崩溃,输出结构可被 smina 接受。
5. **种子排序**:切片段后,sumlogp 排序结果合理(含完整受体的片段排在无受体片段前)。
6. **可回退**:`--use_fluor_guidance` 关闭时,`generate()` 行为与改造前完全一致(可用 sp1 案例回归)。
7. **性能**:单分子匹配+完整性 < 10ms;替换+smina 整体不显著慢于原流程(替换开销应远小于 smina)。

## 8. 不做范围
- 不改模型/采样/checkpoint
- 不做 known_mols 相似度度量(后续 spec)
- 不做口袋对接评分
- 不实现路线2(纯概率最优匹配)——当前路线1(argmax 匹配 + 概率评分)已满足;路线2留作未来增强

## 9. 运行方式
```bash
cd /home/data/users/gaoletian/rna2smol
# 荧光引导生成(8HZK)
python casestudy/8hzk/run_generation.py --use_fluor_guidance --use_fluor_replace
# 验收
python casestudy/8hzk/verify_guidance.py
# 回退原流程
python casestudy/8hzk/run_generation.py
```
