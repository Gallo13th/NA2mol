# SPEC 06 — 限制原子数生成 + 功能库检索 + 种子复现(主路径)+ 对接突破(同步)

> 丢弃之前所有分子池,从零开始。整合两条线攻克 known_mols 复现。

## 1. 目标
复现 mol4(或任一 known_mol)。主路径:限制原子数生成 1w → 功能库检索回收 → 种子生成。同步:FluoDB 片段对接获得 C=C(C#N) 的口袋 3D(突破模型不生成的受体)。

## 2. 主路径

### 阶段A:限制原子数 + 自由生成 1w 分子(后台,长时间)
**原子数限制(关键新点)**:known_mols 是 22-27 重原子。之前按口袋体积采样生成偏大分子。改 `dataloader/featurizer.py` 的 `GeomSeedMolOnlyPocket.__getitem__`:冷启动分支 `total_n = random.randint(20, 28)`(覆盖 known_mols 范围),替代 `max(6, sample_atom_num(...))`。
- 配置:冷启动 `seed_used=False`,**不带** fluor_guidance(纯原料获取),feat_mode=generalfeat
- 规模:`total_target=10000`(达标目标),`iteration` 跑到 100 轮(产出约 1w 原始分子)
- 后台运行(setsid/nohup)
- 产出:从 `iter_*/raw_output/` 收集所有原始分子(约 1w)

### 阶段B(同步):FluoDB C=C(C#N) 片段对接到 8HZK 口袋
- 从功能 A 库取含 `C=C(C#N)` 的片段(如氰基乙烯、氰基丙烯酸 `C=C(C#N)C(=O)O`)
- 用 `util/smina.static` 对接到 `casestudy/8hzk/8HZK_pocket.pdb`(dock 模式,取 top poses)
- 获得口袋上下文 3D 的 C=C(C#N) 片段,存为 SDF
- 这解决"模型不生成 C=C(C#N)"——从外部引入,对接获 3D

### 阶段C:功能库检索分析(1w 生成完成后)
- 用过滤后的 D/A/π 功能库匹配 1w 原始分子
- 回收完整功能片段(带口袋 3D)
- **统计覆盖**(重点):C=C(C#N) 出现次数?stilbene?噻吩?三级胺?D+A+π 齐全的多少?
- 对比限制原子数后,缺失构件是否改善(限制大小可能让模型生成不同的、更接近 known_mols 的结构)

### 阶段D:种子生成(复现)
种子来源(混合):
1. **对接的 C=C(C#N) 片段**(阶段B,口袋3D)
2. **1w 回收的关键片段**(含 stilbene/胺的,阶段C)
组合/选择种子 → `sample_by_seed`(固定功能单元原子,模型补全)→ 产物
- 优先:用对接的 C=C(C#N) 做受体种子 + 回收的 stilbene+胺做共种子,模型组装
- 筛选:Tanimoto vs mol4、子结构(stilbene+C=C(C#N)+胺三特征)、Murcko

## 3. 验收
1. 原子数限制生效:1w 分子的重原子数分布集中在 20-28
2. 对接:C=C(C#N) 片段成功对接到口袋,获得合理 3D
3. 检索:1w 分子的功能片段覆盖统计(对比之前的 3039)
4. **核心**:种子生成产物中是否有复现 mol4 的(Tanimoto>0.5 或三特征齐全)

## 4. 不做
- 不改模型/checkpoint/采样网络权重
- 不用之前的分子池(从零生成)

## 5. 时间
- 阶段A:100 轮 ≈ 8-12 小时(后台)
- 阶段B:对接快(分钟级),生成跑时同步完成
- 阶段C/D:1w 生成完成后(隔天)
