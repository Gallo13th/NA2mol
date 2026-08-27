# SPEC 07 — 固定骨架迭代优化(精确复现 mol4)

> 依赖 SPEC 06 的 12 个含骨架产物。目标:把 Tanimoto 从 0.33 推到 >0.5 或精确 mol4。

## 1. 策略
12 个产物含 mol4 核心骨架(氰基二苯乙烯+胺),但多了装饰(甲氧基/羟基/哌啶)、供体不准(N(CCO)C→N(C)C)、原子数偏大(28-35 vs 27)。

**迭代种子优化**:提取产物的 mol4 骨架(去装饰)做纯骨架种子 → 固定骨架 → 严限原子数 → 模型只补少量 → 产物更接近 mol4 → 再提骨架迭代。

## 2. 骨架定义与提取
mol4 骨架 SMARTS(连续核心):
- 氰基二苯乙烯(受体+桥一体):`c1ccccc1C(C#N)=Cc2ccccc2`
- 胺供体:三级胺 `[NX3;H0]([#6])([#6])[#6]` 或放宽 `[NX3]`(连在骨架苯环上)

**提取流程**(对每个产物):
1. SMARTS 定位氰基二苯乙烯原子 + 连接的胺原子(并集)
2. 提取这些原子为子结构(**保留产物的 3D conformer 坐标**,用 RWMol 删非骨架原子或 `Chem.MolFragmentToSmiles`+手动建 conformer)
3. 骨架片段(约 15-20 重原子,带口袋 3D)= 种子核心

## 3. 迭代生成(每轮)

### 种子构造
- 取上轮产物的 mol4 骨架(或首轮用 12 产物的骨架)
- 骨架原子 = fixed_atom(fixed_atom_mask=True)
- **total_n 严格限制 25-27**(改 featurizer `random.randint(25,27)`,或 sample_by_seed 的 n_nodes)

### sample_by_seed 生成
- 固定骨架原子(坐标+类型来自骨架种子)
- 模型补 total_n - origin_n 个原子(约 5-10 个,装饰空间大减)
- sample_steps=100
- 每个骨架种子生成 ~30-50 产物

### 筛选与迭代
- 产物筛:含 mol4 骨架(氰基二苯乙烯)的
- 按 Tanimoto vs mol4 排序
- 取 top-K(Tanimoto 最高的若干)→ 提取它们的骨架 → 下轮种子
- 迭代 3-5 轮,或 Tanimoto>0.5,或收敛(不再提升)

## 4. 验收
1. 骨架提取:能从产物提取 mol4 骨架(氰基二苯乙烯+胺),保留 3D
2. 固定生成:sample_by_seed 固定骨架跑通,产物含骨架
3. **核心**:Tanimoto 是否随迭代提升?是否达 >0.5?是否有精确 mol4(Tanimoto>0.9 或 SMILES 相同)?
4. 装饰是否减少(原子数趋近 27、甲氧基/羟基减少)?

## 5. 配置
- 复用 run_generation.py 的模型加载/patch
- 原子数限制:临时改 featurizer `random.randint(25,27)`(测后可保留或恢复)
- 迭代脚本独立:`casestudy/8hzk/spec07_iterative/`
- 选 GPU

## 6. 不做
- 不改模型/checkpoint/采样权重
- 不泄露 known_mols(骨架从产物提取,产物是模型生成的)

## 7. 灵活
- 若固定骨架后模型补不出 N(CCO)C(供体不准),可从对接的 CCN 种子或 FluoDB 同类补充供体片段
- 若迭代收敛但未达 0.5,报告最终 Tanimoto 和最接近产物
