# SPEC 03 — 聚焦库 + 种子循环的复现验证

> 机制全部就绪(SPEC 02 已实现种子 sumlogp 排序)。本 spec 是**配置 + 运行 + 验证**,无新核心代码。

## 1. 目标
验证"聚焦高频构件库 + 种子循环迭代"能否让产物收敛向 known_mols 同类(对比 SPEC 02 端到端测试的冷启动结果:Tanimoto≤0.16、无共享骨架)。

## 2. 依据
- 频率筛选探查:freq≥50 保留 69 个构件(1853→69,削减 96%),4 个 known_mols 全 D/A/π 覆盖。
- 种子循环从未真正跑过(上次 seed_used=False 且 iter≤8)。种子来自**口袋内生成的片段**(3D 构象正确,符合 sample_by_seed 要求),sumlogp 排序让高置信荧光构件优先做种子。
- 构件 3D 构象不可直接做种子(孤立生成、无口袋上下文),故用方案 A 而非 B。

## 3. 执行步骤

### 3.1 生成聚焦库
脚本从 `targeted_fluorophore.pkl` 筛选 `freq>=50` 的构件,存为 `fragment_library/targeted_fluorophore_focused.pkl`(保持原结构:donors/acceptors/bridges + MoleculeRefiner 兼容视图)。验证 4 个 known_mols 仍全覆盖。

### 3.2 配置改动(全部临时,记录原值,测后恢复)
| 位置 | 原值 | 临时值 | 目的 |
|---|---|---|---|
| `genbypkt.py:675` `_FLUOR_GUIDANCE_LIB` | targeted_fluorophore.pkl | targeted_fluorophore_focused.pkl | 接入聚焦库(或在 run_generation 里设 `genbypkt._FLUOR_GUIDANCE_LIB`) |
| `genbypkt.py:797` total_target | 10000 | 40 | 控制规模 |
| `genbypkt.py:809` iteration 上限 | 150 | 22 | 控制时间 |
| `genbypkt.py:818` 冷启动门槛 | iteration <= 15 | iteration <= 8 | 加快种子模式启动(种子库空时自动回冷启动,安全) |
| `run_generation.py:177` seed_used | False | True | 启用种子循环 |

### 3.3 运行
```bash
export CUDA_VISIBLE_DEVICES=<空闲GPU>
nohup python casestudy/8hzk/run_generation.py --use_fluor_guidance --use_fluor_replace \
    > casestudy/8hzk/run_seedcycle.log 2>&1 &
```

### 3.4 监控重点
- iter 1-8(冷启动):种子库积累情况、完整 D/A/π 片段数
- iter 9-22(种子模式):**确认走了 sample_by_seed 路径**(非冷启动)、种子库大小变化、种子 _SeedScore 是否为 sumlogp 值
- 每轮产物:完整性、替换触发、smina 成功
- 任何错误立即 kill 报告

### 3.5 分析产物
对 `output/8hzk_output/dummy_ligand/dummy_ligand.sdf`:
- 分子数、affinity 分布、D/A/π 完整性
- **与 4 个 known_mols 的 ECFP4 Tanimoto**(max/mean,>0.4 的数量)——对比冷启动的 ≤0.16
- Murcko 骨架与 known_mols 的重叠
- 种子库 `_finalseed.sdf` 的种子数、top 种子的 sumlogp/smiles
- top-5 代表产物

## 4. 验收
1. 种子循环真正启动(iter>8 后日志显示 sample_by_seed 而非冷启动)
2. 种子 _SeedScore 为 sumlogp(非旧 LBE/QED 分)
3. 聚焦库 69 构件、known_mols 全覆盖
4. 产物 Tanimoto **是否优于冷启动的 0.16**(核心指标)
5. 所有配置已恢复原值

## 5. 恢复(硬性)
测后必须恢复:genbypkt.py 的 4 处改动 + run_generation.py 的 seed_used。Read 确认。

## 6. 时间预期
22 轮 × (推理 + 100×smina + 片段smina) ≈ 1.5-2.5 小时。若过慢,subagent 可将 iteration 上限降到 18(但种子模式至少跑 8 轮才有意义)。
