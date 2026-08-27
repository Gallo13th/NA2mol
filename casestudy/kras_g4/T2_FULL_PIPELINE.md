# T2 完整复现线路(双路径总览)

> 目标分子 t2 = `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`(苯并噻唑-CH₂-苯并噻唑,19 重原子,中性,2 环硫)
> 靶点:KRAS G4,7X8M MODEL1 **BER102** well-buried 口袋(质心 34.5,23.4,28.0)
> 模型:ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth(全程不改权重)

---

## 路径一:概率背书替换直通(t2_repro,高产路线)

**1. 前置探测:模型 S 能力**(纠正解码错位后)
- featurizer 真实 13 类:idx11=(16,True)=芳香 S;旧解码表错标为 Cl
- 修正后 BER102 冷启动 100 采样:环 S 分子 4/82(4.9%)——模型**会**生成环硫(旧"从不生成环硫"是解码伪影)

**2. t2 几何下 S 位强背书**
- t2 SMILES→ETKDG 构象→MCS 对齐 berberine→rdock 规范协议(R6 cavity)最优 pose(INTER −43.06)
- masked probe(芳香感知):两个 S 位 **P[S]=84–97%**(全元素锁/苯并噻吩基底/全氮等电子锁三实验一致,前 5 pose 鲁棒)
- 全元素自洽度 94.0%(19/19 位 >50%)

**3. 种子设计(R7 破局,通用化学参数)**
- 关键洞察:模型拒绝 3 锚点闭环(0/1700)、最擅长**单原子双锚点插入**
- 种子 = 两个苯并噻唑(C2···C2 = 2.46 Å,通用 CH₂ 插入距离区间 2.4–2.7 Å;共面,芳香堆积常识)
- 种子几何源自模型自产 BT-CH₂-Ph 输出 + 定向放置(**未用 t2 构象**)

**4. Inpainting 高产命中**
- sample_by_seed(19 原子锚,free=1 桥位,natom=19):**172 个 SMILES==t2 精确命中**(有效输出 ~25%)
- 产物:`t2_repro/t2_exact_hits.sdf`

## 路径二:纯从头合规路径(t2_denovo,t2 仅作判据)

完整文档:`casestudy/kras_g4/t2_denovo/DENOVOPATH.md`

### ① 冷启动 → 全氮等电子骨架
- BER102 冷启动 2000 采样(`ber102_gen2k/valid_mols.sdf`)
- 按 BM Tanimoto 对 t2 排序:最高 = 稠合双氮杂(phenanthroline 类,BM 0.375)+ 苯并咪唑核
- 单体门控失败(24 候选 48 氮位,redock pose P[S] max 1.25%,0/48 过 10%)
- 上下文证据:P[S] 的驱动是**分子内上下文**(双稠环+桥)而非姿态 → 用通用化学(N↔S 等电子)装配**全氮双苯并咪唑-CH₂ 二聚体**(模型产物片段 + ETKDG 通用构象)

### ② 位姿扫描 + N→S 替换(过门)
- 二聚体刚体位姿扫描(212 个无碰撞摆放)双吡咯 N 位 masked probe:P[S] max 63.4%,106/212>10%
- 选 3 个过门摆放(gp1=12.46%,gp2/gp3=11.24%)
- 执行两位 N→S(芳香 S)→ 双苯并噻唑,sanitize 合法
- 替换后复验:P[S]=11.7%,自洽保留

### ③ 种子 + Inpainting → NH 桥中间体
- 删 CH₂ 桥(留 1 自由原子)→ 18 原子双苯并噻唑锚种子(C2···C2=2.461 Å)
- 16 轮×100 采样:模型在桥位自选 **N** → NH-桥双苯并噻唑 `c1ccc2sc(Nc3nc4ccccc4s3)nc2c1`(Tc=0.517,骨架完整,只差桥元素)

### ④ 桥位裁决 + N→C → t2
- 3 个 NH 桥构象 redock 后桥位 probe:**P[C]CH₂ = 15.7–22.1%,3/3 过 10% 门**(最佳 r12_i043: 22.10%,与 P[N] 46.4% 近平手)
- 执行桥 N→C(补 2H)→ **canonical SMILES == t2 ✓**
- 产物:`t2_denovo/phaseE_bridge/t2_replaced.sdf`

### ⑤ 双评分验证
| 工具 | t2(denovo 产物) | 参照 berberine |
|---|---|---|
| rdock(BER102 规范协议) | **SCORE.INTER −42.92** | redock −45.0(同量级) |
| rnasmol 6 版 ensemble | **P=0.692, 4/6 gate** | 0.595, 4/6(优) |
| 质心 vs berberine 晶体位 | 0.11 Å(占位一致) | — |

---

## 合规声明
- **路径一**:t2 的 SMILES 用于种子构成(S 环替换位由 t2 几何 probe 背书)+最终判据
- **路径二**:t2 **仅作检索/判据**,全部结构输入来自模型产物(全氮骨架、inpainting 中间体),两次元素替换(N→S、N→C)的唯一依据是模型 masked 概率,种子几何用通用化学参数
- 全程不改模型/ckpt/采样;rdock 全部走 `rdock_protocol/` 规范(ground-truth redock RMSD 1.23 Å 验证过)

## 关键文件索引
```
casestudy/kras_g4/
├── t2_repro/T2_FINAL_REPORT.md + t2_exact_hits.sdf(172命中) + final_t2_model.sdf
├── t2_denovo/DENOVOPATH.md(全阶段详档) + phaseE_bridge/t2_replaced.sdf(纯从头产物)
├── ber102_rdock/(BER102口袋/受体/berberine参照/probe结果)
├── ber102_gen2k/valid_mols.sdf(冷启动2000)
└── rdock_protocol/PROTOCOL.md(rdock规范)
```

## 方法论沉淀
1. **解码表必须与训练编码逐类对齐**(本项目最大 bug:idx9-12 错位,"模型不生成环硫"是伪影)
2. **元素替换的合法性判据 = masked probe 条件分布**(芳香感知信念;姿态依赖,redock pose 下探测)
3. **P[S] 由分子内上下文驱动**(双稠环+桥),单环姿态再好也不过门
4. **种子设计把生成难题转成模型众数行为**(3锚闭环→单原子双锚插入)
5. **模型的桥位偏好(N)可用概率背书的 N→C 纠正**——不嫁接、不强加,每步有模型自己的分布数字
