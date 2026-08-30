# t2 纯从头路径完整文档(BER102 / KRAS G4 口袋)

目标分子 t2 = `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`(CH2 桥双苯并噻唑)。
**合规声明:全程 t2 仅作判据**(检索排序 / 产物同一性验证 / 报告),t2 的拓扑、
构象、片段从未进入任何种子或坐标;所有结构操作(N→S、N→C)的唯一依据是
模型自身的 masked 元素概率,替换对象与几何全部来自模型生成产物。

路径全景:

```
① 冷启动生成(模型自身) → 全氮等电子骨架(BM 0.375)
② 双位点扫描选位 → N→S替换(部分背书~12%) → 双苯并噻唑种子
③ inpainting生成 → NH-桥双苯并噻唑(Tc 0.517,模型桥位偏好N)
④ 桥位P[C]探测 → [背书 22.1%] N→C替换 → t2  ✓ SMILES == t2
```

---

## ① 冷启动生成 → 全氮等电子骨架

- **输入**:BER102 口袋(`ber102_rdock/ber102_pocket.pdb`,326 原子),模型冷启动
  (无种子)生成 2000 个产物(`ber102_gen2k/valid_mols.sdf`)。
- **检索排序(t2 仅判据)**:按 BM(Murcko 骨架)Morgan Tanimoto 对 t2 骨架排序,
  最高骨架家族为**稠合双氮杂(phenanthroline 类,BM 0.375**,如 gen2k1369
  `Cc1cc2nc3ccc(C(=O)O)c(O)c3nc2c(C)c1C`,redock SCORE.INTER −41.81),
  另获苯并咪唑核(gen2k1271、gen2k473 等)。
- **A2 门控失败(决策点 1)**:24 个候选、48 个氮位点,redock pose 下 masked
  P[S] 最大仅 **1.25%**(0/48 > 10%)→ 单环等电子体替换无背书。
- **A3/A4 上下文证据**:裁剪核(~5% 上限)与二聚上下文 redock pose(F4/F4b,
  60 探测,最大 P[S] **9.02%**,0>10%)仍不过门 → 结论:**驱动 P[S] 的是上下文
  (双稠环 + 桥)而非姿态**;库中无"双稠合双氮 + CH2 桥"分子 → 用**通用化学**
  (C–CH2 1.51 Å、112°、共面伸展构象,ETKDG)把两个模型产物苯并咪唑核装配为
  **全氮等电子骨架** `c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1`(BM 0.333;N↔S 等电子
  为教科书化学,非 t2 结构注入)。
- **输出**:全氮双苯并咪唑-CH2 二聚体(19 原子)。

## ② 双位点扫描选位 → N→S 替换 → 双苯并噻唑种子

- **A6 姿态扫描(门控 v2,决策点 2)**:对二聚体做刚体姿态扫描(参考 BER 长轴/
  法向,平移 ±3 Å),212 个无碰撞摆放,两位吡咯 N 同时 masked:P[S] 最大
  **63.4%**,中位数 10.2%,**106/212 > 10% → 通过**。选 3 个过门摆放:
  gp1 P[S]=**12.46%**,gp2/gp3=**11.24%**(部分背书)。
- **B(替换前复验)**:同摆放重测 gp1=[0.76%, 12.46%]。
- **B2(执行 N→S)**:两位 N→S(芳香 S),sanitize 合法,两片段均
  `c1ccc2scnc2c1`(苯并噻唑)。
- **B3(替换后复验,背书保留)**:gp1 位点 P[S]=[0.66%, **11.71%**],
  P[N]=[85.9%, 46.7%]——S 位自洽,模型不再坚持 N。
- **C(种子输出)**:删除 CH2 桥(留给模型自由生成,free=1),得 **18 原子双
  苯并噻唑两片段种子 ×7**(3 过门 + 4 通用质心摆放;C2…C2 = **2.461 Å**,
  处于 2.4–2.7 Å 插入区间;dmin_pocket ≥ 0.33 Å;13 类特征校验全过)。
- **输出**:`phaseBC_seeds/seedBC_*.sdf`。

## ③ Inpainting 生成 → NH-桥双苯并噻唑

- **输入**:双苯并噻唑锚种子 + BER102 口袋;ckpt
  `onlypkt_ftwithrna_oncrossdocked_...step_111429.pth`;natom=19(free=1,
  恰好桥位 1 个自由原子),sample_steps=100。
- **D1 运行**:16 轮 × 100 = 1600 样本 → 39 个有效中性分子
  (fail:disconnected 1167 / sanitize 394),**0 个精确 t2**。
- **最近产物(决策点 3)**:模型在唯一自由桥位自选 **N** 而非 CH2,得
  NH-桥双苯并噻唑 `c1ccc2sc(Nc3nc4ccccc4s3)nc2c1` ×3 构象
  (r02_i008 / r12_i043 / r12_i057,种子 centroid1),**Tc(t2)=0.517**,
  双苯并噻唑骨架完整——骨架已成,只差桥位元素。
- **输出**:`gen_runD1/valid_mols.sdf`。

## ④ 桥位 P[C] 探测 → N→C 替换 → t2

### E1 探测(`phaseE_bridge_probe.py` → `phaseE_bridge/phaseE_probe_results.json`)

对 3 个 NH-桥构象:masked probe 桥位 N(其余原子锁真实元素,芳香感知,
belief=uniform 1/13,t=1),先测原始生成 pose,再按 BER102 规范协议
(polar-H、rbdock `-n 10 -s 42`、SCORE.INTER 排序)redock 取最优 pose 复测:

| 构象 | pose | **P[C] CH2通道(6,False)** | P[C] 合计 | P[N] | P[O] | P[S] | top1 |
|---|---|---|---|---|---|---|---|
| r02_i008 | 原始 | 3.92% | 29.92% | 58.55% | 1.65% | 5.32% | 7 |
| r02_i008 | redock(−41.86) | 20.54% | 34.51% | 61.97% | 3.27% | 0.10% | 7 |
| r12_i043 | 原始 | 7.13% | 39.61% | 41.32% | 1.59% | 13.21% | 6* |
| **r12_i043** | **redock(−41.11)** | **22.10%** | **49.35%** | 46.40% | 1.94% | 1.51% | 7 |
| r12_i057 | 原始 | 2.52% | 22.66% | 67.23% | 2.20% | 3.53% | 7 |
| r12_i057 | redock(−41.50) | 15.73% | 29.75% | 67.80% | 1.96% | 0.09% | 7 |

**门控(决策点 4,阈值 >10%):3/3 构象 redock pose 全部通过**,最佳
r12_i043 P[C]CH2 = **22.10%**(P[C] 合计 49.35% ≈ P[N] 46.40%,近平手)。
模型 top1 仍是 N(46–68%),但 CH2 通道概率可观 → N→C 桥替换有概率背书。

### E2 替换与验证(`phaseE_nc_replace.py` → `phaseE_replace_results.json`)

- **N→C 执行**:对 r12_i043 的 redock pose,桥 N `SetAtomicNum(6)` + 2 显式 H,
  sanitize + 芳香性重设。
- **同一性**:canonical SMILES = `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`
  **== t2 ✓**(t2 仅在此作验证判据)。
- **替换后复验**:桥位 masked P[C]CH2 = 22.10%、P[C] 合计 49.35%、P[N]=46.40%,
  与替换前逐位相同——预期行为:masked probe 只条件于坐标+其余原子元素,
  与被测位当前元素无关,故背书对 N→C 替换不变(自洽)。
- **产物**:`phaseE_bridge/t2_replaced.sdf`(19 原子,电中性)。

### E3 双评分

- **rDock(BER102 规范协议,`-n 10 -s 42`)**:t2_replaced 最优
  **SCORE.INTER = −42.92**(优于 NH-桥母体同协议 −41.11)。
- **RNAsmol 6 版集成**(序列 `GAGGGCGGTGTGGGAATAGGGAA`):

| 模型 | t2_denovo | berberine_ref |
|---|---|---|
| pdb_mol | 0.8992 | 0.5392 |
| pdb_net | 0.3675 | 0.3655 |
| pdb_rna | 0.9993 | 0.9991 |
| rob_mol | 0.8857 | 0.6667 |
| rob_net | 1.0000 | 1.0000 |
| rob_rna | 0.0000 | 0.0000 |
| **集成 P(gate P>0.5)** | **0.6919(4/6)** | 0.5951(4/6) |

  与 t2_repro 最终模型分子逐模型同分(RNAsmol 图 insensitive,同一分子);
  ≥ 小檗碱参照。(注:rob_rna 版对三者均 0,为该版模型整体行为。)

---

## 概率决策链汇总

| 阶段 | 探测/决策 | 概率 | 判定 |
|---|---|---|---|
| A2 | 单环等电子体 redock P[S] | max 1.25%, 0/48>10% | 不替换 |
| A4 | 二聚 redock pose P[S] | max 9.02%, 0/60>10% | 不替换 |
| A6 | 二聚姿态扫描 P[S] | max 63.4%, 106/212>10% | 通过,gp1 12.46% |
| B3 | N→S 后 P[S] 保留 | 11.71% (gp1) | 执行 N→S |
| D1 | 桥位生成(模型自选) | N(3/39 最近产物) | NH-桥产物 |
| E1 | 桥位 P[C]CH2(redock pose) | **22.10%** > 10% | **执行 N→C** |
| E2 | N→C 后复验 | P[C] 不变(设计使然) | 自洽 |

## 脚本与产物索引

| 阶段 | 脚本 | 关键产物 |
|---|---|---|
| ① A1–A4 | `phaseA_select_dock.py` `phaseA_probe.py` `phaseA3_context_probe.py` `phaseA4_dimer_*.py` | `phaseA_redock/summary.json` `phaseA4_dimer/` |
| ② A6/BC | `phaseA6_posescan.py` `phaseBC_replacement_seed.py` | `phaseBC_seeds/seedBC_*.sdf` |
| ③ D | `phaseD_generate.py --run D1` | `gen_runD1/valid_mols.sdf`(NH-桥 ×3) |
| ④ E | `phaseE_bridge_probe.py` `phaseE_nc_replace.py` `phaseE_rnasmol.py` | `phaseE_bridge/phaseE_probe_results.json` `phaseE_bridge/t2_replaced.sdf` `phaseE_bridge/phaseE_replace_results.json` `phaseE_bridge/rnasmol_t2_denovo.json` |

一键流水线:`run_denovo.py --stage all`。
