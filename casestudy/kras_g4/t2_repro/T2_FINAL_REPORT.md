# t2 精确复现最终报告 — `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`(苯并噻唑-甲烯-苯并噻唑)

日期: 2026-08-15 | 环境: conda `rna2smol` | 全程未改模型/checkpoint

## 结论

**精确复现达成。** Run U + Run V 共产出 **172 个精确 t2 分子**(canonical SMILES 逐分子校验一致),
均为模型在 BER102 口袋内生成的带 3D 构象产物。最优构象桥键长 1.519/1.516 Å,
质心距 berberine 晶体位 **0.11 Å**。rDock redock −41.28,rnasmol 6 版集成 P=0.692(4/6 门控,
高于 berberine 参照 0.595)。

---

## Phase 1 — 等电子体检索 + 概率探测(决策点)

### 检索
| 库 | BM Tanimoto top | 分子 Tanimoto top | 精确命中 |
|---|---|---|---|
| ber102_gen2k (1579) | 0.375 ×4(三环 phenanthroline 型 `c1ccc2nc3ccccc3nc2c1`) | 0.222 | 0 |
| library_10k (10094) | — | 0.224 | 0 |
| 库内 CH2-桥稠环等电子体 | 苯并咪唑(16个,BER102 pocket 内 d=0.8–3.0 Å)、quinazoline-CH2 等 114 个 | | |

### t2 构象与 masked probe(芳香感知,featurizer 13类表,BER102 pocket)
t2 ETKDG→rDock(R6 验证系统): best SCORE.INTER = **−43.06**,质心距 berberine 0.61 Å。

| 实验 | S4 P[S] | S15 P[S] | 说明 |
|---|---|---|---|
| exp1 苯并噻吩基底(N盲) | **84.5%** | **96.7%** | top1=16*(芳香S) |
| exp2 全 t2 元素锁 | **84.6%** | **96.8%** | mean P[true]=94.0%,19/19>50%,echo 19/19 |
| exp3 全氮等电子锁(S→N[H]) | **83.7%** | **96.6%** | 被告知是N仍重采样为S |
| exp4 前5 pose 稳定性 | 83–97% | 48–96% | pose 鲁棒 |
| 库内真实苯并咪唑(12个,pocket 原位) | 0.02–0.26% | | **P[S]@nH≈0.1%** |

**判定:P[S] 在 t2 几何下强背书(≫10%)→ 走 Phase2A;但元素场强依赖上下文/姿态。**

### 方法论发现(重要)
1. **模型 13 类表 = `dataloader/featurizer.py` 版(S脂肪=10, S芳香=11, Cl=12, 无Br类)**;
   旧 probe 脚本引用的 `util/mol_utils.py` 表(S=9/10)与之不一致——旧结论只涉 C/N 通道故不受影响。
2. gen2k/`sampletraj_to_sdf*` 用 mol_utils 表**误解码**:模型芳香 S(11)被记为 Cl,Cl(12)被记为 Br。
   BER102 库 76 个"Br"分子实为模型 Cl 输出;**"模型从不生成环硫"部分是解码伪影**。本轮
   自写重建循环用 featurizer 表解码。

## Phase 2 — 种子引导生成(7 轮迭代,~5600 有效分子,种子全为 t2 子图/等电子体+独立坐标)

| 轮 | 种子设计 | 有效/采样 | t2 | 失败机理/结论 |
|---|---|---|---|---|
| R1 A/B/C | 苯并噻唑(9原子)/MeBT(10),ETKDG+库内N→S 替换,natom 17–21/19 | 1729 | 0 | 模型只做极性装饰,C2 延伸链(OH/COOH/CH2CH2N);从不自建第二个环硫 |
| R2 D/E/F | BT-CH2-Ph(16)**拓扑错误**(非 t2 子图:CH2 须键两个噻唑C2,非苯环碳);另发现 exec 命名空间捕获旧 `sample_atom_num` 的 bug(已修) | ~1500 | 0 | 教训:t2 中 CH2 连接的是两个**环外**噻唑 C2 |
| R3 G/H/I | 正确子图:BT+Ph(15)/BT+Ph+CH4(16) | ~900 | 0 | 3片段桥接 0/800;2片段仅 1–2 原子链桥 |
| R4 J/K/L | W1 MeBT+Ph(ETKDG类似物切键,C2↔Ph 4.68 Å) | ~500 | 0 | 3原子协同闭环不发生 |
| R5 M/N/O/Q | X18 MeBT+2-氨基苯硫酚(18,自由1=噻唑C2)/X17 变体;1000步、dist_tol 0.55 | ~1300 | 0 | **模型75%输出=链插入** `Sc1ccccc1NCCc1nc2ccccc2s1`;不闭环 |
| R6 R/S2/T | 同 X18 但几何=rDock 对接类似物(−37.62)切键 | ~1900 | 0 | 同上,姿态不改变链插入偏好 |
| **R7 U/V** | **Y种子 = 两个苯并噻唑(18原子,2片段),C2···C2=2.52 Å,自由=1(CH2)**;几何=模型自产BT-CH2-Ph(删CH2)+第二BT定向放置(C2钉于原ipso位,环心沿−CH2方向,法向对齐),及 ETKDG 独立变体 | **711** | **172** | **模型把单原子插入两锚点之间(其最擅长操作)→ C选择≈25% → t2;N插入→BT-NH-BT(23%);S插入→BT-S-BT(6%);环稠合异构体(30%)** |

**成键机理**:R7 将"模型从不做的 3 锚点闭环"重构为"模型最擅长(75%+)的单原子双锚点插入"。
U: 86/342 精确、V: 86/369 精确(每轮 15–23 个)。

## Phase 3 — 精确复现验证

- **SMILES == t2**:172/172(canonical `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`,C15H10N2S2,中性,19 原子)
- 桥键长:171/172 在 1.3–1.8 Å;最优 1.519/1.516 Å;质心距 berberine 晶体位 0.11 Å
- **rDock(BER102 R6 规范协议)**:模型构象 redock n=20, best SCORE.INTER = **−41.28**,
  pose 仍为 t2,质心 0.91 Å(对照:ETKDG 输入 t2 dock −43.06;berberine redock −44.99;花菁 −46.12)
- **rnasmol 6 版集成**(7X8M 23nt `GAGGGCGGTGTGGGAATAGGGAA`):
  t2 ensemble_P = **0.6919,门控 4/6**(pdb_mol 0.899 / pdb_net 0.368 / pdb_rna 0.999 /
  rob_mol 0.886 / rob_net 1.000 / rob_rna 0.000);berberine 参照 0.595(4/6)

## 合规性
- 模型/ckpt 未动;全部运行时 monkey-patch(不改源文件)
- 种子来源:模型产物(R1输出、R7的BT-CH2-Ph模型输出)/ 库内等电子体 N→S / ETKDG独立坐标(含
  连接类似物切键、rDock 自主对接姿态)/ 标准几何叠合;**未使用 t2 自身 dock 构象作种子坐标**(probe 用除外,系用户认可链)
- 解码用正确 featurizer 表;如实报告 0 命中的 6 轮失败

## 文件
- `t2_exact_hits.sdf`(172 精确命中)/ `final_t2_model.sdf`(交付构象)
- `probe_t2_results.json` / `probe_lib_benzimidazole.json`(Phase1)
- `inpaint_seeds_r{1..7}/`、`make_seeds_r{1..7}*.py`(种子谱系)
- `gen_run{A..V}/`(全部生成轨迹)/ `run*.log`
- `t2_dock.sd`、`t2_model_dock.sd`(rDock)/ `rnasmol_t2_final.json`
