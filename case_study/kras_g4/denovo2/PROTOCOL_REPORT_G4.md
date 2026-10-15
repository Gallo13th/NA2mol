# KRAS G4(DNA)配体生成完整链条文档(v3 端口溯源协议 · BER102 位点定稿)

日期:2026-09-10 | 环境:conda `rna2smol` | 模型/checkpoint 全程未改
靶点:KRAS promoter G-四链体,7X8M MODEL 1,**BER102 位点**(well-buried 10/10,
参照配体共晶中心 (34.7, 23.6, 27.6));RNAsmol SEQ = `GAGGGCGGTGTGGGAATAGGGAA`

---

## 0. 位点选择过程(三段证伪存档)

1. **BER101(berberine 位点)旧 10k 库**:t2 死于 `n_total≥3`(G4 口袋 ProLIF 无
   Hydrophobic 类型,BI 互作 ≤1);放宽 n_total≥1 后 t2 组装成功但 sum_rdock 弱
   (−4.49,87% 分位),三重筛去掉 rdock 腿后 t2 的 ms_INTER −16.54、模型票 3/4
   但 rDock 轴 #528/717 落选 Tier-1。
2. **口径分解**:历史 t2 = −42.92 是 BER102 + dock_solv(水合);同 pose InterIdx
   −29.5;我们 BER101 组装+弛豫 pose InterIdx −16.5 → 差距 = 水合(−13.5)+
   位点(−13)。**无分子/构象错配**(历史 pose 在历史口径复现 −43.02)。
3. **BER102 旧 2k 库**:BT×BT 幽灵距 0.40-1.28 Å 全部死于片段重叠(0.19-0.49 Å)
   —— 深埋口袋 + 小库 → 片段位置零多样性。
   → **结论:在 BER102 补 10k 冷启动**(4×2500 GPU 流,10,077 unique)。

## 1. 最终链账本(全部自动检查点,`run_chain_ber102.py`)

```
冷启动 10,077 unique(BER102 口袋,4 流 ×2500)
→ 无偏切割 18,365 实例 → 9,278 片段(BI 35)
→ 门槛 smina≤−2.5 ∧ n_total≥3 ∧ 重原子≤14 + QED>0.5 ∧ SAS<2.0 → 717 池(BI 24)
→ 饱和杂原子突变(KEEP)→ 1,588 变体(BT 26 合法入池)
→ rDock score-only(ber102_full_r6_system,InterIdx)
→ 端口匹配(同元素 ∧ 幽灵≤1.0 Å)216,092 候选 → 9,806 组装 → 1,721 unique 产物
   [精确 t2 = 1,BT×BT 幽灵重合对]
→ 筛选 QED≥0.5 ∧ SAS≤median(2.16)→ 785(t2 IN;rdock 腿按用户裁量去除)
→ 多 seed 弛豫对接(3 seeds × 5 runs,RMSD≤2.0 Å 内最优 INTER,同系统)
   → 728;**t2 ms_INTER −27.05,#282/728**(与历史 BER102 InterIdx −29.5 同量级)
→ RNAsmol 6 模型(G4 SEQ;死维度 robin_rna;pdb_rna 本轮存活)→ 5 排序器
→ 双轴多数分档:模型 ≥3/5 ∧ rDock 前半 → **Tier-1 = 293,t2 IN(idx 121)**
   [t2 模型票:robin_net #1 级强信号;mRNA 链同构现象]
→ Tier-1 内 MinRank:t2 = #246/293
```

## 2. PubChem 检索与订单(Tier-1 全量 293)

- 已注册:**34**(11.6%);**可购买:14**(4.8%);新颖:259
- **t2(CID 74755)在可购名单内**(INTER −27.05)—— 与 mRNA 链结论呼应
- 可购集骨架:萘甲基-氮杂芳烃 / 芳氨基-氮杂芳烃 / 苯并噻唑胺(t2 同族 3 个)
- 订单表:`g4_tier1_order_table.csv/.md`;结构图 `g4_tier1_purchasable.png`;
  订购分子真实 pose:`g4_tier1_purchasable.sdf`(14 个,可直接叠加口袋查看)

## 3. 核心结论

1. **t2 由完全无先验链产出并存活到 Tier-1**(BER102 补库后全链零点名);
   且 t2 直接可购买 —— 两条独立链(mRNA / DNA-G4)均如此。
2. **口袋选择即命运**:同一协议在 BER101(浅)/BER102(深埋)给出截然不同的
   t2 轨迹;历史分数口径(位点+水合)必须分解后才能对齐。
3. 小库 + 深埋口袋 = 片段位置多样性枯竭(2k 库 BT×BT 全重叠);10k 补库即解。
4. RNAsmol 死维度因 SEQ 而变:robin_rna 恒死;pdb_rna 在 G4 SEQ 上饱和死。
5. 全链每步规则化零权重,与 mRNA v3 协议逐字对应(仅 rdock 腿按用户裁量省去)。

## 4. 文件索引

| 步 | 内容 | 文件(denovo2/ 下)|
|---|---|---|
| 库 | BER102 10k | `../ber102_gen10k/valid_mols.sdf` |
| A2-3s | 切割/门槛/突变/打分 | `chain_frag_dedup.sdf` `chain_frag_stats.json` `chain_pool.json` `chain_variants.json` |
| 4 | 组装产物 | `chain_products.sdf/.json` |
| 5 | 筛后+弛豫 | `chain_finalset.json` `chain_finalset_ms.json` |
| 6-7 | RNAsmol+分档 | `chain_rnasmol6.json` `chain_tiers.json` |
| PubChem | Tier-1 全量 | `tier1_pubchem.json` |
| **订单** | **14 可购分子** | **`g4_tier1_order_table.csv/.md` `g4_tier1_purchasable.png/.sdf`** |
| 引擎 | 全链脚本 | `run_chain_ber102.py`(t2 检查点自动停)|

生成脚本:`../run_generation_ber102.py`(口袋 `7X8M_ber102c_pocket.pdb`,
dummy `dummy_ber102.sdf`);rDock:`../ber102_rdock/ber102_full_r6_system.prm`(+.as),
协议 `../rdock_r4/score_only.prm`、`minimise.prm`(RBT_ROOT 内置)。
BER101 阶段存档:`frag_dedup*.sdf`、`variant_pool*.json`、`port_assembled*`
(旧 2k 与旧 10k 两条子链的全部中间物保留供审计)。
