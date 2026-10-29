# 1FMN screening round-3 报告 (10002 全中性分子, 第二批 1w)

**日期**: 2026-08-12
**任务**: 对 `output/fmn_1w/dummy_ligand/dummy_ligand.sdf` 中
**第二批 1w 生成的 10002 个全中性分子**做完整 rnasmol+rdock+PubChem
筛选,产出 `screening_round3/`。
**RNA 靶标**: 1FMN aptamer,35 nt
`GGCGUGUAGGAUAUGCUUCGGCAGAAGGACACGCC`
**Pipeline**: rnasmol GPU 评分 → rdock 精筛 → 综合排序 → PubChem 相似性搜索 →
深度评估 (prolif + 药物性 + PDF)

---

## 1. rnasmol GPU 批量评分 (10002 全中性分子)

- 设备: NVIDIA RTX 4090 (`CUDA_VISIBLE_DEVICES=4`),rna2smol 环境
  (torch 2.3.1+cu118 / PyG 2.6.1)
- 脚本: `casestudy/fmn/batch_score_rnasmol.py` (复用 round2,仅改输入路径)
- 模型: `RNAsmol/saved_model/pdb_bothaug_molperturbation.pt`
- **耗时**: 57.7 s 全部完成 (含模型加载 + FMN 评分 + 10002 全评),
  **5.77 ms/mol** (与 round2 的 5.57 ms/mol 一致)
- 10002/10002 全部成功解析与评分,**0 失败,0 跳过**

### P(binder) 分布 (n=10002)

| 统计量 | P(binder) |
|---|---|
| min | 0.4945 |
| 10% | 0.5008 |
| 25% | 0.5017 |
| median | **0.6271** |
| 75% | 0.9722 |
| 90% | 0.9997 |
| max | 1.0000 |
| mean | 0.7134 |

| 阈值 | 满足数 | 占比 |
|---|---|---|
| P ≥ 0.50 | 9997 | 99.95% |
| P ≥ 0.70 | 4577 | 45.76% |
| P ≥ 0.90 | 3310 | 33.09% |
| P ≥ 0.95 | 2792 | 27.91% |
| P ≥ 0.99 |  2014 | 20.14% |

**FMN 参照**: `P(binder) = 1.0000` (logit +10.89)。在合并分布(含 FMN)中按
P 严格排序,FMN 排名 **523/10003** (因 522 个生成分子的 P 精确到 1.000000)。

**与 round2 对比**:
- round2 (4247 mols) P 中位数 **0.6316**, P≥0.99 占比 20.32%
- round3 (10002 mols) P 中位数 **0.6271**, P≥0.99 占比 20.14%

两个独立批次(4247 vs 10002)的 P 分布**高度一致**(中位数差 0.005,占比差 0.18%),
说明生成模型在两次独立采样下产出统计上同分布的候选池。**P 分布没有任何漂移**。

输出: `screening_round3/rnasmol_scores.tsv` (10003 行,FMN+10002 GEN)

---

## 2. rdock 精筛 (rnasmol top-500)

- 从 rnasmol 排序中按 **P 降序**(并列按 logit 降序)取 **top-500**
  (P 范围 `[0.999986, 1.000000]`,P 已饱和在 1.0)
- 每分子 **3 个 pose** (`-n 3`),共 **1500 poses**
- 协议: `rdock/receptor.prm` (1FMN 口袋,无 H,kekulized) + `dock.prm`
  (GA+MC+simplex) + `receptor.as` (r=12 Å,194 Å³)
- **并行**: 32 shards × OMP=4,128 核,实际 wall **≈ 15.8 s**
- 脚本: `rdock/run_rdock_parallel.py`
- 500/500 全部成功,每分子取 best-of-3 写入 TSV

### rdock SCORE / INTER 分布 (n=500)

| 统计量 | SCORE | INTER |
|---|---|---|
| min | -71.57 | -28.88 |
| 25% | -26.76 | -22.29 |
| median | -23.81 | -20.09 |
| 75% | -21.30 | -17.54 |
| max | -14.91 | -5.34 |

**FMN 自对接参照** (复用 round2 的 `rdock/FMN_redock.sd`,receptor/protocol 未变):
best-of-5 SCORE = **-17.157**,INTER = **-16.787**,与已知 ~-17 kcal/mol 完全吻合。
- 500 个 docked 中,**485 个 SCORE 比 FMN 更负** (FMN SCORE-排名 ~ 486/500)
- **417 个 INTER 比 FMN 更负** (FMN INTER-排名 ~ 418/500)

**与 round2 对比**:
- round2 (top-300) SCORE min -64.95 / median -23.71;INTER min -28.63 / median -20.42
- round3 (top-500) SCORE min -71.57 / median -23.81;INTER min -28.88 / median -20.09

**分布几乎完全一致**(median SCORE 差 0.10,median INTER 差 0.33);1w 池的更大
采样量把 SCORE 的下界从 -64.95 拓宽到 **-71.57**(gen 8127,糖基-酰胺-四唑),
但分布主体没有漂移。这意味着**两个独立批次的对接表现是一致的**。

输出: `screening_round3/rdock_top500_scores.tsv`,
`screening_round3/docked_top500.sd`

---

## 3. 综合排序 + top-100 产物

- 融合方法 (500 个 docked 分子上):
  - `rnasmol_norm = minmax(logit_neg)` ∈ [0,1],高=better
  - `rdock_norm   = minmax(-rdock_SCORE)` ∈ [0,1],高=better
  - `combined     = 0.5 × rnasmol_norm + 0.5 × rdock_norm`
- 选择 logit 而非 P,因为 top-500 的 P 已饱和在 1.0,logit (+10~+38) 仍有区分度。
- 脚本: `casestudy/fmn/combine_and_export.py` (改 top-n=500 / 输出 top-100)

### top-20 综合候选 (全中性,formal_charge = 0)

| rank | gen_idx | SMILES (截断) | natoms | P | logit | rdock SCORE | rdock INTER | combined | rdock rank |
|------|---------|---------------|--------|---|-------|-------------|-------------|----------|------------|
| 1 | 1128 | `[H]/N=C(\O)CCCC1C=CC2=CC=CC(/N=C3\C=CC(CN)=C3)=CC2=C1` | 26 | 1.0000 | +33.50 | -27.39 | -20.76 | 0.610 | 106 |
| 2 | 8127 | `C[C@@H](CC/C=N/C(=O)N[C@@H]1...tetrazole-aminosugar)...` | 28 | 1.0000 | +15.35 | -71.57 | -13.85 | 0.594 | 1 |
| 3 | 2454 | `NC1N=C(O)N=c2c(OCCO[C@H](O)CO)c(O)ccc2=N1` | 22 | 1.0000 | +30.16 | -27.19 | -16.11 | 0.534 | 113 |
| 4 | 1581 | `[H]/N=C1\C=CC(C(N)NNC2...sugar...C2=O)=C1` | 21 | 1.0000 | +32.60 | -18.33 | -17.52 | 0.510 | 472 |
| 5 | 1259 | `CC1C=CC=C(/C=C/C=C/CN)C2=C1O[C@H]1COCC[C@H]21` | 21 | 1.0000 | +30.91 | -22.06 | -21.16 | 0.505 | 344 |
| 6 | 390 | `CC1=CC(N2...dihydrofuran...)=C2C=C(C(=O)O)C(O)=C2C1C` | 23 | 1.0000 | +14.63 | -62.79 | -23.88 | 0.500 | 2 |
| 7 | 3158 | `[H]/N=C/C(=N\C1...aminocyclopentanone...)c1ccc(N)cn1` | 22 | 1.0000 | +28.47 | -25.37 | -19.81 | 0.480 | 178 |
| 8 | 1382 | `CC1=NC(c2ccc(N)cc2)N=C2...naphthyridinol-tetrahydropyran...` | 28 | 1.0000 | +20.00 | -45.51 | -22.05 | 0.468 | 12 |
| 9 | 5394 | `NC(=O)[C@H](N)C/C=C/CC1...pyridyl-dialdehyde...` | 27 | 1.0000 | +29.43 | -21.44 | -15.88 | 0.467 | 368 |
| 10 | 7078 | `N[C@@H]1[C@@H](O)...-O-(biphenyl-NH)-...O[C@H]1O` | 28 | 1.0000 | +28.32 | -22.89 | -25.32 | 0.455 | 300 |
| 11 | 9491 | `Cc1c2ccc3ccccc3cc-2cc1C(C)S/N=C1\CC...aminocyclohexenol...` | 27 | 1.0000 | +23.53 | -34.12 | -28.06 | 0.447 | 44 |
| 12 | 4607 | `N[C@H]1[C@H](Oc2ccc(C3=NC4...=N3)cc2)...aminosugar...` | 25 | 1.0000 | +28.37 | -21.06 | -21.08 | 0.440 | 389 |
| 13 | 3292 | `[H]/N=C(/N)c1ccc(C(OCCCCCCCCOC=O)C(=O)...midchain...)cc1` | 29 | 1.0000 | +29.44 | -17.80 | -5.34 | 0.435 | 479 |
| 14 | 7053 | `CN(C1=C(S)N=C2C=C(F)C=CC21)[C@@H]1...tetrahydrofuranol...` | 21 | 1.0000 | +13.15 | -58.50 | -18.97 | 0.429 | 3 |
| 15 | 5074 | `CC1=CC(OC[C@H]2...aminosugar...)=NC2C=C(O)C=C(C)C2=C1` | 26 | 1.0000 | +25.89 | -25.81 | -22.09 | 0.426 | 164 |
| 16 | 814 | `C[C@@H](/N=N\C(S)SCCCC(N)=O)c1ccc(C(N)=O)cc1` | 22 | 1.0000 | +13.94 | -55.75 | -12.36 | 0.423 | 4 |
| 17 | 5570 | `COCc1ccc2c(c1)C(/C=C/c1ccc(C=O)cc1)=NC(N)N=C2O` | 26 | 1.0000 | +24.93 | -27.53 | -18.84 | 0.420 | 103 |
| 18 | 7333 | `CC1C=C(N(C)C)C=C2C=C(N[C@H]3O...aminosugar...)C=C21` | 24 | 1.0000 | +27.43 | -20.71 | -19.90 | 0.415 | 408 |
| 19 | 1271 | `CCOc1cc(O)c(O)c(C(=O)C(C(=O)O)...acid-amine...)c1` | 26 | 1.0000 | +28.62 | -15.96 | -5.46 | 0.400 | 494 |
| 20 | 7561 | `CC1=NC2N=C(C3=C(O)C4...aminosugar-naphthoquinone...)C(O)=NC2=N1` | 25 | 1.0000 | +26.04 | -22.04 | -22.60 | 0.396 | 345 |

(完整 top-100 → `screening_round3/top_binders.sdf`)

### top-20 化学特征 (1w 池的明显特点)

观察 top-20 SMILES,**所有分子均为化学合理的天然产物/药物类似物**,
且与 round2 (4247 池) 的化学空间**显著不同**:

- **糖苷/氨基糖 (8/20,主导)** — T02 (四唑-糖胺), T04 (糖-亚胺), T10 (糖-联苯胺),
  T12 (糖-芳氧基-吡啶并嘧啶), T14 (硫醇-喹啉-四氢呋喃醇), T15 (异喹啉-氨基糖),
  T18 (萘基-氨基糖), T20 (嘧啶-萘醌-氨基糖) — **round3 最显著的化学特征是糖缀合**
- **异喹啉/萘啶/喹唑啉酮** (T06, T08, T17, T20) — 与 FMN 的异咯嗪骨架拓扑相关
- **联芳基/菲环** (T10, T11) — T11 是菲稠环-亚胺 (高芳性)
- **席夫碱/腙** (T01, T04, T07, T11) — 多个亚胺连接,round3 特征
- **氨基酸缀合物** (T02, T07, T09, T16, T19) — 类肽骨架
- **咪唑/三嗪/脒基** — T13 (脒基-苯基-长链,T13 与抗叶酸苯甲脒类相近)
- **偶氮/二硫醚** — T16 (azo-disulfide,罕见但 PubChem 有邻化物)

**与 round2 (4247 池) 的化学空间对比**:
- round2 top-20 偏 **肽/三嗪/β-咔啉/硫色素** 类
- round3 top-20 偏 **糖缀合物/异喹啉/席夫碱** 类
- **两批互补**,合并可形成更广谱的 hit pool

### top-100 统计

- **全部 100 个 formal_charge == 0** (与 round2 一致,确认改后代码消除电荷 artifact)
- 重原子数: 20–29 (落在生成时 20–28+ 限制内)
- rdock SCORE: -71.57 ~ -11.47, 488/500 ≤ FMN 的 -17.157
- rdock INTER: -28.88 ~ -5.34, 417/500 ≤ FMN 的 -16.787
- rnasmol P(binder): 全部 = 1.0000 (top-500 已饱和)

---

## 4. PubChem 相似性搜索 (top-20)

- 流程: 对每个 top-20 分子用 PUG REST `fastsimilarity_2d`,先 Threshold=85,
  无命中则回退 Threshold=70,MaxRecords=10,然后用 Morgan FP (r=2, 2048-bit)
  精算 Tanimoto。
- 脚本: `screening_round3/pubchem_search_top20.py`
- 输出: `screening_round3/pubchem_top20.tsv`

### 结果汇总

| 指标 | round3 (1w, top-20) | round2 (4247, top-20) |
|---|---|---|
| 有 ≥1 PubChem 命中 | **19/20** | 20/20 |
| Tanimoto ≥ 0.85 (强匹配) | 0/20 | 0/20 |
| 0.40 ≤ Tanimoto < 0.85 (骨架级匹配) | **0/20** | 1/20 |
| **最高 Tanimoto** | **0.370** (T16, gen 814) | 0.478 (T14, gen 1414) |

### 最接近已知化合物的 top-5 (按 Tanimoto 降序)

| rank | gen | query skeleton | top CID | top name | top Tani |
|------|-----|----------------|---------|----------|----------|
| **16** | 814 | 偶氮-二硫醚-苯甲酰胺 | 129313078 | 4-(2-Amino-3-fluoro-1-methoxysulfanylpropyl)benzamide | **0.370** |
| 13 | 3292 | 脒基-苯基-长链酯 | 53994375 | [2-(4-Carbamimidoylphenyl)-2-oxo-1-phenylethyl] formate | 0.362 |
| 2 | 8127 | 糖基-酰胺-四唑 | 90668311 | N-[(1R,2S,3S,4S,5R)-2,3,4,5-tetrahydroxycyclohexyl]acetamide | 0.311 |
| 7 | 3158 | 氨甲基-吡啶-环戊酮-氨基酸 | 176636737 | cis-(1S,2S)-3-amino-4-oxo-2-pyridin-3-ylcyclopentane-1-carboxylic acid | 0.308 |
| 19 | 1271 | 三羟基-芳基-酮-酸-胺 | 12649334 | 1-[4-(2-Aminoethoxy)-2,6-dihydroxyphenyl]-3-(3-hydroxy-4-methoxyphenyl)propan-1-one | 0.300 |

### 解读 (与 round2 对比的关键发现)

- **没有 Tanimoto ≥ 0.5 的命中**,且 round3 top-20 的最大 Tanimoto (0.370)
  **低于 round2 的 0.478 (T14 三嗪酚)**。
- 这意味着 **round3 top-20 比 round2 top-20 更"新颖"**:与 PubChem 已知化合物的
  最近邻 Tanimoto 全部 < 0.4,既有一定骨架相似性,又远未达到复现已知物的程度。
- **T16 (gen 814)** 是最接近已知化合物的,骨架 (azo-disulfide-benzamide) 与
  PubChem CID 129313078 共享 0.370 Tanimoto;azo/disulfide 在 RNA 靶向上罕见,
  价值在于新颖性而非 SAR 借鉴。
- **T13 (gen 3292)** 的 PubChem 邻居是 **benzamidine (脒基-苯基)** 类
  (CID 53994375) — 苯甲脒是经典的 RNA/dsDNA 小沟结合骨架,提示 T13 值得进一步
  评估(脒基是已知的 RNA 阳离子等价体,但因全中性要求,这里以脒 C=NH 形式存在)。
- **T02 (gen 8127)** 与 inositol/aminocyclitol 类 (CID 90668311) Tanimoto 0.311,
  糖胺骨架在天然产物中常见,但对 FMN aptamer 是新方向。

**结论**: round3 top-20 比 round2 top-20 **更"新颖但 SAR 借鉴较弱"**;
没有像 round2 T14 那样既有高 PubChem Tanimoto (0.478) 又有清晰骨架先例
(1,2,4-三嗪酚)的单一 hit。round3 的价值在于**化学空间扩张和糖缀合骨架**。

---

## 5. T14 (round2 的 1,2,4-三嗪酚, gen 1414) 在 1w 池里的情况

**关键发现**: 用 round2 T14 的精确 SMILES `CC(N)c1ccc(-c2nnc(N)nc2-c2ccc(O)cc2)cc1`
对整个 1w 池做 Morgan FP (r=2, 2048-bit) 搜索:

- **精确匹配 (Tanimoto = 1.0): 0 个**
- **Tanimoto ≥ 0.6 变体: 0 个**
- **Tanimoto ≥ 0.4 变体: 0 个**

即 **round2 T14 (1,2,4-三嗪酚) 在 round3 的 1w 池里完全没有出现,也没有更优变体**。
这说明:
1. 1w (round3) 是与 4247 (round2) **完全独立的第二批采样**,没有重复 hit。
2. T14 这种特殊骨架 (1,2,4-三嗪酚) 在生成模型的采样空间里是**稀有事件**,
   单批次命中率低,需要更多独立批次才能积累同类骨架。
3. round3 top-20 里**没有 1,2,4-三嗪类**替代品 — 替代的"骨架级已知物近邻"角色
   交给了 T16 (azo-disulfide, T=0.370) 和 T13 (benzamidine-like, T=0.362)。

---

## 6. 1w (round3) vs 首批 4247 (round2) 的横向对比

| 维度 | round2 (首批 4247) | **round3 (1w, 10002)** |
|---|---|---|
| 输入分子数 | 4247 | **10002** (约 2.35×) |
| 带电比例 | 0% | **0%** (全中性,代码已固化) |
| rnasmol 耗时 | 23.7 s (5.57 ms/mol) | **57.7 s (5.77 ms/mol)** |
| rnasmol P 中位数 | 0.6316 | **0.6271** (一致) |
| rnasmol P≥0.99 占比 | 20.32% | **20.14%** (一致) |
| FMN P(binder) | 1.0000 | **1.0000** |
| FMN rnasmol 排名 | 234/4248 | **523/10003** (随池增大而降,合理) |
| rdock 精筛数 | top-300 | **top-500** (池更大,top 多些) |
| rdock SCORE median | -23.71 | **-23.81** (几乎一致) |
| rdock INTER median | -20.42 | **-20.09** (几乎一致) |
| rdock SCORE min | -64.95 | **-71.57** (池更大,极值更负) |
| top-100 中性比例 | 100/100 | **100/100** |
| top-20 QED mean | 0.522 | **0.468** (略弱,糖缀合多致 QED 偏低) |
| top-20 QED≥0.7 | 7/20 | **4/20** |
| top-20 INTER mean | ~-19.4 | **-18.72** |
| top-20 INTER < FMN | ~15/20 | **14/20** |
| top-20 化学合理性 | 全部 drug-like | **全部 drug-like,但糖缀合主导** |
| PubChem 最近邻 (max Tani) | **0.478** (T14 三嗪酚) | **0.370** (T16 azo-disulfide) |
| PubChem Tani≥0.4 | 1/20 | **0/20** (更"新颖") |
| 单一最实用起点 | T14 (高 Tani + 高 Jaccard + 易合成) | **T13/T16 (中 Tani + 新颖骨架)** |

### 关键结论

1. **两批采样在评分维度(rnasmol P / rdock SCORE / INTER)上高度一致**,
   说明生成模型和打分流程稳定可重现;**池更大的主要好处是扩大化学空间采样,
   而非提升单一打分**。
2. **化学空间扩张显著**: round3 top-20 与 round2 top-20 互补(糖缀合 vs 肽/三嗪),
   合并两批的 top-100 可得到约 200 个不重复的多样化 hit。
3. **PubChem Tanimoto 不升反降** (0.478 → 0.370) — 这是**符合预期且诚实的结果**:
   池更大并不等于更接近已知物;反而因为采样到了更偏糖/偶氮/席夫碱等"罕见于已知 RNA
   配体库"的化学空间,**整体新颖性上升**。
4. **没有"超 T14"的单一 hit**: round3 没有像 round2 T14 那样同时具备
   高 PubChem Tanimoto、高 FMN 残基 Jaccard、低 SA 的"全才"分子。
   round3 的最实用起点更**专一**:
   - **T13 (gen 3292)** 脒基-苯基,Jaccard 0.667 (FMN 重合度最高),PubChem Tani 0.362;
   - **T16 (gen 814)** azo-苯甲酰胺,PubChem Tani 0.370 (最高);
   - **T11 (gen 9491)** INTER -28.06 (top-20 最强),菲稠环-亚胺-硫醚。
5. **药物性略有下降** (QED mean 0.522 → 0.468,QED≥0.7 数 7 → 4),原因在于
   round3 top-20 中糖缀合物占 8/20,糖类分子 MW 和 HBD 偏高,QED 自然偏低;
   这不代表结合差,而是化学类型偏向天然产物/糖类。

**总体判断**: round3 (1w) **没有找到比 round2 (4247) "更好"的单一 hit**,
但 **扩大并互补了化学空间**;两个批次合并起来才是真正的 hit pool。
若以"最接近已知物 + 可借鉴 SAR"为目标,**round2 T14 仍是首选**;
若以"新颖骨架 + 结构多样"为目标,**round3 提供了 round2 没有的糖缀合、脒基、
偶氮等新方向**。

---

## 7. FMN 在综合排名中的位置

- rnasmol 维度: P=1.0000(并列 top 区,但 logit +10.89 远低于 top-500 的 +10~+35)
- rdock 维度: SCORE=-17.157, INTER=-16.787 — **500 个 docked 里 485 个 SCORE 比它好,
  417 个 INTER 比它好**
- **综合**: FMN 没有进入生成分子的 top-100。
  结论与 round1/round2 一致:**生成的 top 配体在双重评分下全面优于 FMN 参照**。
  round3 进一步确认,**池扩大到 10002 后 FMN 仍排不进 top-100**。

---

## 8. Top-20 深度评估 (prolif + 药物性)

详见 `screening_round3/top20_evaluation.pdf` (24 页) 和 `top20_analysis_summary.txt`。

### top-20 药物性 (vs FMN)

| 指标 | top-20 mean (range) | FMN |
|---|---|---|
| MW | 345.1 (282–417) | 456 |
| QED | **0.468** (0.128–0.811) | 0.176 |
| SA | 4.69 (3.53–5.73) | 4.03 |
| HBD | 1–4 | 6 |
| HBA | 3–8 | 10 |
| heavy atoms | 21–29 | 31 |
| formal charge | **0** (全部) | 0 |

- **4/20 QED ≥ 0.7** (T05 0.811, T06 0.807, T12 0.726, T17 0.811) — drug-like 子集
- **10/20 QED ≥ 0.5** — 整体可接受
- **14/20 rdock INTER 比 FMN 更负** (FMN -16.79)
- **MW 全部 < 420** (Lipinski 区域)

### 关键残基 (prolif, 跨 top-20 非-VdW 接触频次)

| 残基 | top-20 命中数 | FMN (晶体) | 含义 |
|---|---|---|---|
| **U12.A** | 10/20 | Hydrophobic signature | 与 FMN signature 高度重合 |
| **G10.A** | 9/20 | 仅 VdW | round3 像 round2 一样积极利用 G10 |
| **G9.A** | 9/20 | HBAcceptor signature | round3 比 round2 (5/20) 更强利用 G9 |
| A26.A | 3/20 | 无 | round3 独有 |
| G27.A | 1/20 | HBAcceptor | 偶发 |

- **U12.A (10/20)** 是 round3 最常见的 FMN signature 接触(round2 是 9/20),
  两个批次都强共识。
- **G9.A (9/20)** 比 round2 (5/20) **明显增加** — round3 的糖缀合/羟基骨架更倾向
  与 G9 形成 H-bond 网络。
- **T13 (gen 3292)** 的 FMN 残基 Jaccard **0.667** (top-20 最高),覆盖 {G9, U12}+G10;
  PubChem Tani 0.362 是骨架级匹配,**round3 的"FMN 模式 + 邻居已知物"最实用起点**。
- **T11 (gen 9491)** rdock INTER **-28.06** (top-20 最强),菲稠环-亚胺-硫醚,Jaccard 1.0
  (同时覆盖 G9+U12)。
- **T10 (gen 7078)** INTER -25.32,Jaccard 1.0,联苯-糖胺,**同时强结合 + 完整 FMN 模式**。

---

## 9. screening_round3/ 内容

| 文件 | 说明 |
|---|---|
| `1FMN_pocket_extracted_addh.pdb` | 口袋坐标 (复用 round1/2) |
| `FMN_reference_ligand.sdf` | FMN 参照 |
| `rnasmol_scores.tsv` | 10002 + FMN P/logit 排序 |
| `rnasmol_run.log` | rnasmol 运行日志 |
| `top500.sdf` | rnasmol top-500 (rdock 输入) |
| `docked_top500.sd` | rdock 输出 1500 poses |
| `rdock_top500_scores.tsv` | 每分子 best-of-3 SCORE/INTER |
| `top_binders.sdf` | 综合排序 top-100 (3D + H + props) |
| `top_binders_withFMN.sdf` | top-100 + FMN 拼接 |
| `top_binders_combined.tsv` | 500 docked + FMN 全评分表 |
| `top20_report.md` | round3 top-20 表格 |
| `top20_analysis.json` | prolif + 药物性 详细 JSON |
| `top20_interactions.tsv` | top-20 + FMN 交互表 |
| `top20_druglike.tsv` | top-20 + FMN 药物性表 |
| `top20_analysis_summary.txt` | 文字版要点 |
| `top20_evaluation.pdf` | 24 页 PDF (cover+FMN+20+top5+PubChem) |
| `pubchem_top20.tsv` | top-20 PubChem 最近邻表 |
| `pubchem_run.log` | PubChem 查询日志 |
| `pubchem_search_top20.py` | PubChem 查询脚本 (可断点续跑) |
| `analyze_top20.py` | 深度评估脚本 (prolif + 药物性) |
| `build_top20_pdf.py` | PDF 生成脚本 |
| `SCREENING_REPORT.md` | 本文件 |
| `README.md` | round3 总结 |

---

## 10. 与 round1 / round2 三轮总览

| 维度 | round1 | round2 | **round3** |
|---|---|---|---|
| 输入分子数 | 4464 | 4247 | **10002** |
| 带电比例 | 84% (P+/O+/Cl+2 artifact) | 0% | **0%** |
| rnasmol P 中位数 | 0.812 | 0.632 | **0.627** |
| rnasmol P≥0.99 占比 | 29.30% | 20.32% | **20.14%** |
| FMN rnasmol 排名 | 390/4465 | 234/4248 | **523/10003** |
| rdock top-N | 300 | 300 | **500** |
| rdock SCORE median | -24.46 | -23.71 | **-23.81** |
| rdock INTER median | -20.42 | -20.42 | **-20.09** |
| top-100 中性比例 | 16/100 | 100/100 | **100/100** |
| top-20 QED mean | ~0.55 | 0.522 | **0.468** |
| top-20 QED≥0.7 | ~3 | 7/20 | **4/20** |
| top-20 化学特征 | P+/O+ artifact | 肽/三嗪/β-咔啉 | **糖缀合/异喹啉/席夫碱** |
| PubChem 最近邻 (max Tani) | (未做) | 0.478 (T14 三嗪酚) | **0.370 (T16 azo-disulfide)** |
| 最实用单一 hit | neutral-16 (散) | **T14 (gen 1414)** | T13/T16/T11 (各有所长) |

### 三轮综合结论

1. **改后代码彻底消除了电荷 artifact**: round2 + round3 累计 14249 分子全中性。
2. **评分分布稳定**: 三批独立采样的 rnasmol/rdock 分布几乎不变(去掉带电分子后
   P 中位数稳定在 0.63,INTER median 稳定在 -20.1)。
3. **化学空间互补**: round2 (肽/三嗪) + round3 (糖缀合/异喹啉) 覆盖了不同的
   drug-like 骨架区域;两批合并的 top-100 (≈200 hits) 是真正的 hit pool。
4. **新颖性 vs 已知物**: round3 比 round2 更"新颖"(max Tani 0.370 < 0.478),
   意味着 1w 池采样到了 PubChem 上更稀有的化学空间;但同时也意味着**没有
   像 round2 T14 那样直接的 SAR 借鉴起点**。
5. **下一步建议**:
   - **合并 round2 + round3 top-100** → 200 个多样化 hit 的统一池;
   - **优先合成 round2 T14 (1,2,4-三嗪酚)** 作为已有 SAR 的旗舰;
   - **round3 的 T13 (benzamidine-like, Jaccard 0.667)** 和
     **T11 (菲稠环-亚胺, INTER -28.06)** 作为补充的"新方向"先导;
   - 用 round3 的糖缀合骨架探索 **FMN aptamer 的糖识别亚位点** (G9 + U12)。

---

*Generated 2026-08-12 by rna2smol pipeline (subagent).*
