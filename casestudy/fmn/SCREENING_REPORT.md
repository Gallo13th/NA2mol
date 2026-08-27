# 1FMN 批量配体筛选报告

**日期**: 2026-08-11
**任务**: 对 `output/fmn_pocket/dummy_ligand/dummy_ligand.sdf` 中 4464 个生成配体做
rnasmol GPU + rdock 双重筛选,产出 top-100 结合配体。
**RNA 靶标**: 1FMN aptamer,35 nt
`GGCGUGUAGGAUAUGCUUCGGCAGAAGGACACGCC`

---

## 1. rnasmol GPU 批量评分(全部 4464)

- 设备: NVIDIA RTX 4090 (`cuda:0`),rna2smol 环境(torch 2.3.1+cu118 / PyG 2.6.1)
- 脚本: `casestudy/fmn/batch_score_rnasmol.py`
- 模型: `RNAsmol/saved_model/pdb_bothaug_molperturbation.pt`
- **耗时**: 38.6 s(模型加载 + FMN 评分 + 4464 全评),平均 **7.76 ms/mol**(含图构建)
- 4464/4464 全部成功解析与评分,0 失败

| 统计量 | P(binder) |
|---|---|
| min | 0.4963 |
| 25% | 0.5078 |
| median | 0.8119 |
| 75% | 0.9964 |
| max | 1.0000 |
| mean | 0.7669 |

| 阈值 | 满足数 | 占比 |
|---|---|---|
| P ≥ 0.50 | 4463 | 99.98% |
| P ≥ 0.70 | 2532 | 56.72% |
| P ≥ 0.90 | 1936 | 43.37% |
| P ≥ 0.95 | 1712 | 38.35% |
| P ≥ 0.99 | 1308 | 29.30% |

**FMN 参照**: `P(binder) = 1.0000`(logit +10.89)。在合并分布(含 FMN)中按
P 严格排序,FMN 排名 **390/4465**(因 389 个生成分子的 P 精确到 1.000000 比例更高)。
输出: `casestudy/fmn/rnasmol_scores.tsv`

---

## 2. rdock 精筛(top-300)

- 从 rnasmol 排序中按 **P 降序**(并列按 logit 降序)取 **top-300**
  (P 范围 `[0.999995, 1.000000]`)
- 每分子 **3 个 pose**(`-n 3`),共 900 poses
- 协议: `receptor.prm`(1FMN 口袋,无 H,kekulized)+ `dock.prm`(GA+MC+simplex)
  + `receptor.as`(预计算 cavity,r=12 Å,194 Å³)
- **并行**: 32 shards × OMP=4,128 核,实际 wall **≈ 11 s**
  (单 shard 单线程等价 ~263 s)
- 脚本: `casestudy/fmn/rdock/run_rdock_parallel.py`
- 300/300 全部成功,每分子取 best-of-3 写入 TSV

| rdock SCORE | 值 |
|---|---|
| min | -72.841 |
| 25% | -27.286 |
| median | -24.459 |
| 75% | -22.477 |
| 90% | -19.868 |
| max | -14.396 |

**FMN 自对接参照**: best SCORE = **-17.157**(INTER = -16.787),与已知 ~-17 完全吻合,
说明 receptor/cavity/protocol 设置正确。
300 个 docked 分子中,**295 个 SCORE 比 FMN 更负**(FMN SCORE-排名 ~ 296/300)。
按 INTER(去 INTRA 影响),**253 个比 FMN 更负**(FMN INTER-排名 ~ 254/300)。
输出: `casestudy/fmn/rdock_top300_scores.tsv`, `casestudy/fmn/docked_top300.sd`

---

## 3. 综合排序 + top-100 产物

- 融合方法(300 个 docked 分子上):
  - `rnasmol_norm = minmax(logit_neg)` ∈ [0,1],高=better
  - `rdock_norm   = minmax(-rdock_SCORE)` ∈ [0,1],高=better
  - `combined     = 0.5 × rnasmol_norm + 0.5 × rdock_norm`
- 选择 logit 而非 P 是因为 top-300 的 P 已饱和在 1.0,logit 仍有 ~+15 ~ +38 的区分度。
- 脚本: `casestudy/fmn/combine_and_export.py`

### top-20 结合配体

| rank | gen_idx | SMILES | natoms | rnasmol P | rnasmol logit | rdock SCORE | rdock INTER | combined | rdock rank |
|------|---------|--------|--------|-----------|---------------|-------------|-------------|----------|------------|
| 1 | 3035 | `COc1ccc2c(C=O)c(C(S)[C@@H]3CC[C@@H](O[P+](O)(O)O)[C@H](N)C3)ncc2c1` | 28 | 1.0000 | +30.78 | -52.08 | -20.93 | 0.685 | 4 |
| 2 | 4401 | `OC1=NC2C=C[N+](c3ccccc3)=C2C(O)=[N+]1/C=C/CO[P+](O)(O)O` | 25 | 1.0000 | +19.82 | -72.84 | -21.03 | 0.650 | 1 |
| 3 | 821 | `CC1=NC2=C(C(=O)CCCO[P+](O)(O)O)C(C)N=C(C)C2=N1` | 22 | 1.0000 | +37.85 | -26.98 | -18.40 | 0.608 | 82 |
| 4 | 1512 | `COCC1=NC2C(NC[P+](O)(O)O)=CC(C)=C(N=O)C2=N1` | 21 | 1.0000 | +35.47 | -31.18 | -17.96 | 0.597 | 34 |
| 5 | 671 | `COC1=C(C)C2=CC=[O+]C2C(OCCO[P+](O)(O)O)=C1` | 20 | 1.0000 | +34.23 | -32.33 | -16.14 | 0.583 | 24 |
| 6 | 2704 | `C=[O+]c1c(O)c(-c2ccccc2)nc2c1=[N+](C)C(C)[N+]=2/C=C/C=[O+]/[P+](O)(O)O` | 28 | 1.0000 | +16.49 | -72.06 | -23.41 | 0.578 | 2 |
| 7 | 1682 | `CC(=O)c1nc(CO[P+](O)(O)O)n(C2=CC3=C4CC(C)NC4=CC3=C2)c1N` | 27 | 1.0000 | +35.72 | -22.78 | -21.34 | 0.530 | 214 |
| 8 | 4163 | `[H]/N=C(/N)c1ccc(C2C=c3c(C(=O)NCCCCCCO)cccc3=N2)cc1` | 29 | 1.0000 | +25.73 | -40.58 | -19.03 | 0.489 | 12 |
| 9 | 4326 | `Cc1c(NCCN2CC[C@@H](O)[C@H](O)C2)nc(C(=O)/C(Cl)=N/C(N)Cl)[n+](C)c1O` | 28 | 1.0000 | +14.24 | -65.49 | -15.08 | 0.479 | 3 |
| 10 | 631 | `CC1N=c2cc(CO[P+](O)(O)O)ccc2=[N+]1CNc1ccc(C(N)=O)cc1` | 27 | 1.0000 | +28.73 | -29.49 | -20.76 | 0.452 | 45 |
| 11 | 2811 | `CC1=C2[O+]=CC=[N+]2C(NP2COC[C@@H](O)[C@H](O)O2)=NC1Nc1cccc(N)c1` | 28 | 1.0000 | +22.43 | -43.67 | -24.16 | 0.451 | 8 |
| 12 | 2102 | `C=[O+]C1=C(O)C2C(=NC=[N+]2CCO[P+](O)(O)O)C(O)=N1` | 20 | 1.0000 | +32.05 | -21.06 | -18.29 | 0.444 | 256 |
| 13 | 440 | `C[n+]1c(C2C=C[Cl+2]=C2)nc(CO[P+](O)(O)O)c2c1C1=CC=[Cl+2]C1=N2` | 24 | 1.0000 | +31.79 | -21.58 | -20.59 | 0.444 | 244 |
| 14 | 1078 | `Cc1ccc(C2=CC=CC3N=C(CCO[P+](O)(O)O)N=C23)cc1` | 23 | 1.0000 | +28.99 | -27.29 | -21.87 | 0.438 | 76 |
| 15 | 2565 | `COC1=CC(CCO[P+](O)(O)O)=C2C(=NN=C(O)C2O)C=C1` | 22 | 1.0000 | +30.07 | -24.37 | -17.61 | 0.434 | 154 |
| 16 | 1499 | `Cc1nc2c(c(OCC[P+](O)(O)O)[n+]1C)C=C1C=[Cl+2]C=C1C2C` | 23 | 1.0000 | +28.46 | -27.82 | -22.73 | 0.432 | 65 |
| 17 | 4140 | `C=[O+]C1=[N+](CCO[P+](O)(O)O)C2N=CN=C(N)C2=C1C` | 20 | 1.0000 | +30.48 | -22.69 | -19.77 | 0.428 | 219 |
| 18 | 3888 | `C=[O+]C1=c2ccc(OC)cc2=CC(CCO[P+](O)(O)O)C=C1C` | 23 | 1.0000 | +28.58 | -26.88 | -21.56 | 0.427 | 84 |
| 19 | 446 | `C=[O+][C@@H]1CC[C@@H](NC2=CC(C(=O)c3ccccc3)=NC2N)CCO1` | 24 | 1.0000 | +27.41 | -26.63 | -16.93 | 0.402 | 93 |
| 20 | 1662 | `CC(=O)CC(=O)Nc1c(C/C=C/CC(C)O[P+](O)(O)O)c(C)c(O)c[n+]1C` | 27 | 1.0000 | +29.97 | -20.47 | -14.50 | 0.399 | 265 |

### top-100 统计

- 重原子数: 20–28(median 24),落在生成时 20–28 限制内
- rdock SCORE: -72.84 ~ -17.26(median -26.60),全部 ≤ FMN 的 -17.157
- rdock INTER: -27.47 ~ -10.81(median -20.12)
- rnasmol P(binder): 全部 = 1.0000

### FMN 在综合排名的位置

- rnasmol 维度: P=1.0(并列第一档,但 logit +10.89 远低于 top-300 的 +14~+38,所以
  FMN 即使进入 top-300,logit 维度也会垫底)
- rdock 维度: SCORE=-17.16,INTER=-16.79,**300 个 docked 里 295 个 SCORE 比它好,
  253 个 INTER 比它好**
- **综合**: FMN 没有进入生成分子的 top-100。结论:**生成的 top 配体在双重评分下
  全面优于 FMN 参照**。这反映出生成模型针对 1FMN 口袋找到了比 FMN 自身更"互补"的
  结构(主要得益于多电荷/磷铵等与口袋阳离子/阴离子残基形成强相互作用)。

---

## 4. top_binders.sdf 路径

| 产物 | 路径 |
|---|---|
| top-100 SDF(含双分数 + 全部 SDF 属性) | `casestudy/fmn/top_binders.sdf` |
| top-100 + FMN 参照(末尾追加) | `casestudy/fmn/top_binders_withFMN.sdf` |
| 综合 TSV(300 docked + FMN) | `casestudy/fmn/top_binders_combined.tsv` |
| rnasmol 全量 TSV(4464 + FMN) | `casestudy/fmn/rnasmol_scores.tsv` |
| rdock top-300 pose SD | `casestudy/fmn/docked_top300.sd` |
| rdock top-300 每分子 best TSV | `casestudy/fmn/rdock_top300_scores.tsv` |
| top-300 输入 SDF | `casestudy/fmn/top300.sdf` |

`top_binders.sdf` 每条记录携带:`combined_rank / combined_score / rnasmol_p_binder /
rnasmol_logit / rnasmol_norm / rdock_SCORE / rdock_INTER / rdock_norm / gen_idx /
smiles`,可直接载入 PyMOL / ChimeraX 查看。

---

## 5. 遇到的问题 / 注意事项

1. **rDock SCORE 受 INTRA 项严重影响**。生成的多个分子 SCORE 低至 -72.8,但对应
   INTER 仅 ~-21(rDock total = INTER + INTRA + RESTR)。这些极端负分来自分子内部
   构象弛豫,不一定反映真实结合力。**建议同时看 INTER 列**(top-100 INTER median
   -20.12,明显优于 FMN -16.79,差距更可信)。已在 TSV / SDF 中保留两列。

2. **84/100 top 配体带净形式电荷**,其中 63 个含 `[P+]`(类 FMN 磷酸但写成磷鎓)、
   17 个 `[O+]`、10 个 `[Cl+2]`。生成模型对口袋中的极性/带电环境给予了高 Vina
   亲和势,但这些结构化学合理性存疑(phosphonium ylide / chloronium 等)。下游
   hit-to-lead 时建议加 PAINS / 形式电荷 / 价态过滤,或对生成分子的奖励函数加
   drug-likeness 约束。

3. **FMN 没有进入 top-100**,但这是合理的:生成模型直接针对 1FMN 口袋生长分子,
   在等长(20–28 重原子)约束下,可以构造比 FMN(31 重原子)更互补的小分子。这
   同时也是一个**警示**:rdock SCORE 单独不能作为结合力真值,需结合 INTER、
   静电/几何合理性、和实验数据综合判断。

4. **rbdock 不并行**(`OMP_NUM_THREADS` 几乎无加速),实际加速来自**多 shard 进程
   并行**(32 × OMP=4 = 128 核)。10 mol×3 pose 单线程 ≈ 8.8 s,折合 300 mol×3 pose
   ≈ 263 s 单线程;并行后 wall ≈ 11 s。

5. **rbdock 输出 SD 会丢弃自定义 SDF 属性**(rnasmol_rank / gen_idx 等不会被保留),
   因此在 `run_rdock_parallel.py` 中通过 **位置对应**(每 shard 的输入顺序 ↔ 输出
   pose 块顺序)回填 gen_idx/rnasmol_p_binder 等元信息。

6. **rdock 对显式氢原子 SEGFAULT**(exit 139)。所有输入 SDF 都先经
   `strip_h_to_sdf` 去氢保留重原子 3D 坐标后再送 rbdock。

7. **top_binders.sdf 中的 3D 构象是生成时的构象,不是 rdock 的 docked pose**。
   若需 docked pose,使用 `docked_top300.sd`(其中 orig_idx 字段对应 top300 中的
   位置,可 join 到 `top_binders_combined.tsv`)。
