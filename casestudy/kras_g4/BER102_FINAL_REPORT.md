# BER102 最终报告 — rDock 协议固化 + 花菁对接 + 轨迹原子概率

日期: 2026-08-15 | 环境: conda `rna2smol` (rDock 2013.1 bioconda), RBT_ROOT=`$CONDA_PREFIX/share/rdock-2013.1-1`
所有新建文件: `casestudy/kras_g4/rdock_protocol/`(固化协议) 与 `casestudy/kras_g4/ber102_rdock/`(本案例)

---

## 1. rDock 协议固化(Step1)

### 1a-1b. BER102 口袋与配体提取
- **口袋**(MODEL 1,中心-中心 14 Å,与 BER101 时代 genbypkt 惯例一致): **15 残基**
  (12×DG, 2×DA, 1×DT),493 原子/326 重原子 → `ber102_rdock/ber102_pocket.pdb`
  (对照: BER101 口袋 13 残基 427 原子)
- **配体提取**(SMILES 图映射法,纠正 BER101_ref.sdf 孤立原子教训):
  - 任务给的 SMILES `COc1ccc2cc3[n+](cc2c1)CCc1cc2c(cc1-3)OCO2` 经验证**错误**
    (23 重原子,PDB 为 25)——缺第二个甲氧基;采用 `COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2`
  - 图同构匹配**唯一**,逐原子键级校验通过
  - 验证: C20H18NO4+ / 电荷 +1 / 单一 fragment / 无孤立原子 / 映射键长 1.36–1.53 Å
  - → `ber102_ref.sdf`(25 重原子,无极性 H,符合 rDock 契约)

### 1c-1d. 受体 + cavity + ground truth
| 项 | 数值 |
|---|---|
| 受体 | MODEL-1 全部核苷酸 24 残基(778 原子)→ 全氢 mol2 → 自剥 211 非极性 H + 23 磷酸误加 H,保留 59 极性 H → 567 原子 |
| 磷酸分布电荷 | substructure 名规范化 DG→G 等 + OP1/OP2→O1P/O2P,匹配 RbtIonicAtoms.prm |
| cavity (RADIUS 10) | **6154.9 Å³** — 太松,pose 游走(见下) |
| cavity (RADIUS 6) | **1534.4 Å³** — 采用 |
| 协议 | `$RBT_ROOT/data/scripts/dock_solv.prm`(SF5 含去溶剂化,RNA 靶向必用;已确认存在) |

**Ground truth redock 矩阵**(排序一律 SCORE.INTER,门限 RMSD<2.5 Å):

| 尝试 | RADIUS | n | best SCORE.INTER | RMSD in-place | 判定 |
|---|---|---|---|---|---|
| 口袋受体 | 10 | 10 | -40.52 | 3.59 Å | FAIL |
| 全受体 | 10 | 10 | -40.52 | 3.59 Å | FAIL |
| **全受体** | **6** | **10** | **-44.995** | **1.230 Å**(aligned 0.552) | **PASS** |

→ 协议固化成立。固化脚本 `rdock_protocol/`:
`extract_pocket.py` / `extract_ligand.py` / `fix_mol2.py` / `prepare_receptor.sh` /
`make_cavity.sh` / `redock_validation.py` / **`PROTOCOL.md`**(用法清单+踩坑记录)

### 1e. 1FMN 回归测试(通用性验证)
- FMN 共晶配体: 31 重原子 + 6 极性 H;图同构唯一映射,公式 C17H21N4O9P ✓
- **发现关键契约**: 配体 SD 必须显式极性 H(否则 6×"too few bonds"、供体相互作用全丢);
  且不能带非极性 H(触发 rDock 配体侧 RemoveNonPolarHydrogens segfault)
- RADIUS 6:n=25 FAIL(2.71/5.15 Å;诊断:异咯嗪环恢复 0.55 Å,柔性核糖醇尾漂移)
- **RADIUS 5:n=25 PASS: best INTER -58.976,RMSD 2.009 Å** → `regression_1fmn/`

### 踩坑记录(全部 gdb/二分定位,详见 PROTOCOL.md)
1. **rDock 2013.1 segfault bug**: MOL2/SD 中非极性 H 让 `RemoveNonPolarHydrogens()`
   在 erase-while-iterate 中崩溃——受体和配体两侧都中招;自己预先删除即可
2. obabel `-h` 误质子化磷酸(11–23 个/体系);substructure 名 `DG11` 需剥号并 DNA→RNA 映射
3. mol2 必须保留 obabel 定宽列布局 + 一致的 MOLECULE 计数 + SUBSTRUCTURE 记录
4. `SITE_MAPPER` 是正确键名;`GRID_SPACING` 不存在(应为 `GRIDSTEP`)
5. 芳香 N+ 的 "makes too many bonds" 告警可容忍(BER102 以 1.23 Å PASS);
   但 rDock 输出 SD 键级会被降级——**禁止子结构匹配输出,坐标按位直读贴回**
6. RADIUS 处方 10 Å 对小平面配体产生 6200 Å³ 海绵腔;5–6 Å(1500–2200 Å³)+ redock 验证为准则

---

## 2. 目标花菁 → BER102 对齐 + redock(Step2)

目标: `CCN1C=C/C(=C/C=C/c2cc[n+]3ccccc3n2)c2ccccc21`(25 重原子: 22C+3N,平面)
- 3D 构象: ETKDG(seed 42)+ MMFF 收敛
- MCS 对齐: 公共子结构 = 完整芳香六元环(6 原子),对齐 RMSD **0.076 Å**;
  对齐后质心距 berberine 6.43 Å(环叠合、分子延伸不同所致)
- **redock(n=20, seed 42, R6 已验证系统)**:

| 统计 | 数值 |
|---|---|
| best SCORE.INTER | **-46.124**(run 7) |
| 前 5 pose | -46.12 / -45.47 / -42.39 / -42.35 / -41.87(主收敛) |
| 最优 pose vs berberine 晶体位 | **质心距 0.30 Å / 色素环二面角 6.9° / 最小原子距 0.34 Å** |

**构象结论: 叠加而非位移**——花菁以近乎共面、质心重合的方式占据 BER102 位点
(即 2:1 复合物中 berberine 的堆叠平面),与 berberine 呈 π-堆叠镜像占位;
打分(-46.1)与 berberine redock 自身(-45.0)同量级,位点兼容。

---

## 3. 轨迹原子概率(Step3,BER102 构象)

协议与 BER101 时代完全一致(同 ckpt/口袋 featurize/t=1/β1/γ,芳香感知信念,
masked probe 逐位 mask 其余锁真实元素);骨架 25 原子坐标 = **本次 redock 最优 pose**;
口袋 = BER102 14 Å 口袋(326 重原子)。echo sanity: full-info argmax 19/25(两条件一致)。

### 3N 位概率表(核心)

| N 位(目标序号) | 目标类 | P[N] BER101 时代 | **P[N] BER102 构象** | Δ |
|---|---|---|---|---|
| tgt2 喹啉 N(enamine 端,非芳香) | (7,False) | 57.05% | **9.09%** | −47.96 pp |
| tgt12 桥头鎓 [n+](萘啶端) | (7,True) | 3.81% | **1.02%** | −2.79 pp |
| tgt18 吡啶环 n(萘啶端) | (7,True) | 42.38% | **76.27%** | **+33.89 pp** |

细分通道: tgt2 [N*=5.25%, N=3.84%],P[C]=83.0%;tgt12 [N*=1.00%, N=0.01%],P[C]=98.9%;
tgt18 [N*=76.18%, N=0.10%],P[C]=22.8%(top1 即 7*,76%)

**skel18(tgt12 桥头鎓)在 BER102 好构象下无改善**: 3.81%→1.02%,两口袋均拒绝(该位
在两构象中都被模型判为 C,与其处于共轭桥头的化学环境一致性弱)。

### Exp3 完整目标元素自洽度(3N+22C 全锁,逐位 mask)
- 全 25 位 mean P[true] = **61.47%**(16/25 位 >50%);最低 tgt12 = 1.03%
- N 位 P[true]: tgt2 = 3.74%(BER101 31.13%)/ tgt12 = 1.03%(2.05%)/ **tgt18 = 79.10%**(28.06%)
- 高背书位: 萘啶端芳香环 tgt9–11/13/15–17/19/21–22/24 ≈ 88–99.6%;
  低背书位集中于喹啉端共轭链/二氢环(tgt1–8 区,与 BER101 时代互补)
- 全部 25 位概率表: `ber102_rdock/probe_ber102_results.json`(exp1/exp3 各含 13 类全概率)

---

## 4. 与 BER101 时代对比 + 最终判定

| 维度 | BER101 时代 | BER102 本次 |
|---|---|---|
| 对接引擎 | smina(非 rDock 惯例) | rDock ground-truth 验证协议(redock 1.23 Å PASS) |
| 构象来源 | ETKDG→align→smina | MCS 对齐→rDock redock(与 berberine 晶体位质心差 0.30 Å) |
| 喹啉 N(tgt2) | 57.1%(背书) | 9.1%(撤回) |
| 桥头鎓(tgt12) | 3.8%(拒绝) | 1.0%(拒绝,更坚定) |
| 吡啶 N(tgt18) | 42.4%(弱背书) | **76.3%/79.1%(强背书)** |
| Exp3 自洽 | mean 62.6%,19/25>50% | mean 61.5%,16/25>50% |

**最终判定(目标分子在 well-buried 构象下的模型概率背书全景)**:
在经 ground-truth 验证的 rDock 协议下,花菁以与 berberine 共面叠加的高亲和构象
(SCORE.INTER −46.1)进入 BER102 位点;该构象下 BFN 模型的元素概率明确背书
**萘啶端吡啶 N(tgt18,~76–79%)**、明确拒绝**桥头鎓(tgt12,~1%)**,
而 BER101 时代对喹啉端 enamine N(tgt2)的背书(57%)在 BER102 构象下**消失**(9%)。
即:模型支持"萘啶端保留双 N、喹啉端 N→C"的元素分配方向;三 N 全保留的花菁
在两 pocket 构象下均无完整概率背书(tgt12 始终 <4%)。

## 文件清单
- `rdock_protocol/` — 固化协议(extract_pocket / extract_ligand / fix_mol2 /
  prepare_receptor.sh / make_cavity.sh / redock_validation.py / PROTOCOL.md)
- `rdock_protocol/regression_1fmn/` — 1FMN 回归(PASS 产物;失败尝试在 `_attempts/`)
- `ber102_rdock/` — BER102 全轨迹(口袋/参考配体/受体 R6/cavity/redock PASS 报告/
  花菁对齐+对接/骨架/probe 结果 json+npy;RADIUS 10 失败尝试在 `_attempts_radius10/`)
