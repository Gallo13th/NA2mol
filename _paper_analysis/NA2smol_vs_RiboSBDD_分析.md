# NA2smol（本地论文）与 RiboSBDD（ICLR 2027 投稿 #27761）对比与"撞车"评估

分析时间：2026-11-21（Asia/Shanghai）
输入材料：
- NA2smol：远程 `bicourse-glt:/home/data/users/gaoletian/projects/NA2smol_paper`，本次重新同步后的本地副本
  `_paper_download/NA2smol_paper_20261121/NA2smol_paper/`（96 个文件，24.4 MB；tar md5 `867f7a66…e844c8`）
- RiboSBDD：`C:\Users\letgao\Downloads\27761_RiboSBDD_and_RiboSBDDBen.pdf`（10.4 MB，pdfTeX，PDF 创建时间
  2026-11-16 23:39 -05:00），文本抽取结果见 `ribosbdd_clean.txt`

---

## 0. 结论摘要（TL;DR）

1. **不是抄袭，也不是数据/文本复用。** 两篇的模型架构、训练数据、评测流程、基准构建都不同；互不引用
   （RiboSBDD 公开时 NA2smol 尚未公开，NA2smol 成稿时也未引用 RiboSBDD）。按现有证据只能认定为
   **同期独立工作**。
2. **但存在实质性的"撞题"（priority collision）。** 两者对研究空白的判断几乎同义：RNA/核酸口袋条件的
   3D 分子生成尚未被解决；两者都主打"蛋白预训练 → RNA 迁移"+"跨域共享表征"，并都在回应该空白。
   两篇各自"首次"的表述互相覆盖。
3. **按可核实的时间线，如果一定要说谁被"抢先"，更接近 NA2smol 被 RiboSBDD 抢先公开**：
   - RiboSBDD：ICLR 2027 投稿 #27761，OpenReview 记录创建于 **2026-09-17（UTC）**，`readers: everyone`
     （匿名但公开可读）；ICLR 2027 论文截稿 2026-09-25。
   - NA2smol：代码仓库 `Gallo13th/NA2mol` **2026-10-29** 才在 GitHub 公开（`created_at 2026-10-29T01:52:49Z`，
     public，MIT）；手稿 2026-11-09～13 才进入 Nature 系列预投稿/投稿流程，无公开预印本。
   - 内部开发更早（checkpoint 日期 2026-06-17、仓库初始提交 2026-08-27）只能证明独立开发，不能构成公开优先权。
4. **但也不是"整篇被 scoop"。** RiboSBDD 没有覆盖 NA2smol 的差异化贡献：ED 图像预训练 + 教师-学生
   对齐、DNA/RNA 通用性、KRAS G4 湿实验验证（ThT，DC50 22.4 µM）、四层虚拟筛选级联。
   NA2smol 仍可独立发表，但"首次"叙事需要弱化，RiboSBDD 需要作为 concurrent work 讨论/引用。
5. **反向不成立**：RiboSBDD 不可能被 NA2smol 抢先（其公开时间早于 NA2mol 仓库公开），且二者方法无关。

---

## 1. 事实基础

### 1.1 本次同步的 NA2smol 版本（2026-11-21）

| 项 | 值 |
|---|---|
| 远程目录 | `/home/data/users/gaoletian/projects/NA2smol_paper` |
| 远端文件数 / 大小 | 96 个文件 / 24 MB |
| 打包 | `/tmp/NA2smol_paper_20261121.tar.gz`，21,712,737 B，md5 `867f7a665824e4c78968330559e844c8` |
| 本地解压 | `_paper_download/NA2smol_paper_20261121/NA2smol_paper/`，96 个文件，24,391,980 B |
| 与 11-07 版本的差异 | 57 → 96 个文件：新增 `extended_data.tex/.pdf`、`supplementary.tex/.pdf`、`figures/figure1_nbt.*`～`figure4_nbt.pdf`、`cover_letter_natmethods.docx`、`NA2smol_presubmission_NMET.pdf`、`si_fig_mapping_control.*`、`si_table_mapping_control.tex`、`table_overlap.tex`、`table_radar_values.tex`、`scripts/split_si_for_nbt.py` 等，以及 `docs/` 下的 NBT 投稿要求、SMARTFlexDB、内部审稿意见等 8 份新文档 |

注：远端 `_papersync/`（本次为绕过同步限制临时创建的解包目录）与本地镜像根目录 `NA2smol_paper/`（remote_sync 的残缺副本）已清理。

### 1.2 RiboSBDD 的来源识别（关键证据）

- PDF 元数据：`Producer = pdfTeX-1.40.29`、`Creator = LaTeX with hyperref`、标题
  "RiboSBDD and RiboSBDDBench: Structure Based Small Molecule Generation for RNA"、作者 "Anonymous authors"、
  页眉 "Under review as a conference paper at ICLR 2027"。
- OpenReview API（`api2.openreview.net/notes/search`）返回该投稿：
  - `forum: n8QuaZwPIm`，`number: 27761`（与文件名 `27761_…` 一致），`venue: ICLR 2027 Conference Submission`
  - `cdate/tcdate = 1789613007904 ms` → **2026-09-17 02:43 UTC**
  - `mdate/tmdate = 1794891132149 ms` → **2026-11-17 04:12 UTC**，`version: 2`
    （即 v2 为 ICLR 答辩期 `Rebuttal_Revision` 修订版；用户手上的 PDF 正是这一版：PDF 生成于 2026-11-16 23:39 EST）
  - 访问权限 `readers: ["everyone"]`——匿名但公开可读。
- ICLR 2027 官方 CFP：摘要截稿 2026-09-18、论文截稿 **2026-09-25**、评审意见 11-05、答辩 11-05～11-18、
  **最终决定 2026-12-16**。

### 1.3 NA2smol 的公开时间线

- 模型 checkpoint 文件时间：**2026-06-17**（说明模型与实验远早于两篇论文的公开时间）。
- `Gallo13th/NA2mol` GitHub：`created_at 2026-10-29T01:52:49Z`、`pushed_at 2026-10-29T02:08:30Z`、
  `visibility: public`、MIT；仓库含模型、数据、case study、图源。
- 本地仓库提交：`2026-08-27 rna2smol: publication release`（初始提交）、`2026-10-29 prepare the repository for public release`。
- 手稿包：`CURRENT.md` 记录 2026-11-06 生成；预投稿相关文件 2026-11-09～13；按
  `docs/NBT_submission_requirements_2026-11-13.md`，最新目标为 Nature Biotechnology（NBT 不接受 presubmission enquiry，
  故改为直接投稿；正文 2,841 词，NBT 上限 3,000 词）。

---

## 2. 两篇论文在做什么

### NA2smol（核酸口袋 → 3D 分子生成，BFN）

- 声称的空白：3D 配体结构此前没有被"直接从显式给定的核酸口袋的几何与物理化学环境"生成过
  （原文："To our knowledge, three-dimensional ligand structures have not previously been generated directly
  from the geometry and physicochemical environment of an explicitly supplied nucleic-acid binding pocket."）。
- 方法：Bayesian Flow Network；口袋只用**跨蛋白/核酸共享的物理化学描述子**（药效团、杂化态、vdW 半径、部分电荷），
  完全不用残基身份；两阶段训练——2.4 M 配体模拟电子密度（ED）预训练得到冻结教师，再用
  "教师-学生对齐 + 跨视图对比"把口袋条件学生对齐到 ED 教师；微调数据 = 126,899 蛋白复合物 + 738 条核酸记录
  （457 RNA / 274 DNA / 7 hybrid；356 训练 / 382 验证）。
- 评测：RNA 测试集 37 个系统（27 个可评估，SMRTnet 方法构建）+ DUD-E 蛋白对照（101 targets）；6 个外部
  蛋白训练生成器（MolCRAFT、Pocket2Mol、ED2Mol、DiffSBDD、GraphBP、FLAG，公开 checkpoint + 伪残基赋值）；
  smina 对接、RMSD、QED、SA、SEI、LLE、PoseCheck 应变能、ECFP4 参考回收、四层筛选级联、骨架多样性。
- 结果：RNA 测试集多目标保留与 ECFP4 回收在 6 个基线中最好；最严过滤保留 239 个分子 / 17 个 target；
  DUD-E 保留 3,726 / 83 个 target。
- 实验：KRAS 启动子 G 四链体 BER102（PDB 7X8M）案例——10,077 个生成分子 → 293 个 Tier-1 → 14 个可购买 →
  候选 CID 74755（2,2′-亚甲基双苯并噻唑）经 ThT 竞争实验验证，DC50 = 22.4 µM。

### RiboSBDD（RNA 口袋 → 3D 配体生成 + RiboSBDDBench）

- 声称的空白：RNA 的 SBDD 受"结构稀缺 + 口袋柔性"限制；蛋白有成熟的 pocket-conditioned 生成模型，
  RNA 没有；"We solve the inverse design problem: given a 3D RNA binding pocket, RiboSBDD generates novel
  3D small molecules directly inside that cavity."
- 方法：SE(3) 等变扩散（TargetDiff 式坐标 + 离散原子类型）；**统一 all-atom 表征**（20 aa + 4 nt 的
  24 类残基词表，共享原子特征）把蛋白学到的几何先验迁移到 RNA；**局部碱基校正**——固定骨架下让核碱基绕
  糖苷键 χ 旋转，用合成旋转做解析逆监督，并通过可微旋转把碱基校正头与配体去噪耦合；**自适应源-目标采样**——
  推理时保留冻结的蛋白源模型，与 RNA 目标模型按"局部等密度投影 + 势函数引导的粒子重采样"合成坐标提议；
  口袋条件原子数预测器 p(N|C)；125 步采样。
- 数据：RiboSBDDBench——7,453 个含 RNA 的 PDB 结构 → 3,670,853 个候选实例 → 8,416 Broad / 7,469 Clean →
  8,006 model-ready（2,258 个 PDB entry）；7,269 Broad 训练实例；**713 个 Clean 测试口袋**（每组 50 次请求
  = 35,650 次）；按 PDB ID + RNA90 序列簇连通分量做分组划分（训练/测试零共享 PDB、零共享 RNA90 簇）。
- 评测：11 个指标——Valid%、QED、归一化 SA、logP、键长/键角 JS 距离、Clash%、Contact 回收（4 Å 参考接触）、
  AnnapuRNA IMP%、Novelty、Diversity；4 种架构 × 3 种训练 regime（P / R / P→R）= 12 个基线
  （TargetDiff、DiffSBDD、Pocket2Mol、GraphBP，统一 CrossDocked2020 10 万对预训练）；不含对接打分与应变能。
- 结果：Contact 0.76、Clash 5.00%、IMP 70.12%（均优于 12 个基线）；蛋白预训练在 4 种架构上都改善共价几何；
  RNA 微调提高 Contact 但会使 Clash 上升（TargetDiff 0.70→0.73 Contact，5.73%→9.90% Clash），据此提出保留
  蛋白源做自适应采样；OOD 子集（严格无蛋白口袋、低相似配体、未见骨架、未见 Rfam、TM<0.5）仍领先。
- 验证：**无湿实验**，Limitations 明确写 "free-energy and wet-lab validation define next steps"；
  代码/数据/权重"录用后开源"。

---

## 3. 逐项差异

| 维度 | NA2smol | RiboSBDD |
|---|---|---|
| 类型 / 去向 | Nature 系列期刊稿件（NCS / Nature Methods / NBT 三个 cover letter 变体；2026-11-13 起以 NBT 要求为准） | ICLR 2027 双盲会议投稿 #27761 |
| 公开状态 | GitHub 2026-10-29 公开（MIT）；手稿未公开 | OpenReview 匿名公开（v1 2026-09-17，v2 2026-11-17） |
| 核心 gap | 核酸口袋 3D 生成 + 多类核酸数据稀缺 | RNA 口袋 3D 生成 + 数据稀缺 + 口袋柔性 |
| 生成框架 | 贝叶斯流网络（BFN），1000 步采样，kNN 图 k=32 | TargetDiff 式 SE(3) 扩散，125 步采样 + 粒子重采样 |
| 口袋表征 | 共享物理化学描述子（药效团/杂化/vdW/电荷），无残基身份 | 统一 all-atom 图，24 类残基词表，共享原子特征 |
| 迁移机制 | 2.4 M 配体 ED 预训练（冻结教师）+ 教师-学生输出对齐 + 跨视图对比 | CrossDocked2020 10 万蛋白复合物预训练 → RNA 微调；推理保留冻结蛋白源 |
| 口袋柔性 | 未建模（刚性受体） | 糖苷键 χ 旋转 + 可微碱基校正头 + 合成旋转自监督；14 对 apo/holo 验证 |
| 原子数 | 空腔体积 → 分段高斯尺寸模型 | 口袋条件计数预测器 p(N\|C) |
| 微调/训练数据 | 738 条核酸记录（457 RNA / 274 DNA / 7 hybrid；356 训练）+ 126,899 蛋白复合物 | 7,269 个 RNA 训练实例（来自 8,006 model-ready，2,258 PDB entries） |
| 测试集 | 37 个 RNA 系统 → 27 个可评估 target；DUD-E 101 targets（蛋白对照） | 713 个 Clean 测试口袋；35,650 次生成请求 |
| 基线 | 6 个蛋白训练生成器：MolCRAFT、Pocket2Mol、ED2Mol、DiffSBDD、GraphBP、FLAG（公开 checkpoint，RNA 上伪残基赋值） | 4 架构 × 3 regime = 12：TargetDiff、DiffSBDD、Pocket2Mol、GraphBP（统一蛋白预训练，P/R/P→R 对照） |
| 评测指标 | smina 对接、RMSD、QED、SA、SEI、LLE、PoseCheck 应变、ECFP4 参考回收、四层筛选级联、骨架多样性 | QED、SA、logP、键长/键角 JSD、Clash%、Contact 回收、AnnapuRNA IMP%、Novelty、Diversity、Valid% |
| 头号结果 | RNA 多目标筛选保留与 ECFP4 回收最佳；最严过滤 239 分子/17 target；DUD-E 3,726/83 | Contact 0.76、Clash 5.00%、IMP 70.12%，全面优于 12 基线；OOD 子集领先 |
| 湿实验 | 有：KRAS G4（7X8M），ThT 竞争，DC50 22.4 µM | 无（明确列为 future work） |
| 靶点范围 | 核酸（RNA + DNA + hybrid），案例为 DNA G4 | 仅 RNA |
| 数据处理严谨性 | 自认训练/验证记录与 13 个基准系统共享结构 ID（按记录级划分） | PDB + RNA90 分组划分，训练/测试零共享；另设 Rfam/骨架/TM 等 OOD 子集 |
| 代码/数据 | 已公开（MIT）；数据表 CC BY 4.0，"发表后" | "accepted 后开源"；OpenReview 匿名可读 |
| 定位强弱 | 强在"可用性/转化"：湿实验 + 筛选级联 + 跨 DNA/RNA | 强在"基准/系统评测"：8,006 实例、713 测试口袋、12 基线、严格划分 |

---

## 4. 重叠度评估（主观打分，附依据）

| 维度 | 重叠度 | 依据 |
|---|---|---|
| 问题设定与动机 | ~90% | 同一空白（RNA/核酸口袋 3D 生成未解决）、同一痛点（结构稀缺、残基专属编码不可迁移）、同一"蛋白→RNA 迁移"思路 |
| 声称的首创 | ~85% | 两文都把自己定位成该方向的第一个方法（NA2smol 用 "to our knowledge …"；RiboSBDD 用 "We solve the inverse design problem…"） |
| 技术路线 | ~50% | 共享"跨域统一口袋表征 + 蛋白预训练后 RNA 微调"；但 BFN vs 扩散、ED 教师对齐 vs 碱基柔性校正、两套完全不同的采样器 |
| 数据/基准 | ~30% | 都从 PDB/HARIBOSS 体系构建 RNA 复合物数据；但 NA2smol 738 条记录、复用 SMRTnet 测试集；RiboSBDD 8,006 实例、自建分组划分，规模与划分范式不同 |
| 评测指标 | ~40% | 共有 QED、SA、Contact/Clash 类相互作用评估、Novelty/Diversity；NA2smol 另用对接 + 应变 + ECFP4 回收 + 级联；RiboSBDD 用 AnnapuRNA + JS 几何距离 |
| 结论/发现 | ~30% | NA2smol 发现预训练提分但应变升高、对齐修复；RiboSBDD 发现蛋白预训练改善几何、RNA 微调提高接触但增加冲突、需保留源模型——方向相似但结论表述不同 |
| 文本/图/数据复用 | 0 | 无任何共享段落、图表或数据表；参考结构不同 |
| 作者/机构 | 未知/无证据 | RiboSBDD 双盲；NA2smol 作者为清华/动物所/人大团队 |

---

## 5. 时间线与"首发权"

| 日期（本地/UTC） | 事件 | 来源 |
|---|---|---|
| 2026-06-17 | NA2smol 模型 checkpoint 生成（内部） | 远端 `*.pth` 文件时间 |
| 2026-08-27 | `rna2smol` 仓库初始提交（内部/未公开） | git log |
| **2026-09-17 02:43 UTC** | **RiboSBDD 投稿在 OpenReview 创建（公开可读，匿名）** | OpenReview API `cdate` |
| 2026-09-18 / 09-25 | ICLR 2027 摘要 / 论文截稿 | ICLR 2027 CFP |
| 2026-09-03 | 仓库 `main` 分支提交 | git for-each-ref |
| 2026-10-13 | `origin/paper-release` 远端分支更新（仓库仍在私有阶段/未公开） | git refs |
| **2026-10-29 01:52 UTC** | **`Gallo13th/NA2mol` 在 GitHub 公开（public, MIT）** | GitHub API `created_at` |
| 2026-11-05 | ICLR 2027 评审意见发布 | ICLR 2027 CFP |
| 2026-11-06 | NA2smol 论文包定稿（CURRENT.md） | 本地文件 |
| 2026-11-09～13 | NA2smol 预投稿/投稿材料（NBT/NCS/Nat Methods 变体） | 文件时间戳 |
| **2026-11-17 04:12 UTC** | **RiboSBDD v2（答辩修订）公开** | OpenReview API `mdate` |
| 2026-12-16 | ICLR 2027 最终决定 | ICLR 2027 CFP |

**严格按公开时间：RiboSBDD（2026-09-17）早于 NA2smol（2026-10-29 代码公开；手稿 11 月）。**

需要注意的限定：
- 无法从本机核验 RiboSBDD v1（9 月版）的正文内容是否已包含现在的全部主张；但其标题、投稿号未变，
  且核心方法（扩散 + 碱基校正 + 自适应采样）与 9 月投稿周期一致，按常理 v1 已包含核心主张。
- NA2smol 的内部时间更早（6 月模型、8 月仓库），但对期刊/会议而言，**公开披露时间才是优先权依据**；
  内部记录只能用于向编辑说明独立开发，不能推翻公开先后。

---

## 6. "是否被 scoop"的三层判定

**口径一：学术不端（抄袭/剽窃）——否。**
无文本、图、数据表、代码的重叠证据；方法路线不同；双方互不可见（RiboSBDD 9 月投稿时 NA2smol 未公开；
NA2smol 10 月底公开时 RiboSBDD 已公开，但 NA2smol 稿中未引用，说明大概率确实不知道）。

**口径二：概念撞车（scooped 的通俗用法）——是。**
"RNA/核酸口袋条件 3D 分子生成 + 蛋白→RNA 迁移"这一核心卖点被两篇同时占据；两边都把同一空白写成了自己的
主要创新。对读者/审稿人而言，这属于"同一问题、同一年、两个独立解法"。

**口径三：首发权（priority）——是，且方向是 NA2smol 落后。**
RiboSBDD 于 2026-09-17 在 OpenReview 公开，NA2smol 于 2026-10-29 才公开代码、11 月才投稿。
因此：
- **NA2smol 的"首次直接从核酸口袋生成 3D 配体"表述，在 2026-09-17 之后已经不成立**（至少对 RNA 子集）。
  这是 NBT/NCS 编辑或审稿人最可能 challenge 的点。
- **RiboSBDD 不存在被 NA2smol scoop 的问题**（时间更早，且两者无交集）。

但"被 scoop"不等于"不能发表"：RiboSBDD 覆盖的是"方法 + 基准"这一层，NA2smol 的 ED 预训练/师生对齐、
DNA/RNA 通用性、筛选级联、以及**唯一的湿实验验证**都未被覆盖；NA2smol 仍是该方向第一个有实验验证的工作。

---

## 7. 对 NA2smol 投稿的具体建议

1. **改写新颖性表述**（最高优先级）：
   - 把 "To our knowledge, three-dimensional ligand structures have not previously been generated directly from …"
     改为不点名"首次"的写法（如 "we present …" / "among the first …"），或限定到"核酸（含 DNA）口袋 + 湿实验验证"。
   - 摘要与引言中所有 "first / highest / unmatched" 类措辞逐条核对。
2. **引用 RiboSBDD 作为 concurrent work**（等其身份公开后即可引用；2026-12-16 决定后无论录用与否，
   OpenReview 记录都会保留并署名，可引用为 `Anonymous, 2027, ICLR submission`，或在正式版引用）。
   引用位置：Introduction 第三段（"generative category remains least developed"）与 Discussion。
3. **在 Discussion 里加一段与 RiboSBDD 的异同**：承认同为"蛋白→RNA 迁移 + 共享表征"路线，强调差异
   （BFN + ED 预训练 vs 扩散 + 碱基柔性；DNA/RNA 通用 vs RNA-only；湿实验 vs 纯计算）。
4. **补强最容易被比较的短板**：
   - 数据规模与划分（738 条 vs 8,006 实例；NA2smol 自认训练/验证与 13 个基准系统共享结构 ID）。
     可在 SI 里补充"若在 RiboSBDDBench 划分下评测"的对照，或至少讨论为什么 738 条足够。
   - 评测口径不同（NA2smol 用对接 + ECFP4；RiboSBDD 用 AnnapuRNA + Contact/Clash）。两套结果不可直接比较，
     建议在 rebuttal 材料中给出映射或说明。
5. **保留独立开发证据**：checkpoint（6 月）、仓库历史、内部审稿记录（11-07）等，用于向编辑说明
   "独立并行工作"，但不要指望用它们主张公开优先权。
6. **反向风险**：如果 RiboSBDD 的 ICLR 版本（12-16 后）被接收并公开，NA2smol 在 NBT 的 novelty 会被进一步压缩；
   建议把投稿时间点与"实验验证 + DNA 通用性"作为主要卖点，而不是"第一个 RNA 口袋生成器"。

---

## 8. 附：本次使用的可核验证据

- OpenReview 投稿：https://openreview.net/forum?id=n8QuaZwPIm （`number: 27761`，`cdate` 2026-09-17，`mdate` 2026-11-17，`readers: everyone`）
- NA2smol 代码：https://github.com/Gallo13th/NA2mol （`created_at 2026-10-29T01:52:49Z`，public，MIT）
- ICLR 2027 CFP：https://iclr.cc/Conferences/2027/CallForPapers
- 提取的 RiboSBDD 全文：`_paper_analysis/ribosbdd_clean.txt`（3,399 行；抽取脚本 `extract_pdf.py`）
- 同步后的 NA2smol 论文：`_paper_download/NA2smol_paper_20261121/NA2smol_paper/`
