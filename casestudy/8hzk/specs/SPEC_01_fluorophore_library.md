# SPEC 01 — 靶向荧光构件库构建 (targeted_fluorophore_library)

> 本 spec 属于 8HZK case study 的第一个独立交付物。它只构建构件库,不涉及模型生成流程的修改(后者为后续 spec)。

## 1. 背景与目标

8HZK 是一个 RNA 口袋(纯核苷酸 A/G/C/U),目标是用项目核心 BFN 模型生成能结合该口袋且具荧光活性的分子。已知的 4 个实验验证荧光分子(`casestudy/8hzk/known_mols.txt`)均为 **D-π-A 推拉电子染料**(供体-共轭桥-受体)。

经验证,项目自带的通用荧光检测库 `fluor_util_2` 对这类染料存在**系统性假阴性**(三级胺供体、绕丹宁 C=S 受体、噻吩共轭桥均未被识别),4 个 known_mols 全部被判"非荧光"。因此需要一个**从大规模已知荧光分子库(FluoDB)统计构建的靶向构件库**,既覆盖 known_mols 所属的 D-π-A 染料家族,又能用于:
- **打分引导**:评估生成分子的 D-π-A 完整性
- **片段替换**:对接 `util/mol_refine.py` 的 `MoleculeRefiner`,对生成骨架做靶向精修

## 2. 硬性约束

1. **数据源唯一**:构件库只能从 `fluor_util_2/FluoDB-Full.csv` 统计构建。**构建脚本不得读取、import、引用 `known_mols.txt`**(代码与运行日志中均不得出现该文件名)。known_mols 仅用于**独立的事后验收**。
2. **3D 构象全部自生成**:**FluoDB 仅含 SMILES,不含任何 3D 坐标**。构件库所需的 3D 构象必须用 RDKit 从 SMILES 生成(见 §5.6),不得假设库中存在预计算构象。
3. **不改模型、不改 `genbypkt.py` 主流程、不碰核心 checkpoint**。
4. **不破坏现有项目文件**:新增文件为主;若需读取现有 `fragment_library/index.pkl.gz` 仅作格式参考,不得覆写它。

## 3. 输入 / 输出

### 输入
- `fluor_util_2/FluoDB-Full.csv`(55168 行,字段:`absorption/nm,emission/nm,plqy,e/m-1cm-1,smiles,solvent,reference(doi),source,tag_name`)

### 输出(全部新建)
| 路径 | 说明 |
|------|------|
| `fragment_library/targeted_fluorophore.pkl` | 主库(数据结构见 §4) |
| `fragment_library/targeted_fluorophore_report.md` | 统计与验收报告 |
| `casestudy/8hzk/build_fluorophore_library.py` | 构建脚本(本 spec 的主交付代码) |
| `casestudy/8hzk/verify_library.py` | 独立验收脚本(允许读 known_mols) |

## 4. 主库数据结构

`targeted_fluorophore.pkl` 为 dict,需同时满足"易打分"与"可替换"两种用途:

```python
{
  # ---- 用途 A:子结构匹配打分 ----
  'donors':    [ Component, ... ],   # 按 freq 降序
  'acceptors': [ Component, ... ],
  'bridges':   [ Component, ... ],

  # ---- 用途 B:MoleculeRefiner 兼容视图(几何替换) ----
  # 格式严格对齐 fragment_library/index.pkl.gz
  'skel_freq':     [ (skeleton_smiles, freq), ... ],
  'skel_to_frags': { skeleton_smiles: [frag_smiles, ...] },
  'fragments':     { frag_smiles: {'molblocks': [mb1, mb2, ...]} },

  # ---- 元信息 ----
  'meta': {
      'source_csv': 'fluor_util_2/FluoDB-Full.csv',
      'subset_size': int,           # D-π-A 聚焦子集分子数
      'total_fluordb': int,         # FluoDB 有效分子数
      'build_time': str,
      'role_rules': str,            # 角色判定规则摘要(见 §5.4)
  }
}
```

其中 `Component` 为 dict:
```python
{
  'smiles':  str,        # 去虚拟原子后的真实子结构 SMILES(用于子结构匹配)
  'smarts':  str,        # 与 smiles 等价的 SMARTS(供 RDKit MolFromSmarts)
  'molblocks': [str],   # 1~3 个带 3D 构象的 molblock(供 MoleculeRefiner 刚性叠合)
  'freq':    int,        # 在聚焦子集中的出现次数
  'n_ref':   int,        # 贡献该构件的不同源分子数
  'role':    str,        # 'donor' | 'acceptor' | 'bridge'
  'skeleton': str,       # get_skeleton_key(mol) 结果,用于对接 MoleculeRefiner
}
```

> `get_skeleton_key` 的定义见 `util/mol_refine.py:172`,构建时直接复用该函数,保证与现有 MoleculeRefiner 对齐。

## 5. 构建流程(代码要求)

### 5.1 加载与解析
- 读 CSV,逐行 `Chem.MolFromSmiles`;统计有效/无效。预期 ~55168 有效。

### 5.2 聚焦 D-π-A 子集
同时满足以下三类的分子纳入子集(探查显示约 2075 个):
- **donor_signal**:命中 `[NX3;H0]`(三级胺) 或 `[NX3;H1]` 且连芳环(芳胺) 或三苯胺 `c1ccccc1N(c2ccccc2)c3ccccc3` 或咔唑/吲哚等富电子稠环
- **bridge_signal**:芳香原子数 > 0(噻吩/苯/稠环等)
- **acceptor_signal**:命中 `C=S`(硫代羰基) 或 `C#N` 或硝基 `[N+](=O)[O-]` 或氰基丙烯酸 `C(=C/C#N)C(=O)O` 或巴比妥/绕丹宁类 `O=C1NC(=S)` 等强受体

> 子集定义的宽窄以"自然覆盖 known_mols 所属染料家族"为准;不要针对 known_mols 特调。

### 5.3 BRICS 切分 + 频率统计
- 对子集每个分子 `BRICS.BreakBRICSBonds` → `Chem.GetMolFrags(asMols=True)`
- 每个片段:去虚拟原子(`atomic_num==0` 删除并 SanitizeMol)→ 得到真实子结构 SMILES
- 统计每个真实子结构的 `freq`(总出现次数)与 `n_ref`(不同源分子数)

### 5.4 角色标注(关键,修复 fluor_util_2 盲区)
**按功能基团优先级判定**,避免"绕丹宁因含 N 被误判为供体"的错误:

```
优先级: acceptor > donor > bridge

step1 acceptor 判定(任一命中即 A):
  - C(=S) 硫代羰基
  - C#N 氰基
  - [N+](=O)[O-] 硝基
  - 氰基丙烯酸 / 巴比妥 / 绕丹宁 / 酰亚胺 等已知受体 SMARTS 集
step2 donor 判定(非 acceptor 前提下,任一命中即 D):
  - [NX3;H0] 三级胺
  - [NX3;H1] 且邻接芳环(芳胺)
  - 咔唑/吲哚/吩噻嗪/三苯胺/吩噁嗪/二苯并呋喃/二苯并噻吩/芴 等富电子稠环
step3 bridge 判定(非 A 非 D 前提下):
  - 芳香原子数 > 0,或含 C=C / C#C 共轭
step4 其余 → 丢弃(噪声)
```

> 重要:绕丹宁 `O=C1NC(=S)S/C1=` 同时含 N 与 C(=S),按优先级判为 **acceptor**。这正是 fluor_util_2 失效的根因,必须判对。

### 5.5 去噪与筛选
丢弃以下片段:
- 重原子数 < 3
- 纯烷基链(无芳环、无杂原子、无共轭)
- `freq == 1` 的孤立片段(可选保留,默认丢弃以降噪)
保留其余,按角色分桶,各桶内按 `freq` 降序。**不设数量上限**(用户要求:库大小不限)。

### 5.6 3D 构象生成(硬性要求,FluoDB 无 3D)
对每个保留构件:
1. `Chem.MolFromSmiles(smiles)` → `AddHs`
2. 用 **ETKDG**(EmbedParams `useRandomCoords=True`, 或默认 `ETKDGv3()`)生成构象,`randomSeed` 取多个种子
3. 力场优化:优先 `MMFFOptimizeMolecule(maxIters=500)`;若无 MMFF 参数则 `UFFOptimizeMolecule`
4. `RemoveHs` → 取 conformer → `Chem.MolToMolBlock`
5. 每个构件生成 **1 个主构象**(若资源允许可最多 3 个不同种子的构象以增加替换柔性,但 1 个即可达标)
6. 若构象生成失败(罕见),记录告警,该构件 `molblocks=[]` 但仍保留 SMILES(打分仍可用,替换降级跳过)

> 所有 molblock 的 3D 坐标均为本步骤生成,**不得从 FluoDB 取**(FluoDB 根本没有)。

### 5.7 MoleculeRefiner 兼容视图
- 用 `get_skeleton_key` 计算每个构件的 skeleton
- 汇总 `skel_freq`(按 freq 降序)、`skel_to_frags`、`fragments`(每个 frag 含 `{'molblocks':[...]}`)
- 格式必须与 `fragment_library/index.pkl.gz` 一致(开发前先 `pickle.load` 该文件 dump 结构对照)

## 6. 打分接口(随库一同交付)

在 `build_fluorophore_library.py` 或同目录新建 `fluorophore_scorer.py` 中提供:

```python
class FluorophoreScorer:
    def __init__(self, library_path='fragment_library/targeted_fluorophore.pkl'): ...
    def score_molecule(self, mol) -> dict:
        """
        返回:
        {
          'donor_hits':    int,   # 匹配的供体构件数
          'acceptor_hits': int,
          'bridge_hits':   int,
          'is_complete_dpa': bool,  # 三类均 >0
          'matched_donors':    [smiles...],
          'matched_acceptors': [smiles...],
          'matched_bridges':   [smiles...],
        }
        """
```
匹配方式:对每个构件用其 `smarts` 做 `mol.GetSubstructMatch`,命中即计。

## 7. 审计规则(强制)

1. **源码审计**:`build_fluorophore_library.py` 全文不得出现字符串 `known_mols`、`known_mols.txt`、`known` 等指向验收集的字样。
2. **运行日志**:构建运行输出不得打印 known_mols 相关路径。
3. **数据来源**:库的 `meta.source_csv` 必须为 FluoDB;构件必须可追溯到 FluoDB 的频率统计,而非手工硬编码。
4. **3D 来源**:所有 molblock 必须由 RDKit EmbedMolecule 生成,不得引入外部 3D 文件。

## 8. 验收标准(`verify_library.py`,允许读 known_mols)

`verify_library.py` 加载库,执行以下检查并打印 PASS/FAIL:

1. **角色覆盖(核心)**:对 `known_mols.txt` 的 4 个分子,每个都能用 `FluorophoreScorer.score_molecule` 得到 `is_complete_dpa == True`(三类构件各 ≥1 命中)。
2. **库非空且分布合理**:`donors`/`acceptors`/`bridges` 三桶均非空;`acceptors` 桶中必须能匹配到绕丹宁类(含 `C(=S)`)与氰基类(`C#N`)构件;`donors` 桶中必须含三级胺类构件。
3. **MoleculeRefiner 可加载**:用兼容视图初始化 `MoleculeRefiner`(传入库路径),不报错;对一个 demo 分子(从 known_mols 任取)调用 `process_molecule` 能跑通(替换与否不限,关键是流程不崩)。
4. **3D 完整性**:`donors`/`acceptors`/`bridges` 中 freq≥5 的构件,`molblocks` 非空率 ≥ 90%。
5. 打印报告:子集规模、三桶数量、top-10 构件(含 freq/role/smile)、known_mols 逐个的命中明细。

## 9. 不做范围(留给后续 spec)
- 不修改 `genbypkt.py` 的生成主流程
- 不实现"生成中的在线荧光打分引导"
- 不实现 known_mols 相似度度量
- 不做口袋对接/亲和力评估

## 10. 运行方式
```bash
cd /home/data/users/gaoletian/rna2smol
python casestudy/8hzk/build_fluorophore_library.py
python casestudy/8hzk/verify_library.py
```
构建应在单机 CPU 上 10~30 分钟内完成(主要耗时在 3D 构象生成;可用 `multiprocessing` 并行)。
