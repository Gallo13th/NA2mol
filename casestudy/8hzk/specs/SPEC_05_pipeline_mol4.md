# SPEC 05 — 功能库匹配回收 + 组合种子 + 复现 mol4

> 依赖 SPEC 04 的 D/A/π 功能库。目标:复现 mol4(无 S 的二苯乙烯-氰基-三级胺)。

## 1. 目标
通过"功能库匹配回收 → 构造 mol4-like 种子 → sample_by_seed 生成",产出与 mol4 相似的分子(Tanimoto>0.5 或含 mol4 核心子结构 stilbene+C=C(C#N)+三级胺)。

## 2. 阶段

### 阶段1:过滤三库(确保 mol4 覆盖)
对 `fragment_library/functional_blocks/{donors,acceptors,bridges}.pkl`:
- 删除含 `.`(断开)的片段
- 删除原子数 <3(D/A)或 <4(π)的
- 删除频率=1 的孤立片段(可选)
- **硬性**:过滤后必须仍含 mol4 三单元——D 库含 `N(C)CCO` 类、A 库含 `C=C(C#N)`(氰基乙烯)、π 库含 stilbene。验证不通过则放宽过滤直到通过。
- 存为 `functional_blocks/{donors,acceptors,bridges}_filtered.pkl`

### 阶段2:匹配回收(用现有 3039 raw 分子)
原料:`output/8hzk_output/dummy_ligand/iter_*/raw_output/*.sdf`(之前生成,有口袋 3D)。
对每个分子:
1. 用过滤后 D/A/π 库 SMARTS 匹配,记录命中的功能单元(原子索引 + 角色 + 3D)
2. 计算 **mol4 覆盖分**:含 N(C)CCO 类(D)+ 氰基/C=C(C#N)(A)+ stilbene/苯共轭(π)的情况
3. 排序:优先"三单元齐全"且"含 stilbene"的分子

### 阶段3:种子构造
- **首选**:同时含 D+A+π 三单元、且含 stilbene 的完整生成分子 → 直接做种子(3D 正确,无需组装)
- **次选**:含 D+A(或 D+π)+ stilbene 的分子
- **兜底**:若现有分子里 C=C(C#N)=0(模型未生成),用含 C#N(氰基)+stilbene+胺 的分子做种子,模型在其上补全时可能产出 C=C(C#N)
- 把选出的种子分子存为 SDF(保留口袋 3D),作为 sample_by_seed 的 seed_library

### 阶段4:种子生成
- 用构造的种子集跑 `sample_by_seed`(seed_used=True,迭代生成)
- 模型固定种子功能单元原子,补全其余 → 产物含 mol4 骨架
- 接入 fluor_guidance 完整性筛选(可选)

### 阶段5:筛选复现 mol4
对产物:
- ECFP4 Tanimoto vs mol4(目标 >0.5)
- 子结构检查:是否含 stilbene + C=C(C#N) + 三级胺(mol4 三特征同时命中)
- Murcko 骨架 vs mol4
- 列出最接近 mol4 的 top 产物

## 3. 配置
- 复用 genbypkt 的 generate(seed_used=True,用阶段3种子),或写独立种子生成脚本
- 临时调整规模(total_target、iteration 上限)控制时间,测后恢复
- 种子生成用聚焦库/功能库做完整性匹配均可

## 4. 验收
1. 过滤后三库含 mol4 三单元
2. 阶段2匹配回收:统计多少分子含 D/A/π 各单元(尤其含 stilbene 的数量)
3. 阶段3种子构造:产出种子集(数量、是否含 stilbene 类)
4. 阶段4种子生成:sample_by_seed 跑通,产物含 mol4 骨架的比例
5. **核心**:产物中是否有 Tanimoto>0.5 或含 mol4 三特征子结构的分子(复现 mol4)

## 5. 不做
- 不改模型/checkpoint/采样
- 跨分子组装 3D(用完整分子避免)

## 6. 灵活
- 若现有 3039 分子里 stilbene+C=C(C#N)+胺 同时存在的为 0,且种子生成仍复现不了,可继续生成更多原料(后台),但优先用现有。
