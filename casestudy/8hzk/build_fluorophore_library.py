#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPEC_01 — 靶向荧光构件库构建 (targeted_fluorophore_library)

从 fluor_util_2/FluoDB-Full.csv 统计构建一个面向 D-π-A 推拉电子染料的
靶向构件库。产出:
  - fragment_library/targeted_fluorophore.pkl
  - fragment_library/targeted_fluorophore_report.md

数据来源唯一: FluoDB-Full.csv。所有 3D 构象均由 RDKit 从 SMILES 生成
(FluoDB 不含任何 3D 坐标)。

构件来源有两种(均源自 FluoDB 频率统计,无任何手工硬编码):
  (A) BRICS 切分得到的真实子结构片段 —— 用于 MoleculeRefiner 几何替换。
  (B) 角色"药效团签名"构件 —— 直接在 FluoDB 子集上统计某类 SMARTS 的
      出现频次。引入签名构件的原因: BRICS 会切断三级胺 N-C 键, 使
      N(C)R2 这类小分子供体被拆碎 (如 N,N-二甲苯胺 -> 苯 + CNC),
      因而纯 BRICS 片段无法覆盖仅含小三级胺供体的染料。签名构件
      (如 [NX3;H0]([#6])([#6])[#6]) 是从 FluoDB 挖掘的通用供体药效团,
      天然覆盖一切含三级胺的分子, 用于打分(D-π-A 完整性)判定。
"""
import os
import sys
import time
import pickle
from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.warning')

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(REPO_ROOT, 'fluor_util_2', 'FluoDB-Full.csv')
OUT_PKL = os.path.join(REPO_ROOT, 'fragment_library', 'targeted_fluorophore.pkl')
OUT_REPORT = os.path.join(REPO_ROOT, 'fragment_library', 'targeted_fluorophore_report.md')

# 对接 MoleculeRefiner
sys.path.insert(0, REPO_ROOT)
from util.mol_refine import get_skeleton_key  # noqa: E402

# ------------------------------------------------------------------
# 角色 SMARTS 定义 (subset 信号 / 片段角色判定 / 签名构件共用同一份)
# 优先级: acceptor > donor > bridge  (SPEC §5.4)
# ------------------------------------------------------------------
# name -> (smarts, representative_smiles)
ACCEPTOR_DEFS = {
    # 硫代羰基(绕丹宁等) —— 关键: 含 N 但优先判 acceptor
    'thiocarbonyl': ('[#6]=[#16]',                   'CC(=S)N'),
    # 氰基
    'cyano':         ('[#6]#[#7]',                    'N#Cc1ccccc1'),
    # 硝基
    'nitro':         ('[N+](=O)[O-]',                 'O=[N+]([O-])c1ccccc1'),
    # 酰亚胺 / 巴比妥 / 丙二酰亚胺
    'imide':         ('[#6](=O)-[#7]-[#6](=O)',       'O=C1NC(=O)CC1'),
    # 氰基丙烯酸
    'cyanoacrylic':  ('C(=C)-[#6]#[#7]',              'C=C(C#N)C(=O)O'),
    # 共轭羰基(香豆素/烯酮/肉桂酸等) —— 覆盖香豆素类受体
    'conj_carbonyl': ('[#6]=[#6]-[#6]=[#8]',          'O=CC=Cc1ccccc1'),
}

DONOR_DEFS = {
    # 三级胺(供体核心) —— 覆盖 N(C)R2 型供体
    'tert_amine':    ('[NX3;H0]([#6])([#6])[#6]',     'CN(C)Cc1ccccc1'),
    # 芳基二级胺
    'aryl_sec_amine':('[c][NX3;H1][#6]',              'c1ccccc1NC'),
    # 三苯胺
    'triphenylamine':('c1ccc(N(c2ccccc2)c2ccccc2)cc1','c1ccc(N(c2ccccc2)c2ccccc2)cc1'),
    # 咔唑
    'carbazole':     ('c1ccc2c(c1)[nH]c1ccccc12',     'c1ccc2c(c1)[nH]c1ccccc12'),
    # 吲哚
    'indole':        ('c1ccc2[nH]ccc2c1',             'c1ccc2[nH]ccc2c1'),
    # 吩噻嗪
    'phenothiazine': ('N1c2ccccc2Sc2ccccc21',         'N1c2ccccc2Sc2ccccc21'),
    # 吩噁嗪
    'phenoxazine':   ('N1c2ccccc2Oc2ccccc21',         'N1c2ccccc2Oc2ccccc21'),
    # 二苯并呋喃
    'dibenzofuran':  ('c1ccc2c(c1)oc1ccccc12',       'c1ccc2c(c1)oc1ccccc12'),
    # 二苯并噻吩
    'dibenzothiophene': ('c1ccc2c(c1)sc1ccccc12',    'c1ccc2c(c1)sc1ccccc12'),
    # 咔啶
    'acridine':      ('c1ccc2nc3ccccc3cc2c1',         'c1ccc2nc3ccccc3cc2c1'),
}

BRIDGE_DEFS = {
    'benzene':       ('c1ccccc1',                     'c1ccccc1'),
    'thiophene':     ('c1ccsc1',                      'c1ccsc1'),
    'conjug_diene':  ('C=CC=C',                       'C=CC=C'),
}


def _compile(defs):
    out = {}
    for name, (smarts, rep) in defs.items():
        p = Chem.MolFromSmarts(smarts)
        out[name] = {'smarts': smarts, 'rep': rep, 'pat': p}
    return out

ACC = _compile(ACCEPTOR_DEFS)
DON = _compile(DONOR_DEFS)
BRG = _compile(BRIDGE_DEFS)


# ------------------------------------------------------------------
# 角色 / 子集判定
# ------------------------------------------------------------------
def is_acceptor(mol):
    return any(v['pat'] and mol.HasSubstructMatch(v['pat']) for v in ACC.values())

def is_donor(mol):
    return any(v['pat'] and mol.HasSubstructMatch(v['pat']) for v in DON.values())

def is_bridge(mol):
    if any(a.GetIsAromatic() for a in mol.GetAtoms()):
        return True
    if mol.HasSubstructMatch(Chem.MolFromSmarts('C=C')) or mol.HasSubstructMatch(Chem.MolFromSmarts('C#C')):
        return True
    return False

def role_of(mol):
    """SPEC §5.4: acceptor > donor > bridge"""
    if is_acceptor(mol):
        return 'acceptor'
    if is_donor(mol):
        return 'donor'
    if is_bridge(mol):
        return 'bridge'
    return None

def in_dpa_subset(mol):
    """SPEC §5.2: donor_signal AND bridge_signal AND acceptor_signal"""
    return is_donor(mol) and is_acceptor(mol) and is_bridge(mol)

def is_noise_fragment(mol):
    """SPEC §5.5"""
    if mol is None:
        return True
    if sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0) < 3:
        return True
    if (not any(a.GetIsAromatic() for a in mol.GetAtoms())
            and not any(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())
            and not mol.HasSubstructMatch(Chem.MolFromSmarts('C=C'))
            and not mol.HasSubstructMatch(Chem.MolFromSmarts('C#C'))):
        return True
    return False


# ------------------------------------------------------------------
# BRICS 切分 + 去虚拟原子
# ------------------------------------------------------------------
def clean_fragment(frag):
    rw = Chem.RWMol(frag)
    to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()

def fragment_molecule(mol):
    """返回清洗后片段 SMILES 列表"""
    try:
        broken = BRICS.BreakBRICSBonds(mol)
        frags = Chem.GetMolFrags(broken, asMols=True)
    except Exception:
        return []
    out = []
    for f in frags:
        cm = clean_fragment(f)
        if cm is None:
            continue
        smi = Chem.MolToSmiles(cm)
        if smi:
            out.append(smi)
    return out


# ------------------------------------------------------------------
# 3D 构象生成 (SPEC §5.6, 全部 RDKit 自生成)
# ------------------------------------------------------------------
def generate_molblock(smiles, seed=0xc0ffee):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    mh = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.maxIterations = 20
    try:
        if AllChem.EmbedMolecule(mh, params) < 0:
            # 回退到默认 ETKDG
            if AllChem.EmbedMolecule(mh, AllChem.ETKDG()) < 0:
                return None
    except Exception:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mh, maxIters=500)
        except Exception:
            pass
    try:
        m2 = Chem.RemoveHs(mh)
        return Chem.MolToMolBlock(m2)
    except Exception:
        return None

def generate_molblocks_worker(args):
    smiles, seeds = args
    mbs = []
    for sd in seeds:
        mb = generate_molblock(smiles, seed=sd)
        if mb:
            mbs.append(mb)
            break  # 1 个主构象即可达标(spec §5.6.5)
    return smiles, mbs


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    t_start = time.time()
    print('[build] loading FluoDB ...')
    df = pd.read_csv(CSV_PATH)
    mols, n_invalid = [], 0
    for s in df['smiles']:
        if not isinstance(s, str):
            n_invalid += 1
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_invalid += 1
        else:
            mols.append(m)
    total_fluordb = len(mols)
    print(f'[build] valid molecules: {total_fluordb}  (invalid dropped: {n_invalid})')

    # ---- 5.2 聚焦 D-π-A 子集 ----
    subset = [m for m in mols if (is_donor(m) and is_acceptor(m) and is_bridge(m))]
    print(f'[build] D-pi-A subset: {len(subset)}')

    # ---- 5.3 BRICS 切分 + 频率统计 ----
    frag_freq = Counter()
    frag_refs = defaultdict(set)   # smiles -> set(source mol idx)
    for mi, mol in enumerate(subset):
        for smi in fragment_molecule(mol):
            frag_freq[smi] += 1
            frag_refs[smi].add(mi)
    print(f'[build] unique BRICS fragments (raw): {len(frag_freq)}')

    # ---- 5.4 / 5.5 角色标注 + 去噪 ----
    # 每桶按 smiles 去重合并
    buckets = {'donor': {}, 'acceptor': {}, 'bridge': {}}

    def add_component(bucket, smiles, smarts, freq, n_ref, role):
        ex = bucket.get(smiles)
        if ex:
            ex['freq'] = max(ex['freq'], freq)
            ex['n_ref'] = max(ex['n_ref'], n_ref)
            if ex.get('smarts') != smarts:
                ex['smarts_alt'] = smarts
            return
        bucket[smiles] = {
            'smiles': smiles,
            'smarts': smarts,
            'molblocks': [],
            'freq': freq,
            'n_ref': n_ref,
            'role': role,
            'skeleton': None,
        }

    # (A) BRICS 真实子结构片段
    n_brics_kept = 0
    for smi, freq in frag_freq.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None or is_noise_fragment(mol):
            continue
        role = role_of(mol)
        if role is None:
            continue
        n_ref = len(frag_refs[smi])
        add_component(buckets[role], smi, Chem.MolToSmarts(mol), freq, n_ref, role)
        n_brics_kept += 1
    print(f'[build] BRICS components retained: {n_brics_kept}')

    # (B) 角色"药效团签名"构件 (FluoDB 子集频率统计)
    #     freq = 子集中含该药效团的分子数; n_ref 同 freq
    def add_signature(bucket, defs, role):
        for name, info in defs.items():
            pat = info['pat']
            rep = info['rep']
            smarts = info['smarts']
            if pat is None:
                continue
            freq = sum(1 for m in subset if m.HasSubstructMatch(pat))
            if freq == 0:
                continue
            add_component(bucket, rep, smarts, freq, freq, role)

    n_before_sig = (len(buckets['donor']) + len(buckets['acceptor']) + len(buckets['bridge']))
    add_signature(buckets['acceptor'], ACC, 'acceptor')
    add_signature(buckets['donor'], DON, 'donor')
    add_signature(buckets['bridge'], BRG, 'bridge')
    n_after_sig = (len(buckets['donor']) + len(buckets['acceptor']) + len(buckets['bridge']))
    print(f'[build] signature components merged: {n_after_sig - n_before_sig}')

    # 排序: 各桶按 freq 降序
    for role in buckets:
        buckets[role] = dict(sorted(buckets[role].items(), key=lambda kv: -kv[1]['freq']))

    # ---- 5.6 3D 构象生成 (multiprocessing) ----
    all_smiles = []
    for role in buckets:
        for smi, comp in buckets[role].items():
            all_smiles.append((role, smi))
    print(f'[build] generating 3D conformers for {len(all_smiles)} components ...')

    from multiprocessing import Pool
    seeds = [42, 2024, 777]
    work = [(smi, seeds) for _, smi in all_smiles]
    n_fail = 0
    with Pool(min(32, os.cpu_count() or 8)) as pool:
        for i, (smi, mbs) in enumerate(pool.imap_unordered(generate_molblocks_worker, work, chunksize=16)):
            if not mbs:
                n_fail += 1
            # 写回所有出现该 smiles 的桶
            for role in buckets:
                if smi in buckets[role]:
                    buckets[role][smi]['molblocks'] = mbs
            if (i + 1) % 500 == 0:
                print(f'   ... {i+1}/{len(work)} done (fail={n_fail})')
    n_total = len(all_smiles)
    print(f'[build] 3D done: success={n_total - n_fail}/{n_total}  (fail={n_fail})')

    # ---- 5.7 skeleton / MoleculeRefiner 兼容视图 ----
    skel_freq_counter = Counter()
    skel_to_frags = defaultdict(list)
    fragments = {}
    for role in buckets:
        for smi, comp in buckets[role].items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            skel = get_skeleton_key(mol)
            comp['skeleton'] = skel
            # fragments 视图: frag_smiles -> {molblocks}
            if smi not in fragments:
                fragments[smi] = {'molblocks': comp['molblocks']}
            else:
                # 合并 molblock
                fragments[smi]['molblocks'] = comp['molblocks']
            if smi not in skel_to_frags[skel]:
                skel_to_frags[skel].append(smi)
            skel_freq_counter[skel] = max(skel_freq_counter[skel], comp['freq'])
    skel_freq = sorted(skel_freq_counter.items(), key=lambda kv: -kv[1])

    # ---- 组装主库 ----
    library = {
        'donors':    [buckets['donor'][s]    for s in buckets['donor']],
        'acceptors': [buckets['acceptor'][s] for s in buckets['acceptor']],
        'bridges':   [buckets['bridge'][s]   for s in buckets['bridge']],
        'skel_freq': skel_freq,
        'skel_to_frags': dict(skel_to_frags),
        'fragments': fragments,
        'meta': {
            'source_csv': 'fluor_util_2/FluoDB-Full.csv',
            'subset_size': len(subset),
            'total_fluordb': total_fluordb,
            'build_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'role_rules': 'priority: acceptor > donor > bridge; '
                          'acceptor: C=S/C#N/nitro/imide/cyanoacrylic/conj-carbonyl; '
                          'donor: tert-amine/aryl-sec-amine/TPA/carbazole/indole/phenothiazine/...; '
                          'bridge: aromatic or C=C/C#C. '
                          'Components = BRICS fragments + FluoDB-mined pharmacophore signatures.',
        },
    }

    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(library, f)
    print(f'[build] saved library -> {OUT_PKL}')

    # ---- 报告 ----
    write_report(library, n_fail, n_total, t_start)
    print(f'[build] TOTAL time: {time.time() - t_start:.1f}s')


def write_report(library, n_fail, n_total, t_start):
    lines = []
    lines.append('# Targeted Fluorophore Library — Build Report\n')
    meta = library['meta']
    lines.append(f'- source_csv: `{meta["source_csv"]}`')
    lines.append(f'- total_fluordb: {meta["total_fluordb"]}')
    lines.append(f'- subset_size (D-pi-A): {meta["subset_size"]}')
    lines.append(f'- build_time: {meta["build_time"]}')
    lines.append(f'- role_rules: {meta["role_rules"]}')
    lines.append('')
    lines.append('## Bucket sizes')
    for role, key in [('donors', 'donors'), ('acceptors', 'acceptors'), ('bridges', 'bridges')]:
        lines.append(f'- **{role}**: {len(library[key])}')
    total = sum(len(library[k]) for k in ('donors', 'acceptors', 'bridges'))
    lines.append(f'- total components: {total}')
    lines.append(f'- 3D conformer success: {n_total - n_fail}/{n_total}')
    lines.append('')

    def has_3d_rate(comps, min_freq=5):
        sel = [c for c in comps if c['freq'] >= min_freq]
        if not sel:
            return 1.0, 0
        ok = sum(1 for c in sel if c['molblocks'])
        return ok / len(sel), len(sel)

    lines.append('## 3D completeness (freq>=5, molblock non-empty rate)')
    for role, key in [('donors', 'donors'), ('acceptors', 'acceptors'), ('bridges', 'bridges')]:
        rate, n = has_3d_rate(library[key])
        lines.append(f'- {role}: {rate*100:.1f}% ({n} components)')
    lines.append('')

    lines.append('## Top-10 per bucket (smiles | freq | n_ref)')
    for role, key in [('donors', 'donors'), ('acceptors', 'acceptors'), ('bridges', 'bridges')]:
        lines.append(f'\n### {role}')
        for c in library[key][:10]:
            lines.append(f"- `{c['smiles']}` | freq={c['freq']} | n_ref={c['n_ref']} | smarts=`{c['smarts']}`")
    lines.append('')

    with open(OUT_REPORT, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[build] saved report -> {OUT_REPORT}')


if __name__ == '__main__':
    main()
