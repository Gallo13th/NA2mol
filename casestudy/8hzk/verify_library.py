#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPEC_01 §8 — 靶向荧光构件库验收

允许读取 casestudy/8hzk/known_mols.txt 作为事后验收工具(构建脚本不允许,
本脚本作为独立验收不受 §7 审计约束)。

检查项:
  1. 角色覆盖: 4 个 known_mols 用 FluorophoreScorer 均 is_complete_dpa
  2. 库非空且分布合理 (acceptors 含 C=S 与 C#N; donors 含三级胺)
  3. MoleculeRefiner 可加载 + process_molecule 跑通
  4. 3D 完整性: freq>=5 构件 molblock 非空率 >= 90%
  5. 打印报告
"""
import os
import sys
import pickle

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(REPO_ROOT, 'fragment_library', 'targeted_fluorophore.pkl')
KNOWN_PATH = os.path.join(REPO_ROOT, 'casestudy', '8hzk', 'known_mols.txt')

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluorophore_scorer import FluorophoreScorer

CHECKS = []


def record(name, passed, detail=''):
    CHECKS.append((name, passed, detail))
    tag = 'PASS' if passed else 'FAIL'
    print(f'[{tag}] {name}' + (f' -- {detail}' if detail else ''))
    return passed


def main():
    print('=' * 70)
    print('SPEC_01 靶向荧光构件库验收')
    print('=' * 70)

    if not os.path.exists(LIB_PATH):
        record('library exists', False, f'missing {LIB_PATH}')
        return
    with open(LIB_PATH, 'rb') as f:
        lib = pickle.load(f)
    record('library loadable', True)

    donors = lib['donors']
    acceptors = lib['acceptors']
    bridges = lib['bridges']
    meta = lib.get('meta', {})

    scorer = FluorophoreScorer(LIB_PATH)

    # ---- 读 known_mols (本脚本允许) ----
    known_smiles = []
    with open(KNOWN_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                known_smiles.append(line)
    known_mols = [Chem.MolFromSmiles(s) for s in known_smiles]

    # ---- Check 1: 角色覆盖 ----
    print('\n--- Check 1: 角色覆盖 (known_mols is_complete_dpa) ---')
    all_complete = True
    per_mol_detail = []
    for i, m in enumerate(known_mols):
        res = scorer.score_molecule(m)
        complete = res['is_complete_dpa']
        all_complete = all_complete and complete
        per_mol_detail.append((i, known_smiles[i], res))
        print(f'  mol{i}: complete={complete} '
              f'D={res["donor_hits"]} A={res["acceptor_hits"]} B={res["bridge_hits"]}')
        # 命中明细 (取前 3 个)
        if res['matched_donors']:
            print(f'        donor hits : {res["matched_donors"][:3]}')
        if res['matched_acceptors']:
            print(f'        accept hits: {res["matched_acceptors"][:3]}')
        if res['matched_bridges']:
            print(f'        bridge hits: {res["matched_bridges"][:3]}')
    record('Check1: all known_mols is_complete_dpa', all_complete)

    # ---- Check 2: 库非空且分布合理 ----
    print('\n--- Check 2: 库分布 ---')
    nonempty = bool(donors) and bool(acceptors) and bool(bridges)

    # acceptors 桶含 C=S (绕丹宁类) 与 C#N (氰基类)
    has_CS = any(Chem.MolFromSmiles(c['smiles']) and
                 Chem.MolFromSmiles(c['smiles']).HasSubstructMatch(Chem.MolFromSmarts('[#6]=[#16]'))
                 for c in acceptors)
    # 也可直接看 smarts 字段
    has_CS = has_CS or any('[#6]=[#16]' in c.get('smarts', '') for c in acceptors)
    has_CN = any(Chem.MolFromSmiles(c['smiles']) and
                 Chem.MolFromSmiles(c['smiles']).HasSubstructMatch(Chem.MolFromSmarts('[#6]#[#7]'))
                 for c in acceptors)
    has_CN = has_CN or any('[#6]#[#7]' in c.get('smarts', '') for c in acceptors)
    # donors 桶含三级胺
    has_tertN = any('[NX3;H0]' in c.get('smarts', '') for c in donors) or any(
        Chem.MolFromSmiles(c['smiles']) and
        Chem.MolFromSmiles(c['smiles']).HasSubstructMatch(Chem.MolFromSmarts('[NX3;H0]'))
        for c in donors)
    print(f'  donors={len(donors)} acceptors={len(acceptors)} bridges={len(bridges)}')
    print(f'  acceptors has C=S: {has_CS}, has C#N: {has_CN}; donors has tert-N: {has_tertN}')
    record('Check2: buckets nonempty', nonempty)
    record('Check2: acceptors cover C=S & C#N', has_CS and has_CN,
           f'C={has_CS} N={has_CN}')
    record('Check2: donors cover tertiary amine', has_tertN)

    # ---- Check 3: MoleculeRefiner 可加载 + process_molecule ----
    print('\n--- Check 3: MoleculeRefiner ---')
    try:
        from util.mol_refine import MoleculeRefiner
        refiner = MoleculeRefiner(LIB_PATH)
        # demo 分子需带 3D 构象(process_molecule 内部读 conformer)
        demo = Chem.Mol(known_mols[0])
        demo = Chem.AddHs(demo)
        AllChem.EmbedMolecule(demo, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(demo, maxIters=50)
        except Exception:
            pass
        demo = Chem.RemoveHs(demo)
        out = refiner.process_molecule(demo)
        ok3 = True
        detail3 = 'process_molecule ran (replaced=%s)' % (out is not None)
    except Exception as e:
        ok3 = False
        detail3 = f'ERROR: {e}'
    record('Check3: MoleculeRefiner init + process_molecule', ok3, detail3)

    # ---- Check 4: 3D 完整性 ----
    print('\n--- Check 4: 3D 完整性 (freq>=5 molblock>=90%) ---')
    rate_overall = None
    ok4 = True
    for role, comps in [('donors', donors), ('acceptors', acceptors), ('bridges', bridges)]:
        sel = [c for c in comps if c['freq'] >= 5]
        if not sel:
            rate = 1.0
        else:
            ok = sum(1 for c in sel if c.get('molblocks'))
            rate = ok / len(sel)
        print(f'  {role}: {rate*100:.1f}% ({len(sel)} freq>=5 comps)')
        if rate < 0.90:
            ok4 = False
    record('Check4: 3D completeness >=90% (all buckets)', ok4)

    # ---- Check 5: 报告 ----
    print('\n--- Check 5: 报告 ---')
    print(f'  source_csv     : {meta.get("source_csv")}')
    print(f'  total_fluordb  : {meta.get("total_fluordb")}')
    print(f'  subset_size    : {meta.get("subset_size")}')
    print(f'  build_time     : {meta.get("build_time")}')

    print('\n=== per-bucket top-10 ===')
    for role, comps in [('donors', donors), ('acceptors', acceptors), ('bridges', bridges)]:
        print(f'\n[{role}]')
        for c in comps[:10]:
            mb = 'Y' if c.get('molblocks') else 'N'
            print(f"  freq={c['freq']:>5} nref={c['n_ref']:>4} 3D={mb}  "
                  f"{c['smiles']}  | smarts={c['smarts']}")

    # ---- 汇总 ----
    print('\n' + '=' * 70)
    n_pass = sum(1 for _, p, _ in CHECKS if p)
    n_all = len(CHECKS)
    print(f'SUMMARY: {n_pass}/{n_all} checks PASSED')
    print('=' * 70)
    for name, p, detail in CHECKS:
        print(f"  {'PASS' if p else 'FAIL'}  {name}" + (f' ({detail})' if detail else ''))

    return 0 if all(p for _, p, _ in CHECKS) else 1


if __name__ == '__main__':
    sys.exit(main())
