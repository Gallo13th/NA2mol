#!/usr/bin/env python3
"""Step4: the model's strongly-endorsed molecule at the KPT102-docked cyanine pose.

From Step3 Exp1 (masked per-position element distributions, aromatic-aware):
  1. full argmax assembly: set every skeleton atom to its argmax element class
     -> SMILES (the molecule the model "wants" in this pocket, given the
     cyanine carbon skeleton shape/coordinates)
  2. strong-correction positions: argmax prob > 50% AND argmax class differs
     from the T1 target element class (T1: N at 2/12/18, C elsewhere,
     aromaticity per skeleton)
  3. T1 variant with only those strong corrections applied -> SMILES
     + Tanimoto coefficient (Morgan/ECFP4) vs T1
  4. identical analysis recomputed from the archived BER102 probe json for
     head-to-head comparison
"""
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_SMILES = 'CCN1C=C/C(=C/C=C/c2cc[n+]3ccccc3n2)c2ccccc21'
N_SITES = {2: (7, False), 12: (7, True), 18: (7, True)}
K13_TO_Z = {0: 1, 1: 6, 2: 6, 3: 7, 4: 7, 5: 8, 6: 8, 7: 9, 8: 15, 9: 15, 10: 16, 11: 16, 12: 17}
CLS_NAME = {0: 'H', 1: '6', 2: '6*', 3: '7', 4: '7*', 5: '8', 6: '8*', 7: '9',
            8: '15', 9: '15*', 10: '16', 11: '16*', 12: '17'}


AROM_CLS = {2, 4, 6, 9, 11}


def make_skeleton():
    tgt = Chem.MolFromSmiles(TARGET_SMILES)
    rw = Chem.RWMol(tgt)
    for a in rw.GetAtoms():
        a.SetAtomicNum(6)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
    skel = rw.GetMol()
    Chem.SanitizeMol(skel)
    return skel


def _build(skeleton_classes, elem_classes, strict):
    skel = make_skeleton()
    rw = Chem.RWMol(skel)
    arom_disagree = []
    for i, cls in enumerate(elem_classes):
        z = K13_TO_Z[cls]
        arom = cls in AROM_CLS
        skel_arom = int(skeleton_classes[i]) == 2
        if arom != skel_arom:
            arom_disagree.append(i)
        a = rw.GetAtomWithIdx(i)
        a.SetAtomicNum(z)
        a.SetFormalCharge(0)
        if strict:
            a.SetIsAromatic(arom)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
    if strict:
        for b in rw.GetBonds():
            if b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic():
                b.SetBondType(Chem.BondType.AROMATIC)
                b.SetIsAromatic(True)
    mol = rw.GetMol()
    for a in mol.GetAtoms():  # aromatic bridgehead N (3 ring neighbours) -> n+
        if a.GetSymbol() == 'N' and a.GetIsAromatic() and a.GetDegree() == 3:
            a.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    return mol, Chem.MolToSmiles(mol), arom_disagree


def assemble(skeleton_classes, elem_classes):
    """elements per argmax class; tier A strict (element+aromaticity, bonds
    aromatized where both ends aromatic) -> tier B fallback (element only,
    skeleton aromaticity). Returns (mol, smiles, tier, arom-disagreements)."""
    try:
        mol, smi, dis = _build(skeleton_classes, elem_classes, strict=True)
        return mol, smi, 'A(strict: element+aromaticity)', dis
    except Exception:
        pass
    try:
        mol, smi, dis = _build(skeleton_classes, elem_classes, strict=False)
        return mol, smi, 'B(fallback: element only)', dis
    except Exception as e:
        return None, f'<sanitize failed: {e}>', '-', []


def fp_tc(smiles_a, smiles_b):
    if smiles_a.startswith('<') or smiles_b.startswith('<'):
        return None
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return DataStructs.TanimotoSimilarity(fa, fb)


def analyse(tag, exp1, arom_ktrue):
    arom_ktrue = [int(k) for k in arom_ktrue]
    probs = np.array([exp1[str(i)]['probs'] for i in range(25)])
    argmax = probs.argmax(1)
    pmax = probs.max(1)

    # target class per position (T1 element + skeleton aromaticity)
    tgt_cls = []
    for i in range(25):
        z, _ = N_SITES.get(i, (6, None))
        tgt_cls.append({(6, False): 1, (6, True): 2, (7, False): 3, (7, True): 4}[(z, arom_ktrue[i] == 2)])

    print(f'\n===== [{tag}] full argmax assembly =====')
    print('pos argmax (p%):')
    print('  ' + ' | '.join(f'{i}:{CLS_NAME[int(argmax[i])]}({pmax[i]*100:.0f}%)' for i in range(25)))
    asm_mol, asm_smi, asm_tier, dis = assemble(arom_ktrue, argmax)
    print(f'  assembled SMILES [{asm_tier}]: {asm_smi}')
    if dis:
        print(f'  (argmax aromaticity != skeleton at positions {dis})')
    if asm_mol is not None:
        print(f'  formula: {CalcMolFormula(asm_mol)}  charge: {Chem.GetFormalCharge(asm_mol)}')
        tc = fp_tc(asm_smi, TARGET_SMILES)
        print(f'  Tc vs T1: {tc:.3f}' if tc is not None else '  Tc vs T1: n/a')

    corrections = [i for i in range(25)
                   if pmax[i] > 0.5 and int(argmax[i]) != tgt_cls[i]]
    print(f'\n----- [{tag}] strong-correction positions (argmax>50% & != T1 target) -----')
    for i in corrections:
        t = N_SITES.get(i, (6, None))
        print(f'  pos {i:>2}: T1 {CLS_NAME[tgt_cls[i]]} -> {CLS_NAME[int(argmax[i])]} '
              f'({pmax[i]*100:.1f}%)   [P(N)={exp1[str(i)]["PN"]*100:.1f}%]')
    if not corrections:
        print('  (none)')

    # T1 variant with only the strong corrections applied
    variant = list(tgt_cls)
    for i in corrections:
        variant[i] = int(argmax[i])
    var_mol, var_smi, var_tier, _ = assemble(arom_ktrue, variant)
    print(f'  T1+corrections SMILES [{var_tier}]: {var_smi}')
    if var_mol is not None:
        print(f'  formula: {CalcMolFormula(var_mol)}  charge: {Chem.GetFormalCharge(var_mol)}')
        tc = fp_tc(var_smi, TARGET_SMILES)
        print(f'  Tc vs T1: {tc:.3f}' if tc is not None else '  Tc: n/a')

    n_endorsed = sum(1 for i in range(25) if pmax[i] > 0.5 and int(argmax[i]) == tgt_cls[i])
    return {'argmax': [int(a) for a in argmax], 'pmax': [round(float(p), 4) for p in pmax],
            'tgt_cls': tgt_cls,
            'assembled_smiles': asm_smi, 'assembled_tier': asm_tier,
            'assembled_formula': CalcMolFormula(asm_mol) if asm_mol else None,
            'corrections': [{'pos': i, 'from': CLS_NAME[tgt_cls[i]],
                             'to': CLS_NAME[int(argmax[i])], 'p': round(float(pmax[i]), 4)}
                            for i in corrections],
            'variant_smiles': var_smi, 'variant_tier': var_tier,
            'variant_formula': CalcMolFormula(var_mol) if var_mol else None,
            'Tc_variant_vs_t1': fp_tc(var_smi, TARGET_SMILES) if var_mol else None,
            'Tc_assembled_vs_t1': fp_tc(asm_smi, TARGET_SMILES) if asm_mol else None,
            'n_endorsed_ge50': n_endorsed}


def main():
    kpt = json.load(open(os.path.join(HERE, 'probe_kpt102_results.json')))
    ber = json.load(open(os.path.join(HERE, '..', 'ber102_rdock', 'probe_ber102_results.json')))
    out = {}
    out['kpt102'] = analyse('KPT102 (7X8O)', kpt['exp1'], kpt['arom_ktrue'])
    out['ber102'] = analyse('BER102 (7X8M)', ber['exp1'], ber['arom_ktrue'])

    print('\n===== head-to-head =====')
    for key in ('assembled_smiles', 'variant_smiles', 'Tc_variant_vs_t1', 'Tc_assembled_vs_t1'):
        print(f'  {key}: KPT {out["kpt102"][key]}  |  BER {out["ber102"][key]}')
    kc = {c['pos'] for c in out['kpt102']['corrections']}
    bc = {c['pos'] for c in out['ber102']['corrections']}
    print(f'  correction positions: KPT {sorted(kc)}  |  BER {sorted(bc)}  | shared {sorted(kc & bc)}')

    with open(os.path.join(HERE, 'step4_endorsed_molecule.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n[saved] {HERE}/step4_endorsed_molecule.json')


if __name__ == '__main__':
    main()
