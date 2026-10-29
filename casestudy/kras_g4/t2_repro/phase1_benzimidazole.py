#!/usr/bin/env python3
"""Phase1 supplement: locate benzimidazole (5-ring diaza-fused) + CH2 molecules in both
libraries, with BER102-pocket coords check; candidates for lib-seed extraction + N->S."""
import os
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np

BENZIMID = Chem.MolFromSmarts('c1ccc2nc[nX2]c2c1')   # benzimidazole core (any N order)
BENZIMID2 = Chem.MolFromSmarts('c1ccc2[nX2]cnc2c1')
BRIDGE = Chem.MolFromSmarts('[#6R]-[#6H2]-[#6R]')

libs = [
    ('ber102', 'casestudy/kras_g4/ber102_gen2k/valid_mols.sdf'),
    ('lib10k', 'casestudy/kras_g4/library_10k.sdf'),
]
# BER102 pocket center = berberine reference centroid
ref = next(Chem.SDMolSupplier('casestudy/kras_g4/ber102_rdock/ber102_ref.sdf', removeHs=True))
BER = ref.GetConformer().GetPositions().mean(axis=0)
print('BER102 ref centroid:', np.round(BER, 2))

for name, path in libs:
    print(f'\n===== {name} =====')
    supp = Chem.SDMolSupplier(path, removeHs=True)
    n_bim, hits = 0, []
    for i, m in enumerate(supp):
        if m is None or not m.GetNumConformers():
            continue
        ms = m.GetSubstructMatches(BENZIMID) or m.GetSubstructMatches(BENZIMID2)
        if not ms:
            continue
        n_bim += 1
        has_bridge = m.HasSubstructMatch(BRIDGE)
        pos = m.GetConformer().GetPositions()[list(ms[0])]
        c = pos.mean(axis=0)
        d = float(np.linalg.norm(c - BER))
        smi = Chem.MolToSmiles(m)
        nS = sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'S')
        hits.append((has_bridge, d, i, smi, nS, m.GetNumAtoms()))
    print(f'benzimidazole-containing: {n_bim}')
    w_bridge = [h for h in hits if h[0]]
    print(f'  with aryl-CH2 bridge: {len(w_bridge)}')
    for h in sorted(w_bridge, key=lambda x: x[1])[:8]:
        print(f'    d(BER)={h[1]:5.1f}  #{h[2]}  {h[3]}  (S:{h[4]}, atoms:{h[5]})')
    wo = [h for h in hits if not h[0]]
    print('  closest-to-pocket without bridge:')
    for h in sorted(wo, key=lambda x: x[1])[:5]:
        print(f'    d(BER)={h[1]:5.1f}  #{h[2]}  {h[3]}  (S:{h[4]}, atoms:{h[5]})')
