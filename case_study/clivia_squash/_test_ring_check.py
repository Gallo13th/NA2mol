#!/usr/bin/env python3
"""Validate the three-ring-system check on targets (must pass) and fused hits (must fail)."""
import importlib.util
import os
import sys

ROOT = '/home/data/users/gaoletian/rna2smol'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

spec = importlib.util.spec_from_file_location('rs', f'{ROOT}/case_study/clivia_squash/ring_system_check.py')
RSC = importlib.util.module_from_spec(spec); sys.modules['rs'] = RSC; spec.loader.exec_module(RSC)

TARGETS = {
    'mol1': 'COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1',
    'mol2': 'COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1',
    'mol3': 'COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1',
    'mol4': 'COc1ccc(-c2cc(-c3ccncc3)nc3ccc(OC)cc23)cc1',
}
FUSED = {
    'fused hit A': 'COc1cc2c3c(cc(-c4ccc(O)cc4)nc3c1)-c1ccc(O)cc1-2',
    'fused hit B': 'COc1ccc(-c2nc3ccc(OC)cc3c3c(O)cc(O)cc23)cc1',
    'fused hit C': 'COc1ccc(-c2ccc3c(O)cc(-c4cccc(OC)c4)nc3c2)cc1',
}
print('=== targets (must pass) ===')
ok_t = 0
for k, smi in TARGETS.items():
    m = Chem.MolFromSmiles(smi)
    ok, why = RSC.check_three_ring_systems(m)
    ok_t += ok
    print(f'  {"PASS" if ok else "FAIL"}  {k}  {why}')

print('\n=== fused four-ring hits (must be rejected) ===')
rej = 0
for k, smi in FUSED.items():
    m = Chem.MolFromSmiles(smi)
    if m is None:
        print(f'  {k}: unparsable'); continue
    ok, why = RSC.check_three_ring_systems(m)
    rej += (not ok)
    print(f'  {"PASS" if ok else "REJECT"}  {k}  {why}')

print(f'\ntargets accepted {ok_t}/{len(TARGETS)}   fused rejected {rej}/{len(FUSED)}')
