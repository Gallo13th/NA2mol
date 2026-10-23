#!/usr/bin/env python3
"""Draw the best three-ring-system hits against the target series."""
import glob
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

HERE = Path('/home/data/users/gaoletian/rna2smol/case_study/clivia_squash')
spec = importlib.util.spec_from_file_location('rs', str(HERE / 'ring_system_check.py'))
RSC = importlib.util.module_from_spec(spec); sys.modules['rs'] = RSC; spec.loader.exec_module(RSC)

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
T = {'mol1': ('COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1', 'C23H19NO3'),
     'mol2': ('COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1', 'C23H20N2O2'),
     'mol3': ('COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1', 'C25H24N2O2')}
ref = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v[0])) for k, v in T.items()}

seen = {}
for f in glob.glob(str(HERE / 'output_traj/st*/molecules.sdf')):
    if os.path.getsize(f) == 0:
        continue
    for m in Chem.SDMolSupplier(f, sanitize=False):
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m)
        except Exception:
            continue
        if RSC.check_three_ring_systems(m)[0]:
            seen.setdefault(Chem.MolToSmiles(m), m)
rows = sorted(((max(DataStructs.TanimotoSimilarity(GEN.GetFingerprint(m), v) for v in ref.values()),
                Chem.MolToSmiles(m), m) for m in seen.values()), key=lambda x: -x[0])

cells = []
for k, (smi, formula) in T.items():
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); AllChem.Compute2DCoords(m)
    cells.append((Chem.RemoveHs(m), f'{k} TARGET {formula}'))
for s, smi, m in rows[:3]:
    mm = Chem.AddHs(m); AllChem.Compute2DCoords(mm)
    cells.append((Chem.RemoveHs(mm),
                  f'ECFP4 {s:.3f}  MW {Descriptors.MolWt(m):.0f}  3 ring systems'))
o = Draw.MolDrawOptions(); o.bondLineWidth = 1.4
img = Draw.MolsToGridImage([c[0] for c in cells], molsPerRow=3, subImgSize=(360, 320),
                           legends=[c[1] for c in cells], drawOptions=o)
img.save(str(HERE / 'figures' / 'strict_core_best.png'))
print('wrote', HERE / 'figures' / 'strict_core_best.png')
for s, smi, m in rows[:6]:
    print(f'  {s:.3f}  {smi}')
