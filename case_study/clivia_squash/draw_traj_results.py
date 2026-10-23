#!/usr/bin/env python3
"""Draw the best core-fixed resampling hits against the target series."""
import glob
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
QUIN = Chem.MolFromSmarts('c1ccc2ncccc2c1')
def pos24(m):
    out = set()
    for mt in m.GetSubstructMatches(QUIN):
        for lab, i in (('C2', 5), ('C4', 7)):
            for nb in m.GetAtomWithIdx(mt[i]).GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in mt and nb.GetSymbol() in 'CN':
                    out.add(lab)
    return out
T = {'mol1': ('COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1', 'C23H19NO3'),
     'mol2': ('COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1', 'C23H20N2O2'),
     'mol3': ('COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1', 'C25H24N2O2')}
ref = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v[0])) for k, v in T.items()}
seen = set(); rows = []
for f in sorted(glob.glob(str(HERE / 'output_traj/*/molecules.sdf'))):
    for m in Chem.SDMolSupplier(f, sanitize=False):
        if m is None: continue
        try: Chem.SanitizeMol(m)
        except Exception: continue
        if pos24(m) != {'C2', 'C4'}: continue
        smi = Chem.MolToSmiles(m)
        if smi in seen: continue
        seen.add(smi)
        s = max(DataStructs.TanimotoSimilarity(GEN.GetFingerprint(m), v) for v in ref.values())
        rows.append((s, m))
rows.sort(key=lambda x: -x[0])
cells = []
for k, (smi, formula) in T.items():
    m = Chem.MolFromSmiles(smi); m = Chem.AddHs(m); AllChem.Compute2DCoords(m)
    cells.append((Chem.RemoveHs(m), f'{k} TARGET {formula}'))
for s, m in rows[:3]:
    mm = Chem.AddHs(m); AllChem.Compute2DCoords(mm)
    cells.append((Chem.RemoveHs(mm),
                  f'core-fixed hit  ECFP4 {s:.3f}  MW {Descriptors.MolWt(m):.0f}'))
o = Draw.MolDrawOptions(); o.bondLineWidth = 1.4
img = Draw.MolsToGridImage([c[0] for c in cells], molsPerRow=3, subImgSize=(360, 320),
                           legends=[c[1] for c in cells], drawOptions=o)
img.save(str(OUT / 'core_fixed_best.png'))
print('wrote', OUT / 'core_fixed_best.png')
