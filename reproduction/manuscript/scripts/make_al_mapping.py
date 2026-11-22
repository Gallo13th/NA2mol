#!/usr/bin/env python3
"""Mapping figure: collaborator Al-1/Al-2/Al-3 vs our internal designations."""
import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
OUT = 'docs/clivia_8hze/figures'
os.makedirs(OUT, exist_ok=True)

AL = [
    ('Al-1', 'COc1ccc(-c2cc(-c3ccc(O)cc3)c3ccc(OC)cc3n2)cc1',
     'our "target 1"  ·  OMe | OH | OMe', 'in 46-molecule shortlist, rank 19'),
    ('Al-2', 'COc1ccc(-c2cc(-c3ccc(N)cc3)c3ccc(OC)cc3n2)cc1',
     'our "query C"  ·  OMe | NH$_2$ | OMe', 'outside the 2% pruned library (NH$_2$ at C2 = 0.06%)'),
    ('Al-3', 'COc1ccc(-c2cc(-c3ccncc3)c3ccc(OC)cc3n2)cc1',
     'our "query B"  ·  4-pyridyl at C2', 'outside the enumeration (ring replacement, not a substituent swap)'),
]
mols, legends = [], []
for name, smi, internal, status in AL:
    m = Chem.MolFromSmiles(smi)
    mols.append(m)
    legends.append(f'{name}  ({rdMolDescriptors.CalcMolFormula(m)}, {Descriptors.MolWt(m):.1f} Da)')
img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(360, 300), legends=legends)
img.save(f'{OUT}/al_numbering.png')

rows = []
for name, smi, internal, status in AL:
    m = Chem.MolFromSmiles(smi)
    rows.append(dict(compound=name, internal_name=internal, smiles=Chem.MolToSmiles(m),
                     inchikey=Chem.MolToInchiKey(m),
                     formula=rdMolDescriptors.CalcMolFormula(m),
                     mw=round(Descriptors.MolWt(m), 2), library_status=status))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/al_numbering.csv', index=False)
print(df.to_string(index=False))
