#!/usr/bin/env python3
"""Full m1-route overview (v2, global-scan edition).

One wide SVG row: 18641-library global scan pick (chroman core gen[16145] x
phenol gen[9945]) -> extraction -> assembly seed -> mutation ladder ->
m1/m2/m3, with rDock/RNAsmol dual scores annotated.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDraw2DSVG
RDLogger.DisableLog('rdApp.*')

OUT = 'casestudy/figures/overview_m1_route.svg'
D = 'casestudy/sp1/m1_route'

def load(p):
    return next(Chem.SDMolSupplier(p, removeHs=True))

cells = [
    ('COLD START\nlibrary 18641\n(global scan pick)', 'casestudy/sp1/m1_route/fragment_sources/gen16145.sdf'),
    ('core: chroman\ngen[16145] native', 'dummy'),
    ('COLD START\nphenol B-ring\ngen[9945] native', 'casestudy/sp1/m1_route/fragment_sources/gen9945.sdf'),
    ('B-ring: phenol\n(4\'-OH)', 'dummy'),
    ('ASSEMBLY seed\n(2-phenylchroman,\nSATURATED 6+6)', f'{D}/global_scan/seed_g0.sdf'),
    ('BOND-ORDER STEP\n+C4=O (chrom.-4-one)', f'{D}/mutg0_ox.sdf'),
    ('BOND-ORDER STEP\n+C2=C3 (chromone)', f'{D}/mutg0_ox_dehyd.sdf'),
    ('+C3-OH', f'{D}/mutg0_ox_dehyd_3OH.sdf'),
    ('+A5,7-OH\n== m1  (rdock -50.53)', f'{D}/mutg0_m1.sdf'),
    ('+2\'-OH\n== m2  (rdock -38.98)', f'{D}/mutg0_m2.sdf'),
    ('+3\'-OH\n== m3  (rdock -39.29)', f'{D}/mutg0_m3.sdf'),
]

cells_fig = [c for c in cells if c[1] != 'dummy']
W_CELL, H_CELL = 320, 280
W = len(cells_fig) * W_CELL + 40
H = H_CELL + 80
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="20" y="28" font-size="22" font-weight="bold" fill="#222">sp1 m1-route v2 (18641-library global scan)</text>']

fi = 0
for i, (label, path) in enumerate(cells):
    if path == 'dummy':
        continue
    x0 = 20 + fi * W_CELL
    m = load(path)
    rdDepictor.SetPreferCoordGen(True); rdDepictor.Compute2DCoords(m)
    d = MolDraw2DSVG(W_CELL - 30, 190)
    d.drawOptions().bondLineWidth = 2
    d.DrawMolecule(m)
    d.FinishDrawing()
    inner = d.GetDrawingText()
    # strip the inner <svg ...> opening tag itself (keep its children)
    i0 = inner.find('<svg')
    i1 = inner.find('>', i0)
    inner = inner[i1+1:]
    inner = inner[:inner.rfind('</svg>')]
    parts.append(f'<g transform="translate({x0},60)">{inner}</g>')
    label_parts = label.split('\n')
    for li, lp in enumerate(label_parts):
        parts.append(f'<text x="{x0+8}" y="{265+li*18}" font-size="13" font-weight="bold" fill="#333">{lp}</text>')
    if fi < len(cells_fig) - 1:
        parts.append(f'<path d="M {x0+W_CELL-14} 140 L {x0+W_CELL-4} 140" stroke="#888" stroke-width="2"/>')
    fi += 1

parts.append('</svg>')
with open(OUT, 'w') as f:
    f.write('\n'.join(parts))
print(f'[saved] {OUT}')
