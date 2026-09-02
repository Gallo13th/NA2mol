#!/usr/bin/env python3
"""Full-route overview figure: cold start -> fragments -> assembly -> mutation
ladder -> m1/m2. Produces a single wide SVG (molecule depictions + arrows +
stage labels)."""
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

mols = []
loads = [
    ('COLD START\ngen[3192] (complete)', 'casestudy/sp1/m1_route/fragment_sources/gen3192.sdf'),
    ('fragment A: chroman core\n(native coords)', 'dummy'),
    ('COLD START\ngen[9945] (complete)', 'casestudy/sp1/m1_route/fragment_sources/gen9945.sdf'),
    ('fragment B: phenol ring\n(native coords)', 'dummy'),
    ('ASSEMBLY\n2-phenylchroman seed', f'{D}/m1_route_seed.sdf'),
    ('+C4 carbonyl\n(chroman-4-one)', f'{D}/mut_ox.sdf'),
    ('+C2=C3\n(chromone)', f'{D}/mut_ox_dehyd.sdf'),
    ('+C3-OH', f'{D}/mut_ox_dehyd_3OH.sdf'),
    ('+A5,A7-OH\n== m1', f'{D}/mut_m1.sdf'),
    ('+2\'-OH\n== m2', f'{D}/mut_m2.sdf'),
    ('+3\'-OH\n== m3', f'{D}/mut_m3.sdf'),
]

svgparts = []
W_CELL, H_CELL = 300, 260
N_CELL = len(loads)
W = N_CELL * W_CELL + 40
H = H_CELL + 90
svgparts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
svgparts.append('<rect width="100%" height="100%" fill="white"/>')
svgparts.append(f'<text x="20" y="30" font-size="22" font-weight="bold" fill="#222">sp1 cold-start route to m1 / m2 / m3</text>')

for i, (label, path) in enumerate(loads):
    x0 = 20 + i * W_CELL
    if path == 'dummy':
        continue
    m = load(path)
    rdDepictor.SetPreferCoordGen(True); rdDepictor.Compute2DCoords(m)
    d = MolDraw2DSVG(W_CELL - 30, 180)
    d.drawOptions().bondLineWidth = 2
    d.drawOptions().addStereoAnnotation = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # extract inner <svg> content
    inner = svg[svg.find('<svg', 10):]
    inner = inner[inner.find('>')+1:]
    inner = inner[:inner.rfind('</svg>')]
    svgparts.append(f'<g transform="translate({x0},55)">{inner}</g>')
    svgparts.append(f'<text x="{x0+8}" y="255" font-size="13" font-weight="bold" fill="#333">{label}</text>')
    if i < len(loads) - 1:
        svgparts.append(f'<path d="M {x0+W_CELL-14} 150 L {x0+W_CELL-4} 150" stroke="#888" stroke-width="2" marker-end="url(#arr)"/>')

svgparts.append('<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#888"/></marker></defs>')
svgparts.append('</svg>')
with open(OUT, 'w') as f:
    f.write('\n'.join(svgparts))
print(f'[saved] {OUT}')
