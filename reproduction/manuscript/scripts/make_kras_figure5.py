#!/usr/bin/env python3
"""Rebuild Figure 5 (KRAS case study).

Discipline: every panel is produced as its own artwork at its final printed size, then the
panels are placed on one canvas and the panel letters are added last, all on a common baseline.
"""
import os
import subprocess
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.optimize import curve_fit
from PIL import Image
RDLogger.DisableLog('rdApp.*')

OUT = 'docs/clivia_8hze/figures'
os.makedirs(OUT, exist_ok=True)
TEAL = '#14615c'

# ============================================================ panel f: candidate grid
df = pd.read_csv('figures/figure4_candidate_labels.csv').sort_values('candidate')
DPI = 600
NCOL, NROW = 5, 7
CELL_W_MM, CELL_H_MM = 23.0, 16.4      # cell incl. the vote line
MOL_H_MM = 12.4                        # drawing area inside the cell
W = int(round(CELL_W_MM / 25.4 * DPI))
H = int(round(MOL_H_MM / 25.4 * DPI))
BIG_W, BIG_H, BOND = 2400, 1400, 96
FONT_BOLD = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf',
                         'DejaVuSans-Bold.ttf')


def _render(smi):
    """Draw on a large canvas at one fixed bond length; return the cropped ink."""
    m = Chem.MolFromSmiles(smi)
    AllChem.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(BIG_W, BIG_H)
    o = d.drawOptions()
    o.fixedBondLength = BOND
    o.bondLineWidth = 11
    o.fontFile = FONT_BOLD
    o.minFontSize = 104
    o.maxFontSize = 112
    o.padding = 0.02
    o.clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    im = Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB')
    arr = np.asarray(im)
    ys, xs = np.where((arr < 250).any(axis=2))
    pad = 8
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(BIG_W, xs.max() + pad), min(BIG_H, ys.max() + pad)))


_raw = [_render(r['smiles']) for _, r in df.iterrows()][:NCOL * NROW]
_max_w = max(im.size[0] for im in _raw)
_max_h = max(im.size[1] for im in _raw)
_SCALE = min(0.92 * W / _max_w, 0.88 * H / _max_h)


def _cell(im):
    w, h = int(im.size[0] * _SCALE), int(im.size[1] * _SCALE)
    im = im.resize((w, h), Image.LANCZOS)
    cell = Image.new('RGB', (W, H), 'white')
    cell.paste(im, ((W - w) // 2, (H - h) // 2))
    return np.asarray(cell)


FIG_W_MM = NCOL * CELL_W_MM
FIG_H_MM = NROW * CELL_H_MM
fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
for i, (im, (_, r)) in enumerate(zip(_raw, df.iterrows())):
    if i >= NCOL * NROW:
        break
    col, row = i % NCOL, i // NCOL
    cell_top = 1.0 - row / NROW
    mol_h = (MOL_H_MM / CELL_H_MM) / NROW
    ax = fig.add_axes([col / NCOL, cell_top - mol_h - 0.004, 1 / NCOL, mol_h])
    ax.imshow(_cell(im))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    votes = int(r['model_votes'])
    colour = TEAL if votes >= 4 else '#555555'
    fig.text((col + 0.5) / NCOL, cell_top - mol_h - 0.010, f'{votes}/5 votes',
             ha='center', va='top', fontsize=5.2, color=colour,
             fontweight='bold' if votes >= 4 else 'normal')
# frame the experimentally tested, highest-vote candidate (first cell)
fig.patches.append(Rectangle((0.0015, 1 - 1 / NROW + 0.003), 1 / NCOL - 0.003, 1 / NROW - 0.006,
                             transform=fig.transFigure, fill=False, edgecolor=TEAL, lw=1.5, zorder=10))
fig.savefig(f'{OUT}/panel_f_grid.pdf', dpi=DPI)
fig.savefig(f'{OUT}/panel_f_grid.png', dpi=200)
plt.close(fig)
print(f'panel f: {FIG_W_MM:.0f} x {FIG_H_MM:.0f} mm, {len(_raw)} structures, votes shown')

# ============================================================ panel e legend (replaces inline text)
fig, ax = plt.subplots(figsize=(19.0 / 25.4, 3.8 / 25.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')
ax.scatter([0.05], [0.5], s=16, color=TEAL, lw=0)
ax.text(0.13, 0.5, 'Tier-1', fontsize=5.6, va='center')
ax.scatter([0.62], [0.5], s=16, color='#d0d0d0', lw=0)
ax.text(0.70, 0.5, 'Other', fontsize=5.6, va='center')
fig.savefig(f'{OUT}/panel_e_legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0.01)
fig.savefig(f'{OUT}/panel_e_legend.png', dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.01)
plt.close(fig)

# ============================================================ canvas assembly
PAGE_W, PAGE_H = 183.0, 243.0
ML, MT = 4.0, 8.0          # left margin, top margin (room for the letters)
SRC = os.path.abspath('figures/figure4_noletters.pdf')
SRC_W, SRC_H = 237.8, 279.0

body = []


def img_put(x, top, w, h, path_or_trim):
    """Place a panel with its top-left corner at (x, top) measured from the page top."""
    y = PAGE_H - top - h
    if isinstance(path_or_trim, tuple):
        left, right, top_t, bottom = path_or_trim
        inc = (f'\\includegraphics[width={w:.2f}mm,trim={left:.1f}mm {SRC_H - bottom:.1f}mm '
               f'{SRC_W - right:.1f}mm {top_t:.1f}mm,clip]{{{SRC}}}')
    else:
        inc = f'\\includegraphics[width={w:.2f}mm]{{{path_or_trim}}}'
    body.append(f'\\put({x:.2f},{y:.2f}){{{inc}}}')
    return y, h


def letter_put(letter, x, y, h):
    body.append(f'\\put({x:.2f},{y + h + 1.4:.2f})'
                f'{{\\fontsize{{9}}{{10}}\\selectfont\\bfseries {letter}}}')


# ---- row 1: a (workflow) -------------------------------------------------
a_trim = (10.5, 226.6, 10.5, 74.8)
A_W = PAGE_W - 2 * ML
A_H = A_W * (a_trim[3] - a_trim[2]) / (a_trim[1] - a_trim[0])
y, h = img_put(ML, MT, A_W, A_H, a_trim)
letter_put('a', ML, y, h)
row1_bottom = MT + A_H

# ---- row 2: b, c, d (was e), e (was f) -----------------------------------
GAP = 5.0
ROW2_TOP = row1_bottom + GAP + 2.0
SMALL_H = 50.0
cols = [('b', (6.5, 60.1, 94.1, 159.3)), ('c', (68.9, 117.7, 94.1, 159.3)),
        ('d', (126.8, 175.5, 94.1, 159.3)), ('e', (187.1, 233.2, 94.1, 159.3))]
x = ML
row2_h = 0.0
for letter, trim in cols:
    w = SMALL_H * (trim[1] - trim[0]) / (trim[3] - trim[2])
    y, h = img_put(x, ROW2_TOP, w, SMALL_H, trim)
    letter_put(letter, x, y, h)
    x += w + GAP
    row2_h = max(row2_h, h)

# ---- row 3: f (grid) left; g (ThT) and h (qPCR) stacked right ------------
ROW3_TOP = ROW2_TOP + row2_h + GAP + 2.0
GRID_W, GRID_H = FIG_W_MM, FIG_H_MM
y, h = img_put(ML, ROW3_TOP, GRID_W, GRID_H, os.path.abspath(f'{OUT}/panel_f_grid.pdf'))
letter_put('f', ML, y, h)

# legend for panel e: vector text + filled dots, right-aligned in the panel's top band
E_X = None
E_W = None
E_TOP = ROW2_TOP
_x = ML
for letter, trim in cols:
    w = SMALL_H * (trim[1] - trim[0]) / (trim[3] - trim[2])
    if letter == 'e':
        E_X, E_W = _x, w
    _x += w + GAP

_ly = PAGE_H - (E_TOP + 6.6)
_xr = E_X + E_W - 1.2
body.append(r'\put(%.2f,%.2f){\makebox[0pt][r]{\fontsize{5.6}{6.5}\selectfont Other}}' % (_xr, _ly))
body.append(r'\put(%.2f,%.2f){\color[gray]{0.80}\circle*{1.5}}' % (_xr - 5.4, _ly + 0.35))
body.append(r'\put(%.2f,%.2f){\makebox[0pt][r]{\fontsize{5.6}{6.5}\selectfont Tier-1}}' % (_xr - 6.8, _ly))
body.append(r'\put(%.2f,%.2f){\color{tealbox}\circle*{1.5}}' % (_xr - 13.2, _ly + 0.35))

RX = ML + GRID_W + 14.0
G_W, G_H = 50.0, 43.1
y, h = img_put(RX, ROW3_TOP, G_W, G_H, os.path.abspath(f'{OUT}/panel_g_tht.pdf'))
letter_put('g', RX, y, h)
H_TOP = ROW3_TOP + G_H + 12.0
H_W, H_H = 46.0, 43.0
y, h = img_put(RX, H_TOP, H_W, H_H, os.path.abspath(f'{OUT}/kras_qpcr_panel_h.pdf'))
letter_put('h', RX, y, h)

tex = f'''\\documentclass[10pt]{{article}}
\\usepackage[paperwidth={PAGE_W}mm,paperheight={PAGE_H}mm,margin=0mm]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{helvet}}
\\usepackage{{xcolor}}
\\definecolor{{tealbox}}{{RGB}}{{20,97,92}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}
\\pagestyle{{empty}}
\\setlength{{\\parindent}}{{0pt}}
\\begin{{document}}
\\setlength{{\\unitlength}}{{1mm}}
\\begin{{picture}}({PAGE_W},{PAGE_H})(0,0)
''' + '\n'.join(body) + '''
\\end{picture}
\\end{document}
'''
tex_path = f'{OUT}/figure5_kras.tex'
open(tex_path, 'w').write(tex)
subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                '-output-directory', OUT, tex_path], check=True, capture_output=True)
subprocess.run(['cp', f'{OUT}/figure5_kras.pdf', 'figures/figure5_kras.pdf'], check=True)
print(f'wrote figures/figure5_kras.pdf  ({PAGE_W} x {PAGE_H} mm)')
