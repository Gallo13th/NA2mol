#!/usr/bin/env python3
"""Assemble the Clivia chapter main figure.

All panels in a row are placed at the same height, so rows line up and the panel
letters/titles share a baseline. Widths follow each panel's own aspect ratio.
"""
import os
import subprocess
from PIL import Image

PANELS = 'docs/clivia_8hze/figures/panels'
OUT = 'docs/clivia_8hze/figures'
MAXW = 178.0
ROW_GAP = 11.0
COL_GAP = 7.0
LEFT = 2.0

# (letter, file, row, row_height_mm, title)
LAYOUT = [
    ('a', 'panel_pocket_8HZE_NI4',        1, 47.0, r'RNA pocket (8HZE\_A109)'),
    ('b', 'panel_filter_scaffold',        1, 47.0, 'Scaffold selection'),
    ('c', 'panel_core',                   1, 50.0, '2,4-diarylquinoline (Al-1)'),
    ('d', 'panel_competition',            2, 47.0, 'Competition assay'),
    ('e', 'panel_imaging',                2, 47.0, 'Cell imaging'),
    ('f', 'panel_imaging_quantification', 2, 58.0, 'Cellular fluorescence'),
]


def ar(name):
    im = Image.open(f'{PANELS}/{name}.png')
    return im.size[0] / im.size[1]


rows = {}
for letter, name, row, h, title in LAYOUT:
    rows.setdefault(row, []).append([letter, name, h, title, ar(name)])

placed, y = [], 0.0
for r in sorted(rows):
    items = rows[r]
    H = items[0][2]
    total = sum(H * it[4] for it in items) + COL_GAP * (len(items) - 1)
    if total > MAXW:
        H *= MAXW / total
        total = MAXW
    x = LEFT + max(0.0, (MAXW - total) / 2.0)
    for letter, name, _, title, a in items:
        w = H * a
        placed.append((letter, name, title, x, y, w, H))
        x += w + COL_GAP
    y += H + ROW_GAP
TOTAL_H = y - ROW_GAP
TOP_PAD = 6.0
H_CANVAS = TOTAL_H + TOP_PAD

body = []
for letter, name, title, x, yy, w, h in placed:
    y_top = H_CANVAS - yy
    path = os.path.abspath(f'{PANELS}/{name}.png')
    body.append(f'\\node[anchor=north west,inner sep=0pt] at ({x:.2f}mm,{y_top:.2f}mm) '
                f'{{\\includegraphics[height={h:.2f}mm]{{{path}}}}};')
    body.append(f'\\node[anchor=south west,inner sep=0pt,font=\\bfseries\\fontsize{{7.5}}{{9}}\\selectfont] '
                f'at ({x:.2f}mm,{y_top + 1.1:.2f}mm) {{{letter}}};')
    body.append(f'\\node[anchor=south west,inner sep=0pt,font=\\fontsize{{7}}{{8}}\\selectfont] '
                f'at ({x + 3.6:.2f}mm,{y_top + 1.15:.2f}mm) {{{title}}};')

tex = r'''\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{graphicx}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[x=1mm,y=1mm]
''' + '\n'.join(body) + '\n\\end{tikzpicture}\n\\end{document}\n'

open(f'{OUT}/main_figure.tex', 'w').write(tex)
print(f'canvas {max(p[3]+p[5] for p in placed):.1f} x {H_CANVAS:.1f} mm, {len(placed)} panels')
subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                '-output-directory', OUT, f'{OUT}/main_figure.tex'], check=True, capture_output=True)
subprocess.run(['pdftoppm', '-r', '220', '-png', '-singlefile',
                f'{OUT}/main_figure.pdf', f'{OUT}/main_figure'], check=True)
print('wrote main_figure.pdf / .png')
