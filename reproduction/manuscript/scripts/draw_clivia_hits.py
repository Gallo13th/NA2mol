"""Draw the generated analogues closest to the two Clivia/8HZE target fluorophores.

Row layout (one block per target):
    [target]  [rank 1]  [rank 2]  [rank 3]  [rank 4]  [rank 5]

The two target structures are drawn with the substructure they share with the generated hit
highlighted, so it is visible what the model reproduced (the ring system) and what it did not
(the linker / substituent pattern).

    python scripts/draw_clivia_hits.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, rdDepictor

ROOT = Path(__file__).resolve().parent.parent
import sys
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs' / 'clivia_8hze'
SUFFIX = '' if D.name == 'clivia_8hze' else '_' + D.name
TARGETS = {
    'clivia_target1': 'COC1=CC=C(C=C1)C3=CC(=NC2=CC(OC)=CC=C23)C4=CC=C(O)C=C4',
    'clivia_target2': r'CN(C)C=1C=CC=C(C=1)\C=C(C#N)/C2=CC=C(C=C2)C3=CC=CC=C3',
}
CELL = 330
PAD = 8
HEADER = 46


def draw(mol, legend, highlight=None, sub=()):
    rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=(CELL, CELL), legend=legend,
                           highlightAtoms=list(highlight or []), highlightBonds=list(sub),
                           kekulize=False)


def main() -> int:
    hits = pd.read_csv(D / 'closest_to_targets.csv')
    rows = []
    for name, smi in TARGETS.items():
        t = Chem.MolFromSmiles(smi)
        tfp = AllChem.GetMorganFingerprintAsBitVect(t, 2, nBits=2048)
        g = hits[hits['target'] == name].head(5)
        cells = [('target (reference)', Chem.MolToSmiles(t), 1.000)]
        for r in g.itertuples():
            cells.append((f'gen rank {r.rank}', r.smiles, r.tc))
        rows.append((name, cells))

    cols = 6
    width = PAD + cols * (CELL + PAD)
    height = HEADER + len(rows) * (CELL + PAD + 18) + PAD
    canvas = Image.new('RGB', (width, height), 'white')
    draw_img = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
        font_b = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 16)
    except OSError:
        font = font_b = ImageFont.load_default()

    y = HEADER
    for name, cells in rows:
        label = ('Target 1 — quinoline / bis-aryl / phenol' if name.endswith('1')
                 else 'Target 2 — NMe$_2$ / acrylonitrile / biphenyl').replace('$_2$', '2')
        draw_img.text((PAD, y - 34 if y == HEADER else y - 32), label, fill='black', font=font_b)
        x = PAD
        for i, (tag, smi, tc) in enumerate(cells):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                x += CELL + PAD
                continue
            if i == 0:
                legend = f'{tag}'
                img = draw(m, legend)
            else:
                legend = f'{tag}  Tc={tc:.2f}'
                img = draw(m, legend)
            canvas.paste(img, (x, y))
            x += CELL + PAD
        y += CELL + PAD + 18

    out = D / f'clivia_8hze_top_hits{SUFFIX}.png'
    canvas.save(out)
    print('wrote', out)

    # --- the few molecules that do carry the acrylonitrile ---
    samples = pd.read_csv(D / 'samples.csv')
    patt = Chem.MolFromSmarts('[CX3]=[CX3][CX2]#N')
    hits_ccn = [s for s in samples['smiles'] if Chem.MolFromSmiles(s) and
                Chem.MolFromSmiles(s).HasSubstructMatch(patt)]
    print('acrylonitrile molecules:', len(hits_ccn))
    cw = PAD + max(len(hits_ccn), 1) * (CELL + PAD)
    cv2 = Image.new('RGB', (cw, HEADER + CELL + PAD), 'white')
    d2 = ImageDraw.Draw(cv2)
    d2.text((PAD, 12), f'All {len(hits_ccn)} generated molecules containing C=C-C#N '
                       '(motif highlighted)', fill='black', font=font_b)
    x = PAD
    for smi in hits_ccn:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        match = m.GetSubstructMatch(patt)
        bonds = [b.GetIdx() for b in m.GetBonds()
                 if b.GetBeginAtomIdx() in match and b.GetEndAtomIdx() in match]
        img = draw(m, 'C=C-C#N', highlight=match, sub=bonds)
        cv2.paste(img, (x, HEADER))
        x += CELL + PAD
    out2 = D / f'clivia_8hze_acrylonitrile_hits{SUFFIX}.png'
    cv2.save(out2)
    print('wrote', out2)
    for smi in hits_ccn:
        print('   ', smi)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
