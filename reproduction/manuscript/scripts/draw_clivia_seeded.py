"""Draw scaffold-anchored designs next to the target, highlighting the fixed ring system.

Because the ring skeleton is held fixed during sampling, every panel of the design row
carries the complete ring system of the target; the interesting variation is in the
decoration the model adds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

ROOT = Path(__file__).resolve().parent.parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs' / 'clivia_8hze'
TARGET_SMILES = {
    '1': 'COC1=CC=C(C=C1)C3=CC(=NC2=CC(OC)=CC=C23)C4=CC=C(O)C=C4',
    '2': r'CN(C)C=1C=CC=C(C=1)\C=C(C#N)/C2=CC=C(C=C2)C3=CC=CC=C3',
}
RINGSYS = {
    '1': 'c1ccc2nc(-c3ccccc3)cc(-c3ccccc3)c2c1',
    '2': 'c1ccc(-c2ccc(/C=C/c3ccccc3)cc2)cc1',
}
CELL, PAD, HEADER = 330, 8, 46


def render(mol, legend, highlight):
    rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=(CELL, CELL), legend=legend, highlightAtoms=highlight,
                           kekulize=False)


def main() -> int:
    try:
        font_b = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 16)
    except OSError:
        font_b = ImageFont.load_default()

    rows = []
    for t in ('1', '2'):
        csv = D / f'seeded_t{t}.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv).sort_values('ecfp4', ascending=False).head(5)
        rows.append((t, df))

    cols = 6
    width = PAD + cols * (CELL + PAD)
    height = HEADER + len(rows) * (CELL + PAD + 18) + PAD
    canvas = Image.new('RGB', (width, height), 'white')
    dr = ImageDraw.Draw(canvas)
    y = HEADER
    for t, df in rows:
        dr.text((PAD, y - 34), f'Target {t} — ring system fixed, model decorates',
                fill='black', font=font_b)
        patt = Chem.MolFromSmarts(RINGSYS[t])
        cells = [('target (reference)', TARGET_SMILES[t], None)]
        for r in df.itertuples():
            cells.append((f'Tc={r.ecfp4:.2f}', r.smiles, patt))
        x = PAD
        for tag, smi, highlight_patt in cells:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                x += CELL + PAD
                continue
            hl = []
            if highlight_patt is not None:
                hit = m.GetSubstructMatch(highlight_patt)
                hl = list(hit)
            canvas.paste(render(m, tag, hl), (x, y))
            x += CELL + PAD
        y += CELL + PAD + 18
    out = D / 'clivia_8hze_seeded_designs.png'
    canvas.save(out)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
