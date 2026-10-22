#!/usr/bin/env python3
"""Draw the target series against the closest generated molecules.

Produces a visual comparison so the differences can be confirmed by eye:
  row 1  mol1-mol5, the fluorescent-active series
  row 2  the five closest Squash hits (ECFP4 to their nearest target shown)
  row 3  the five closest Clivia hits

Also renders the most informative pair - the Clivia 2,6-diaryl quinoline against mol2 -
with the scaffold-substitution positions marked, since the scaffold class matches and only
the regiochemistry differs.
"""
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
HERE = Path(__file__).resolve().parent
REPORTS = HERE / 'reports'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)

TARGETS = [
    ('mol1', 'C23H19NO3', '357.41', 'COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1'),
    ('mol2', 'C23H20N2O2', '356.43', 'COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1'),
    ('mol3', 'C25H24N2O2', '384.48', 'COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1'),
    ('mol4', 'C22H18N2O2', '342.40', 'COc1ccc(-c2cc(-c3ccncc3)nc3ccc(OC)cc23)cc1'),
    ('mol5', 'C23H19NO4', '373.41', 'COc1ccc2c(c1)C(-c1ccc(O)cc1)C=C(-c1cccc(C(=O)O)c1)N2'),
]


def top_hits(csv_name, n=5):
    import csv
    path = REPORTS / csv_name
    if not path.is_file():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: -float(r['best_ecfp4']))
    return rows[:n]


def mol_with_label(smi, label):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, label
    m = Chem.AddHs(m)
    AllChem.Compute2DCoords(m)
    return m, label


def draw_grid(cells, path, legends_per_row, mols_per_image=1):
    """cells: list of (mol, label) laid out in rows."""
    opts = Draw.MolDrawOptions()
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 1.4
    n = len(cells)
    drawn = [Chem.RemoveHs(c[0]) if c[0] is not None else None for c in cells]
    legends = [c[1] for c in cells]
    img = Draw.MolsToGridImage(
        drawn, molsPerRow=legends_per_row, subImgSize=(330, 300),
        legends=legends, drawOptions=opts, useSVG=False)
    img.save(str(path))


def main() -> int:
    # ---- rows 1-3: targets, squash hits, clivia hits ----
    cells = []
    for name, formula, mw, smi in TARGETS:
        m, _ = mol_with_label(smi, '')
        cells.append((m, f'{name}  {formula}  MW {mw}\nTARGET (fluorescent-active)'))

    for tag, csv_name in [('Squash', 'comparison_squash_run2k.csv'),
                          ('Clivia', 'comparison_clivia_run2k.csv')]:
        for r in top_hits(csv_name, 5):
            m, _ = mol_with_label(r['smiles'], '')
            cells.append((m, f'{tag} hit  {r["mw"]}  {r["heavy"]} heavy\n'
                               f'ECFP4 {r["best_ecfp4"]} to {r["best_target"]}'))

    draw_grid(cells, OUT / 'targets_vs_hits.png', legends_per_row=5)
    print('wrote', (OUT / 'targets_vs_hits.png').relative_to(HERE))

    # ---- the most informative pair: regiochemistry ----
    # mol2 (2,4-diaryl) vs the best Clivia hit (2,6-diaryl)
    cliv = top_hits('comparison_clivia_run2k.csv', 1)
    pair = [(Chem.MolFromSmiles(TARGETS[1][3]),
             'mol2  TARGET\n2-aryl + 4-aryl quinoline'),
            (Chem.MolFromSmiles(cliv[0]['smiles']) if cliv else None,
             f'Clivia hit  ECFP4 {cliv[0]["best_ecfp4"]} to mol2\n2-aryl + 6-aryl quinoline'
             if cliv else 'n/a')]
    imgs = []
    for m, leg in pair:
        if m is None:
            continue
        m = Chem.AddHs(m)
        AllChem.Compute2DCoords(m)
        m = Chem.RemoveHs(m)
        # highlight the quinoline core so the attachment positions are obvious
        core = Chem.MolFromSmarts('c1ccc2ncccc2c1')
        match = m.GetSubstructMatch(core)
        d = rdMolDraw2D.MolDraw2DCairo(420, 360)
        opts = d.drawOptions()
        opts.bondLineWidth = 1.6
        opts.legendFontSize = 15
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, legend=leg,
            highlightAtoms=list(match) if match else [])
        d.FinishDrawing()
        p = OUT / f'pair_{len(imgs)}.png'
        p.write_bytes(d.GetDrawingText())
        imgs.append(p)

    if len(imgs) == 2:
        from PIL import Image
        a, b = (Image.open(p) for p in imgs)
        canvas = Image.new('RGB', (a.width + b.width, max(a.height, b.height)), 'white')
        canvas.paste(a, (0, 0)); canvas.paste(b, (a.width, 0))
        canvas.save(OUT / 'closest_pair.png')
        print('wrote', (OUT / 'closest_pair.png').relative_to(HERE))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
