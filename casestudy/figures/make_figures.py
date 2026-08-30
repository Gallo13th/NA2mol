#!/usr/bin/env python3
"""Generate 2D topology figures (with key-fragment atom highlights) and
per-molecule SDFs for every molecule on the reproduction routes.

Outputs -> casestudy/figures/
  <name>.png   2D depiction, fragments highlighted
  <name>.sdf   single-molecule SDF (from the staged-route evidence files)
  index.md     legend

Highlight scheme (consistent across figures):
  cyan     fused 5+6 / 6+6 ring system, fragment 1
  blue     second ring system / standalone six-membered ring
  red      sites of the endorsed N->S ring conversion (S atoms in products,
           the ring N atoms they replace in precursors)
  green    bridge atom (N->C conversion site)
  orange   hydroxyl oxygens (sp1 flavones)
"""
import os, sys
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDraw2DCairo
from rdkit.Geometry import Point2D
RDLogger.DisableLog('rdApp.*')

ROOT = '/home/data/users/gaoletian/rna2smol'
os.chdir(ROOT)
OUT = 'casestudy/figures'
os.makedirs(OUT, exist_ok=True)

CYAN, BLUE, RED, GREEN, ORANGE = (0.55, 0.95, 0.95), (0.60, 0.75, 1.0), (1.0, 0.45, 0.45), (0.55, 1.0, 0.55), (1.0, 0.75, 0.35)

def first_mol(path):
    return next(Chem.SDMolSupplier(path, removeHs=True))

def draw(mol, name, highlight_atoms=None, colors=None, legend=''):
    m = Chem.Mol(mol)
    Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(m)
    d = MolDraw2DCairo(640, 440)
    opts = d.drawOptions()
    opts.legendFontSize = 22
    opts.bondLineWidth = 2.5
    d.DrawMolecule(m, legend=legend or name,
                   highlightAtoms=highlight_atoms or [],
                   highlightAtomColors=colors or {})
    d.FinishDrawing()
    with open(f'{OUT}/{name}.png', 'wb') as f:
        f.write(d.GetDrawingText())
    w = Chem.SDWriter(f'{OUT}/{name}.sdf')
    m.SetProp('_Name', name)
    w.write(m)
    w.close()
    print(f'[fig] {name}: {Chem.MolToSmiles(m)}')

def atoms_of_smarts(mol, smarts, which='all'):
    patt = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return []
    if which == 'all':
        return sorted(set(i for m in matches for i in m))
    return sorted(matches[0])

# ── T2 staged de-novo chain ────────────────────────────────────────────────
BT = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'          # target/product
BT_NH = 'c1ccc2sc(Nc3nc4ccccc4s3)nc2c1'        # stage-3 intermediate
BT_RING = 'c1ccc2scnc2c1'                       # benzothiazole 5+6 ring
BI_RING = 'c1ccc2[nH]cnc2c1'                    # benzimidazole ring (precursor)

# Stage 1: cold-start library representative (gen2k1369, phenanthroline-type)
g1 = Chem.MolFromSmiles('Cc1cc2nc3ccc(C(=O)O)c(O)c3nc2c(C)c1C')
draw(g1, 'stage1_gen2k1369', highlight_atoms=[], colors={},
     legend='Stage 1: cold-start hit (gen2k1369)')

# Stage 2 precursor: all-nitrogen isoelectronic dimer
dimer = first_mol('casestudy/kras_g4/t2_denovo/phaseA4_dimer/dimer0_gen2k1271xgen2k1271_p0_F4.sdf')
rings = dimer.GetSubstructMatches(Chem.MolFromSmarts(BI_RING))
colors, hl = {}, []
for k, match in enumerate(rings[:2]):
    c = CYAN if k == 0 else BLUE
    for a in match:
        hl.append(a); colors[a] = c
# ring N atoms that will undergo N->S
for a in dimer.GetAtomWithIdx.__self__.GetAtoms() if False else []:
    pass
for a in dimer.GetAtoms():
    if a.GetSymbol() == 'N' and any(a.GetIdx() in m for m in rings):
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
draw(dimer, 'stage2a_dimer_allN', hl, colors,
     legend='Stage 2a: all-N isoelectronic dimer (red = N->S sites)')

# Stage 2 product: two-benzothiazole anchor seed
seed = first_mol('casestudy/kras_g4/t2_denovo/phaseBC_seeds/seedBC_centroid1_generic.sdf')
rings = seed.GetSubstructMatches(Chem.MolFromSmarts(BT_RING))
colors, hl = {}, []
for k, match in enumerate(rings[:2]):
    c = CYAN if k == 0 else BLUE
    for a in match:
        hl.append(a); colors[a] = c
for a in seed.GetAtoms():
    if a.GetSymbol() == 'S':
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
draw(seed, 'stage2b_seed_twoBT', hl, colors,
     legend='Stage 2b: two benzothiazole (5+6) fragments (red = converted S)')

# Stage 3: NH-bridged intermediate
nh = first_mol('casestudy/kras_g4/t2_denovo/phaseE_bridge/denovoD1_r12_i043_s3054_orig.sdf')
rings = nh.GetSubstructMatches(Chem.MolFromSmarts(BT_RING))
colors, hl = {}, []
for k, match in enumerate(rings[:2]):
    c = CYAN if k == 0 else BLUE
    for a in match:
        hl.append(a); colors[a] = c
# bridge N: the N bonded to both ring systems
ring_atoms = set(i for m in rings for i in m)
for a in nh.GetAtoms():
    if a.GetSymbol() == 'N' and a.GetIdx() not in ring_atoms and \
       sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ring_atoms) == 2:
        colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
draw(nh, 'stage3_nh_bridged', hl, colors,
     legend='Stage 3: NH-bridged product (green = bridge atom)')

# Stage 4: final product == T2
t2 = first_mol('casestudy/kras_g4/t2_denovo/phaseE_bridge/t2_replaced.sdf')
rings = t2.GetSubstructMatches(Chem.MolFromSmarts(BT_RING))
colors, hl = {}, []
for k, match in enumerate(rings[:2]):
    c = CYAN if k == 0 else BLUE
    for a in match:
        hl.append(a); colors[a] = c
ring_atoms = set(i for m in rings for i in m)
for a in t2.GetAtoms():
    if a.GetSymbol() == 'S':
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    if a.GetSymbol() == 'C' and a.GetIdx() not in ring_atoms and \
       sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ring_atoms) == 2:
        colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
draw(t2, 'stage4_t2_product', hl, colors,
     legend='Stage 4: final product == T2 (green = N->C bridge)')

# Target reference
tgt = Chem.MolFromSmiles(BT)
rings = tgt.GetSubstructMatches(Chem.MolFromSmarts(BT_RING))
colors, hl = {}, []
for k, match in enumerate(rings[:2]):
    c = CYAN if k == 0 else BLUE
    for a in match:
        hl.append(a); colors[a] = c
draw(tgt, 'target_T2', hl, colors, legend='Target T2')

# ── sp1 flavones ───────────────────────────────────────────────────────────
CHROMONE_FUSED = 'O=c1ccoc2ccccc12'   # chromone 6+6 fused core (incl. ring O)
for name, smi in [('m1', 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'),
                  ('m2', 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'),
                  ('m3', 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12')]:
    m = Chem.MolFromSmiles(smi)
    core = m.GetSubstructMatch(Chem.MolFromSmarts('c1cccc2c1occc2'))  # fused bicycle
    if not core:
        core = m.GetSubstructMatch(Chem.MolFromSmarts('O=c1ccoc2ccccc12'))
    bring = None
    # standalone six-membered ring not in core
    ri = m.GetRingInfo()
    for r in ri.AtomRings():
        if len(r) == 6 and not (set(r) & set(core)):
            bring = r; break
    colors, hl = {}, []
    for a in core:
        hl.append(a); colors[a] = CYAN
    if bring:
        for a in bring:
            hl.append(a); colors[a] = BLUE
    for a in m.GetAtoms():
        if a.GetSymbol() == 'O' and a.GetDegree() == 1:
            colors[a.GetIdx()] = ORANGE; hl.append(a.GetIdx())
    tag = 'm1 (reproduced 159x)' if name == 'm1' else ('m2 (reproduced 16x)' if name == 'm2' else 'm3 (not reproduced)')
    draw(m, f'sp1_{name}', hl, colors, legend=f'sp1 {tag}')

# ── index ──────────────────────────────────────────────────────────────────
with open(f'{OUT}/index.md', 'w') as f:
    f.write('''# Reproduction-route molecules: 2D figures and SDFs

| figure | molecule | role |
|---|---|---|
| stage1_gen2k1369 | cold-start hit (phenanthroline-type core) | Stage-1 library representative |
| stage2a_dimer_allN | all-N isoelectronic dimer | Stage-2 precursor; red = ring N sites of the endorsed N->S conversion |
| stage2b_seed_twoBT | two benzothiazole fragments (18 atoms) | Stage-2 anchor seed; red = converted S |
| stage3_nh_bridged | NH-bridged bis-benzothiazole | Stage-3 inpainting product; green = bridge atom |
| stage4_t2_product | `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1` | Stage-4 final product (SMILES == target); green = N->C bridge |
| target_T2 | same | reference target |
| sp1_m1 / sp1_m2 / sp1_m3 | flavones | sp1 targets (m1, m2 reproduced exactly) |

Highlights: cyan = first fused ring system (5+6 benzothiazole / 6+6 chromene core);
blue = second ring system (standalone six-membered ring); red = N->S conversion
sites; green = bridge atom (N->C conversion); orange = hydroxyl oxygens.

Each molecule is also written as a single-molecule SDF next to its figure.
''')
print(f'[done] figures -> {OUT}/')
