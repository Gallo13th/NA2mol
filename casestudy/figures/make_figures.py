#!/usr/bin/env python3
"""Curated figure set (SVG) for the reproduction routes.

Keeps only the molecules on the final reproduction story lines:

sp1 flavone chain:   targets m1/m2/m3, 21-atom skeleton, seeds s1-s4,
                     representative exact generated hits (m1, m2)
T2 two-fragment chain: two same-pocket source molecules (gen2k1271/473,
                     benzimidazole core highlighted), the two-fragment
                     anchor seed, the model product (all-N isostere),
                     the S target, and the O element-scan variant.

Highlight scheme: cyan/blue = the two ring systems; red = sites analysed
for element conversion (S-corresponding ring N / converted S); green =
model-generated bridge atom; orange = hydroxyl / seed-retained O.
"""
import os, sys, json
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDraw2DSVG
RDLogger.DisableLog('rdApp.*')

ROOT = '/home/data/users/gaoletian/rna2smol'
os.chdir(ROOT)
OUT = 'casestudy/figures'
os.makedirs(OUT, exist_ok=True)

CYAN = (0.55, 0.95, 0.95)
BLUE = (0.60, 0.75, 1.0)
RED = (1.0, 0.45, 0.45)
GREEN = (0.55, 1.0, 0.55)
ORANGE = (1.0, 0.75, 0.35)

BT = Chem.MolFromSmarts('c1ccc2scnc2c1')
BI = Chem.MolFromSmarts('c1ccc2[nH]cnc2c1')
BO = Chem.MolFromSmarts('c1ccc2ocnc2c1')

def draw(mol, name, legend='', hl=None, colors=None):
    # 2D depiction on a COPY (SVG only); the exported SDF keeps the ORIGINAL
    # coordinates (3D poses / fragment assembly geometry).
    m2d = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(m2d)
    d = MolDraw2DSVG(680, 440)
    d.drawOptions().legendFontSize = 20
    d.drawOptions().bondLineWidth = 2.5
    d.DrawMolecule(m2d, legend=legend or name,
                   highlightAtoms=hl or [], highlightAtomColors=colors or {})
    d.FinishDrawing()
    with open(f'{OUT}/{name}.svg', 'w') as f:
        f.write(d.GetDrawingText())
    print(f'[fig] {name}: {Chem.MolToSmiles(Chem.Mol(mol))}')

def two_rings(m, patt):
    matches = m.GetSubstructMatches(patt)
    hl, colors = [], {}
    for k, mt in enumerate(matches[:2]):
        c = CYAN if k == 0 else BLUE
        for a in mt:
            hl.append(a); colors[a] = c
    return hl, colors, set(i for mt in matches for i in mt)

# ══ sp1 chain ══════════════════════════════════════════════════════════════
def flav_hl(m):
    core = m.GetSubstructMatch(Chem.MolFromSmarts('c1ccc2c(c1)occc2')) or \
           m.GetSubstructMatch(Chem.MolFromSmarts('O=c1ccoc2ccccc12'))
    ri = m.GetRingInfo()
    bring = next((r for r in ri.AtomRings()
                  if len(r) == 6 and not (set(r) & set(core))), None)
    hl, colors = [], {}
    for a in core:
        hl.append(a); colors[a] = CYAN
    if bring:
        for a in bring:
            hl.append(a); colors[a] = BLUE
    for a in m.GetAtoms():
        if a.GetSymbol() == 'O' and a.GetDegree() == 1:
            colors[a.GetIdx()] = ORANGE; hl.append(a.GetIdx())
    return hl, colors

for nm, smi, leg in [
        ('sp1_target_m1', 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12', 'Target m1 (reproduced 159x)'),
        ('sp1_target_m2', 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12', 'Target m2 (reproduced 16x)'),
        ('sp1_target_m3', 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12', 'Target m3 (not reproduced)')]:
    m = Chem.MolFromSmiles(smi)
    hl, colors = flav_hl(m)
    draw(m, nm, leg, hl, colors)

# skeleton
skel = next(Chem.SDMolSupplier('casestudy/sp1/skeleton_flavone_sp1.sdf', removeHs=True))
ri = skel.GetRingInfo()
ring_sets = [set(r) for r in ri.AtomRings()]
fp = next(((s1, s2) for i, s1 in enumerate(ring_sets) for j, s2 in enumerate(ring_sets)
           if j > i and len(s1 & s2) >= 2 and len(s1) == 6 and len(s2) == 6), None)
st = next((s for s in ring_sets if s not in fp and len(s) == 6), None)
hl, colors = [], {}
for a in fp[0] | fp[1]:
    hl.append(a); colors[a] = CYAN
if st:
    for a in st:
        hl.append(a); colors[a] = BLUE
draw(skel, 'sp1_skeleton', '21-atom skeleton (6+6 fused core + six-membered ring)', hl, colors)

# seeds s1-s4
for s in ['s1', 's2', 's3', 's4']:
    m = next(Chem.SDMolSupplier(f'casestudy/sp1/inpaint_seeds/seed_{s}_flavone.sdf', removeHs=True))
    ri = m.GetRingInfo()
    ring_sets = [set(r) for r in ri.AtomRings()]
    fp = next(((s1, s2) for i, s1 in enumerate(ring_sets) for j, s2 in enumerate(ring_sets)
               if j > i and len(s1 & s2) >= 2 and len(s1) == 6 and len(s2) == 6), None)
    st = next((x for x in ring_sets if x not in fp and len(x) == 6), None)
    hl, colors = [], {}
    for a in fp[0] | fp[1]:
        hl.append(a); colors[a] = CYAN
    if st:
        for a in st:
            hl.append(a); colors[a] = BLUE
    for a in m.GetAtoms():
        if a.GetSymbol() == 'O':
            colors[a.GetIdx()] = ORANGE; hl.append(a.GetIdx())
    draw(m, f'sp1_seed_{s}', f'sp1 seed {s}', hl, colors)

# generated exact hits (one representative each)
recs = json.load(open('casestudy/sp1/gen_runA/results.json'))['results']
M1 = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
M2 = 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'
mols_by_name = {}
for m in Chem.SDMolSupplier('casestudy/sp1/gen_runA/valid_mols.sdf', removeHs=True):
    if m is not None and m.HasProp('_Name'):
        mols_by_name[m.GetProp('_Name')] = m
for tag, smi, nm in [('m1', M1, 'sp1_hit_m1'), ('m2', M2, 'sp1_hit_m2')]:
    r = next((x for x in recs if x['smiles'] == smi), None)
    if r is None:
        continue
    m = mols_by_name.get(r['name'])
    if m is None:
        continue
    hl, colors = flav_hl(m)
    draw(m, nm, f'generated {tag} exact hit', hl, colors)

# ══ T2 two-fragment chain ══════════════════════════════════════════════════
# sources: canonical copies live in t2_denovo/fragment_sources/
LIB = {'t2_source_1271': (1271, 'gen2k[1271]'),
       't2_source_473': (473, 'gen2k[473]'),
       't2_source_423': (423, 'gen2k[423]'),
       't2_source_65': (65, 'gen2k[65]')}
for nm, (idx, _src) in LIB.items():
    m = next(Chem.SDMolSupplier(f'casestudy/kras_g4/t2_denovo/fragment_sources/gen2k_{idx}.sdf', removeHs=True))
    core = m.GetSubstructMatches(BI)
    hl, colors = [], {}
    for k, mt in enumerate(core[:2]):
        c = CYAN if k == 0 else BLUE
        for a in mt:
            hl.append(a); colors[a] = c
    draw(m, nm, f"{_src} (BER102 cold start; core highlighted)", hl, colors)

# two-fragment anchor seed
seed = next(Chem.SDMolSupplier('casestudy/kras_g4/t2_denovo/xfrag_seeds/seed_xfrag_1271x473.sdf', removeHs=True))
hl, colors, _ = two_rings(seed, BI)
for a in seed.GetAtoms():
    if a.GetSymbol() == 'N' and a.GetNumExplicitHs() >= 1:
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
draw(seed, 't2_xfrag_seed', 'Two-fragment anchor seed (1271x473; red = probed N sites)', hl, colors)

# model product (all-N isostere)
nn = next(Chem.SDMolSupplier('casestudy/kras_g4/t2_denovo/phaseG_elem_scan/t2_dimer_NN.sdf', removeHs=True))  # keep 3D pose
hl, colors, ra = two_rings(nn, BI)
for a in nn.GetAtoms():
    if a.GetSymbol() == 'N' and a.GetNumExplicitHs() >= 1:
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    if a.GetSymbol() == 'C' and a.GetIdx() not in ra and \
       sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ra) == 2:
        colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
draw(nn, 't2_dimer_NN', 'Model product: all-N isostere (green = model CH2 bridge)', hl, colors)

# target (S version)
t2 = Chem.MolFromSmiles('c1ccc2sc(Cc3nc4ccccc4s3)nc2c1')
hl, colors, ra = two_rings(t2, BT)
for a in t2.GetAtoms():
    if a.GetSymbol() == 'S':
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    if a.GetSymbol() == 'C' and a.GetIdx() not in ra and \
       sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ra) == 2:
        colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
draw(t2, 't2_target_S', 'Target T2 (S version)', hl, colors)

# element-scan variant OO
oo = next(Chem.SDMolSupplier('casestudy/kras_g4/t2_denovo/phaseG_elem_scan/t2_dimer_OO.sdf', removeHs=True))  # keep 3D pose
hl, colors, ra = two_rings(oo, BO)
for a in oo.GetAtoms():
    if a.GetSymbol() == 'O' and a.GetIdx() in ra:
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    if a.GetSymbol() == 'C' and a.GetIdx() not in ra and \
       sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ra) == 2:
        colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
draw(oo, 't2_dimer_OO', 'Element-scan variant OO (benzoxazole)', hl, colors)

# positive reference (berberine, co-crystallized ligand of 7X8M BER102)
berb = next(Chem.SDMolSupplier('casestudy/kras_g4/t2_denovo/phaseG_elem_scan/berberine_ref.sdf', removeHs=True))
# highlight only the benzodioxole fused 5+6 on one side (same 5+6 fused-ring
# motif as the route's fragments, heteroatoms = O)
BDX = Chem.MolFromSmarts('c1ccc2c(c1)OCO2')
mt = berb.GetSubstructMatches(BDX)
hl, colors = [], {}
if mt:
    for a in mt[0]:
        hl.append(a); colors[a] = CYAN
draw(berb, 't2_positive_ref_berberine', 'Positive reference: berberine (co-crystal, BER102)', hl, colors)

with open(f'{OUT}/index.md', 'w') as f:
    f.write('''# Reproduction-route figures (SVG)

## sp1 flavone chain
| figure | role |
|---|---|
| sp1_target_m1 / m2 / m3 | targets (m1 159x, m2 16x reproduced) |
| sp1_skeleton | 21-atom all-C skeleton (6+6 fused core + six-membered ring) |
| sp1_seed_s1..s4 | inpainting seeds (intact rings; orange = retained O) |
| sp1_hit_m1 / sp1_hit_m2 | representative exact generated hits |

## T2 two-fragment chain
| figure | role |
|---|---|
| t2_source_1271 / t2_source_473 | two independent same-pocket (BER102) cold-start molecules; benzimidazole cores highlighted |
| t2_xfrag_seed | two-fragment anchor seed (native conformations, clash-free assembly); red = probed ring-N sites |
| t2_dimer_NN | model product: all-N isostere, green = model-generated CH2 bridge (568/915 poses) |
| t2_target_S | target T2 (S version) |
| t2_dimer_OO | element-scan variant (benzoxazole) |
| t2_positive_ref_berberine | co-crystallized positive reference of the 7X8M BER102 site (RNAsmol 0.595 / rDock -44.995) |

Dual scores (phaseG): NN 0.674 (4/6) / -37.71; SS(=target) 0.692 (4/6) / -36.42;
OO 0.520 (3/6) / -28.13.

Highlights: cyan/blue = the two ring systems; red = element-conversion sites
analysed; green = model-generated bridge; orange = hydroxyl/retained O.
Each molecule also exported as a single-molecule SDF.
''')
print(f'[done] -> {OUT}/')
