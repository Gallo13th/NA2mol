#!/usr/bin/env python3
"""Complete process-molecule figure set for auditing the reproduction routes.

Every molecule actually used at every stage is drawn (2D, key fragments
highlighted) and exported as a single-molecule SDF.

T2 staged de-novo chain (casestudy/kras_g4/):
  S1  library molecules with benzimidazole cores (gen2k1271/473/348)
  S1  trimmed 5+6 cores (9 atoms each)
  S2a all-N isoelectronic dimers (1271x1271, 1271x348, 1271x473)
  S2b gated poses gp1/gp2/gp3 (pre-replacement dimers)
  S2c two-benzothiazole anchor seeds (endorsed gp*, centroid variants)
  S3  NH-bridged D1 products (3 conformers)
  S4  final N->C product == T2 (+ target reference)

sp1 flavone chain (casestudy/sp1/):
  target m1/m2, 21-atom skeleton, seeds s1-s4, exact m1 hits (poses),
  exact m2 hit, top BM variants.

Highlights: cyan = ring system 1 (fused core), blue = ring system 2,
red = N->S conversion sites, green = bridge atom (N->C), orange = O atoms
(seed-retained / hydroxyls), magenta = free/undetermined atoms in seeds.
"""
import os, sys, json, glob
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import MolDraw2DCairo
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
MAGENTA = (1.0, 0.65, 1.0)

BT_RING = Chem.MolFromSmarts('c1ccc2scnc2c1')
BI_RING = Chem.MolFromSmarts('c1ccc2[nH]cnc2c1')

def first_mol(path):
    m = next(Chem.SDMolSupplier(path, removeHs=True), None)
    assert m is not None, path
    return m

def draw(mol, name, hl=None, colors=None, legend=''):
    m = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(m)
    d = MolDraw2DCairo(680, 460)
    d.drawOptions().legendFontSize = 20
    d.drawOptions().bondLineWidth = 2.5
    d.DrawMolecule(m, legend=legend or name,
                   highlightAtoms=hl or [], highlightAtomColors=colors or {})
    d.FinishDrawing()
    open(f'{OUT}/{name}.png', 'wb').write(d.GetDrawingText())
    m.SetProp('_Name', name)
    w = Chem.SDWriter(f'{OUT}/{name}.sdf'); w.write(m); w.close()
    print(f'[fig] {name}: {Chem.MolToSmiles(m)}')

def two_ring_highlights(mol, patt):
    """highlight the two matched ring systems cyan/blue; return (hl, colors)."""
    matches = mol.GetSubstructMatches(patt)
    hl, colors = [], {}
    for k, match in enumerate(matches[:2]):
        c = CYAN if k == 0 else BLUE
        for a in match:
            hl.append(a); colors[a] = c
    return hl, colors, set(i for m in matches for i in m)

# ═════════════════════════ T2 chain ════════════════════════════════════════
print('── T2 staged de-novo chain ──')

# S1: library molecules (highlight the benzimidazole 5+6 core)
LIB_SMILES = {
    'gen2k1271': 'O=C(O)NCc1ccc2cc(-c3nc4ccc([C@H](O)CO)cc4[nH]3)ccc2c1',
    'gen2k473':  'O=C1CCN(c2ccc3[nH]c(-c4ccc(O)cc4)nc3c2)C1',
    'gen2k348':  'N=C(N)c1ccc2nc(C3=C4N=c5cc(CO)ccc5=C4N=C3)[nH]c2c1',
}
for nm, smi in LIB_SMILES.items():
    m = Chem.MolFromSmiles(smi)
    core = m.GetSubstructMatches(BI_RING)
    hl, colors = [], {}
    for k, mt in enumerate(core[:2]):
        c = CYAN if k == 0 else BLUE
        for a in mt:
            hl.append(a); colors[a] = c
    draw(m, f't2_s1_lib_{nm}', hl, colors, f'S1 cold-start {nm} (core highlighted)')

# S1: trimmed cores
for nm in ['gen2k1271', 'gen2k473', 'gen2k348']:
    m = first_mol(f'casestudy/kras_g4/t2_denovo/phaseA3_contexts/F1_{nm}_bim.sdf')
    mt = m.GetSubstructMatches(BI_RING)[0]
    hl, colors = [], {}
    for a in mt:
        hl.append(a); colors[a] = CYAN
    for a in m.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetIdx() in mt:
            colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    draw(m, f't2_s1_core_{nm}', hl, colors,
         f'S1 trimmed core of {nm} (red = N->S site)')

# S2a: all-N dimers
for nm, f in [('dimer0_1271x1271', 'dimer0_gen2k1271xgen2k1271_p0_F4.sdf'),
              ('dimer1_1271x348',  'dimer1_gen2k1271xgen2k348_p0_F4.sdf'),
              ('dimer2_1271x473',  'dimer2_gen2k1271xgen2k473_p0_F4.sdf')]:
    m = first_mol(f'casestudy/kras_g4/t2_denovo/phaseA4_dimer/{f}')
    hl, colors, ring_atoms = two_ring_highlights(m, BI_RING)
    for a in m.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetIdx() in ring_atoms:
            colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    draw(m, f't2_s2a_{nm}', hl, colors,
         f'S2a all-N dimer {nm} (red = N->S sites)')

# S2b: gated poses (pre-replacement dimers from the A6 scan)
for gp in ['gp1', 'gp2', 'gp3']:
    m = first_mol(f'casestudy/kras_g4/t2_denovo/phaseBC_seeds/seedBC_{gp}_endorsed_pre.sdf')
    hl, colors, ring_atoms = two_ring_highlights(m, BI_RING)
    for a in m.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetIdx() in ring_atoms:
            colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    draw(m, f't2_s2b_pose_{gp}', hl, colors,
         f'S2b gated pose {gp} (pre-replacement, red = N->S sites)')

# S2c: replacement seeds (two benzothiazole fragments)
for tag in ['gp1_endorsed', 'gp2_endorsed', 'gp3_endorsed',
            'centroid0_generic', 'centroid1_generic', 'centroid2_generic',
            'centroid3_generic']:
    m = first_mol(f'casestudy/kras_g4/t2_denovo/phaseBC_seeds/seedBC_{tag}.sdf')
    hl, colors, ring_atoms = two_ring_highlights(m, BT_RING)
    for a in m.GetAtoms():
        if a.GetSymbol() == 'S':
            colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    draw(m, f't2_s2c_seed_{tag}', hl, colors,
         f'S2c two-BT seed {tag} (red = converted S, 2x intact 5+6)')

# S3: NH-bridged products (3 conformers)
for conf in ['r02_i008_s3044', 'r12_i043_s3054', 'r12_i057_s3054']:
    try:
        m = first_mol(f'casestudy/kras_g4/t2_denovo/phaseE_bridge/denovoD1_{conf}_orig.sdf')
    except Exception:
        cands = glob.glob(f'casestudy/kras_g4/t2_denovo/phaseE_bridge/denovoD1_{conf}*.sdf'.replace('_s30', '_s30'))
        m = first_mol([c for c in cands if c.endswith('_orig.sdf')][0])
    hl, colors, ring_atoms = two_ring_highlights(m, BT_RING)
    for a in m.GetAtoms():
        if a.GetSymbol() == 'N' and a.GetIdx() not in ring_atoms and \
           sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ring_atoms) == 2:
            colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
    draw(m, f't2_s3_nh_{conf}', hl, colors,
         f'S3 NH-bridged {conf} (green = bridge N)')

# S4: final product + target
t2 = first_mol('casestudy/kras_g4/t2_denovo/phaseE_bridge/t2_replaced.sdf')
hl, colors, ring_atoms = two_ring_highlights(t2, BT_RING)
for a in t2.GetAtoms():
    if a.GetSymbol() == 'S':
        colors[a.GetIdx()] = RED; hl.append(a.GetIdx())
    if a.GetSymbol() == 'C' and a.GetIdx() not in ring_atoms and \
       sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ring_atoms) == 2:
        colors[a.GetIdx()] = GREEN; hl.append(a.GetIdx())
draw(t2, 't2_s4_product', hl, colors,
     'S4 final product == T2 (green = N->C bridge)')
tgt = Chem.MolFromSmiles('c1ccc2sc(Cc3nc4ccccc4s3)nc2c1')
hl, colors, _ = two_ring_highlights(tgt, BT_RING)
draw(tgt, 't2_target', hl, colors, 'Target T2')

# ═════════════════════════ sp1 chain ══════════════════════════════════════
print('── sp1 flavone chain ──')

def flav_highlights(m):
    core = m.GetSubstructMatch(Chem.MolFromSmarts('c1ccc2c(c1)occc2'))
    if not core:
        core = m.GetSubstructMatch(Chem.MolFromSmarts('O=c1ccoc2ccccc12'))
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
    hl, colors = flav_highlights(m)
    draw(m, nm, hl, colors, leg)

# skeleton (all-C smear, 21 atoms): highlight the three rings
skel = first_mol('casestudy/sp1/skeleton_flavone_sp1.sdf')
ri = skel.GetRingInfo()
hl, colors = [], {}
fused = [set(r) for r in ri.AtomRings()]
# identify the 6+6 fused pair and the standalone ring
ring_sets = [set(r) for r in ri.AtomRings()]
standalone = None
for i, s1 in enumerate(ring_sets):
    for j, s2 in enumerate(ring_sets):
        if j <= i: continue
        if len(s1 & s2) >= 2 and len(s1) == 6 and len(s2) == 6:
            fused_pair = (s1, s2)
standalone = next((s for k, s in enumerate(ring_sets)
                   if s not in fused_pair and len(s) == 6), None)
for a in fused_pair[0] | fused_pair[1]:
    hl.append(a); colors[a] = CYAN
if standalone:
    for a in standalone:
        hl.append(a); colors[a] = BLUE
draw(skel, 'sp1_s0_skeleton', hl, colors,
     'S0 21-atom all-C skeleton (6+6 fused core + six-membered ring)')

# seeds s1-s4 (O-retained atoms orange)
for s in ['s1', 's2', 's3', 's4']:
    m = first_mol(f'casestudy/sp1/inpaint_seeds/seed_{s}_flavone.sdf')
    ri = m.GetRingInfo()
    ring_sets = [set(r) for r in ri.AtomRings()]
    fp = None
    for i, s1 in enumerate(ring_sets):
        for j, s2 in enumerate(ring_sets):
            if j <= i: continue
            if len(s1 & s2) >= 2 and len(s1) == 6 and len(s2) == 6:
                fp = (s1, s2)
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
    draw(m, f'sp1_s1_seed_{s}', hl, colors,
         f'sp1 seed {s} (intact rings; orange = retained O)')

# exact hits from gen_runA (m1 x2 poses, m2 x1) + top BM variants
recs = json.load(open('casestudy/sp1/gen_runA/results.json'))['results']
M1 = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
M2 = 'O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12'
mols_by_name = {}
for m in Chem.SDMolSupplier('casestudy/sp1/gen_runA/valid_mols.sdf', removeHs=True):
    if m is not None and m.HasProp('_Name'):
        mols_by_name[m.GetProp('_Name')] = m
m1_hits = [r for r in recs if r['smiles'] == M1][:2]
m2_hits = [r for r in recs if r['smiles'] == M2][:1]
for i, r in enumerate(m1_hits, 1):
    m = mols_by_name.get(r['name'])
    if m is None: continue
    hl, colors = flav_highlights(m)
    draw(m, f'sp1_p_m1_hit{i}', hl, colors, f'generated m1 exact hit ({r["name"]})')
for r in m2_hits:
    m = mols_by_name.get(r['name'])
    if m is None: continue
    hl, colors = flav_highlights(m)
    draw(m, 'sp1_p_m2_hit1', hl, colors, f'generated m2 exact hit ({r["name"]})')
# top BM variants (diversity of the same scaffold)
seen = set()
variants = []
for r in sorted(recs, key=lambda x: -max(x['tc'].values())):
    if r['smiles'] in (M1, M2) or r['smiles'] in seen or not r['bm_match']:
        continue
    seen.add(r['smiles']); variants.append(r)
    if len(variants) >= 3: break
for i, r in enumerate(variants, 1):
    m = Chem.MolFromSmiles(r['smiles'])
    hl, colors = flav_highlights(m)
    draw(m, f'sp1_p_variant{i}', hl, colors,
         f'generated scaffold variant (Tc {max(r["tc"].values()):.2f})')

# index
with open(f'{OUT}/index.md', 'w') as f:
    f.write('''# Process-molecule audit figures

Every molecule used at every stage of the reproduction routes, drawn as 2D
topology with key-fragment highlights; each also exported as a single-molecule
SDF next to its figure.

## T2 staged de-novo chain (casestudy/kras_g4/t2_denovo/)

| prefix | stage | molecules |
|---|---|---|
| t2_s1_lib_* | S1 cold-start library members (full structures) | gen2k1271 / gen2k473 / gen2k348 |
| t2_s1_core_* | S1 trimmed 5+6 cores (9 atoms) | one per library member; red = N->S site |
| t2_s2a_dimer* | S2a all-N isoelectronic dimers | 1271x1271, 1271x348, 1271x473 |
| t2_s2b_pose_gp* | S2b gated pocket poses | gp1/gp2/gp3 (pre-replacement) |
| t2_s2c_seed_* | S2c two-benzothiazole anchor seeds | gp1-3 endorsed + centroid0-3 |
| t2_s3_nh_* | S3 inpainting products | NH-bridged, 3 conformers (green = bridge N) |
| t2_s4_product | S4 final N->C product | canonical SMILES == target T2 |
| t2_target | reference | |

## sp1 flavone chain (casestudy/sp1/)

| prefix | stage | molecules |
|---|---|---|
| sp1_target_m1/m2/m3 | targets | m1 159x, m2 16x, m3 not reproduced |
| sp1_s0_skeleton | S0 21-atom all-C skeleton | 6+6 fused core + six-membered ring |
| sp1_s1_seed_s1..s4 | inpainting seeds | intact rings; orange = retained O |
| sp1_p_m1_hit* / sp1_p_m2_hit1 | exact generated hits | poses from gen_runA |
| sp1_p_variant1..3 | scaffold variants | same BM scaffold, model-chosen substituents |

## Highlight scheme

cyan = ring system 1 (fused 5+6 / 6+6 core); blue = ring system 2;
red = N->S conversion sites; green = bridge atom (N->C); orange = oxygen atoms
(hydroxyls / seed-retained); magenta = reserved.
''')
print(f'[done] -> {OUT}/')
