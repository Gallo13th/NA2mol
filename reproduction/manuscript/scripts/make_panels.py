#!/usr/bin/env python3
"""Emit every Clivia figure panel as its own file, all sharing one visual style."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, leaves_list
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from panel_style import (apply_style, clean, C_DATA, C_KEEP, C_HILITE, C_DYE, C_AL1, C_AL2,
                         C_GREY, C_TEXT, SZ_LABEL, SZ_TICK, SZ_LEGEND, SZ_NOTE,
                         LW_AXIS, LW_THRESH, LW_REF, LW_FIT, MS_SMALL, MS_BIG)
RDLogger.DisableLog('rdApp.*')
apply_style()
OUT = 'docs/clivia_8hze/figures/panels'
# final print sizes in mm (width, height) — panels are drawn at this exact size so that
# the 8.5/7 pt type is the real printed size
SIZES = {
    'panel_pocket_rmsd': (52, 44),
    'panel_conjugation_histogram': (48, 52),
    'panel_scaffold_ranking': (61, 52),
    'panel_gate_qed_sa': (54, 60),
    'panel_substituent_C4': (40, 42),
    'panel_substituent_C2': (36, 40),
    'panel_substituent_C7': (36, 42),
    'panel_rdock_histogram': (62, 48),
    'panel_competition': (55, 48),
}


def mm(name, default=(4.0, 3.5)):
    w, h = SIZES.get(name, default)
    return (w / 25.4, h / 25.4)

os.makedirs(OUT, exist_ok=True)
DYE_COL = {'NI4': C_KEEP, 'O2I': C_AL1, 'NJL': C_AL2, 'O00': C_DYE}


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf')
    fig.savefig(f'{OUT}/{name}.png', dpi=300)
    plt.close(fig)
    print('wrote', name)


# ================= pocket =================
mat = list(csv.reader(open('docs/clivia_8hze/pocket_rmsd_matrix.csv')))
names = [n.replace('_copy', '') for n in mat[0][1:]]
M = np.array([[float(x) for x in row[1:]] for row in mat[1:]])
clusters = {r['pocket']: r for r in csv.DictReader(open('docs/clivia_8hze/pocket_clusters.csv'))}
ref = list(csv.DictReader(open('docs/clivia_8hze/pocket_vs_reference.csv')))
dyes = [clusters[n]['dye'] if n in clusters else '?' for n in names]
short = [n.split('_')[0].replace('8HZ', '') + n.split('_')[1] for n in names]
Z = linkage(M[np.triu_indices_from(M, 1)], method='average')
order = leaves_list(Z)
Mo = M[np.ix_(order, order)]
short_o = [short[i] for i in order]
dyes_o = [dyes[i] for i in order]

fig, ax = plt.subplots(figsize=mm('panel_pocket_rmsd'))
from matplotlib.colors import LinearSegmentedColormap
_teal = LinearSegmentedColormap.from_list('tealramp', ['#0a3d3a', '#14615c', '#9fb8d6', '#ffffff'])
im = ax.imshow(Mo, cmap=_teal, vmin=0, vmax=2.2)
ax.set_xticks(range(len(short_o))); ax.set_yticks(range(len(short_o)))
ax.set_xticklabels(short_o, rotation=90, fontsize=4.6)
ax.set_yticklabels(short_o, fontsize=4.6)
for lbl, d in zip(ax.get_xticklabels(), dyes_o):
    lbl.set_color(DYE_COL.get(d, C_TEXT))
for lbl, d in zip(ax.get_yticklabels(), dyes_o):
    lbl.set_color(DYE_COL.get(d, C_TEXT))
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label('pairwise RMSD (Å)')
cb.ax.tick_params(labelsize=SZ_TICK, length=2.5)
cb.outline.set_visible(False)
save(fig, 'panel_pocket_rmsd')

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ref_sorted = sorted(ref, key=lambda r: float(r['rmsd_to_ref_A']))
xs = np.arange(len(ref_sorted))
vals = [float(r['rmsd_to_ref_A']) for r in ref_sorted]
ax.bar(xs, vals, color=[DYE_COL.get(r['dye'], C_GREY) for r in ref_sorted], width=0.7, lw=0)
ax.set_xticks(xs)
ax.set_xticklabels([r['pocket'].replace('8HZ', '') for r in ref_sorted], rotation=90)
ax.set_ylabel('RMSD to 8HZE_A109 (Å)')
clean(ax)
save(fig, 'panel_pocket_deviation')

# ================= scaffold selection =================
rows = []
for r in csv.DictReader(open('docs/clivia_8hze/cold10k_score.csv')):
    m = Chem.MolFromSmiles(r['smiles'])
    if m is None or int(r['radicals']) or abs(int(r['charge'])) > 1:
        continue
    try:
        r['pi'] = int(r['pi_atoms']); r['n_arom'] = int(r['n_arom'])
        r['mw'] = float(r['mw']); r['sa'] = float(r['sa_score'])
        r['tpsa'] = float(r['tpsa']); r['rot'] = float(r['rot'])
    except ValueError:
        continue
    r['scaffold'] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    rows.append(r)
PI_MIN = 20
QUIN = 'c1ccc(-c2cc(-c3ccccc3)c3ccccc3n2)cc1'
kept = [r for r in rows if r['pi'] >= PI_MIN and r['n_arom'] >= 2 and r['sa'] <= 4.0
        and 250 <= r['mw'] <= 600 and r['tpsa'] <= 120 and r['rot'] <= 9]
cnt = Counter(r['scaffold'] for r in kept if r['scaffold'] not in ('', 'C', 'c1ccccc1', 'C1CCCCC1'))
ranked = cnt.most_common()

fig, ax = plt.subplots(figsize=mm('panel_conjugation_histogram'))
_pi = np.array([r['pi'] for r in rows])
_bins = np.arange(0, 34, 1)
_counts, _edges = np.histogram(_pi, bins=_bins)
_cols = [C_GREY if _edges[i] < PI_MIN else C_DATA for i in range(len(_counts))]
ax.bar(_edges[:-1], _counts, width=1.0, align='edge', color=_cols, lw=0)
ax.axvline(PI_MIN, color=C_HILITE, lw=LW_REF)
ax.text(PI_MIN + 0.6, ax.get_ylim()[1] * 0.94, f'{PI_MIN}', color=C_HILITE, fontsize=SZ_NOTE, va='top')
ax.set_xlabel('largest sp² system (atoms)')
ax.set_ylabel('molecules')
clean(ax)
save(fig, 'panel_conjugation_histogram')

from PIL import Image, ImageChops
from rdkit.Chem import Draw as _Draw
from rdkit.Chem import AllChem as _AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io as _io


def draw_mol(smi, w=1300, h=430, highlight=None, hcol=None, notes=None,
             bond_length=46, lw=3.2):
    m = Chem.MolFromSmiles(smi)
    _AllChem.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.fixedBondLength = bond_length
    o.bondLineWidth = lw
    o.padding = 0.06
    o.clearBackground = True
    o.minFontSize = 22
    o.maxFontSize = 26
    o.additionalAtomLabelPadding = 0.12
    if notes:
        for idx, t in notes.items():
            m.GetAtomWithIdx(idx).SetProp('atomNote', t)
    kw = {}
    if highlight:
        kw['highlightAtoms'] = highlight
        kw['highlightAtomColors'] = hcol
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, **kw)
    d.FinishDrawing()
    im = Image.open(_io.BytesIO(d.GetDrawingText())).convert('RGB')
    bb = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    if bb:
        pad = 6
        im = im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
                      min(im.size[0], bb[2] + pad), min(im.size[1], bb[3] + pad)))
    return np.asarray(im)


# --- merged panel: conjugation filter + full scaffold-frequency distribution ---
FIG_W, FIG_H = 112.0, 50.0
fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

# ---- left: conjugation-size distribution with the 20-atom cut ----
ax = fig.add_subplot(gs[0, 0])
_pi = np.array([r['pi'] for r in rows])
_bins = np.arange(0, 34, 1)
_counts, _edges = np.histogram(_pi, bins=_bins)
_cols = [C_GREY if _edges[i] < PI_MIN else C_DATA for i in range(len(_counts))]
ax.bar(_edges[:-1], _counts, width=1.0, align='edge', color=_cols, lw=0)
ax.axvline(PI_MIN, color=C_HILITE, lw=LW_REF)
ax.annotate(f'{PI_MIN}', xy=(PI_MIN, ax.get_ylim()[1]), xytext=(3, -2),
            textcoords='offset points', ha='left', va='top',
            fontsize=SZ_NOTE, color=C_HILITE, fontweight='bold')
ax.set_xlabel('largest sp² system (atoms)', fontsize=SZ_LABEL - 0.5)
ax.set_ylabel('molecules', fontsize=SZ_LABEL - 0.5)
ax.set_xlim(0, 34)
clean(ax)

# ---- right: how often each retained scaffold is repeated ----
axr = fig.add_subplot(gs[0, 1])
_reps = np.array([n for _, n in ranked])
_maxrep = int(_reps.max())
_hist = np.bincount(_reps, minlength=_maxrep + 1)[1:]
_xs = np.arange(1, len(_hist) + 1)
axr.bar(_xs, _hist, width=0.72, color=C_KEEP, lw=0)
axr.set_yscale('log')
axr.set_xlabel('molecules per scaffold', fontsize=SZ_LABEL - 0.5)
axr.set_ylabel('scaffolds', fontsize=SZ_LABEL - 0.5)
axr.set_xlim(0.3, _maxrep + 0.8)
axr.set_ylim(0.7, _hist.max() * 2.2)
axr.set_xticks(range(1, _maxrep + 1, 2))
clean(axr)

save(fig, 'panel_filter_scaffold')

# ================= substituent frequencies (one combined panel) =================
N_GEN = 4948
SITES = [('C4-aryl', {'H': 3044, 'OH': 530, 'Cl': 485, 'Me': 352, 'OMe': 330, 'F': 63, 'NH2': 30, 'NHMe': 3}, C_KEEP),
         ('C2-aryl', {'OMe': 1914, 'OH': 497, 'H': 318, 'Cl': 124, 'Me': 109, 'OEt': 104, 'F': 9, 'NMe2': 6, 'NH2': 3}, C_AL1),
         ('C7', {'H': 3770, 'OMe': 382, 'OH': 217, 'Me': 141, 'Cl': 114, 'OEt': 33, 'CF3': 12, 'NH2': 9, 'F': 3}, C_AL2)]

# --- bare 2,4-diarylquinoline core with the three substitution sites marked ---
_CORE = 'c1ccc(-c2cc(-c3ccccc3)c3ccccc3n2)cc1'
_m = Chem.MolFromSmiles(_CORE)
_AllChem.Compute2DCoords(_m)
_rings = [set(r) for r in _m.GetRingInfo().AtomRings()]
_N = [a.GetIdx() for a in _m.GetAtoms() if a.GetSymbol() == 'N'][0]
_q = set()
for _r in _rings:
    if _N in _r:
        _q |= _r
for _r in _rings:
    if (_r & _q) and _N not in _r:
        _q |= _r
_pos = {}


def _ring_dist(ring, start):
    dist = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for aa in frontier:
            for nb in _m.GetAtomWithIdx(aa).GetNeighbors():
                bb = nb.GetIdx()
                if bb in ring and bb not in dist:
                    dist[bb] = dist[aa] + 1
                    nxt.append(bb)
        frontier = nxt
    return dist


for _r in _rings:
    if _r & _q:
        continue
    _attach = _q_attach = None
    for _j in _r:
        for _nb in _m.GetAtomWithIdx(_j).GetNeighbors():
            if _nb.GetIdx() in _q:
                _attach, _q_attach = _j, _nb.GetIdx()
    _dist = _ring_dist(_r, _attach)
    _para = next(a for a, dd in _dist.items() if dd == 3)
    _is_C2 = any(nb.GetIdx() == _N for nb in _m.GetAtomWithIdx(_q_attach).GetNeighbors())
    _pos['C2' if _is_C2 else 'C4'] = _para

_fusion = [a.GetIdx() for a in _m.GetAtoms()
           if a.GetIdx() in _q and sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in _q) == 3]
_ben = next(set(r) for r in _rings if (r & _q) and _N not in r)
_anchor = next(a for a in _ben if a in _fusion)
_ddist = _ring_dist(_ben, _anchor)
_pos['C7'] = next(a for a in _ben if a != _anchor and _ddist.get(a, 99) == 3)

_COLS = {'C4': (0.08, 0.38, 0.36), 'C2': (0.72, 0.42, 0.12), 'C7': (0.56, 0.42, 0.69)}
_W, _H = 1200, 780
_d = rdMolDraw2D.MolDraw2DCairo(_W, _H)
_o = _d.drawOptions()
_o.fixedBondLength = 64
_o.bondLineWidth = 3.2
_o.padding = 0.02
_o.minFontSize = 24
_o.maxFontSize = 28
rdMolDraw2D.PrepareAndDrawMolecule(_d, _m)
_d.FinishDrawing()
_core_img = np.asarray(Image.open(_io.BytesIO(_d.GetDrawingText())).convert('RGB'))
_cb = ImageChops.difference(Image.fromarray(_core_img),
                            Image.new('RGB', (_W, _H), (255, 255, 255))).getbbox()
if _cb:
    _core_img = _core_img[max(0, _cb[1] - 8):_cb[3] + 8, max(0, _cb[0] - 8):_cb[2] + 8]
_xy = {k: (_d.GetDrawCoords(i).x - (max(0, _cb[0] - 8) if _cb else 0),
           _d.GetDrawCoords(i).y - (max(0, _cb[1] - 8) if _cb else 0)) for k, i in _pos.items()}
print('substitution sites:', _pos)

fig = plt.figure(figsize=(242 / 25.4, 50 / 25.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.38, 1.0, 1.0, 1.0], wspace=0.46)
_ax_core = fig.add_subplot(gs[0, 0])
_hh, _ww = _core_img.shape[:2]
_ax_core.imshow(_core_img)
_cx = _ww / 2.0
_cy = _hh / 2.0
for _k, (_px, _py) in _xy.items():
    _dx, _dy = _px - _cx, _py - _cy
    _nrm = max((_dx ** 2 + _dy ** 2) ** 0.5, 1e-6)
    _tx, _ty = _px + 88 * _dx / _nrm, _py + 88 * _dy / _nrm
    _ax_core.annotate(_k, xy=(_px, _py), xytext=(_tx, _ty), annotation_clip=False,
                      fontsize=SZ_LABEL - 0.3, fontweight='bold', color=_COLS[_k],
                      ha='center', va='center',
                      arrowprops=dict(arrowstyle='-', lw=0.9, color=_COLS[_k],
                                      shrinkA=0, shrinkB=5))
_ax_core.set_xlim(-45, _ww + 45)
_ax_core.set_ylim(_hh + 55, -55)
_ax_core.set_xticks([]); _ax_core.set_yticks([])
for _sp in _ax_core.spines.values():
    _sp.set_visible(False)
axes = [fig.add_subplot(gs[0, i]) for i in (1, 2, 3)]
for ax, (site, d, colour) in zip(axes, SITES):
    items = sorted(d.items(), key=lambda kv: kv[1])
    names_ = [k for k, _ in items]
    vals_ = [v for _, v in items]
    yy = np.arange(len(names_))
    ax.barh(yy, vals_, color=colour, height=0.64, lw=0)
    ax.set_yticks(yy)
    ax.set_yticklabels(names_, fontsize=SZ_TICK)
    ax.set_xlim(0, max(vals_) * 1.42)
    for yi, v in zip(yy, vals_):
        ax.text(v + max(vals_) * 0.04, yi, f'{v}', va='center', fontsize=SZ_TICK - 1.0)
    ax.set_xlabel('molecules', fontsize=SZ_LABEL - 1.0)
    ax.tick_params(axis='x', labelsize=SZ_TICK - 1.0)
    ax.set_title(site, fontsize=SZ_LABEL - 1.0, pad=3)
    clean(ax)
    ax.tick_params(axis='y', length=0)
save(fig, 'panel_substituents')

# ================= screening =================
rows_e = list(csv.DictReader(open('docs/clivia_8hze/unified_enum.csv')))
for r in rows_e:
    r['qed'] = float(r['qed']); r['sa'] = float(r['sa'])
    r['rdock_min'] = float(r['rdock_min']) if r['rdock_min'] not in ('', None) else None
pruned = list(csv.DictReader(open('docs/clivia_8hze/pruned_observed_ranked.csv')))
for r in pruned:
    r['rank'] = int(r['rank']); r['rdock_min'] = float(r['rdock_min'])
    r['qed'] = float(r['qed']); r['sa'] = float(r['sa'])
SETS = {'C4': {'OH', 'Cl', 'Me', 'OMe', 'F', 'NH2', 'NHMe'},
        'C2': {'OMe', 'OH', 'Cl', 'Me', 'OEt', 'F', 'NMe2', 'NH2'},
        'C7': {'OMe', 'OH', 'Me', 'Cl', 'OEt', 'CF3', 'NH2', 'F'}}
lib = [r for r in rows_e if r['C4'] in SETS['C4'] and r['C2'] in SETS['C2'] and r['Q'] in SETS['C7']]
al1 = next(r for r in pruned if (r['C4'], r['C2'], r['Q']) == ('OMe', 'OH', 'OMe'))
DYES = {'NI4': -38.86, 'NJL': -37.26, 'O2I': -35.83, 'O00': -33.53}

fig, ax = plt.subplots(figsize=mm('panel_gate_qed_sa'))
passg = [r for r in lib if r['qed'] >= 0.44 and r['sa'] <= 2.00]
failg = [r for r in lib if not (r['qed'] >= 0.44 and r['sa'] <= 2.00)]
ax.scatter([r['qed'] for r in failg], [r['sa'] for r in failg], s=7, color=C_GREY, lw=0, label='removed')
ax.scatter([r['qed'] for r in passg], [r['sa'] for r in passg], s=7, color=C_KEEP, lw=0, label='retained')
ax.axvline(0.44, color=C_TEXT, ls='--', lw=LW_THRESH); ax.axhline(2.00, color=C_TEXT, ls='--', lw=LW_THRESH)
a1 = next(x for x in lib if x['key'] == al1['key'])
ax.scatter([a1['qed']], [a1['sa']], s=28, color=C_AL1, marker='D', zorder=6, lw=0, label='Al-1')
ax.set_xlabel('QED'); ax.set_ylabel('SA score')
clean(ax)
ax.legend(loc='upper left', handletextpad=0.3, borderpad=0.2, labelspacing=0.25)
save(fig, 'panel_gate_qed_sa')

fig, ax = plt.subplots(figsize=mm('panel_rdock_histogram'))
rm = np.array([r['rdock_min'] for r in rows_e if r['rdock_min'] is not None])
valid = rm[rm < 0]
ax.hist(valid, bins=40, range=(-48, 0), color=C_DATA, lw=0)
for i, (name, v) in enumerate(DYES.items()):
    ax.axvline(v, color='#1a1a1a', lw=LW_REF, ls='-' if name == 'NI4' else '--', alpha=0.9,
               label='crystal dyes' if i == 0 else None)
for r, c, lab in ((al1, C_AL1, 'Al-1'),):
    ax.axvline(r['rdock_min'], color=c, lw=LW_REF + 0.2, label=lab)
ax.set_xlim(-48, 0)
ax.set_xlabel('rDock score')
ax.set_ylabel('molecules')
clean(ax)
ax.legend(loc='upper right', handletextpad=0.5, borderpad=0.2, labelspacing=0.25)
save(fig, 'panel_rdock_histogram')

img = Draw.MolsToGridImage([Chem.MolFromSmiles(al1['smiles'])],
                           molsPerRow=1, subImgSize=(520, 380), legends=['Al-1'])
img.save(f'{OUT}/panel_structures_al1.png')
print('wrote panel_structures_al1 (png)')

# ================= competition =================
SRC = {'Al-1': '/Users/letiangao/Documents/AI-1-Compatative.xlsx'}
COL = {'Al-1': C_AL1}
def four_pl(x, b, t, ic50, h):
    return b + (t - b) / (1.0 + (x / ic50) ** h)
fig, ax = plt.subplots(figsize=mm('panel_competition'))
xs_fit = np.logspace(np.log10(0.005), np.log10(250), 300)
for name, path in SRC.items():
    df = pd.read_excel(path)
    df.columns = ['conc_nM', 'r1', 'r2', 'r3', 'n1', 'n2', 'n3', 'mean', 'sd']
    df = df.sort_values('conc_nM')
    xu = df['conc_nM'].values / 1000.0
    popt, _ = curve_fit(four_pl, xu, df['mean'].values, p0=[0.05, 0.9, 3.0, 1.5],
                        bounds=([0.0, 0.5, 1e-4, 0.2], [0.5, 1.5, 1e3, 6.0]), maxfev=40000)
    ax.errorbar(xu, df['mean'].values, yerr=df['sd'].values, fmt='o', ms=2.6, color=COL[name],
                ecolor=C_TEXT, elinewidth=0.5, capsize=1.0, zorder=4, lw=0)
    ax.plot(xs_fit, four_pl(xs_fit, *popt), '-', color=COL[name], lw=LW_FIT, zorder=3,
            label=f'{name}: {popt[2]:.1f} µM')
ax.set_xscale('log')
ax.set_xlabel('Concentration (µM)')
ax.set_ylabel('Normalised fluorescence')
clean(ax)
ax.set_ylim(-0.02, 1.08)
ax.legend(loc='lower left', handletextpad=0.5, borderpad=0.2, labelspacing=0.2,
          fontsize=SZ_LEGEND - 0.5)
save(fig, 'panel_competition')

# ================= imaging (raster, cropped to a consistent margin) =================
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
im = Image.open('/Users/letiangao/Documents/Clivia-Competition.tif').convert('RGB')
im.crop((0, 0, 921, 900)).save(f'{OUT}/panel_imaging.png')
print('wrote panel_imaging (png)')
