#!/usr/bin/env python3
"""Screening figure: drug-likeness gate, non-docked score landscape, the two reference compounds."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.DisableLog('rdApp.*')
TEAL, GREY, ORANGE, PURPLE = '#14615c', '#c9c9c9', '#b5651d', '#8e6cb0'

rows = list(csv.DictReader(open('docs/clivia_8hze/unified_enum.csv')))
for r in rows:
    r['qed'] = float(r['qed']); r['sa'] = float(r['sa'])
    r['rdock_min'] = float(r['rdock_min']) if r['rdock_min'] not in ('', None) else None
pruned = list(csv.DictReader(open('docs/clivia_8hze/pruned_observed_ranked.csv')))
for r in pruned:
    r['rank'] = int(r['rank']); r['rdock_min'] = float(r['rdock_min'])
SETS = {'C4': {'OH', 'Cl', 'Me', 'OMe', 'F', 'NH2', 'NHMe'},
        'C2': {'OMe', 'OH', 'Cl', 'Me', 'OEt', 'F', 'NMe2', 'NH2'},
        'C7': {'OMe', 'OH', 'Me', 'Cl', 'OEt', 'CF3', 'NH2', 'F'}}
lib = [r for r in rows if r['C4'] in SETS['C4'] and r['C2'] in SETS['C2'] and r['Q'] in SETS['C7']]
DYES = {'NI4': -38.86, 'NJL': -37.26, 'O2I': -35.83, 'O00': -33.53}
al1 = next(r for r in pruned if (r['C4'], r['C2'], r['Q']) == ('OMe', 'OH', 'OMe'))
al2 = next(r for r in pruned if (r['C4'], r['C2'], r['Q']) == ('OMe', 'NH2', 'OMe'))

fig = plt.figure(figsize=(10.6, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], width_ratios=[1.0, 1.32],
                      hspace=0.38, wspace=0.22, left=0.075, right=0.975, top=0.94, bottom=0.40)

# ---- a: drug-likeness gate, reference compounds coloured ----
ax = fig.add_subplot(gs[0, 0])
passg = [r for r in lib if r['qed'] >= 0.44 and r['sa'] <= 2.00]
failg = [r for r in lib if not (r['qed'] >= 0.44 and r['sa'] <= 2.00)]
ax.scatter([r['qed'] for r in failg], [r['sa'] for r in failg], s=7, color=GREY, lw=0)
ax.scatter([r['qed'] for r in passg], [r['sa'] for r in passg], s=7, color=TEAL, lw=0,
           label='retained')
ax.axvline(0.44, color='#333333', ls='--', lw=0.8); ax.axhline(2.00, color='#333333', ls='--', lw=0.8)
a1 = next(x for x in lib if x['key'] == al1['key'])
a2 = next(x for x in lib if x['key'] == al2['key'])
ax.scatter([a1['qed']], [a1['sa']], s=70, color=ORANGE, marker='D', zorder=6, label='Al-1')
ax.scatter([a2['qed']], [a2['sa']], s=70, color=PURPLE, marker='s', zorder=6, label='Al-2')
ax.set_xlabel('QED', fontsize=8.5); ax.set_ylabel('SA score', fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.set_title(f'Drug-likeness gate: {len(passg)} of {len(lib)} retained', fontsize=9)
ax.legend(frameon=False, fontsize=7.2, loc='upper left')

# ---- b: non-docked score landscape ----
ax = fig.add_subplot(gs[0, 1])
rm = np.array([r['rdock_min'] for r in rows if r['rdock_min'] is not None])
valid = rm[rm < 0]
ax.hist(valid, bins=40, range=(-48, 0), color='#9fb8d6', edgecolor='none')
for i, (name, v) in enumerate(DYES.items()):
    ax.axvline(v, color='#c0392b', lw=1.1, ls='-' if name == 'NI4' else '--', alpha=0.9,
               label='crystal dyes' if i == 0 else None)
for r, c, lab in ((al1, ORANGE, 'Al-1'), (al2, PURPLE, 'Al-2')):
    ax.axvline(r['rdock_min'], color=c, lw=1.3, label=lab)
ax.set_xlim(-48, 0)
ax.set_xlabel('rDock score after in-place minimisation (lower = better)', fontsize=8.5)
ax.set_ylabel('molecules', fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.set_title('Non-docked scoring', fontsize=9)
ax.legend(frameon=False, fontsize=7.2, loc='upper left')

# ---- c: the two compounds ----
sub = fig.add_axes([0.10, 0.005, 0.80, 0.38]); sub.axis('off')
cells = [(al1['smiles'], 'Al-1'), (al2['smiles'], 'Al-2')]
mols = [Chem.MolFromSmiles(s) for s, _ in cells]
img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(420, 340), legends=[t for _, t in cells])
img.save('/tmp/screen_pair.png')
import numpy as _np
from PIL import Image as _Image
_arr = _np.array(_Image.open('/tmp/screen_pair.png').convert('RGB'))
_nonwhite = _np.where((_arr < 250).any(axis=2))
_r0, _r1 = _nonwhite[0].min(), _nonwhite[0].max()
_c0, _c1 = _nonwhite[1].min(), _nonwhite[1].max()
sub.imshow(_arr[max(0, _r0 - 5):_r1 + 5, max(0, _c0 - 5):_c1 + 5])
fig.savefig('docs/clivia_8hze/figures/fig_screening.png', dpi=300)
fig.savefig('docs/clivia_8hze/figures/fig_screening.pdf')
print('wrote fig_screening.png/.pdf')
