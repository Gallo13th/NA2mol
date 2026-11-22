#!/usr/bin/env python3
"""Pocket-selection figure: dye-defined pockets, their superposition and the representative."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

mat = list(csv.reader(open('docs/clivia_8hze/pocket_rmsd_matrix.csv')))
names = [n.replace('_copy', '') for n in mat[0][1:]]
M = np.array([[float(x) for x in row[1:]] for row in mat[1:]])
clusters = {r['pocket']: r for r in csv.DictReader(open('docs/clivia_8hze/pocket_clusters.csv'))}
ref = list(csv.DictReader(open('docs/clivia_8hze/pocket_vs_reference.csv')))
DYE_COL = {'NI4': '#14615c', 'O2I': '#b5651d', 'NJL': '#8e6cb0', 'O00': '#c0392b'}
dyes = [clusters[n]['dye'] if n in clusters else '?' for n in names]
short = [n.split('_')[0].replace('8HZ', '') + n.split('_')[1] for n in names]

# hierarchical ordering
Z = linkage(M[np.triu_indices_from(M, 1)], method='average')
order = leaves_list(Z)
Mo = M[np.ix_(order, order)]
names_o = [names[i] for i in order]
dyes_o = [dyes[i] for i in order]
short_o = [short[i] for i in order]

fig = plt.figure(figsize=(11.2, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.30, left=0.075, right=0.975, top=0.88, bottom=0.18)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Mo, cmap='viridis', vmin=0, vmax=2.2)
ax.set_xticks(range(len(names_o)))
ax.set_yticks(range(len(names_o)))
ax.set_xticklabels(short_o, rotation=90, fontsize=5.6)
ax.set_yticklabels(short_o, fontsize=5.6)
for lbl, d in zip(ax.get_xticklabels(), dyes_o):
    lbl.set_color(DYE_COL.get(d, '#333333'))
for lbl, d in zip(ax.get_yticklabels(), dyes_o):
    lbl.set_color(DYE_COL.get(d, '#333333'))
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('pairwise pocket RMSD (Å)', fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)
ax.set_title('18 dye-defined pockets, hierarchically ordered', fontsize=8.5)
handles = [plt.Line2D([], [], marker='s', ls='', color=c, ms=5) for c in DYE_COL.values()]
ax.legend(handles, list(DYE_COL), frameon=False, fontsize=6, loc='lower left',
          bbox_to_anchor=(0.0, -0.30), ncol=4, handletextpad=0.3, columnspacing=1.2)

ax = fig.add_subplot(gs[0, 1])
ref_sorted = sorted(ref, key=lambda r: float(r['rmsd_to_ref_A']))
xs = np.arange(len(ref_sorted))
vals = [float(r['rmsd_to_ref_A']) for r in ref_sorted]
cols = [DYE_COL.get(r['dye'], '#777777') for r in ref_sorted]
ax.bar(xs, vals, color=cols, width=0.72)
ax.set_xticks(xs)
ax.set_xticklabels([r['pocket'].replace('8HZ', '') for r in ref_sorted], rotation=90, fontsize=5.6)
ax.set_ylabel('RMSD to the 8HZE_A109 pocket (Å)', fontsize=7)
ax.tick_params(axis='y', labelsize=6)
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(1.0, color='#666666', ls=':', lw=0.8)
ax.text(len(xs) - 0.5, 1.15, '1 Å', fontsize=6, color='#666666', ha='right')
for i, r in enumerate(ref_sorted):
    v = float(r['rmsd_to_ref_A'])
    if v > 1.0:
        ax.text(i, v + 0.25, f'{v:.1f}', ha='center', fontsize=5.8, color='#333333')
ax.set_title('Deviation from the representative pocket', fontsize=8.5)
fig.savefig('docs/clivia_8hze/figures/fig_pocket_selection.png', dpi=300)
fig.savefig('docs/clivia_8hze/figures/fig_pocket_selection.pdf')
print('wrote fig_pocket_selection.png/.pdf')
print('median pairwise RMSD:', round(float(np.median(M[np.triu_indices_from(M, 1)])), 2))
print('pockets within 1 A of the representative:', sum(1 for r in ref if float(r['rmsd_to_ref_A']) <= 1.0), '/', len(ref))
