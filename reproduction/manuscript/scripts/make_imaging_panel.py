#!/usr/bin/env python3
"""Quantification panel for the Clivia cell-imaging experiment (from the Prism table)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_style import apply_style, clean, C_KEEP, C_AL1, C_AL2, C_GREY, C_TEXT, SZ_LABEL, SZ_TICK
from scipy import stats

vals = np.load('/tmp/clivia_imaging.npy')
labels = ['Mock', 'Al-1\n1 µM', 'Al-1\n10 µM', 'Al-2\n1 µM', 'Al-2\n10 µM']
cols = ['#8a8a8a', C_AL1, C_AL1, C_AL2, C_AL2]
fills = ['#e2e6e5', '#e8c8a4', '#d9a86f', '#d6c8e6', '#bca6d6']
mock = vals[:, 0]
N_COND = 5
mean = np.array([vals[:, j].mean() for j in range(N_COND)])
sem = np.array([vals[:, j].std(ddof=1) / np.sqrt(vals.shape[0]) for j in range(N_COND)])
rel = 100 * mean / mock.mean()
rel_sem = 100 * sem / mock.mean()

apply_style()
fig, ax = plt.subplots(figsize=(55/25.4, 48/25.4))
xs = np.arange(N_COND)
ax.bar(xs, rel, width=0.66, color=fills, edgecolor=cols, lw=0.8, zorder=2)
ax.errorbar(xs, rel, yerr=rel_sem, fmt='none', ecolor=C_TEXT, elinewidth=0.8, capsize=2.0, zorder=4)
rng = np.random.RandomState(0)
for j in range(N_COND):
    y = 100 * vals[:, j] / mock.mean()
    ax.scatter(j + rng.uniform(-0.18, 0.18, len(y)), y, s=1.6, color=cols[j], alpha=0.55, lw=0, zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=5.6)
ax.set_ylabel('Fluorescence\n(% of Mock)', fontsize=7.0)

ax.set_ylim(0, 215)
clean(ax)
fig.tight_layout(pad=0.3)
out = 'docs/clivia_8hze/figures'
fig.savefig(f'{out}/imaging_quantification.pdf')
fig.savefig(f'{out}/panels/panel_imaging_quantification.pdf')
fig.savefig(f'{out}/panels/panel_imaging_quantification.png', dpi=300)
fig.savefig(f'{out}/imaging_quantification.png', dpi=400)
print('wrote imaging_quantification.png/.pdf')
for j, lab in enumerate(labels):
    print('{:10s} mean {:6.1f}  rel {:5.1f}%  sem {:.1f}'.format(lab.replace(chr(10), ' '), mean[j], rel[j], rel_sem[j]))
