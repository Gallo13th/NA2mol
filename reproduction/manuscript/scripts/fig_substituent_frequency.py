#!/usr/bin/env python3
"""Substituent-frequency figure: what the inpainting model places at each of the three positions."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_GEN = 4948
SITES = [
    ('C4-aryl para', {'H': 3044, 'OH': 530, 'Cl': 485, 'Me': 352, 'OMe': 330, 'F': 63, 'NH2': 30, 'NHMe': 3}, '#2b6cb0'),
    ('C2-aryl para', {'OMe': 1914, 'OH': 497, 'H': 318, 'Cl': 124, 'Me': 109, 'OEt': 104, 'F': 9, 'NMe2': 6, 'NH2': 3}, '#c05621'),
    ('quinoline C7', {'H': 3770, 'OMe': 382, 'OH': 217, 'Me': 141, 'Cl': 114, 'OEt': 33, 'CF3': 12, 'NH2': 9, 'F': 3}, '#2f855a'),
]

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharex=False)
for ax, (site, d, colour) in zip(axes, SITES):
    items = sorted(d.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colour, height=0.66, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel('molecules', fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.28)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, f'{v}', va='center', fontsize=7.5, color='#333333')
    ax.set_title(f'{site}\n({len(d)} substituents observed)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, 0.005, f'counted over {N_GEN:,} products that retained the intact three-ring system',
         ha='center', fontsize=8, color='#555555')
fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=2.0)
fig.savefig('docs/clivia_8hze/figures/fig_substituent_frequency.png', dpi=300)
fig.savefig('docs/clivia_8hze/figures/fig_substituent_frequency.pdf')
print('wrote fig_substituent_frequency.png/.pdf')
