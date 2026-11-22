#!/usr/bin/env python3
"""KRAS DNA qPCR bar panel (panel h, Figure 4): log-scaled bars, all concentrations visible."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = {
    0: [0.508152466084094, 1.22546844252913, 1.60584576372675, 0.873572895916695, 1.14472416059868],
    5: [0.1641780694848, 0.0832349549638697, 0.064405751270581, 0.0593338825594074, 0.0459115197899471],
    10: [0.0172570937616305, 0.0254416021126781, 0.0657590530125448, 0.0123016871338938, 0.0181359986622456],
    20: [0.00899496468906234, 0.00905752961621555, 0.00918396801784072, 0.00641204381883201,
         0.00645664311057978, 0.00654677448959336],
    30: [0.00918396801784069, 0.00777648186799399, 0.0071558225802363, 0.00654677448959335,
         0.00554345061015775, 0.00510101479331607],
    40: [0.022770871735421, 0.0242366369036521, 0.0159902170624496, 0.016232173489026,
         0.0172770414581292, 0.0113985964393764],
}
conc = sorted(DATA)
mean = np.array([100 * np.mean(DATA[c]) for c in conc])
sd = np.array([100 * np.std(DATA[c], ddof=1) for c in conc])
TEAL = '#14615c'
fig, ax = plt.subplots(figsize=(2.3, 2.15))
x = np.arange(len(conc))
ax.bar(x, mean, width=0.64, color='#bcd3d1', edgecolor=TEAL, lw=0.8, zorder=2, bottom=0.25)
ax.errorbar(x, mean, yerr=sd, fmt='none', ecolor='#333333', elinewidth=0.9, capsize=2.0, zorder=4)
for i, c in enumerate(conc):
    pts = 100 * np.array(DATA[c])
    jit = np.random.RandomState(c).uniform(-0.12, 0.12, len(pts))
    ax.scatter(i + jit, pts, s=6, facecolor='white', edgecolor=TEAL, lw=0.5, zorder=5)
ax.axhline(100, color='#888888', ls=':', lw=0.8, zorder=1)
ax.text(len(conc) - 0.45, 108, 'control', fontsize=5.8, color='#666666', ha='right')
ax.set_yscale('log')
ax.set_ylim(0.25, 260)
ax.set_xticks(x)
ax.set_xticklabels([str(c) for c in conc], fontsize=6.2)
ax.set_xlabel('Concentration (µM)', fontsize=6.8)
ax.set_ylabel('KRAS DNA level (% of control)', fontsize=6.8)
ax.tick_params(axis='y', labelsize=6)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.25)
out = 'docs/clivia_8hze/figures'
fig.savefig(f'{out}/kras_qpcr_panel_h.pdf')
fig.savefig(f'{out}/kras_qpcr_panel_h.png', dpi=400)
print('panel h (log bars) written')
