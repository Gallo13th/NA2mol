#!/usr/bin/env python3
"""Rebuild Figure 4 panels d (candidate grid), g (ThT dose-response) and h (qPCR bars)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.DisableLog('rdApp.*')
OUT = 'docs/clivia_8hze/figures'
os.makedirs(OUT, exist_ok=True)

# ---------------- panel d: 5 columns x 7 rows ----------------
df = pd.read_csv('figures/figure4_candidate_labels.csv')
mols, legends = [], []
for _, r in df.sort_values('candidate').iterrows():
    m = Chem.MolFromSmiles(r['smiles'])
    if m is None:
        continue
    mols.append(m)
    star = '*' if bool(r['has_vendors']) else ''
    legends.append('{:02d} · {}{}'.format(int(r['candidate']), int(r['cid']), star))
    legends[-1] += '\n{}/5 votes'.format(int(r['model_votes']))
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 190), legends=legends)
img.save(f'{OUT}/panel_d_grid.png')
w, h = img.size
print('panel d grid', w, 'x', h, 'px')

# ---------------- panel g: ThT dose-response ----------------
assay = pd.read_csv('figures/figure4_assay_plotted_values.csv')
x = assay['concentration_um'].values
y = assay['normalised_fluorescence'].values
sd = assay['sd'].values


def four_pl(x, bottom, top, dc50, hill):
    return bottom + (top - bottom) / (1.0 + (x / dc50) ** hill)


from scipy.optimize import curve_fit
p0 = [0.0, 0.97, 22.4, 1.5]
bounds = ([-0.05, 0.80, 1.0, 0.2], [0.20, 1.10, 200.0, 10.0])
popt, _ = curve_fit(lambda xx, b, t, d, hh: four_pl(10 ** xx, b, t, d, hh),
                    np.log10(x), y, p0=p0, bounds=bounds, maxfev=40000)
TEAL = '#14615c'
fig, ax = plt.subplots(figsize=(2.3, 2.15))
ax.errorbar(x, y, yerr=sd, fmt='o', ms=3, color=TEAL, ecolor='#555555', elinewidth=0.7,
            capsize=1.5, zorder=4)
xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
ax.plot(xs, four_pl(xs, *popt), '-', color=TEAL, lw=1.2, zorder=3)
ax.set_xscale('log')
ax.set_xlabel('Concentration (µM)', fontsize=6.8)
ax.set_ylabel('Normalised ThT fluorescence', fontsize=6.8)
ax.tick_params(labelsize=6)
ax.set_ylim(-0.03, 1.08)
ax.spines[['top', 'right']].set_visible(False)
ax.text(0.97, 0.30, 'DC50 = %.1f uM' % popt[2], transform=ax.transAxes,
        ha='right', fontsize=6.2, color=TEAL)
fig.tight_layout(pad=0.25)
fig.savefig(f'{OUT}/panel_g_tht.pdf')
fig.savefig(f'{OUT}/panel_g_tht.png', dpi=400)
print('panel g: bottom=%.3f top=%.3f DC50=%.2f uM hill=%.2f' % tuple(popt))
