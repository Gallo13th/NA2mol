#!/usr/bin/env python3
"""Competition dose-response curves for Al-1 and Al-2: 4PL fits and figure."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SRC = {
    'Al-1': '/Users/letiangao/Documents/AI-1-Compatative.xlsx',
    'Al-2': '/Users/letiangao/Documents/AI-2-Compatative.xlsx',
}
COLOURS = {'Al-1': '#14615c', 'Al-2': '#b5651d'}


def four_pl(x, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1.0 + (x / ic50) ** hill)


fits = {}
for name, path in SRC.items():
    df = pd.read_excel(path)
    df.columns = ['conc_nM', 'r1', 'r2', 'r3', 'n1', 'n2', 'n3', 'mean', 'sd']
    df = df.sort_values('conc_nM')
    x_uM = df['conc_nM'].values / 1000.0
    y = df['mean'].values
    sd = df['sd'].values
    p0 = [y.max() * 0.05, y.max(), np.median(x_uM), 1.0]
    bounds = ([0.0, 0.5, 1e-4, 0.2], [0.5, 1.5, 1e3, 6.0])
    popt, _ = curve_fit(four_pl, x_uM, y, p0=p0, bounds=bounds, maxfev=40000)
    resid = y - four_pl(x_uM, *popt)
    fits[name] = dict(x=x_uM, y=y, sd=sd, popt=popt, rms=float(np.sqrt((resid ** 2).mean())))
    print(f'{name}: n={len(x_uM)}  bottom={popt[0]:.3f}  top={popt[1]:.3f}  '
          f'IC50={popt[2]:.2f} uM  hill={popt[3]:.2f}  RMS={fits[name]["rms"]:.4f}')

# duplicates check: fold difference
print('\nIC50 ratio Al-2/Al-1 = {:.2f}'.format(fits['Al-2']['popt'][2] / fits['Al-1']['popt'][2]))

fig, ax = plt.subplots(figsize=(3.1, 2.6))
xs = np.logspace(np.log10(0.005), np.log10(250), 400)
for name, d in fits.items():
    c = COLOURS[name]
    ax.errorbar(d['x'], d['y'], yerr=d['sd'], fmt='o', ms=3.2, color=c,
                ecolor='#888888', elinewidth=0.6, capsize=1.2, zorder=4)
    ax.plot(xs, four_pl(xs, *d['popt']), '-', color=c, lw=1.2, zorder=3,
            label=f"{name}  IC$_{{50}}$ = {d['popt'][2]:.1f} µM")
ax.set_xscale('log')
ax.set_xlabel('Concentration (µM)', fontsize=7.5)
ax.set_ylabel('Normalised fluorescence', fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=6.2, loc='lower left')
ax.set_ylim(-0.02, 1.08)
fig.tight_layout(pad=0.3)
out = 'docs/clivia_8hze/figures'
fig.savefig(f'{out}/competition_al1_al2.pdf')
fig.savefig(f'{out}/competition_al1_al2.png', dpi=400)
print('wrote competition_al1_al2.png/.pdf')

# also per-compound panels (if the journal wants them separate)
for name, d in fits.items():
    fig, ax = plt.subplots(figsize=(2.2, 2.1))
    c = COLOURS[name]
    ax.errorbar(d['x'], d['y'], yerr=d['sd'], fmt='o', ms=3, color=c, ecolor='#888888',
                elinewidth=0.6, capsize=1.2, zorder=4)
    ax.plot(xs, four_pl(xs, *d['popt']), '-', color=c, lw=1.1, zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Concentration (µM)', fontsize=7)
    ax.set_ylabel('Normalised fluorescence', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-0.02, 1.08)
    ax.text(0.05, 0.08, f'IC50 = {d["popt"][2]:.1f} µM', transform=ax.transAxes, fontsize=6.5, color=c)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{out}/competition_{name.replace("-", "").lower()}.png', dpi=400)
    fig.savefig(f'{out}/competition_{name.replace("-", "").lower()}.pdf')
print('wrote per-compound panels')
