#!/usr/bin/env python3
"""Clivia chapter figure: workflow, drug-likeness gate, substituent vocabulary, shortlist, competition."""
import csv, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from scipy.optimize import curve_fit
RDLogger.DisableLog('rdApp.*')
OUT = 'docs/clivia_8hze/figures'
TEAL, GREY, ORANGE = '#14615c', '#c9c9c9', '#b5651d'

rows = list(csv.DictReader(open('docs/clivia_8hze/unified_enum.csv')))
for r in rows:
    r['qed'] = float(r['qed']); r['sa'] = float(r['sa'])
pruned = list(csv.DictReader(open('docs/clivia_8hze/pruned_observed_ranked.csv')))
for r in pruned:
    r['rank'] = int(r['rank']); r['rdock_min'] = float(r['rdock_min'])
    r['qed'] = float(r['qed']); r['sa'] = float(r['sa'])
FREQ = {'C4-aryl': {'OMe': 330, 'OH': 530, 'OEt': 0, 'NMe2': 0, 'NHMe': 3, 'NH2': 30, 'CN': 0,
                    'F': 63, 'Cl': 485, 'Me': 352, 'CF3': 0, 'COOH': 0, 'CONH2': 0},
        'C2-aryl': {'OMe': 1914, 'OH': 497, 'OEt': 104, 'NMe2': 6, 'NHMe': 0, 'NH2': 3, 'CN': 0,
                    'F': 9, 'Cl': 124, 'Me': 109, 'CF3': 0, 'COOH': 0, 'CONH2': 0},
        'quinoline C7': {'OMe': 382, 'OH': 217, 'OEt': 33, 'NMe2': 0, 'NHMe': 0, 'NH2': 9, 'CN': 0,
                         'F': 3, 'Cl': 114, 'Me': 141, 'CF3': 12, 'COOH': 0, 'CONH2': 0}}
N_GEN = 4948
SETS = {'C4': {'OH', 'Cl', 'Me', 'OMe', 'F', 'NH2', 'NHMe'},
        'C2': {'OMe', 'OH', 'Cl', 'Me', 'OEt', 'F', 'NMe2', 'NH2'},
        'C7': {'OMe', 'OH', 'Me', 'Cl', 'OEt', 'CF3', 'NH2', 'F'}}
lib = [r for r in rows if r['C4'] in SETS['C4'] and r['C2'] in SETS['C2'] and r['Q'] in SETS['C7']]

fig = plt.figure(figsize=(13.2, 10.0))
gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.05, 1.45], hspace=0.42, wspace=0.30,
                      left=0.055, right=0.975, top=0.96, bottom=0.04)

# ---- a ----
ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis('off')
boxes = [(0.15, 'RNA pocket', 'Clivia aptamer\n8HZE, 23 nt\n24 dye-site pockets'),
         (3.1, 'Generative sampling', '10,000 molecules\nBFN inpainting\n4,948 keep the ring system'),
         (6.05, 'Scaffold selection', '1,085 scaffolds\n2,4-diarylquinoline\nrank 17 / 1,085 (top 1.6%)'),
         (9.0, 'Substituent vocabulary', '3 sites counted in\n4,948 products\nobserved >= 1x -> 448 combinations')]
for x, title, body in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.75), 2.75, 2.05, boxstyle='round,pad=0.06',
                                fc='#eef4f3', ec=TEAL, lw=1.0))
    ax.text(x + 1.37, 2.55, title, ha='center', va='center', fontsize=9.2, fontweight='bold', color=TEAL)
    ax.text(x + 1.37, 1.6, body, ha='center', va='center', fontsize=8.0, color='#222222')
for x in (2.92, 5.9, 8.85):
    ax.add_patch(FancyArrowPatch((x, 1.75), (x + 0.2, 1.75), arrowstyle='-|>', mutation_scale=13, color='#555555'))
ax.text(0.0, 3.05, 'a', fontsize=13, fontweight='bold')

# ---- b ----
ax = fig.add_subplot(gs[1, 0])
passg = [r for r in lib if r['qed'] >= 0.44 and r['sa'] <= 2.00]
failg = [r for r in lib if not (r['qed'] >= 0.44 and r['sa'] <= 2.00)]
ax.scatter([r['qed'] for r in failg], [r['sa'] for r in failg], s=6, color=GREY, lw=0)
ax.scatter([r['qed'] for r in passg], [r['sa'] for r in passg], s=6, color=TEAL, lw=0)
ax.axvline(0.44, color='#333333', ls='--', lw=0.8); ax.axhline(2.00, color='#333333', ls='--', lw=0.8)
ax.set_xlabel('QED', fontsize=8); ax.set_ylabel('SA score', fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title(f'Drug-likeness gate ({len(passg)} / {len(lib)} retained)', fontsize=8.5)
ax.text(-0.03, 1.07, 'b', transform=ax.transAxes, fontsize=13, fontweight='bold', ha='right')

# ---- c ----
ax = fig.add_subplot(gs[1, 1:])
ALL = ['OMe', 'OH', 'OEt', 'NMe2', 'NHMe', 'NH2', 'CN', 'F', 'Cl', 'Me', 'CF3', 'COOH', 'CONH2']
xs = np.arange(len(ALL)); w = 0.26
for i, (site, d) in enumerate(FREQ.items()):
    ax.bar(xs + (i - 1) * w, [100.0 * d.get(s, 0) / N_GEN for s in ALL], width=w,
           label=site, color=[TEAL, ORANGE, '#7f9c96'][i])
ax.axhline(100.0 / N_GEN, color='#c0392b', ls='--', lw=0.9)
ax.text(len(ALL) - 0.6, 0.028, 'one occurrence', fontsize=6.8, color='#c0392b', ha='right')
i_nh2 = ALL.index('NH2')
ax.annotate('Al-2 substituent', xy=(i_nh2 + w, 100 * 3 / N_GEN), xytext=(i_nh2 + 1.1, 0.10),
            fontsize=6.6, color='#c0392b', arrowprops=dict(arrowstyle='->', lw=0.7, color='#c0392b'))
ax.set_yscale('log'); ax.set_ylim(0.01, 300)
ax.set_xticks(xs); ax.set_xticklabels(ALL, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('frequency (% of 4,948 products)', fontsize=8)
ax.set_title('Substituent vocabulary per site', fontsize=8.5)
ax.legend(frameon=False, fontsize=6.5, ncol=3, loc='upper right')
ax.text(-0.02, 1.07, 'c', transform=ax.transAxes, fontsize=13, fontweight='bold', ha='right')

# ---- d (structures) ----
picks = pruned[:5]
al1 = next(r for r in pruned if (r['C4'], r['C2'], r['Q']) == ('OMe', 'OH', 'OMe'))
al2 = next(r for r in pruned if (r['C4'], r['C2'], r['Q']) == ('OMe', 'NH2', 'OMe'))
cells = [(r['smiles'], f"#{r['rank']}  {r['C4']}|{r['C2']}|{r['Q']}") for r in picks]
cells.append((al1['smiles'], f"Al-1  (#{al1['rank']})"))
cells.append((al2['smiles'], f"Al-2  (#{al2['rank']})"))
mols = [Chem.MolFromSmiles(s) for s, _ in cells]
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 220), legends=[t for _, t in cells])
img.save('/tmp/clivia_grid3.png')
sub = fig.add_axes([0.045, 0.015, 0.60, 0.30]); sub.axis('off')
sub.imshow(plt.imread('/tmp/clivia_grid3.png'))
sub.text(-0.04, 1.12, 'd', fontsize=13, fontweight='bold', transform=sub.transAxes)

# ---- e (competition dose-response) ----
ax = fig.add_subplot(gs[2, 2])
SRC = {'Al-1': '/Users/letiangao/Documents/AI-1-Compatative.xlsx',
       'Al-2': '/Users/letiangao/Documents/AI-2-Compatative.xlsx'}
COLOURS = {'Al-1': TEAL, 'Al-2': ORANGE}


def four_pl(x, b, t, ic50, h):
    return b + (t - b) / (1.0 + (x / ic50) ** h)


xs_fit = np.logspace(np.log10(0.005), np.log10(250), 300)
for name, path in SRC.items():
    df = pd.read_excel(path)
    df.columns = ['conc_nM', 'r1', 'r2', 'r3', 'n1', 'n2', 'n3', 'mean', 'sd']
    df = df.sort_values('conc_nM')
    xu = df['conc_nM'].values / 1000.0
    y = df['mean'].values
    sd = df['sd'].values
    popt, _ = curve_fit(four_pl, xu, y, p0=[0.05, 0.9, 3.0, 1.5],
                        bounds=([0.0, 0.5, 1e-4, 0.2], [0.5, 1.5, 1e3, 6.0]), maxfev=40000)
    ax.errorbar(xu, y, yerr=sd, fmt='o', ms=2.6, color=COLOURS[name], ecolor='#999999',
                elinewidth=0.5, capsize=1.0, zorder=4)
    ax.plot(xs_fit, four_pl(xs_fit, *popt), '-', color=COLOURS[name], lw=1.1, zorder=3,
            label=f"{name}: IC$_{{50}}$ = {popt[2]:.1f} µM")
ax.set_xscale('log')
ax.set_xlabel('Concentration (µM)', fontsize=8)
ax.set_ylabel('Normalised fluorescence', fontsize=8)
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(-0.02, 1.08)
ax.legend(frameon=False, fontsize=6.2, loc='lower left')
ax.text(-0.03, 1.07, 'e', transform=ax.transAxes, fontsize=13, fontweight='bold', ha='right')

fig.savefig(f'{OUT}/fig_clivia_chapter.png', dpi=220)
fig.savefig(f'{OUT}/fig_clivia_chapter.pdf')
print('wrote chapter figure with competition panel e')
print('Al-1 rank', al1['rank'], '| Al-2 rank', al2['rank'])
