#!/usr/bin/env python3
"""Scaffold-selection figure using hard, reference-anchored criteria only (no weighted score)."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
TEAL, ORANGE = '#14615c', '#b5651d'
QUIN = 'c1ccc(-c2cc(-c3ccccc3)c3ccccc3n2)cc1'
JUNK = {'', 'C', 'c1ccccc1', 'C1CCCCC1'}

rows = []
for r in csv.DictReader(open('docs/clivia_8hze/cold10k_score.csv')):
    m = Chem.MolFromSmiles(r['smiles'])
    if m is None or int(r['radicals']) or abs(int(r['charge'])) > 1:
        continue
    try:
        r['pi'] = int(r['pi_atoms']); r['n_arom'] = int(r['n_arom'])
        r['mw'] = float(r['mw']); r['sa'] = float(r['sa_score']); r['tpsa'] = float(r['tpsa'])
        r['rot'] = float(r['rot']); r['logp'] = float(r['logp'])
    except ValueError:
        continue
    r['scaffold'] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    rows.append(r)

PI_MIN = 20          # the four Clivia crystal dyes all have a 20-atom sp2 system
def passes(r):
    return (r['pi'] >= PI_MIN and r['n_arom'] >= 2 and r['sa'] <= 4.0
            and 250 <= r['mw'] <= 600 and r['tpsa'] <= 120 and r['rot'] <= 9)

kept = [r for r in rows if passes(r)]
cnt = Counter(r['scaffold'] for r in kept if r['scaffold'] not in JUNK)
ranked = cnt.most_common()
rank = next(i for i, (s, _) in enumerate(ranked, 1) if s == QUIN)
print(f'library {len(rows)} -> kept {len(kept)} -> {len(cnt)} scaffolds; quinoline n={cnt[QUIN]}, rank {rank}')

fig = plt.figure(figsize=(11.6, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.28, left=0.07, right=0.975, top=0.92, bottom=0.16)

# a: pi-system size distribution with the dye-anchored threshold
ax = fig.add_subplot(gs[0, 0])
pi_all = [r['pi'] for r in rows]
ax.hist(pi_all, bins=np.arange(0, 34, 1), color='#c9d6d4', edgecolor='none', label='generated library')
ax.axvline(PI_MIN, color=ORANGE, lw=1.6)
ax.text(PI_MIN + 0.5, ax.get_ylim()[1] * 0.93, f'{PI_MIN}', fontsize=8, color=ORANGE, va='top', fontweight='bold')
ax.fill_betweenx([0, ax.get_ylim()[1]], PI_MIN, 34, color=ORANGE, alpha=0.08)
ax.set_xlabel('largest sp²-connected system (atoms)', fontsize=8)
ax.set_ylabel('molecules', fontsize=8)
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Size of the largest conjugated system', fontsize=8.5)

# b: scaffold repeat ranking among the retained molecules
ax = fig.add_subplot(gs[0, 1])
top = ranked[:12]
labels = [s for s, _ in top]; vals = [n for _, n in top]
y = np.arange(len(labels))[::-1]
colors = [ORANGE if s == QUIN else TEAL for s in labels]
ax.barh(y, vals, color=colors, height=0.72)
ax.set_yticks(y)
ax.set_yticklabels([f'#{i}' for i in range(1, len(labels) + 1)], fontsize=7)
ax.set_xlabel('number of molecules sharing the scaffold', fontsize=8)
ax.tick_params(axis='x', labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
for i, (s, v) in enumerate(top):
    ax.text(v + 0.15, y[i], str(v), va='center', fontsize=6.5, color='#333333')
    ax.text(v + 1.4, y[i], s if len(s) < 46 else s[:44] + '…', va='center', fontsize=5.6, color='#333333')
ax.set_xlim(0, max(vals) * 2.6)
ax.set_title(f'Most frequently generated scaffolds after the hard filters (n = {len(kept)} molecules)', fontsize=8.5)
fig.savefig('docs/clivia_8hze/figures/fig_scaffold_ranking.png', dpi=300)
fig.savefig('docs/clivia_8hze/figures/fig_scaffold_ranking.pdf')
print('wrote fig_scaffold_ranking.png/.pdf')
for i, (s, v) in enumerate(top[:6], 1):
    print(f'  #{i} n={v:3d} {s}')
