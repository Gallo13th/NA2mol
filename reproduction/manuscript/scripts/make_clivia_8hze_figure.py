"""Figure for the 8HZE (Clivia aptamer) pocket-conditioned generation run.

Panels
  a  ECFP4 similarity of the 997 generated molecules to each target, with the top hits marked
  b  substructure coverage: how often each chemotype of the two targets appears at all
  c  property distributions next to the two targets
  d  smina minimisation score of the generated poses, with the blind-docked targets marked

Run from the workspace root after copying the analysis CSVs into docs/clivia_8hze/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs' / 'clivia_8hze'
SUFFIX = '' if D.name == 'clivia_8hze' else '_' + D.name

TEAL = '#246B70'
ORANGE = '#B5751F'
GREY = '#6E6E6E'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.55, 'ytick.major.width': 0.55,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'pdf.fonttype': 42,
})


def main() -> int:
    samples = pd.read_csv(D / 'samples.csv')
    summary = json.loads((D / 'summary.json').read_text())
    closest = pd.read_csv(D / 'closest_to_targets.csv')
    smin = pd.read_csv(D / 'smina_minimize.csv')
    dock = pd.read_csv(D / 'smina_dock_targets.csv')

    fig = plt.figure(figsize=(7.1, 4.5))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.30,
                          left=0.075, right=0.985, top=0.92, bottom=0.11)

    # ---- a: similarity distributions ----
    ax = fig.add_subplot(gs[0, 0])
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    bins = np.linspace(0, 0.7, 29)
    for name, colour in (('clivia_target1', TEAL), ('clivia_target2', ORANGE)):
        smi = summary['targets'][name]['smiles']
        tfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048)
        sims = np.array([
            DataStructs.TanimotoSimilarity(
                tfp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048))
            for s in samples['smiles']])
        ax.hist(sims, bins=bins, histtype='step', lw=1.1, color=colour,
                label=f'{name.replace("clivia_", "")}  max {sims.max():.2f}')
    ax.set_xlabel('ECFP4 Tanimoto to target')
    ax.set_ylabel('molecules')
    ax.legend(frameon=False, fontsize=5.6, loc='upper right')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(0.0, 1.03, f"n = {summary['n_molecules']:,} generated, {summary['n_unique']:,} unique", transform=ax.transAxes,
            fontsize=5.6, color='0.35', va='bottom')

    # ---- b: substructure coverage ----
    ax = fig.add_subplot(gs[0, 1])
    cover = summary['substructure_coverage']
    keys = ['quinoline', 'methoxy_aryl', 'phenol', 'dimethylamino', 'biphenyl', 'acrylonitrile']
    labels = ['quinoline', 'aryl-OMe', 'phenol', 'NMe$_2$', 'biphenyl', 'C=C-C#N']
    vals = [100 * cover[k] / summary['n_molecules'] for k in keys]
    bars = ax.barh(range(len(keys)), vals, color=[TEAL if v > 5 else '#C2453F' for v in vals], height=0.62)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f'{v:.1f}%', va='center', fontsize=5.6, color='0.3')
    ax.set_xlabel('molecules carrying the motif (%)')
    ax.set_xlim(0, max(vals) * 1.25)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- c: size / aromaticity, the two axes where the targets sit outside the generated set ----
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.42)
    for k, (col, lab, targets) in enumerate([
            ('heavy', 'heavy atoms', (27, 25)),
            ('arom_rings', 'aromatic rings', (4, 3))]):
        axc = fig.add_subplot(sub[0, k])
        vals = samples[col].values
        bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1)
        axc.hist(vals, bins=bins, color='0.72', lw=0)
        for t, colour, nm in zip(targets, (TEAL, ORANGE), ('t1', 't2')):
            axc.axvline(t, color=colour, lw=1.0, ls='--')
            axc.text(t, axc.get_ylim()[1] * 0.98, f' {nm}', color=colour, fontsize=5.4,
                     rotation=90, va='top', ha='left')
        axc.set_xlabel(lab, fontsize=6)
        if k == 0:
            axc.set_ylabel('molecules', fontsize=6)
        axc.tick_params(labelsize=5.6)
        for s_ in ('top', 'right'):
            axc.spines[s_].set_visible(False)

    # ---- d: affinity ----
    ax = fig.add_subplot(gs[1, 1])
    aff = smin['affinity'].dropna()
    ax.hist(aff, bins=np.linspace(-13, -3, 41), color='0.72', lw=0)
    top = ax.get_ylim()[1]
    for row, colour, lab in ((dock.iloc[0], TEAL, 'target 1'), (dock.iloc[1], ORANGE, 'target 2')):
        ax.axvline(row['affinity'], color=colour, lw=1.1, ls='--')
        ax.text(row['affinity'] + 0.12, top * 0.97, f"{lab}\n{row['affinity']:.1f}",
                color=colour, fontsize=5.4, va='top', ha='left')
    ax.set_xlabel('smina score (kcal mol$^{-1}$; lower is better)')
    ax.set_ylabel('generated molecules')
    ax.text(0.02, 0.95, f'median {aff.median():.2f}', transform=ax.transAxes,
            fontsize=5.6, color='0.3', va='top')
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)

    fig.text(0.008, 0.985, 'a', fontsize=8, fontweight='bold', va='top')
    fig.text(0.512, 0.985, 'b', fontsize=8, fontweight='bold', va='top')
    fig.text(0.008, 0.505, 'c', fontsize=8, fontweight='bold', va='top')
    fig.text(0.512, 0.505, 'd', fontsize=8, fontweight='bold', va='top')

    out = D / f'clivia_8hze_generation{SUFFIX}.png'
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix('.pdf'))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
